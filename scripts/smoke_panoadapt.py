#!/usr/bin/env python3
"""Smoke test: verify PanoAdapt hang fix."""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import signal, sys, time

TIMEOUT = 300
t0 = time.time()


def timeout_handler(signum, frame):
    print(f"\n❌ TIMEOUT after {TIMEOUT}s", flush=True)
    sys.exit(1)


signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(TIMEOUT)


def cp(msg):
    print(f"  [{time.time()-t0:6.1f}s] {msg}", flush=True)


print("=" * 60, flush=True)
print("PanoAdapt Smoke Test (after hang fix)", flush=True)
print("=" * 60, flush=True)

cp("Loading config...")
import yaml
from cora.baseline.config import BaselineConfig

with open("configs/baseline/panoadapt_internvl35_2b.yaml") as f:
    cfg_dict = yaml.safe_load(f)

cfg_dict.setdefault("data", {})["max_train_samples"] = 4
cfg_dict["training"]["dataloader_num_workers"] = 0
config = BaselineConfig(**cfg_dict)

cp("Loading model...")
from cora.baseline.models import BaselineModelRegistry

model, processor, tokenizer = BaselineModelRegistry.load_model(config.model)

cp("Applying LoRA...")
from peft import LoraConfig, TaskType, get_peft_model

targets = BaselineModelRegistry.get_default_lora_targets(config.model.model_type)
peft_config = LoraConfig(
    r=32, lora_alpha=64, lora_dropout=0.1,
    target_modules=targets, task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_config)
model.enable_input_require_grads()

cp("Building dataset...")
from cora.baseline.finetune import VLMDataset, _collate_causal

train_ds = VLMDataset(
    csv_path=config.data_train_csv, processor=processor, tokenizer=tokenizer,
    model_type=config.model.model_type, image_column=config.data.image_column,
    instruction_column=config.data.instruction_column,
    response_column=config.data.response_column,
    pano_view_config=config.effective_pano_view, max_samples=4,
)
collate_fn = _collate_causal(processor, tokenizer, max_length=config.training.max_length)

cp("Creating TrainingArguments...")
from cora.baseline.finetune import _resolve_precision
from transformers import TrainingArguments

prec = _resolve_precision(config.training.mixed_precision)
training_args = TrainingArguments(
    output_dir="/tmp/smoke_panoadapt/checkpoints",
    num_train_epochs=1, per_device_train_batch_size=1,
    gradient_accumulation_steps=1, learning_rate=5e-5,
    fp16=prec["fp16"], bf16=prec["bf16"],
    gradient_checkpointing=True, logging_steps=1, save_strategy="no",
    report_to=[], seed=42, remove_unused_columns=False,
    dataloader_num_workers=0, dataloader_pin_memory=False, max_steps=2,
)

trainer_kwargs = dict(
    model=model, args=training_args, train_dataset=train_ds,
    eval_dataset=None, processing_class=tokenizer, data_collator=collate_fn,
)

cp("Creating _PanoAdaptTrainer (was hanging before fix)...")
from cora.baseline.finetune import _PanoAdaptTrainer

pa_cfg = config.panoadapt
trainer = _PanoAdaptTrainer(
    panoadapt_config=pa_cfg,
    pano_view_config=config.effective_pano_view,
    **trainer_kwargs,
)
cp("✅ _PanoAdaptTrainer created OK — hang fix works!")

cp("Training for 2 steps...")
trainer.train()
cp("✅ Training completed!")

trainer.cleanup()
signal.alarm(0)
print("=" * 60, flush=True)
print("SMOKE TEST PASSED ✅", flush=True)
print("=" * 60, flush=True)
