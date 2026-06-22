# CORA 모델 아키텍처

## 개요

**CORA** (Contrastive Overlap Representation Alignment)는 360° 파노라마 이미지에 특화된 Vision-Language Model이다. 핵심 아이디어는 인접 뷰 간 **겹치는 영역(overlap)**의 표현을 정합(align)하여, 파노라마의 공간적 연속성을 학습하는 것이다.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         CORA Architecture                            │
│                                                                      │
│  ERP Image (360°)                                                    │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────┐    9 views (1 global + 8 tiles)                     │
│  │ E2P Tiling  │──────────────────────────────────┐                  │
│  └─────────────┘                                  │                  │
│       │                                           │                  │
│       ▼                                           ▼                  │
│  ┌──────────────┐  [B×9, 256, 1152]   ┌──────────────────────────┐   │
│  │ SigLIP2      │────────────────────▶│ Resampler (BiMamba)       │   │
│  │ (frozen)     │                     │ [B×9, 256, 1024]          │   │
│  └──────────────┘                     └────────┬─────────────────┘   │
│                                                │                     │
│                     ┌──────────────────────────┼──────────┐          │
│                     │ Stage 1                  │ Stage 2,3│          │
│                     ▼                          ▼          │          │
│              ┌──────────────┐          ┌──────────────┐   │          │
│              │ VICReg Proj  │          │ PanoramaProj │   │          │
│              │ [B×8, 256, D]│          │ (PE+Stitch)  │   │          │
│              └──────┬───────┘          │ [B, T, 1024] │   │          │
│                     │                  └──────┬───────┘   │          │
│                     ▼                         ▼           │          │
│              ┌──────────────┐          ┌──────────────┐   │          │
│              │ Overlap Loss │          │ LanguageFusion│   │          │
│              │ (VICReg/     │          │ (<|vision|>)  │   │          │
│              │  InfoNCE/    │          └──────┬───────┘   │          │
│              │  DenseCL)    │                 ▼           │          │
│              └──────────────┘          ┌──────────────┐   │          │
│                                        │ Qwen3-0.6B   │   │          │
│                                        │ (LoRA)       │   │          │
│                                        └──────────────┘   │          │
│                                                           │          │
└───────────────────────────────────────────────────────────┘──────────┘
```

## 입력 처리

### ERP → E2P 타일링

360° ERP(Equirectangular Projection) 이미지를 9개의 뷰로 분할:

```
┌─────────────────────────────────────────────────┐
│                  ERP Image (360°)                │
│                                                   │
│  View 1   View 2   View 3   ...   View 8         │
│  ←──45°──→                                        │
│       ←──overlap──→                               │
│            (50%)                                  │
└─────────────────────────────────────────────────┘

생성되는 뷰:
  - Global view: ERP 전체를 256×256으로 리사이즈 (1개)
  - E2P tiles: pitch=0°, yaw=0°/45°/90°/.../315° (8개)
    - FOV=90°, stride=45° → overlap=50%
    - 각 타일 256×256 pixels
```

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `crop_strategy` | `anyres_e2p` | ERP→Perspective 투영 |
| `fov_deg` | 90° | 각 타일의 시야각 |
| Stride | 45° | 인접 타일 간격 |
| `overlap_ratio` | 0.5 (50%) | 물리적 겹침 비율 |
| Views | 9 | 1 global + 8 tiles |
| Tile size | 256×256 | SigLIP2 입력 해상도 |

## 컴포넌트 상세

### 1. Vision Encoder — `VisionBackbone`

| 항목 | 설정 |
|------|------|
| **모델** | `google/siglip2-so400m-patch16-256` |
| **파라미터** | ~400M |
| **패치 크기** | 16×16 pixels |
| **출력 그리드** | 16×16 = 256 patches per view |
| **출력 차원** | 1152 |
| **상태** | 기본 frozen, Stage 1에서 마지막 2 block만 unfreeze |

```
입력: [B×9, 3, 256, 256]
출력: [B×9, 256, 1152]
       ─────  ───  ────
       views  S    D_vision
```

**역할**: 각 뷰를 독립적으로 인코딩하여 패치 수준의 시각 특징을 추출한다. Pretrained 가중치를 최대한 보존하면서, 마지막 2개 block만 학습하여 파노라마 도메인에 미세 적응한다.

파일: `src/cora/model/vision_encoder.py`

### 2. Resampler — `BiMambaResampler` / `MLPResampler`

Vision features를 latent space로 변환하는 모듈. **토큰 수는 유지**하면서 차원만 변환한다.

#### BiMamba (기본)

```
Input Proj: Linear(1152 → 1024)
    ↓
BiMamba Block ×4:
    ├── Forward Mamba SSM ──→ ─┐
    │                          │ average
    └── Backward Mamba SSM ←── ┘
        + Residual + LayerNorm
    ↓
Final LayerNorm
    ↓
Output Proj: Linear(1024 → 1024)
```

| 항목 | 설정 |
|------|------|
| **hidden_dim** | 1024 |
| **num_layers** | 4 |
| **d_state** | 64 (SSM state dimension) |
| **d_conv** | 4 (convolution kernel) |
| **expand** | 2.0 |
| **파라미터** | ~66M |

**핵심 차별점**: MLP는 각 토큰을 독립적으로 변환하지만, BiMamba는 **양방향 SSM**으로 같은 뷰 내 256개 토큰 간의 공간적 의존성을 포착한다.

#### MLP (비교군)

```
Linear(1152 → 1536) → LayerNorm → GELU
    ↓
Linear(1536 → 1536) → LayerNorm → GELU
    ↓
Linear(1536 → 1024)
```

**특징**: LLaVA-1.5, InternVL2 등 주류 VLM에서 채택하는 표준 구조. 토큰 간 상호작용 없음.

#### C-Abstractor (비교군, 구현 완료)

```
Input Proj: Linear(1152 → 1024)
    ↓
Reshape to 2D: [BV, 1024, 16, 16]
    ↓
Stage 1: SpatialBlock ×3
    ├── DepthwiseConv2d(k=7) — 로컬 공간 패턴
    ├── LayerNorm2d
    ├── PointwiseConv(1024 → 4096) → SiLU
    ├── SE Attention — 채널 재보정
    └── PointwiseConv(4096 → 1024) + Residual
    ↓
[Optional: AdaptiveAvgPool2d — 토큰 압축]
    ↓
Stage 2: SpatialBlock ×3
    ↓
Reshape back: [BV, 256, 1024]
    ↓
LayerNorm → MLP Readout
```

**핵심 차별점**: Depthwise convolution + Squeeze-and-Excitation으로 2D 공간 구조를 직접 활용한다. Cambrian-1에서 채안.

파일: `src/cora/model/resampler/`

```
입력: [B×9, 256, 1152]
출력: [B×9, 256, 1024]
       ─────  ───  ────
       views  S    D_latent
```

### 3. VICReg Projector — `VICRegProjector`

Stage 1에서 **self-supervised loss 계산**을 위한 projection head.

```
Linear(1024 → 1024) → LayerNorm → GELU [→ Dropout]
    ↓
Linear(1024 → 1024)
```

| 항목 | 설정 |
|------|------|
| **depth** | 2 layers |
| **dropout** | 0.0 (VICReg) / 0.1 (Contrastive) |
| **파라미터** | ~2M |

**Dropout 역할** (Contrastive 모드): 같은 features에 dropout mask를 2번 적용하여 두 개의 stochastic view를 생성 → InfoNCE의 positive pair로 사용.

파일: `src/cora/model/projectors.py`

### 4. PanoramaProjector — `PanoramaProjector`

Resampler 출력을 **LLM 입력 공간**으로 변환하고, 멀티뷰를 **하나의 시퀀스**로 합치는 모듈. Stage 2, 3에서 사용.

```
[B×V, S, D_latent]
    ↓
① Panorama PE (yaw + spatial sinusoidal, additive)
    ↓
② Linear(D_latent → D_lm)
    ↓
③ View Stitching (overlap strip 제거 후 concat)
    ↓
[B, T, D_lm]
```

#### ① Positional Encoding — `PanoramaPositionalEncoding`

```
Yaw Encoding:
  각 뷰의 글로벌 경도(longitude) 위치를 인코딩.
  overlap_ratio를 반영하여 겹치는 열은 동일한 PE 값을 가짐.

  view 0: yaw = [0°, ..., 90°]
  view 1: yaw = [45°, ..., 135°]  ← 좌측 절반이 view 0의 우측 절반과 동일
  ...
  view 7: yaw = [315°, ..., 405°=45°]  ← 360° wrap-around

Spatial Encoding:
  뷰 내부의 2D 그리드 위치 (row + column).
  글로벌 column 좌표를 사용하여 yaw continuity 보장.

최종 PE = Yaw + Spatial (additive)
```

**핵심 설계**: 같은 물리적 위치를 가리키는 **인접 뷰의 겹치는 열**에 **동일한 PE 값**이 할당된다. 이로써 모델이 공간적 일관성을 유지할 수 있다.

#### ③ View Stitching (`stride_views` 모드)

```
8 tiles, 각 16×16, overlap k=8 columns

View 0: [col 0 ~ 15]  전체 사용
View 1: [col 8 ~ 15]  앞 8열(overlap) 제거
View 2: [col 8 ~ 15]  앞 8열 제거
...
View 7: [col 8 ~ 15]  앞 8열 제거

결과: 16 + 8×7 = 72 unique columns → 16(H) × 72(W) = 1,152 tokens
+ Global view: 256 tokens
= 총 1,408 vision tokens → LLM에 입력
```

파일: `src/cora/model/projectors.py`, `src/cora/model/positional.py`

### 5. Language Model — `LanguageModel`

| 항목 | 설정 |
|------|------|
| **모델** | `Qwen/Qwen3-0.6B` |
| **hidden_size** | 1024 |
| **파라미터** | ~600M |
| **Attention** | SDPA (Scaled Dot-Product Attention) |
| **LoRA** | r=32, α=64, dropout=0.1 |
| **LoRA target** | q, k, v, o, gate, up, down proj |
| **LoRA 학습 파라미터** | ~2M |

**특수 토큰**: `<|vision|>` — vision token이 삽입될 위치를 표시.

파일: `src/cora/model/language_model.py`

### 6. Language Fusion — `LanguageFusion`

Vision tokens을 text token stream에 삽입하는 유틸리티.

```
Text:    [BOS] <|vision|> Describe the panorama [EOS]
                   ↓
         ┌─────────────────┐
         │ Vision Tokens   │  [1,408 tokens, D_lm]
         │ (from Projector)│
         └─────────────────┘
                   ↓
Fused:   [BOS] [v1][v2]...[v1408] Describe the panorama [EOS]

Labels:  [-100][-100]...[-100]    Describe the panorama [EOS]
         ──────────────────────   ─────────────────────────────
         vision tokens: masked    text tokens: computed loss
```

파일: `src/cora/model/language_fusion.py`

## 3-Stage Progressive Training

### Stage 별 Freeze/Unfreeze 전략

```
Component              Stage 1 (Vision)    Stage 2 (Resampler)    Stage 3 (Finetune)
─────────────────────  ─────────────────   ────────────────────   ──────────────────
Vision Encoder         ❄️ (last 2: 🔥)    ❄️                     ❄️
Resampler              🔥                  🔥 (low lr)            ❄️
VICReg Projector       🔥                  ❄️ (gradient 통과)     ❄️
PanoramaProjector      ❄️                  🔥                     🔥
LLM                    ❄️                  ❄️                     🔥 (LoRA)
```

### Stage 1: Vision Alignment

**목적**: Resampler가 인접 뷰의 겹치는 영역에서 **동일한 표현**을 출력하도록 학습.

```
입력: ERP 이미지 → 9 views
경로: Vision Encoder → Resampler → VICReg Projector → Loss

Loss 계산 (global view 제외, tiles만):
  - 인접 타일 쌍 (v0,v1), (v1,v2), ..., (v7,v0)
  - 각 쌍에서 overlap strip 추출 (k=4 columns, 25%)

  View i:  [...  col12  col13  col14  col15]  ← 오른쪽 k열
  View i+1:[col0   col1   col2   col3  ...]   ← 왼쪽 k열
            ─────────────────────────────────
            이 두 strip이 같아져야 함
```

**Loss 선택지** (비교 실험):

| Loss | 수식 | 특징 |
|------|------|------|
| VICReg (batchwise) | `25·inv + 25·var(dim=0) + 1·cov` | 배치 전체 통계 → gradient 소실 |
| VICReg (pairwise) | `25·inv + 25·var(dim=1) + 1·cov` | 쌍별 공간 다양성 강제 |
| InfoNCE | `−log(exp(sim⁺/τ) / Σexp(sim/τ))` | 부정 쌍 대비 + within-tile loss |
| DenseCL | overlap-only InfoNCE | 가장 단순, tile loss 없음 |

**핵심 발견**: Pretrained features 위의 alignment 문제이므로 **1 epoch이면 충분**. 추가 epoch은 collapse 유발.

### Stage 2: Resampler + LM Alignment

**목적**: PanoramaProjector가 vision tokens을 LLM 입력 공간에 정렬하도록 학습. 동시에 VICReg loss로 Resampler의 공간 일관성을 유지.

```
입력: ERP 이미지 + 텍스트 (query + annotation)

Branch A (VICReg): Resampler → VICReg Proj (frozen) → Loss (weight=0.1)
Branch B (LM):     Resampler → PanoramaProjector → Fusion → LLM → CE Loss

Total Loss = LM_loss + 0.1 × VICReg_loss
```

| 항목 | 설정 |
|------|------|
| **Epochs** | 1 |
| **VICReg weight** | 0.1 |
| **LR** | 1e-4 |
| **Accumulate** | 8 steps |

### Stage 3: Finetune (LoRA)

**목적**: LLM이 panorama-specific 캡션을 생성하도록 LoRA로 fine-tune.

```
입력: ERP 이미지 + 텍스트

경로: Vision Encoder → Resampler (frozen) → PanoramaProjector → Fusion → LLM (LoRA)

Loss = Cross-Entropy (autoregressive, vision tokens masked)
```

| 항목 | 설정 |
|------|------|
| **Epochs** | 1 |
| **LR** | 2e-6 |
| **LoRA rank** | 32 |
| **LoRA alpha** | 64 |

## 파라미터 요약

| Component | Params | Stage 1 | Stage 2 | Stage 3 |
|-----------|-------:|:---:|:---:|:---:|
| Vision Encoder (SigLIP2) | ~400M | last 2 blocks 🔥 | ❄️ | ❄️ |
| Resampler (BiMamba) | ~66M | 🔥 | 🔥 | ❄️ |
| VICReg Projector | ~2M | 🔥 | ❄️ | ❄️ |
| PanoramaProjector | ~1M | ❄️ | 🔥 | 🔥 |
| LLM (Qwen3-0.6B) | ~600M | ❄️ | ❄️ | LoRA ~2M 🔥 |
| **Total** | **~1.07B** | **~68M 🔥** | **~67M 🔥** | **~3M 🔥** |

## Tensor Shape 흐름

```
ERP Image: [B, 3, H_erp, W_erp]
    ↓ E2P tiling
Pixel Values: [B, 9, 3, 256, 256]
    ↓ flatten views
Vision Input: [B×9, 3, 256, 256]
    ↓ SigLIP2
Vision Features: [B×9, 256, 1152]
    ↓ BiMamba Resampler
Resampled: [B×9, 256, 1024]
    │
    ├─── Stage 1 ────────────────────────────────
    │    ↓ VICReg Projector
    │    VICReg Features: [B×8, 256, 1024]  (global 제외)
    │    ↓ overlap strip 추출 (k=4)
    │    curr/nxt: [B×8, 64, 1024]  (H=16, k=4 → L=64)
    │    ↓ Loss 계산
    │
    └─── Stage 2,3 ──────────────────────────────
         ↓ 분리: global [B, 256, 1024] + tiles [B×8, 256, 1024]
         ↓ PanoramaProjector (PE + Linear + Stitch)
         Tile tokens: [B, 1152, 1024]  (stitched)
         Global tokens: [B, 256, 1024]
         ↓ concat
         Vision Tokens: [B, 1408, 1024]
         ↓ LanguageFusion (replace <|vision|>)
         Fused Embeddings: [B, 1408+L_text, 1024]
         ↓ Qwen3-0.6B
         Output Logits: [B, 1408+L_text, vocab_size]
```

## 데이터

### QuIC-360 Dataset

| Split | Samples | 용도 |
|-------|--------:|------|
| Train | ~5,300 | Stage 2, 3 학습 |
| Valid | ~530 | 검증 |
| Test | 5,349 | 최종 평가 |

### Stage 1 Data (자체 구성)

Stage 1은 이미지만 필요 (텍스트 불필요). QuIC-360 train split의 이미지를 Stage 1용 CSV로 변환하여 사용.

| Split | Samples |
|-------|--------:|
| stage1_train | ~5,300 |
| stage1_val | ~530 |

## 평가 지표

### Stage 1 진단 지표

| Metric | 의미 | 이상적 값 |
|--------|------|-----------|
| `val_adj_cos` | overlap 영역 cosine similarity | → 1.0 |
| `val_adj_mse` | overlap 영역 MSE | → 0.0 |
| `val_overlap_ret_acc` | overlap retrieval accuracy | → 1.0 |
| `val_hungarian_acc` | 위치 구분 능력 (Hungarian matching) | → 1.0 |
| `val_feat_std` | feature 다양성 (collapse 감지) | 높을수록 good |
| `val_eff_rank` | effective rank (표현 차원 활용도) | 높을수록 good |
| `val_r_eff_rank` | resampler output effective rank | 높을수록 good |

### 최종 평가 지표

| Metric | 출처 |
|--------|------|
| BLEU-4 | sacrebleu |
| METEOR | NLTK |
| ROUGE-L | rouge-score |
| CIDEr | pycocoevalcap |
| SPICE | pycocoevalcap |

## 파일 구조

```
src/cora/
├── model/
│   ├── vlm.py               # PanoramaVLM (전체 오케스트레이터)
│   ├── vision_encoder.py     # VisionBackbone (SigLIP2 래퍼)
│   ├── language_model.py     # LanguageModel (Qwen3 + LoRA)
│   ├── language_fusion.py    # LanguageFusion (<|vision|> 토큰 교체)
│   ├── projectors.py         # VICRegProjector, PanoramaProjector
│   ├── positional.py         # PanoramaPositionalEncoding
│   └── resampler/
│       ├── __init__.py       # build_resampler() 팩토리
│       ├── resamplers.py     # MLP, QFormer, Identity, AvgPool, Conv
│       ├── bimamba.py        # BiMambaResampler
│       ├── c_abstractor.py   # CAbstractorResampler (spatial-aware)
│       ├── perceiver.py      # PerceiverResampler (Flamingo-style)
│       ├── spatial_pool.py   # SpatialPoolResampler
│       └── masked_drop.py    # MaskedDropResampler
├── training/
│   ├── module.py             # PanoramaTrainingModule (Lightning)
│   ├── trainer.py            # CORATrainer (3-stage 관리)
│   ├── losses.py             # VICRegLoss, PanoContrastiveLoss, DenseCLLoss
│   ├── autobatch.py          # GPU 메모리 기반 자동 배치
│   └── callbacks.py          # MetadataCallback
├── data/
│   ├── dataset.py            # PanoramaDataset (CSV 기반)
│   └── datamodule.py         # PanoramaDataModule (Lightning)
├── processors/
│   ├── processor.py          # PanoramaProcessor
│   ├── image.py              # PanoramaImageProcessor (E2P 타일링)
│   └── text.py               # UniversalTextFormatter
├── config/
│   ├── schema.py             # Pydantic config 스키마
│   └── manager.py            # ConfigManager (YAML 로드)
└── baseline/
    └── finetune.py           # 베이스라인 VLM 학습/평가
```
