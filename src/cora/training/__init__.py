"""Training infrastructure: 3-stage progressive trainer, Lightning module, losses, callbacks."""

from .trainer import CORATrainer
from .module import PanoramaTrainingModule
from .losses import DenseCLLoss, GlobalLocalLoss, PanoContrastiveLoss, VICRegLoss
from .callbacks import MetadataCallback

__all__ = [
    "CORATrainer",
    "PanoramaTrainingModule",
    "DenseCLLoss",
    "GlobalLocalLoss",
    "PanoContrastiveLoss",
    "VICRegLoss",
    "MetadataCallback",
]
