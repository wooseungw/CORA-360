"""Processors: image (panorama crop strategies) and text (chat template formatting)."""

from cora.processors.images import PanoramaImageProcessor
from cora.processors.processor import PanoramaProcessor
from cora.processors.text import UniversalTextFormatter

__all__ = [
    "PanoramaImageProcessor",
    "PanoramaProcessor",
    "UniversalTextFormatter",
]
