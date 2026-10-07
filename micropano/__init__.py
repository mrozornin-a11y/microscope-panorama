"""Panorama (mosaic) of a thin / polished section from a microscope video."""

from .config import Settings
from .fov import FieldOfView, detect_fov
from .pipeline import Pipeline, Result

__all__ = ["Settings", "FieldOfView", "detect_fov", "Pipeline", "Result"]
__version__ = "0.1.0"
