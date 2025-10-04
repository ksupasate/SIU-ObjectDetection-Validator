"""
SIU (Structured Instance Understanding) Object Detection Validator

This package implements the complete pipeline for validating object detection
results using geometric relationship analysis as described in the research paper:
"Structured Instance Understanding with Boundary Box Relationships in Object Detection System"
"""

__version__ = "1.0.0"
__author__ = "SIU Implementation"

from . import utils
from . import feature_engineering
from . import data_synthesis
from . import train
from . import inference

__all__ = [
    "utils",
    "feature_engineering",
    "data_synthesis",
    "train",
    "inference",
]
