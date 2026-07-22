"""Shared utilities for the EconML and CausalML marketing benchmark."""

from .data import FEATURE_COLUMNS, generate_coupon_retention_data, load_or_create_data
from .evaluation import evaluate_cate_predictions
from .hillstrom import load_hillstrom_raw, load_or_create_hillstrom_contrast
from .rct_evaluation import evaluate_rct_predictions, randomized_difference

__all__ = [
    "FEATURE_COLUMNS",
    "evaluate_cate_predictions",
    "generate_coupon_retention_data",
    "load_hillstrom_raw",
    "load_or_create_data",
    "load_or_create_hillstrom_contrast",
    "evaluate_rct_predictions",
    "randomized_difference",
]
