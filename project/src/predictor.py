"""
predictor.py
-------------
Responsible for running inference on new/unseen candidate pairs using a
trained model, and producing the final submission-ready output.

NOTE: This module currently contains only method stubs.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils import setup_logger


class Predictor:
    """
    Loads a trained model and generates predictions (match / no-match,
    plus confidence scores) for candidate pairs.
    """

    def __init__(self, model=None, logger: Optional[logging.Logger] = None):
        """
        Args:
            model: A pre-loaded trained model instance (optional; can also
                   be loaded later via load_model()).
            logger: Optional pre-configured logger.
        """
        self.logger = logger or setup_logger(self.__class__.__name__)
        self.model = model

    def load_model(self, model_path: Path):
        """
        Load a trained model from disk for inference.

        Args:
            model_path: Path to the saved model file.

        TODO: Implement model loading (mirrors ModelTrainer.load_model).
        """
        raise NotImplementedError("load_model() is not yet implemented.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate binary match / no-match predictions for a feature matrix.

        Args:
            X: Feature matrix for candidate pairs.

        Returns:
            Series of predicted labels (1 = match, 0 = non-match).

        TODO: Implement prediction logic.
        """
        self.logger.info("Predictor.predict() called - placeholder.")
        raise NotImplementedError("predict() is not yet implemented.")

    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate match-probability scores for a feature matrix.

        Args:
            X: Feature matrix for candidate pairs.

        Returns:
            Series of predicted match probabilities.

        TODO: Implement probability prediction logic.
        """
        raise NotImplementedError("predict_proba() is not yet implemented.")

    def generate_submission(self, predictions: pd.DataFrame, output_path: Path):
        """
        Format predictions into the competition's required submission
        format and write to disk.

        Args:
            predictions: DataFrame of predictions (e.g., pair IDs + labels).
            output_path: Destination file path for the submission file.

        TODO: Implement submission file formatting/writing.
        """
        raise NotImplementedError("generate_submission() is not yet implemented.")
