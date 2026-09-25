"""
utils.py
--------
Shared utilities for the Business Entity Resolution pipeline:
    - Logging setup
    - Configuration loading / management
    - Common helper functions used across modules

Keeping these concerns centralized avoids duplicating logging/config
boilerplate in every module.
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

def setup_logger(name: str = "entity_resolution",
                  level: int = logging.INFO,
                  log_file: Optional[str] = None) -> logging.Logger:
    """
    Create and configure a logger with a consistent format across the project.

    Args:
        name: Logger name (usually __name__ of the calling module).
        level: Logging level (e.g., logging.INFO, logging.DEBUG).
        log_file: Optional path to a file where logs should also be written.
                  If None, logs only go to stdout.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if the logger is fetched multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass
class ProjectConfig:
    """
    Central configuration object for the entity resolution pipeline.

    This dataclass holds all paths and pipeline-wide settings in one place
    so that individual modules don't hardcode paths. Extend this with
    model hyperparameters, thresholds, etc. as the project grows.
    """

    # --- Directory paths ---
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    dataset_dir: Path = field(init=False)
    output_dir: Path = field(init=False)
    models_dir: Path = field(init=False)

    # --- File names (defaults; override as needed) ---
    train_file: str = "train.tsv"
    test_file: str = "test.tsv"

    # --- Logging ---
    log_level: int = logging.INFO
    log_file: Optional[str] = None

    # --- Misc pipeline settings (placeholders for future use) ---
    random_seed: int = 42

    def __post_init__(self):
        self.dataset_dir = self.project_root / "dataset"
        self.output_dir = self.project_root / "output"
        self.models_dir = self.project_root / "models"

        # Ensure required directories exist
        for directory in (self.dataset_dir, self.output_dir, self.models_dir):
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def train_path(self) -> Path:
        """Full path to the training TSV file."""
        return self.dataset_dir / self.train_file

    @property
    def test_path(self) -> Path:
        """Full path to the test TSV file."""
        return self.dataset_dir / self.test_file


def load_config(**overrides) -> ProjectConfig:
    """
    Factory function to build a ProjectConfig, optionally overriding defaults.

    Args:
        **overrides: Any ProjectConfig field to override (e.g., train_file="my_train.tsv").

    Returns:
        A ProjectConfig instance.
    """
    return ProjectConfig(**overrides)
