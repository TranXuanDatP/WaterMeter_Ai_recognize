"""
M1 Model Implementation

Implements YOLOv8-based watermeter detection model training and management.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import numpy as np

try:
    import torch
    from ultralytics import YOLO
    import mlflow
    import mlflow.pytorch
except ImportError as e:
    raise ImportError(
        f"Required dependencies not installed: {e}\n"
        "Please install: pip install torch ultralytics mlflow"
    )

from .config import M1_CONFIG, M1ModelConfig, MODEL_CARD, get_config


# Setup logging
logger = logging.getLogger(__name__)


class M1Model:
    """
    YOLOv8 Watermeter Detection Model

    Handles training, validation, and model management for M1 module.
    """

    def __init__(
        self,
        config: Optional[M1ModelConfig] = None,
        model_path: Optional[str] = None,
    ):
        """
        Initialize M1 model.

        Args:
            config: Model configuration (uses default if None)
            model_path: Path to existing model (for fine-tuning or inference)
        """
        self.config = config or M1_CONFIG
        self.model: Optional[YOLO] = None

        # Initialize model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading existing model from {model_path}")
            self.model = YOLO(model_path)
        else:
            logger.info(f"Initializing new {self.config.model_name} model")
            self.model = YOLO(self.config.model_name)

        # Setup MLflow if enabled
        if self.config.mlflow_enabled:
            self._setup_mlflow()

    def _setup_mlflow(self):
        """Configure MLflow tracking."""
        try:
            mlflow.set_experiment(self.config.mlflow_experiment_name)
            logger.info(f"MLflow experiment: {self.config.mlflow_experiment_name}")
        except Exception as e:
            logger.warning(f"MLflow setup failed: {e}")

    def train(
        self,
        data_yaml: str,
        project: str = "runs/detect",
        name: str = "m1_train",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Train YOLOv8 model on watermeter detection dataset.

        Args:
            data_yaml: Path to dataset.yaml configuration
            project: MLflow project name
            name: Run name
            **kwargs: Additional training arguments (overrides config)

        Returns:
            Training results dictionary

        Raises:
            ValueError: If data_yaml doesn't exist
            RuntimeError: If training fails
        """
        if not os.path.exists(data_yaml):
            raise ValueError(f"Dataset config not found: {data_yaml}")

        # Merge config with kwargs
        train_args = {
            "data": data_yaml,
            "epochs": self.config.epochs,
            "batch": self.config.batch_size,
            "imgsz": self.config.input_size,
            "optimizer": self.config.optimizer,
            "lr0": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "lrf": self.config.final_lr_fraction,
            "patience": self.config.patience,
            "project": project,
            "name": name,
            "device": self.config.device,
            "workers": self.config.num_workers,
            "pin_memory": self.config.pin_memory,
            "save": True,
            "save_period": self.config.checkpoint_interval,
            "verbose": True,
            "amp": True,  # Automatic Mixed Precision
        }
        train_args.update(kwargs)

        logger.info(f"Starting training with {train_args['epochs']} epochs")
        logger.info(f"Dataset: {data_yaml}")
        logger.info(f"Batch size: {train_args['batch']}")

        # Start MLflow run
        if self.config.mlflow_enabled:
            mlflow.start_run()
            mlflow.log_params(train_args)
            mlflow.log_text(MODEL_CARD, "model_card.txt")

        try:
            # Train model
            results = self.model.train(**train_args)

            # Log results
            metrics = self._extract_training_metrics(results)
            logger.info(f"Training completed. Final mAP@0.5: {metrics.get('maps50', 'N/A')}")

            if self.config.mlflow_enabled:
                mlflow.log_metrics(metrics)
                mlflow.pytorch.log_model(self.model, "model")

            return metrics

        except Exception as e:
            logger.error(f"Training failed: {e}")
            if self.config.mlflow_enabled:
                mlflow.log_param("error", str(e))
            raise RuntimeError(f"Training failed: {e}") from e

        finally:
            if self.config.mlflow_enabled:
                mlflow.end_run()

    def _extract_training_metrics(self, results) -> Dict[str, float]:
        """
        Extract key metrics from training results.

        Args:
            results: YOLO training results object

        Returns:
            Dictionary of metrics
        """
        try:
            # Results is a YOLO results object with metrics
            # Extract available metrics
            metrics = {}

            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
                metrics['maps50'] = results_dict.get('metrics/mAP50(B)', 0.0)
                metrics['maps95'] = results_dict.get('metrics/mAP50-95(B)', 0.0)
                metrics['precision'] = results_dict.get('metrics/precision(B)', 0.0)
                metrics['recall'] = results_dict.get('metrics/recall(B)', 0.0)
            else:
                # Fallback: try to access attributes directly
                metrics['maps50'] = getattr(results, 'maps50', 0.0)
                metrics['maps95'] = getattr(results, 'maps95', 0.0)
                metrics['precision'] = getattr(results, 'precision', 0.0)
                metrics['recall'] = getattr(results, 'recall', 0.0)

            return metrics

        except Exception as e:
            logger.warning(f"Failed to extract metrics: {e}")
            return {}

    def validate(
        self,
        data_yaml: str,
        batch: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Validate model on dataset.

        Args:
            data_yaml: Path to dataset.yaml
            batch: Batch size (uses config default if None)
            **kwargs: Additional validation arguments

        Returns:
            Validation metrics dictionary
        """
        if not os.path.exists(data_yaml):
            raise ValueError(f"Dataset config not found: {data_yaml}")

        val_args = {
            "data": data_yaml,
            "batch": batch or self.config.batch_size,
            "imgsz": self.config.input_size,
            "device": self.config.device,
            "verbose": True,
        }
        val_args.update(kwargs)

        logger.info(f"Validating on {data_yaml}")
        results = self.model.val(**val_args)

        # Extract metrics
        metrics = self._extract_training_metrics(results)
        logger.info(f"Validation mAP@0.5: {metrics.get('maps50', 'N/A')}")

        return metrics

    def export(
        self,
        format: str = "torchscript",
        output_path: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Export model to deployment format.

        Args:
            format: Export format (torchscript, onnx, etc.)
            output_path: Output path (auto-generated if None)
            **kwargs: Additional export arguments

        Returns:
            Path to exported model
        """
        if output_path is None:
            output_path = os.path.join(
                self.config.model_save_path,
                f"m1_watermeter_detection.{format}"
            )

        logger.info(f"Exporting model to {format} format: {output_path}")
        exported_path = self.model.export(
            format=format,
            **kwargs,
        )

        logger.info(f"Model exported to: {exported_path}")
        return exported_path

    def save(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Save path
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Model saved to: {path}")

    @staticmethod
    def load(path: str) -> "M1Model":
        """
        Load model from checkpoint.

        Args:
            path: Model path

        Returns:
            M1Model instance
        """
        logger.info(f"Loading model from: {path}")
        return M1Model(model_path=path)
