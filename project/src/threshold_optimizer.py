import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, fbeta_score

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """
    Evaluates a specific sequence of probability thresholds to maximize the F0.5 Score
    for Machine Learning models, prioritizing Precision over Recall.
    """
    def __init__(self):
        # The exact sequence of thresholds requested
        self.thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    def optimize(self, y_true: np.ndarray, y_prob: np.ndarray, 
                 save_path: str = 'output/best_threshold.json') -> dict:
        """
        Iterates through probability thresholds, calculates precision, recall, and F0.5.
        Identifies the optimal threshold maximizing F0.5 and saves the results.
        
        Args:
            y_true: True binary labels (1 or 0) from the validation set
            y_prob: Predicted probability of class 1 output by the model
            save_path: Location to save the JSON output
            
        Returns:
            Dictionary containing the best threshold and all associated metrics.
        """
        logger.info("Starting probability threshold optimization...")
        
        best_threshold = 0.50
        best_f05 = -1.0
        best_metrics = {}
        all_results = []
        
        for thresh in self.thresholds:
            # Generate binary predictions based on the current threshold
            y_pred = (y_prob >= thresh).astype(int)
            
            # Compute evaluation metrics safely avoiding division by zero
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f05 = fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
            
            logger.info(f"Threshold: {thresh:.2f} -> Precision: {precision:.4f} | Recall: {recall:.4f} | F0.5: {f05:.4f}")
            
            result = {
                'threshold': float(thresh),
                'precision': float(precision),
                'recall': float(recall),
                'f0.5_score': float(f05)
            }
            all_results.append(result)
            
            # Track the configuration that yields the highest F0.5 Score
            if f05 > best_f05:
                best_f05 = f05
                best_threshold = thresh
                best_metrics = result
                
        logger.info(f"Optimization complete. Best Threshold: {best_threshold:.2f} yielding an F0.5 Score of {best_f05:.4f}")
        
        # Prepare the final structured output
        output_data = {
            'best_threshold': float(best_threshold),
            'best_metrics': best_metrics,
            'all_threshold_results': all_results
        }
        
        # Persist the configuration and metrics to disk
        logger.info(f"Saving optimal threshold configuration to {save_path}...")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        logger.info("Threshold optimization results saved successfully.")
        return output_data
