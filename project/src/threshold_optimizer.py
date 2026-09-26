import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class ThresholdOptimizer:
    """
    Evaluates probability thresholds from 0.1 to 0.95 to maximize F1 Score
    for Machine Learning models, balancing Precision and Recall.
    """
    def __init__(self):
        # Evaluate thresholds from 0.1 to 0.95
        self.thresholds = np.arange(0.1, 0.96, 0.05)

    def optimize(self, y_true: np.ndarray, y_prob: np.ndarray, 
                 save_path: str = 'output/best_threshold.json') -> dict:
        """
        Iterates through probability thresholds, calculates precision, recall, F1, and ROC AUC.
        Identifies the optimal threshold maximizing F1 and saves the results.
        """
        logger.info("Starting probability threshold optimization...")
        
        best_threshold = 0.50
        best_f1 = -1.0
        best_metrics = {}
        all_results = []
        
        # Calculate ROC AUC globally
        roc_auc = roc_auc_score(y_true, y_prob)
        logger.info(f"Global ROC AUC Score: {roc_auc:.4f}")
        
        logger.info(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
        logger.info("-" * 50)
        
        for thresh in self.thresholds:
            y_pred = (y_prob >= thresh).astype(int)
            
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            logger.info(f"{thresh:<10.2f} | {precision:<10.4f} | {recall:<10.4f} | {f1:<10.4f}")
            
            result = {
                'threshold': float(thresh),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
            all_results.append(result)
            
            # Maximize F1 Score
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
                best_metrics = result
                
        logger.info("-" * 50)
        logger.info(f"Optimization complete. Best Threshold: {best_threshold:.2f} yielding an F1 Score of {best_f1:.4f}")
        
        output_data = {
            'best_threshold': float(best_threshold),
            'global_roc_auc': float(roc_auc),
            'best_metrics': best_metrics,
            'all_threshold_results': all_results
        }
        
        logger.info(f"Saving optimal threshold configuration to {save_path}...")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(output_data, f, indent=4)
            
        logger.info("Threshold optimization results saved successfully.")
        return output_data
