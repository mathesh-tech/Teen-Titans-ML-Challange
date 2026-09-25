import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import joblib
import json
import logging
import os

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self):
        self.params = {
            'objective': 'binary',
            'metric': 'auc',
            'learning_rate': 0.05,
            'num_leaves': 64,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }

    def train_and_evaluate(self, training_data_path: str, model_save_path: str, metrics_save_path: str) -> dict:
        logger.info(f"Loading training data from {training_data_path}...")
        
        # Load training dataset
        df = pd.read_csv(training_data_path)
        
        # Ensure we drop id columns
        exclude_cols = ['source1_entity_id', 'target_entity_id', 'label']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols]
        y = df['label']
        
        logger.info("Splitting dataset 80/20 for Train/Validation...")
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        logger.info(f"Training rows: {len(X_train):,}")
        logger.info(f"Validation rows: {len(X_val):,}")
        logger.info(f"Feature count: {len(feature_cols)}")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        logger.info("Training LightGBM model with optimized parameters...")
        model = lgb.train(
            self.params,
            train_data,
            num_boost_round=500,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False),
                       lgb.log_evaluation(period=50)]
        )
        
        logger.info(f"Saving model to {model_save_path}...")
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        joblib.dump(model, model_save_path)
        
        logger.info("Evaluating model on validation set...")
        y_prob = model.predict(X_val)
        y_pred = (y_prob >= 0.5).astype(int)
        
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_val, y_prob)
        
        logger.info(f"Accuracy:  {acc:.4f}")
        logger.info(f"Precision: {prec:.4f}")
        logger.info(f"Recall:    {rec:.4f}")
        logger.info(f"F1 Score:  {f1:.4f}")
        logger.info(f"ROC AUC:   {roc_auc:.4f}")
        
        metrics = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        logger.info(f"Saving metrics to {metrics_save_path}...")
        with open(metrics_save_path, 'w') as f:
            json.dump(metrics, f, indent=4)
            
        return metrics
