"""
AutoML with Optuna

Automated hyperparameter optimization for all models
"""
import logging
from typing import Dict, Any, Callable
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np


class OptunaOptimizer:
    """
    AutoML hyperparameter optimization using Optuna
    
    Optimizes:
    - XGBoost acceptance model
    - Neural network architectures
    - Causal ML models
    """
    
    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        self.logger = logging.getLogger("EAC.Models.AutoML")
        self.n_trials = n_trials
        self.timeout = timeout
        
        self.study = None
        self.best_params = None
        
        self.logger.info(f"Optuna Optimizer initialized: {n_trials} trials, {timeout}s timeout")
    
    def optimize_xgboost(
        self,
        X_train, y_train,
        X_val, y_val,
        metric: str = "auc"
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            metric: Optimization metric
            
        Returns:
            Best hyperparameters
        """
        self.logger.info("Optimizing XGBoost hyperparameters...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2.0),
                'random_state': 42
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            
            if metric == "auc":
                from sklearn.metrics import roc_auc_score
                y_pred = model.predict_proba(X_val)[:, 1]
                score = roc_auc_score(y_val, y_pred)
            else:
                score = model.score(X_val, y_val)
            
            return score
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        self.study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        self.best_params = self.study.best_params
        
        self.logger.info(f"Optimization complete: best {metric}={self.study.best_value:.4f}")
        self.logger.info(f"Best params: {self.best_params}")
        
        return self.best_params
    
    def optimize_neural_architecture(
        self,
        X_train, y_train,
        X_val, y_val
    ) -> Dict[str, Any]:
        """
        Optimize neural network architecture
        
        Returns:
            Best architecture parameters
        """
        self.logger.info("Optimizing neural network architecture...")
        
        def objective(trial):
            import torch
            import torch.nn as nn
            
            # Architecture hyperparameters
            n_layers = trial.suggest_int('n_layers', 2, 5)
            hidden_dims = [
                trial.suggest_int(f'hidden_dim_{i}', 64, 512, log=True)
                for i in range(n_layers)
            ]
            dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            
            # Build model
            layers = []
            input_dim = X_train.shape[1]
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_dim = hidden_dim
            
            layers.append(nn.Linear(input_dim, 1))
            layers.append(nn.Sigmoid())
            
            model = nn.Sequential(*layers)
            
            # Train
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.BCELoss()
            
            X_train_t = torch.FloatTensor(X_train.values if hasattr(X_train, 'values') else X_train)
            y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
            X_val_t = torch.FloatTensor(X_val.values if hasattr(X_val, 'values') else X_val)
            y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
            
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_t)
                loss = criterion(outputs, y_train_t)
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            
            return -val_loss.item()  # Minimize loss
        
        self.study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42)
        )
        
        self.study.optimize(objective, n_trials=self.n_trials // 2, timeout=self.timeout)
        
        self.best_params = self.study.best_params
        self.logger.info(f"Best architecture: {self.best_params}")
        
        return self.best_params
    
    def get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history"""
        if self.study is None:
            return {}
        
        return {
            'n_trials': len(self.study.trials),
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'trials': [
                {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params
                }
                for trial in self.study.trials
            ]
        }
