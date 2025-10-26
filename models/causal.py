"""
Causal ML for Treatment Effect Estimation

Implements Meta-Learners and Doubly Robust estimation
For estimating heterogeneous treatment effects of recommendations
"""
import logging
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


class CausalMLEstimator:
    """
    Causal ML for heterogeneous treatment effect estimation
    
    Methods:
    1. S-Learner: Single model for all
    2. T-Learner: Separate models for treatment/control
    3. X-Learner: Advanced meta-learner
    4. DR-Learner: Doubly robust estimation
    """
    
    def __init__(self, method: str = "t-learner"):
        self.logger = logging.getLogger("EAC.Models.Causal")
        self.method = method
        
        # Models
        self.treatment_model = None
        self.control_model = None
        self.propensity_model = None
        self.tau_model = None
        
        self.logger.info(f"Causal ML Estimator initialized: {method}")
    
    def fit(
        self,
        X: pd.DataFrame,
        treatment: np.ndarray,
        y: np.ndarray
    ):
        """
        Fit causal model
        
        Args:
            X: Features (user, context, recommendation features)
            treatment: Binary treatment indicator (1=received rec, 0=control)
            y: Outcome (e.g., spend, satisfaction)
        """
        self.logger.info(f"Training {self.method} on {len(X)} observations")
        
        if self.method == "s-learner":
            self._fit_s_learner(X, treatment, y)
        elif self.method == "t-learner":
            self._fit_t_learner(X, treatment, y)
        elif self.method == "x-learner":
            self._fit_x_learner(X, treatment, y)
        elif self.method == "dr-learner":
            self._fit_dr_learner(X, treatment, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.logger.info("Training complete")
    
    def _fit_s_learner(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        """
        S-Learner: Single model with treatment as feature
        
        τ(x) = E[Y|X=x, T=1] - E[Y|X=x, T=0]
        """
        # Add treatment as feature
        X_with_t = X.copy()
        X_with_t['treatment'] = treatment
        
        # Train single model
        self.treatment_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.treatment_model.fit(X_with_t, y)
    
    def _fit_t_learner(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        """
        T-Learner: Separate models for treatment and control
        
        τ(x) = μ_1(x) - μ_0(x)
        """
        # Split data
        X_treatment = X[treatment == 1]
        y_treatment = y[treatment == 1]
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        # Train treatment model
        self.treatment_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.treatment_model.fit(X_treatment, y_treatment)
        
        # Train control model
        self.control_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.control_model.fit(X_control, y_control)
    
    def _fit_x_learner(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        """
        X-Learner: Advanced meta-learner with imputed treatment effects
        
        1. Estimate μ_0(x) and μ_1(x)
        2. Impute individual treatment effects
        3. Model treatment effects
        """
        # Step 1: Fit base models (like T-learner)
        self._fit_t_learner(X, treatment, y)
        
        # Step 2: Impute treatment effects
        X_treatment = X[treatment == 1]
        y_treatment = y[treatment == 1]
        X_control = X[treatment == 0]
        y_control = y[treatment == 0]
        
        # Imputed effects for treated
        tau_treatment = y_treatment - self.control_model.predict(X_treatment)
        
        # Imputed effects for control
        tau_control = self.treatment_model.predict(X_control) - y_control
        
        # Step 3: Model treatment effects
        self.tau_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        
        # Combine and fit
        X_tau = pd.concat([X_treatment, X_control])
        tau_combined = np.concatenate([tau_treatment, tau_control])
        self.tau_model.fit(X_tau, tau_combined)
    
    def _fit_dr_learner(self, X: pd.DataFrame, treatment: np.ndarray, y: np.ndarray):
        """
        Doubly Robust Learner
        
        Combines outcome regression and propensity score weighting
        """
        # Fit propensity model
        self.propensity_model = LogisticRegression(max_iter=1000, random_state=42)
        self.propensity_model.fit(X, treatment)
        propensity_scores = self.propensity_model.predict_proba(X)[:, 1]
        
        # Clip propensity scores to avoid extreme weights
        propensity_scores = np.clip(propensity_scores, 0.01, 0.99)
        
        # Fit outcome models
        self._fit_t_learner(X, treatment, y)
        
        # Compute doubly robust estimates
        mu_1 = self.treatment_model.predict(X)
        mu_0 = self.control_model.predict(X)
        
        # DR formula
        dr_treatment = (treatment * y) / propensity_scores - \
                      ((treatment - propensity_scores) * mu_1) / propensity_scores
        
        dr_control = ((1 - treatment) * y) / (1 - propensity_scores) + \
                    ((treatment - propensity_scores) * mu_0) / (1 - propensity_scores)
        
        # Model the difference
        tau = dr_treatment - dr_control
        self.tau_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        self.tau_model.fit(X, tau)
    
    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict Conditional Average Treatment Effect (CATE)
        
        Args:
            X: Features
            
        Returns:
            Predicted treatment effects
        """
        if self.method == "s-learner":
            # Predict with treatment=1 and treatment=0
            X_t1 = X.copy()
            X_t1['treatment'] = 1
            X_t0 = X.copy()
            X_t0['treatment'] = 0
            
            return self.treatment_model.predict(X_t1) - self.treatment_model.predict(X_t0)
        
        elif self.method == "t-learner":
            return self.treatment_model.predict(X) - self.control_model.predict(X)
        
        elif self.method in ["x-learner", "dr-learner"]:
            return self.tau_model.predict(X)
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def estimate_ate(self, X: pd.DataFrame) -> float:
        """
        Estimate Average Treatment Effect (ATE)
        
        Args:
            X: Features
            
        Returns:
            Average treatment effect
        """
        cate = self.predict_cate(X)
        return float(np.mean(cate))
    
    def estimate_att(
        self, 
        X: pd.DataFrame, 
        treatment: np.ndarray
    ) -> float:
        """
        Estimate Average Treatment Effect on the Treated (ATT)
        
        Args:
            X: Features
            treatment: Treatment indicator
            
        Returns:
            ATT
        """
        cate = self.predict_cate(X)
        return float(np.mean(cate[treatment == 1]))
    
    def get_heterogeneity_analysis(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze treatment effect heterogeneity
        
        Returns:
            Dict with heterogeneity metrics
        """
        cate = self.predict_cate(X)
        
        analysis = {
            'mean_effect': float(np.mean(cate)),
            'std_effect': float(np.std(cate)),
            'min_effect': float(np.min(cate)),
            'max_effect': float(np.max(cate)),
            'median_effect': float(np.median(cate)),
            'heterogeneity_ratio': float(np.std(cate) / np.abs(np.mean(cate))) if np.mean(cate) != 0 else 0
        }
        
        # Quantile analysis
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        analysis['quantiles'] = {
            f'q{int(q*100)}': float(np.quantile(cate, q))
            for q in quantiles
        }
        
        return analysis
