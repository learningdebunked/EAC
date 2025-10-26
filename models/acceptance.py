"""
XGBoost Acceptance Model

Predicts P(user accepts recommendation | features)
Trained on Instacart substitution data (1M+ events)
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss
import joblib


class XGBoostAcceptanceModel:
    """
    XGBoost model for predicting recommendation acceptance
    
    Features:
    - Product similarity (category, brand, price)
    - User history (past acceptance rate, price sensitivity)
    - SDOH signals (food insecurity, financial constraint)
    - Recommendation quality (savings, nutrition improvement)
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.logger = logging.getLogger("EAC.Models.Acceptance")
        self.model = None
        self.feature_names = None
        
        if model_path:
            self.load(model_path)
        else:
            self._initialize_model()
    
    def _initialize_model(self):
        """Initialize XGBoost model with optimal hyperparameters"""
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        
        self.logger.info("XGBoost acceptance model initialized")
    
    def train(self, instacart_data: pd.DataFrame, validation_split: float = 0.2):
        """
        Train on Instacart substitution data
        
        Args:
            instacart_data: DataFrame with columns:
                - user_id, order_id, product_id
                - original_product_id, suggested_product_id
                - accepted (0/1)
                - price_original, price_suggested
                - category_match, brand_match
                - user_past_acceptance_rate
                - user_price_sensitivity
                - sdoh_food_insecurity, sdoh_financial_constraint
                - nutrition_improvement
            validation_split: Fraction for validation
        """
        self.logger.info(f"Training on {len(instacart_data)} substitution events")
        
        # Extract features
        X, y = self._extract_features(instacart_data)
        self.feature_names = X.columns.tolist()
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Train XGBoost
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=10
        )
        
        # Calibrate probabilities
        self.logger.info("Calibrating probabilities...")
        self.model = CalibratedClassifierCV(
            self.model, 
            method='isotonic', 
            cv='prefit'
        )
        self.model.fit(X_val, y_val)
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred_proba)
        logloss = log_loss(y_val, y_pred_proba)
        
        self.logger.info(f"Training complete: AUC={auc:.4f}, LogLoss={logloss:.4f}")
        
        return {
            'auc': auc,
            'log_loss': logloss,
            'feature_importance': self._get_feature_importance()
        }
    
    def _extract_features(self, data: pd.DataFrame) -> tuple:
        """Extract features for training"""
        features = pd.DataFrame()
        
        # Product similarity features
        features['category_match'] = data['category_match'].astype(int)
        features['brand_match'] = data['brand_match'].astype(int)
        features['price_delta_pct'] = (
            (data['price_suggested'] - data['price_original']) / 
            data['price_original']
        )
        features['price_delta_abs'] = data['price_suggested'] - data['price_original']
        
        # Nutrition features
        features['nutrition_improvement'] = data.get('nutrition_improvement', 0)
        features['is_healthier'] = (features['nutrition_improvement'] > 0).astype(int)
        
        # User features
        features['user_past_acceptance_rate'] = data.get('user_past_acceptance_rate', 0.5)
        features['user_price_sensitivity'] = data.get('user_price_sensitivity', 0.5)
        features['user_order_count'] = data.get('user_order_count', 1)
        
        # SDOH features
        features['sdoh_food_insecurity'] = data.get('sdoh_food_insecurity', 0)
        features['sdoh_financial_constraint'] = data.get('sdoh_financial_constraint', 0)
        features['sdoh_mobility_limitation'] = data.get('sdoh_mobility_limitation', 0)
        
        # Interaction features
        features['price_x_sensitivity'] = (
            features['price_delta_pct'] * features['user_price_sensitivity']
        )
        features['nutrition_x_health_risk'] = (
            features['nutrition_improvement'] * data.get('sdoh_health_risk', 0)
        )
        
        # Target
        y = data['accepted'].astype(int)
        
        return features, y
    
    def predict_proba(self, recommendation: Dict[str, Any], user_features: Dict[str, Any]) -> float:
        """
        Predict acceptance probability for a recommendation
        
        Args:
            recommendation: Recommendation dict with product info
            user_features: User features dict
            
        Returns:
            Probability of acceptance [0, 1]
        """
        if self.model is None:
            self.logger.warning("Model not trained, using base rate")
            return 0.6  # Base acceptance rate
        
        # Build feature vector
        features = self._build_feature_vector(recommendation, user_features)
        
        # Predict
        proba = self.model.predict_proba([features])[0][1]
        
        return float(proba)
    
    def _build_feature_vector(
        self, 
        recommendation: Dict[str, Any], 
        user_features: Dict[str, Any]
    ) -> np.ndarray:
        """Build feature vector for prediction"""
        features = []
        
        # Product similarity
        features.append(recommendation.get('category_match', 0))
        features.append(recommendation.get('brand_match', 0))
        features.append(recommendation.get('price_delta_pct', 0))
        features.append(recommendation.get('savings', 0))
        
        # Nutrition
        features.append(recommendation.get('nutrition_improvement', 0))
        features.append(int(recommendation.get('nutrition_improvement', 0) > 0))
        
        # User
        features.append(user_features.get('past_acceptance_rate', 0.5))
        features.append(user_features.get('price_sensitivity', 0.5))
        features.append(user_features.get('order_count', 1))
        
        # SDOH
        features.append(user_features.get('food_insecurity', 0))
        features.append(user_features.get('financial_constraint', 0))
        features.append(user_features.get('mobility_limitation', 0))
        
        # Interactions
        features.append(features[2] * features[7])  # price_delta × sensitivity
        features.append(features[4] * user_features.get('health_risk', 0))  # nutrition × health
        
        return np.array(features)
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        if hasattr(self.model, 'feature_importances_'):
            importance = dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
            return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'feature_names': self.feature_names
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.logger.info(f"Model loaded from {path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get model statistics"""
        return {
            'model_type': 'XGBoost',
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'is_trained': self.model is not None,
            'feature_importance': self._get_feature_importance()
        }
