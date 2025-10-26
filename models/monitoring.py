"""
Model Monitoring with Evidently AI

Detects data drift, model performance degradation, and fairness issues
"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime


class EvidentlyMonitor:
    """
    Model monitoring using Evidently AI
    
    Monitors:
    - Data drift (feature distribution changes)
    - Model performance drift
    - Prediction drift
    - Fairness metrics over time
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EAC.Models.Monitor")
        
        # Reference data (baseline)
        self.reference_data = None
        self.reference_predictions = None
        
        # Monitoring history
        self.drift_history = []
        self.performance_history = []
        self.fairness_history = []
        
        self.logger.info("Evidently Monitor initialized")
    
    def set_reference(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        targets: Optional[np.ndarray] = None
    ):
        """
        Set reference (baseline) data
        
        Args:
            data: Reference feature data
            predictions: Model predictions on reference data
            targets: True labels for reference data
        """
        self.reference_data = data.copy()
        self.reference_predictions = predictions
        self.reference_targets = targets
        
        self.logger.info(f"Reference data set: {len(data)} samples")
    
    def detect_data_drift(
        self,
        current_data: pd.DataFrame,
        threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Detect data drift using statistical tests
        
        Args:
            current_data: Current production data
            threshold: Drift detection threshold
            
        Returns:
            Drift report
        """
        if self.reference_data is None:
            self.logger.warning("No reference data set")
            return {}
        
        self.logger.info("Detecting data drift...")
        
        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'n_features': len(current_data.columns),
            'drifted_features': [],
            'drift_scores': {}
        }
        
        for column in current_data.columns:
            if column not in self.reference_data.columns:
                continue
            
            # Compute drift score (Kolmogorov-Smirnov test)
            drift_score = self._compute_drift_score(
                self.reference_data[column],
                current_data[column]
            )
            
            drift_report['drift_scores'][column] = drift_score
            
            if drift_score > threshold:
                drift_report['drifted_features'].append(column)
        
        drift_report['drift_detected'] = len(drift_report['drifted_features']) > 0
        drift_report['drift_share'] = len(drift_report['drifted_features']) / drift_report['n_features']
        
        # Store in history
        self.drift_history.append(drift_report)
        
        if drift_report['drift_detected']:
            self.logger.warning(
                f"Data drift detected in {len(drift_report['drifted_features'])} features: "
                f"{drift_report['drifted_features']}"
            )
        else:
            self.logger.info("No data drift detected")
        
        return drift_report
    
    def _compute_drift_score(
        self,
        reference: pd.Series,
        current: pd.Series
    ) -> float:
        """
        Compute drift score using Kolmogorov-Smirnov statistic
        
        Returns:
            Drift score [0, 1]
        """
        from scipy.stats import ks_2samp
        
        # Handle categorical features
        if reference.dtype == 'object' or current.dtype == 'object':
            # Use chi-square for categorical
            ref_counts = reference.value_counts(normalize=True)
            cur_counts = current.value_counts(normalize=True)
            
            # Align indices
            all_categories = set(ref_counts.index) | set(cur_counts.index)
            ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
            cur_aligned = [cur_counts.get(cat, 0) for cat in all_categories]
            
            # Compute difference
            drift_score = np.sum(np.abs(np.array(ref_aligned) - np.array(cur_aligned))) / 2
        else:
            # Use KS test for numerical
            statistic, p_value = ks_2samp(reference.dropna(), current.dropna())
            drift_score = statistic
        
        return float(drift_score)
    
    def monitor_performance(
        self,
        current_predictions: np.ndarray,
        current_targets: np.ndarray,
        metric: str = "auc"
    ) -> Dict[str, Any]:
        """
        Monitor model performance over time
        
        Args:
            current_predictions: Current predictions
            current_targets: Current true labels
            metric: Performance metric
            
        Returns:
            Performance report
        """
        self.logger.info("Monitoring model performance...")
        
        # Compute current performance
        if metric == "auc":
            from sklearn.metrics import roc_auc_score
            current_score = roc_auc_score(current_targets, current_predictions)
        elif metric == "accuracy":
            from sklearn.metrics import accuracy_score
            current_score = accuracy_score(current_targets, (current_predictions > 0.5).astype(int))
        else:
            current_score = 0.0
        
        # Compare to reference
        if self.reference_predictions is not None and self.reference_targets is not None:
            if metric == "auc":
                from sklearn.metrics import roc_auc_score
                reference_score = roc_auc_score(self.reference_targets, self.reference_predictions)
            elif metric == "accuracy":
                from sklearn.metrics import accuracy_score
                reference_score = accuracy_score(
                    self.reference_targets,
                    (self.reference_predictions > 0.5).astype(int)
                )
            else:
                reference_score = 0.0
            
            performance_drop = reference_score - current_score
            performance_drop_pct = (performance_drop / reference_score) * 100 if reference_score > 0 else 0
        else:
            reference_score = None
            performance_drop = None
            performance_drop_pct = None
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'metric': metric,
            'current_score': float(current_score),
            'reference_score': float(reference_score) if reference_score else None,
            'performance_drop': float(performance_drop) if performance_drop else None,
            'performance_drop_pct': float(performance_drop_pct) if performance_drop_pct else None,
            'degradation_detected': performance_drop_pct > 10 if performance_drop_pct else False
        }
        
        # Store in history
        self.performance_history.append(report)
        
        if report['degradation_detected']:
            self.logger.warning(
                f"Performance degradation detected: {metric} dropped by {performance_drop_pct:.2f}%"
            )
        else:
            self.logger.info(f"Performance stable: {metric}={current_score:.4f}")
        
        return report
    
    def monitor_fairness(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        protected_attribute: str
    ) -> Dict[str, Any]:
        """
        Monitor fairness metrics over time
        
        Args:
            data: Feature data with protected attributes
            predictions: Model predictions
            protected_attribute: Name of protected attribute column
            
        Returns:
            Fairness report
        """
        self.logger.info(f"Monitoring fairness for {protected_attribute}...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'protected_attribute': protected_attribute,
            'groups': {}
        }
        
        # Compute metrics by group
        for group in data[protected_attribute].unique():
            group_mask = data[protected_attribute] == group
            group_predictions = predictions[group_mask]
            
            report['groups'][str(group)] = {
                'n_samples': int(group_mask.sum()),
                'mean_prediction': float(group_predictions.mean()),
                'acceptance_rate': float((group_predictions > 0.5).mean())
            }
        
        # Compute disparity
        acceptance_rates = [g['acceptance_rate'] for g in report['groups'].values()]
        report['max_disparity'] = float(max(acceptance_rates) - min(acceptance_rates))
        report['fairness_violation'] = report['max_disparity'] > 0.1
        
        # Store in history
        self.fairness_history.append(report)
        
        if report['fairness_violation']:
            self.logger.warning(
                f"Fairness violation detected: max disparity={report['max_disparity']:.3f}"
            )
        else:
            self.logger.info("Fairness metrics within acceptable range")
        
        return report
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get summary of all monitoring"""
        return {
            'drift': {
                'n_checks': len(self.drift_history),
                'n_drifts_detected': sum(1 for d in self.drift_history if d.get('drift_detected')),
                'latest': self.drift_history[-1] if self.drift_history else None
            },
            'performance': {
                'n_checks': len(self.performance_history),
                'n_degradations': sum(1 for p in self.performance_history if p.get('degradation_detected')),
                'latest': self.performance_history[-1] if self.performance_history else None
            },
            'fairness': {
                'n_checks': len(self.fairness_history),
                'n_violations': sum(1 for f in self.fairness_history if f.get('fairness_violation')),
                'latest': self.fairness_history[-1] if self.fairness_history else None
            }
        }
