"""
Drift Detection Module

Monitors data drift using statistical tests.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
import logging
import json
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DriftDetector:
    """Detects data drift in production data using statistical tests."""
    
    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_columns: List[str],
        drift_threshold: float = 0.1
    ):
        """
        Initialize drift detector.
        
        Args:
            reference_data: Reference (training) data
            feature_columns: List of feature column names
            drift_threshold: Threshold for drift detection (0-1)
        """
        self.reference_data = reference_data
        self.feature_columns = feature_columns
        self.drift_threshold = drift_threshold
        
        logger.info(f"Drift Detector initialized with {len(feature_columns)} features")
    
    def _detect_numerical_drift(
        self, 
        ref_col: pd.Series, 
        curr_col: pd.Series,
        column_name: str
    ) -> Dict[str, Any]:
        """Detect drift for numerical columns using KS test."""
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_col.dropna(), curr_col.dropna())
        
        # Mean shift
        mean_shift = abs(curr_col.mean() - ref_col.mean()) / (ref_col.std() + 1e-6)
        
        # Standard deviation shift
        std_shift = abs(curr_col.std() - ref_col.std()) / (ref_col.std() + 1e-6)
        
        # Combined drift score
        drift_score = (ks_statistic + mean_shift * 0.3 + std_shift * 0.3) / 1.6
        
        drift_detected = drift_score > self.drift_threshold
        
        return {
            'column': column_name,
            'type': 'numerical',
            'drift_score': float(drift_score),
            'drift_detected': drift_detected,
            'ks_statistic': float(ks_statistic),
            'ks_pvalue': float(ks_pvalue),
            'mean_shift': float(mean_shift),
            'std_shift': float(std_shift)
        }
    
    def _detect_categorical_drift(
        self, 
        ref_col: pd.Series, 
        curr_col: pd.Series,
        column_name: str
    ) -> Dict[str, Any]:
        """Detect drift for categorical columns using Chi-square test."""
        # Get value counts
        ref_counts = ref_col.value_counts()
        curr_counts = curr_col.value_counts()
        
        # Align categories
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        
        ref_aligned = pd.Series({cat: ref_counts.get(cat, 0) for cat in all_categories})
        curr_aligned = pd.Series({cat: curr_counts.get(cat, 0) for cat in all_categories})
        
        # Chi-square test
        try:
            chi2_stat, chi2_pvalue = stats.chisquare(
                curr_aligned.values + 1,  # Add 1 to avoid division by zero
                ref_aligned.values + 1
            )
        except Exception:
            chi2_stat, chi2_pvalue = 0, 1
        
        # Distribution difference
        ref_dist = ref_aligned / ref_aligned.sum()
        curr_dist = curr_aligned / curr_aligned.sum()
        distribution_shift = abs(curr_dist - ref_dist).mean()
        
        # Drift score
        drift_score = min(1.0, distribution_shift * 2 + (1 - chi2_pvalue) * 0.5)
        drift_detected = drift_score > self.drift_threshold
        
        return {
            'column': column_name,
            'type': 'categorical',
            'drift_score': float(drift_score),
            'drift_detected': drift_detected,
            'chi2_statistic': float(chi2_stat),
            'chi2_pvalue': float(chi2_pvalue),
            'distribution_shift': float(distribution_shift)
        }
    
    def detect_drift(
        self, 
        current_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current production data
            
        Returns:
            Drift detection results
        """
        logger.info("Running drift detection")
        
        results = {
            'drift_detected': False,
            'drift_ratio': 0.0,
            'drifted_columns': [],
            'drift_scores': {},
            'column_details': [],
            'timestamp': datetime.now().isoformat(),
            'threshold': self.drift_threshold
        }
        
        drifted_count = 0
        total_columns = 0
        
        for col in self.feature_columns:
            if col not in current_data.columns or col not in self.reference_data.columns:
                continue
            
            ref_col = self.reference_data[col]
            curr_col = current_data[col]
            
            # Determine column type and run appropriate test
            if pd.api.types.is_numeric_dtype(ref_col):
                col_result = self._detect_numerical_drift(ref_col, curr_col, col)
            else:
                col_result = self._detect_categorical_drift(ref_col, curr_col, col)
            
            results['column_details'].append(col_result)
            results['drift_scores'][col] = col_result['drift_score']
            
            if col_result['drift_detected']:
                drifted_count += 1
                results['drifted_columns'].append(col)
            
            total_columns += 1
        
        # Calculate overall drift
        if total_columns > 0:
            results['drift_ratio'] = drifted_count / total_columns
            results['drift_detected'] = results['drift_ratio'] > self.drift_threshold
            results['n_drifted_columns'] = drifted_count
            results['total_columns'] = total_columns
        
        logger.info(f"Drift detected: {results['drift_detected']}")
        logger.info(f"Drift ratio: {results['drift_ratio']:.2%}")
        if results['drifted_columns']:
            logger.info(f"Drifted columns: {results['drifted_columns']}")
        
        return results
    
    def save_report(
        self, 
        current_data: pd.DataFrame, 
        output_dir: str = "reports"
    ) -> str:
        """
        Generate and save drift report as JSON.
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = self.detect_drift(current_data)
        
        # Convert numpy types
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.bool_, np.integer)):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_clean = convert_types(results)
        
        # Save JSON report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_path / f"drift_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results_clean, f, indent=2)
        
        logger.info(f"Drift report saved to {report_path}")
        
        return str(report_path)


def check_drift(
    reference_path: str = "data/processed/train.csv",
    current_path: str = "data/production/current_data.csv",
    config_path: str = "configs/config.yaml"
) -> bool:
    """
    Main drift check function for CI/CD.
    
    Args:
        reference_path: Path to reference data
        current_path: Path to current production data
        config_path: Path to config file
        
    Returns:
        True if drift detected, False otherwise
    """
    import yaml
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Load data
    reference_data = pd.read_csv(reference_path)
    
    if not Path(current_path).exists():
        logger.info("No production data found. Skipping drift check.")
        print("drift_detected=false")
        return False
    
    current_data = pd.read_csv(current_path)
    
    # Get feature columns
    feature_cols = (
        config['features']['numerical'] + 
        config['features']['categorical']
    )
    
    # Initialize detector
    detector = DriftDetector(
        reference_data=reference_data,
        feature_columns=feature_cols,
        drift_threshold=config['monitoring']['drift_threshold']
    )
    
    # Detect drift
    result = detector.detect_drift(current_data)
    
    # Save report
    detector.save_report(current_data)
    
    # Save result for GitHub Actions
    # Convert numpy types to Python types
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    result_clean = convert_types(result)
    
    with open("drift_result.json", 'w') as f:
        json.dump(result_clean, f, indent=2)
    
    # Output for GitHub Actions
    print(f"drift_detected={'true' if result['drift_detected'] else 'false'}")
    print(f"Drift ratio: {result['drift_ratio']:.2%}")
    print(f"Drifted columns: {result['drifted_columns']}")
    
    return result['drift_detected']


def simulate_production_data(
    reference_path: str = "data/processed/train.csv",
    output_path: str = "data/production/current_data.csv",
    drift_magnitude: float = 0.1
) -> pd.DataFrame:
    """
    Simulate production data with artificial drift for testing.
    """
    reference = pd.read_csv(reference_path)
    drifted = reference.copy()
    
    # Add drift to numerical columns
    numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numerical_cols:
        if col in drifted.columns:
            drift = np.random.normal(
                drift_magnitude * drifted[col].std(),
                drifted[col].std() * 0.1,
                len(drifted)
            )
            drifted[col] = drifted[col] + drift
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    drifted.to_csv(output_path, index=False)
    
    logger.info(f"Simulated production data saved to {output_path}")
    logger.info(f"Drift magnitude: {drift_magnitude}")
    
    return drifted


if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    
    if len(sys.argv) > 1 and sys.argv[1] == "--simulate":
        drift_mag = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
        simulate_production_data(drift_magnitude=drift_mag)
    else:
        drift_detected = check_drift()
        
        if drift_detected:
            print("\n⚠️  DRIFT DETECTED - Consider retraining")
        else:
            print("\n✓ No significant drift detected")
