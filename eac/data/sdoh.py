"""
SDOH Data Loader - Loads and manages Social Determinants of Health data
"""
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class SDOHDataLoader:
    """
    Loads SDOH data from various sources:
    - CDC Social Vulnerability Index (SVI)
    - Area Deprivation Index (ADI)
    - USDA Food Access Research Atlas
    - National Transit Map
    - Census ACS data
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.logger = logging.getLogger("EACAgent.Data.SDOH")
        
        # Load data
        self.tract_data = self._load_tract_data()
        self.zip_to_tract_map = self._load_zip_to_tract_mapping()
        
        self.logger.info(f"SDOH Data loaded: {len(self.tract_data)} census tracts")
    
    def _load_tract_data(self) -> Dict[str, Dict[str, Any]]:
        """
        Load census tract-level SDOH data
        
        Returns:
            Dict mapping tract_id to SDOH indices
        """
        # In production, load from actual data files
        # For now, return synthetic data structure
        
        tract_data = {}
        
        # Example: Load from CSV
        # svi_file = self.data_path / "svi_2022.csv"
        # if svi_file.exists():
        #     df = pd.read_csv(svi_file)
        #     for _, row in df.iterrows():
        #         tract_id = row['FIPS']
        #         tract_data[tract_id] = {
        #             'svi': row['RPL_THEMES'],
        #             ...
        #         }
        
        # Synthetic data for development
        self.logger.warning("Using synthetic SDOH data - replace with real data in production")
        
        return tract_data
    
    def _load_zip_to_tract_mapping(self) -> Dict[str, str]:
        """
        Load ZIP code to census tract mapping
        
        Returns:
            Dict mapping ZIP code to primary census tract
        """
        # In production, load from HUD USPS ZIP-Tract crosswalk
        # For now, return empty dict
        
        return {}
    
    def get_tract_data(self, tract_id: str) -> Optional[Dict[str, Any]]:
        """
        Get SDOH data for a census tract
        
        Args:
            tract_id: Census tract FIPS code
            
        Returns:
            Dict with SDOH indices or None if not found
        """
        if tract_id in self.tract_data:
            return self.tract_data[tract_id]
        
        # Generate synthetic data for development
        return self._generate_synthetic_tract_data(tract_id)
    
    def zip_to_tract(self, zip_code: str) -> Optional[str]:
        """
        Convert ZIP code to census tract
        
        Args:
            zip_code: 5-digit ZIP code
            
        Returns:
            Census tract FIPS code or None
        """
        if zip_code in self.zip_to_tract_map:
            return self.zip_to_tract_map[zip_code]
        
        # For development: generate synthetic tract ID from ZIP
        # Format: state(2) + county(3) + tract(6)
        state = zip_code[:2]
        county = zip_code[2:5]
        tract = f"{int(zip_code) % 1000000:06d}"
        return f"{state}{county}{tract}"
    
    def _generate_synthetic_tract_data(self, tract_id: str) -> Dict[str, Any]:
        """
        Generate synthetic SDOH data for development
        
        In production, this should never be called - all data should be loaded
        """
        # Use tract_id as seed for reproducibility
        seed = int(tract_id[-6:]) if len(tract_id) >= 6 else 0
        np.random.seed(seed)
        
        # Generate correlated SDOH indices
        base_vulnerability = np.random.beta(2, 5)  # Skewed toward lower values
        
        data = {
            'tract_id': tract_id,
            'svi': np.clip(base_vulnerability + np.random.normal(0, 0.1), 0, 1),
            'adi': np.clip(base_vulnerability + np.random.normal(0, 0.15), 0, 1),
            'food_access_score': np.clip(1 - base_vulnerability + np.random.normal(0, 0.2), 0, 1),
            'transit_score': np.clip(0.5 + np.random.normal(0, 0.2), 0, 1),
            'median_income': int(50000 * (1 - base_vulnerability) + np.random.normal(0, 10000)),
            'poverty_rate': np.clip(base_vulnerability + np.random.normal(0, 0.1), 0, 1),
            'unemployment_rate': np.clip(base_vulnerability * 0.5 + np.random.normal(0, 0.05), 0, 1),
            'elderly_percentage': np.clip(0.15 + np.random.normal(0, 0.05), 0, 1),
            'disability_rate': np.clip(0.12 + np.random.normal(0, 0.03), 0, 1),
            'chronic_disease_rate': np.clip(base_vulnerability * 0.6 + np.random.normal(0, 0.1), 0, 1),
            'healthcare_access_score': np.clip(1 - base_vulnerability + np.random.normal(0, 0.15), 0, 1),
            'uninsured_rate': np.clip(base_vulnerability * 0.4 + np.random.normal(0, 0.08), 0, 1),
            'population': int(np.random.uniform(1200, 8000))
        }
        
        return data
