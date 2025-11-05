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
        Load census tract-level SDOH data from real sources
        
        Data sources (Paper Table II):
        1. CDC Social Vulnerability Index (SVI)
        2. Area Deprivation Index (ADI)
        3. USDA Food Access Research Atlas
        4. National Transit Map
        5. U.S. Census ACS API
        
        Returns:
            Dict mapping tract_id (FIPS code) to SDOH indices
        """
        tract_data = {}
        
        # Load CDC SVI data
        svi_file = self.data_path / "cdc_svi" / "SVI2022_US.csv"
        if svi_file.exists():
            self.logger.info(f"Loading CDC SVI from {svi_file}")
            self._load_cdc_svi(tract_data, svi_file)
        else:
            self.logger.warning(f"CDC SVI not found at {svi_file}")
        
        # Load ADI data
        adi_file = self.data_path / "adi" / "US_2022_ADI.csv"
        if adi_file.exists():
            self.logger.info(f"Loading ADI from {adi_file}")
            self._load_adi(tract_data, adi_file)
        else:
            self.logger.warning(f"ADI not found at {adi_file}")
        
        # Load USDA Food Access data
        food_access_file = self.data_path / "usda_food_atlas" / "FoodAccessResearchAtlasData2019.csv"
        if food_access_file.exists():
            self.logger.info(f"Loading USDA Food Atlas from {food_access_file}")
            self._load_food_access(tract_data, food_access_file)
        else:
            self.logger.warning(f"USDA Food Atlas not found at {food_access_file}")
        
        # Load Census ACS data
        acs_file = self.data_path / "census_acs" / "ACSST5Y2022.csv"
        if acs_file.exists():
            self.logger.info(f"Loading Census ACS from {acs_file}")
            self._load_census_acs(tract_data, acs_file)
        else:
            self.logger.warning(f"Census ACS not found at {acs_file}")
        
        # Load Transit data
        transit_file = self.data_path / "transit" / "transit_scores.csv"
        if transit_file.exists():
            self.logger.info(f"Loading transit data from {transit_file}")
            self._load_transit_data(tract_data, transit_file)
        else:
            self.logger.warning(f"Transit data not found at {transit_file}")
        
        if not tract_data:
            self.logger.warning("No real SDOH data found - falling back to synthetic data")
            self.logger.warning("Download datasets: See scripts/download_datasets.sh")
            return {}  # Will use synthetic generation
        
        self.logger.info(f"Loaded SDOH data for {len(tract_data)} census tracts")
        return tract_data
    
    def _load_cdc_svi(self, tract_data: Dict, path: Path):
        """
        Load CDC Social Vulnerability Index
        
        Source: https://www.atsdr.cdc.gov/placeandhealth/svi/data_documentation_download.html
        Dataset: CDC/ATSDR SVI 2022
        """
        try:
            df = pd.read_csv(path, dtype={'FIPS': str})
            
            for _, row in df.iterrows():
                tract_id = row['FIPS']
                
                if tract_id not in tract_data:
                    tract_data[tract_id] = {'tract_id': tract_id}
                
                tract_data[tract_id].update({
                    'svi': row.get('RPL_THEMES', 0.0),  # Overall SVI percentile
                    'svi_socioeconomic': row.get('RPL_THEME1', 0.0),
                    'svi_household': row.get('RPL_THEME2', 0.0),
                    'svi_minority': row.get('RPL_THEME3', 0.0),
                    'svi_housing': row.get('RPL_THEME4', 0.0),
                    'poverty_rate': row.get('EP_POV150', 0.0) / 100.0,
                    'unemployment_rate': row.get('EP_UNEMP', 0.0) / 100.0,
                    'uninsured_rate': row.get('EP_UNINSUR', 0.0) / 100.0,
                    'elderly_percentage': row.get('EP_AGE65', 0.0) / 100.0,
                    'disability_rate': row.get('EP_DISABL', 0.0) / 100.0,
                    'population': row.get('E_TOTPOP', 0)
                })
            
            self.logger.info(f"Loaded CDC SVI for {len(tract_data)} tracts")
        
        except Exception as e:
            self.logger.error(f"Error loading CDC SVI: {e}")
    
    def _load_adi(self, tract_data: Dict, path: Path):
        """
        Load Area Deprivation Index
        
        Source: https://www.neighborhoodatlas.medicine.wisc.edu/
        Dataset: ADI 2022
        """
        try:
            df = pd.read_csv(path, dtype={'FIPS': str})
            
            for _, row in df.iterrows():
                tract_id = row['FIPS']
                
                if tract_id not in tract_data:
                    tract_data[tract_id] = {'tract_id': tract_id}
                
                # ADI is 1-100, normalize to 0-1
                adi_raw = row.get('ADI_NATRANK', 50)
                tract_data[tract_id]['adi'] = adi_raw / 100.0
                tract_data[tract_id]['adi_state_rank'] = row.get('ADI_STATERNK', 50)
            
            self.logger.info(f"Loaded ADI for {len(tract_data)} tracts")
        
        except Exception as e:
            self.logger.error(f"Error loading ADI: {e}")
    
    def _load_food_access(self, tract_data: Dict, path: Path):
        """
        Load USDA Food Access Research Atlas
        
        Source: https://www.ers.usda.gov/data-products/food-access-research-atlas/
        Dataset: Food Access Research Atlas 2019
        """
        try:
            df = pd.read_csv(path, dtype={'CensusTract': str})
            
            for _, row in df.iterrows():
                tract_id = str(row['CensusTract']).zfill(11)  # Pad to 11 digits
                
                if tract_id not in tract_data:
                    tract_data[tract_id] = {'tract_id': tract_id}
                
                # Food access indicators
                low_access = row.get('LATracts_1And10', 0)  # Low access at 1 and 10 miles
                low_income = row.get('LILATracts_1And10', 0)  # Low income + low access
                
                # Compute food access score (higher = better access)
                food_access_score = 1.0 - (low_access * 0.5 + low_income * 0.5)
                
                tract_data[tract_id].update({
                    'food_access_score': food_access_score,
                    'low_access_flag': bool(low_access),
                    'food_desert': bool(low_income),
                    'snap_stores': row.get('TractSNAP', 0)
                })
            
            self.logger.info(f"Loaded food access for {len(tract_data)} tracts")
        
        except Exception as e:
            self.logger.error(f"Error loading food access data: {e}")
    
    def _load_census_acs(self, tract_data: Dict, path: Path):
        """
        Load Census ACS 5-Year Estimates
        
        Source: https://www.census.gov/data/developers/data-sets/acs-5year.html
        Dataset: ACS 5-Year 2022
        """
        try:
            df = pd.read_csv(path, dtype={'GEO_ID': str})
            
            for _, row in df.iterrows():
                # Extract FIPS from GEO_ID (format: 1400000US##########)
                geo_id = row['GEO_ID']
                tract_id = geo_id.split('US')[-1] if 'US' in geo_id else geo_id
                
                if tract_id not in tract_data:
                    tract_data[tract_id] = {'tract_id': tract_id}
                
                # Extract key demographics
                median_income = row.get('B19013_001E', 50000)  # Median household income
                
                tract_data[tract_id].update({
                    'median_income': median_income,
                    'total_population': row.get('B01003_001E', 0),
                    'median_age': row.get('B01002_001E', 0),
                    'pct_broadband': row.get('B28002_004E', 0) / row.get('B28002_001E', 1) if row.get('B28002_001E', 0) > 0 else 0
                })
            
            self.logger.info(f"Loaded Census ACS for {len(tract_data)} tracts")
        
        except Exception as e:
            self.logger.error(f"Error loading Census ACS: {e}")
    
    def _load_transit_data(self, tract_data: Dict, path: Path):
        """
        Load transit accessibility scores
        
        Source: Bureau of Transportation Statistics / AllTransit
        Dataset: Transit scores by census tract
        """
        try:
            df = pd.read_csv(path, dtype={'tract_id': str})
            
            for _, row in df.iterrows():
                tract_id = row['tract_id']
                
                if tract_id not in tract_data:
                    tract_data[tract_id] = {'tract_id': tract_id}
                
                # Transit score (0-1, higher = better access)
                tract_data[tract_id]['transit_score'] = row.get('transit_score', 0.5)
                tract_data[tract_id]['transit_routes'] = row.get('num_routes', 0)
            
            self.logger.info(f"Loaded transit data for {len(tract_data)} tracts")
        
        except Exception as e:
            self.logger.error(f"Error loading transit data: {e}")
    
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
