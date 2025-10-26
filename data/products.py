"""
Product Data Loader - Loads and manages product information
"""
import logging
from typing import Dict, Any, Optional, List
import numpy as np
from pathlib import Path


class ProductDataLoader:
    """
    Loads product data from various sources:
    - USDA FoodData Central (nutrition)
    - Open Food Facts (UPC, labels)
    - WIC Authorized Product Lists
    - Store inventory
    """
    
    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.logger = logging.getLogger("EACAgent.Data.Products")
        
        # Load product database
        self.products = self._load_products()
        
        # Build indices for fast lookup
        self.category_index = self._build_category_index()
        
        self.logger.info(f"Product data loaded: {len(self.products)} products")
    
    def _load_products(self) -> Dict[str, Dict[str, Any]]:
        """
        Load product database
        
        Returns:
            Dict mapping product_id to product info
        """
        # In production, load from database or files
        # For now, return synthetic data
        
        self.logger.warning("Using synthetic product data - replace with real data in production")
        
        return {}
    
    def _build_category_index(self) -> Dict[str, List[str]]:
        """Build index of products by category"""
        index = {}
        for product_id, product in self.products.items():
            category = product.get('category', 'unknown')
            if category not in index:
                index[category] = []
            index[category].append(product_id)
        return index
    
    def get_product(self, product_id: str) -> Dict[str, Any]:
        """
        Get product information
        
        Args:
            product_id: Product ID
            
        Returns:
            Product info dict
        """
        if product_id in self.products:
            return self.products[product_id]
        
        # Generate synthetic product for development
        return self._generate_synthetic_product(product_id)
    
    def find_snap_alternative(
        self, 
        product_id: str, 
        category: str, 
        max_price_delta: float
    ) -> Optional[Dict[str, Any]]:
        """
        Find SNAP-eligible alternative
        
        Args:
            product_id: Original product
            category: Product category
            max_price_delta: Maximum price difference
            
        Returns:
            Alternative product or None
        """
        original = self.get_product(product_id)
        
        # In production, query SNAP-eligible products in same category
        # For now, generate synthetic alternative
        
        if np.random.random() > 0.6:  # 40% chance of finding alternative
            return self._generate_synthetic_product(
                f"{product_id}_snap_alt",
                category=category,
                snap_eligible=True,
                price=original.get('price', 5.0) * 0.9  # 10% cheaper
            )
        
        return None
    
    def find_low_gi_alternative(
        self, 
        product_id: str, 
        category: str, 
        max_gi: int,
        max_price_delta_pct: float
    ) -> Optional[Dict[str, Any]]:
        """
        Find low-glycemic index alternative
        
        Args:
            product_id: Original product
            category: Product category
            max_gi: Maximum glycemic index
            max_price_delta_pct: Maximum price increase %
            
        Returns:
            Alternative product or None
        """
        original = self.get_product(product_id)
        
        if np.random.random() > 0.5:  # 50% chance
            return self._generate_synthetic_product(
                f"{product_id}_lowgi_alt",
                category=category,
                glycemic_index=max_gi - 10,
                price=original.get('price', 5.0) * 1.1  # 10% more expensive
            )
        
        return None
    
    def is_fsa_hsa_eligible(self, product_id: str) -> bool:
        """
        Check if product is FSA/HSA eligible
        
        Args:
            product_id: Product ID
            
        Returns:
            True if eligible
        """
        product = self.get_product(product_id)
        category = product.get('category', '')
        
        # OTC medications and health supplies are typically eligible
        return category in ['otc_medication', 'health_supplies', 'first_aid']
    
    def find_bulk_alternative(
        self, 
        product_id: str, 
        min_savings_pct: float
    ) -> Optional[Dict[str, Any]]:
        """
        Find bulk/larger size alternative
        
        Args:
            product_id: Original product
            min_savings_pct: Minimum savings percentage
            
        Returns:
            Bulk alternative or None
        """
        original = self.get_product(product_id)
        
        if np.random.random() > 0.7:  # 30% chance
            # Bulk version: 2x quantity, 1.7x price (15% savings per unit)
            return self._generate_synthetic_product(
                f"{product_id}_bulk",
                category=original.get('category'),
                price=original.get('price', 5.0) * 1.7,
                quantity=original.get('quantity', 1) * 2
            )
        
        return None
    
    def get_affordable_produce(self) -> Optional[Dict[str, Any]]:
        """
        Get affordable produce recommendation
        
        Returns:
            Produce product
        """
        # Return a common affordable produce item
        return self._generate_synthetic_product(
            'produce_001',
            category='vegetables',
            name='Fresh Seasonal Vegetables',
            price=3.99,
            snap_eligible=True
        )
    
    def _generate_synthetic_product(
        self, 
        product_id: str,
        category: Optional[str] = None,
        name: Optional[str] = None,
        price: Optional[float] = None,
        snap_eligible: Optional[bool] = None,
        glycemic_index: Optional[int] = None,
        quantity: int = 1
    ) -> Dict[str, Any]:
        """
        Generate synthetic product for development
        
        In production, all products should be in database
        """
        # Use product_id as seed
        seed = abs(hash(product_id)) % (2**32)
        np.random.seed(seed)
        
        if category is None:
            categories = ['grains', 'dairy', 'meat', 'produce', 'snacks', 'beverages']
            category = np.random.choice(categories)
        
        if name is None:
            name = f"Product {product_id}"
        
        if price is None:
            price = np.random.uniform(2.0, 15.0)
        
        if snap_eligible is None:
            snap_eligible = np.random.random() > 0.3  # 70% SNAP-eligible
        
        if glycemic_index is None and category in ['grains', 'snacks', 'beverages']:
            glycemic_index = int(np.random.uniform(40, 85))
        
        # Generate nutrition info
        nutrition = {
            'calories': int(np.random.uniform(100, 400)),
            'protein_g': round(np.random.uniform(2, 20), 1),
            'carbs_g': round(np.random.uniform(10, 50), 1),
            'fat_g': round(np.random.uniform(1, 20), 1),
            'fiber_g': round(np.random.uniform(0, 8), 1),
            'sugar_g': round(np.random.uniform(0, 25), 1),
            'sodium_mg': int(np.random.uniform(50, 800))
        }
        
        return {
            'product_id': product_id,
            'name': name,
            'category': category,
            'price': round(price, 2),
            'quantity': quantity,
            'snap_eligible': snap_eligible,
            'glycemic_index': glycemic_index,
            'nutrition': nutrition,
            'in_stock': True,
            'brand': f"Brand_{seed % 10}"
        }
