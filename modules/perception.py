"""
Perception Module - Observes and processes checkout context
"""
import logging
from typing import Dict, List, Any, Optional
import numpy as np

from config import EACConfig
from data.sdoh import SDOHDataLoader
from data.products import ProductDataLoader


# ============================================================================
# NEURAL NETWORK MODULES FOR ADVANCED FEATURE ENGINEERING
# Paper Algorithm 1: CNN + RNN + MLP + Attention
# ============================================================================

class CartCNN(nn.Module):
    """CNN for cart feature extraction (fc)"""
    
    def __init__(self, input_channels: int = 10, embedding_dim: int = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class BehavioralRNN(nn.Module):
    """RNN for behavioral sequence encoding (fb)"""
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.3, bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        x = self.fc(h_combined)
        x = self.dropout(x)
        return x


class SDOHMLP(nn.Module):
    """MLP for SDOH feature encoding (fs)"""
    
    def __init__(self, input_dim: int = 15, hidden_dim: int = 128, embedding_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, embedding_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class LearnedAttention(nn.Module):
    """Learned attention mechanism (α)"""
    
    def __init__(self, embedding_dim: int = 64, n_modalities: int = 3):
        super().__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(embedding_dim * n_modalities, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_modalities),
            nn.Softmax(dim=1)
        )
    
    def forward(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        concat = torch.cat(embeddings, dim=1)
        weights = self.attention_fc(concat)
        return weights


class MultiModalFusion(nn.Module):
    """Complete multi-modal feature fusion (Paper Algorithm 1)"""
    
    def __init__(self, embedding_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.cart_cnn = CartCNN(input_channels=10, embedding_dim=embedding_dim)
        self.behavioral_rnn = BehavioralRNN(input_dim=20, hidden_dim=128, embedding_dim=embedding_dim)
        self.sdoh_mlp = SDOHMLP(input_dim=15, hidden_dim=128, embedding_dim=embedding_dim)
        self.attention = LearnedAttention(embedding_dim, n_modalities=3)
        self.output_fc = nn.Linear(embedding_dim, output_dim)
    
    def forward(
        self,
        cart_features: torch.Tensor,
        behavioral_features: torch.Tensor,
        sdoh_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Step 1: Modality-specific encoding
        fc = self.cart_cnn(cart_features)
        fb = self.behavioral_rnn(behavioral_features)
        fs = self.sdoh_mlp(sdoh_features)
        
        # Step 2: Learned attention (α)
        alpha = self.attention([fc, fb, fs])
        
        # Step 3: Weighted combination
        embeddings = torch.stack([fc, fb, fs], dim=1)
        alpha_expanded = alpha.unsqueeze(2)
        weighted = embeddings * alpha_expanded
        fused = weighted.sum(dim=1)
        
        # Step 4: Final projection
        output = self.output_fc(fused)
        
        return output, alpha


# ============================================================================
# PERCEPTION MODULE
# ============================================================================

class PerceptionModule:
    """
    Perception Module: Observes the checkout environment
    
    Responsibilities:
    1. Parse cart contents
    2. Extract SDOH signals (census tract → indices)
    3. Identify payment methods
    4. Detect constraints (budget, dietary, mobility)
    5. Assess protected attributes (for fairness)
    """
    
    def __init__(self, config: EACConfig):
        self.config = config
        self.logger = logging.getLogger("EACAgent.Perception")
        
        # Load data sources
        self.sdoh_loader = SDOHDataLoader(config.sdoh_data_path)
        self.product_loader = ProductDataLoader(config.product_data_path)
        
        # Initialize advanced feature fusion (if enabled)
        self.use_advanced_features = getattr(config, 'use_advanced_feature_fusion', True)
        
        if self.use_advanced_features:
            try:
                self.fusion_model = MultiModalFusion(
                    embedding_dim=64,
                    output_dim=config.bandit_context_dim
                )
                self.fusion_model.eval()  # Inference mode
                
                # Load pretrained weights if available
                if hasattr(config, 'feature_fusion_model_path') and config.feature_fusion_model_path:
                    try:
                        self.fusion_model.load_state_dict(
                            torch.load(config.feature_fusion_model_path, map_location='cpu')
                        )
                        self.logger.info(f"Loaded pretrained fusion model from {config.feature_fusion_model_path}")
                    except Exception as e:
                        self.logger.warning(f"Could not load pretrained model: {e}, using random initialization")
                
                self.logger.info("Advanced feature fusion enabled (CNN+RNN+MLP+Attention)")
            except Exception as e:
                self.logger.error(f"Failed to initialize fusion model: {e}, falling back to simple features")
                self.use_advanced_features = False
        else:
            self.logger.info("Using simple feature concatenation (legacy mode)")
        
        self.logger.info("Perception Module initialized")
    
    def observe(self, event) -> Dict[str, Any]:
        """
        Observe checkout event and extract context
        
        Args:
            event: CheckoutEvent
            
        Returns:
            Context dictionary with all relevant features
        """
        context = {
            'user_id': event.user_id,
            'timestamp': event.timestamp,
            'cart': self._parse_cart(event.cart),
            'payment_methods': event.payment_methods,
            'delivery_address': event.delivery_address,
            'consent': event.consent
        }
        
        # Extract SDOH signals (if consented)
        if event.consent.get('sdoh_signals', False):
            context['sdoh'] = self._extract_sdoh_signals(event.delivery_address)
        else:
            context['sdoh'] = {}
        
        # Extract behavioral features
        context['behavioral'] = self._extract_behavioral_features(context)
        
        # Detect constraints
        context['constraints'] = self._detect_constraints(context)
        
        # Build feature vector for bandit
        context['features'] = self._build_feature_vector(context)
        
        return context
    
    def _parse_cart(self, cart: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse cart contents and enrich with product data
        
        Returns:
            Parsed cart with nutrition info, categories, etc.
        """
        parsed_items = []
        total_price = 0.0
        total_items = 0
        
        for item in cart:
            product_id = item.get('product_id')
            quantity = item.get('quantity', 1)
            price = item.get('price', 0.0)
            
            # Enrich with product data
            product_info = self.product_loader.get_product(product_id)
            
            parsed_item = {
                'product_id': product_id,
                'quantity': quantity,
                'price': price,
                'total_price': price * quantity,
                'category': product_info.get('category', 'unknown'),
                'nutrition': product_info.get('nutrition', {}),
                'snap_eligible': product_info.get('snap_eligible', False),
                'glycemic_index': product_info.get('glycemic_index', None)
            }
            
            parsed_items.append(parsed_item)
            total_price += price * quantity
            total_items += quantity
        
        # Compute cart-level features
        categories = [item['category'] for item in parsed_items]
        snap_eligible_count = sum(1 for item in parsed_items if item['snap_eligible'])
        
        return {
            'items': parsed_items,
            'total_price': total_price,
            'total_items': total_items,
            'unique_items': len(parsed_items),
            'categories': list(set(categories)),
            'snap_eligible_ratio': snap_eligible_count / len(parsed_items) if parsed_items else 0,
            'avg_price_per_item': total_price / total_items if total_items > 0 else 0
        }
    
    def _extract_sdoh_signals(self, delivery_address: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract SDOH signals from delivery address
        
        Process:
        1. Geocode address to census tract
        2. Join SDOH indices (SVI, ADI, food access, transit)
        3. Apply differential privacy (if configured)
        4. Compute composite risk scores
        """
        # Geocode to census tract
        census_tract = self._geocode_to_census_tract(delivery_address)
        
        if not census_tract:
            self.logger.warning("Could not geocode address to census tract")
            return {}
        
        # Load SDOH indices for this tract
        sdoh_data = self.sdoh_loader.get_tract_data(census_tract)
        
        if not sdoh_data:
            self.logger.warning(f"No SDOH data for census tract {census_tract}")
            return {}
        
        # Apply differential privacy (add Laplace noise)
        if self.config.differential_privacy_epsilon < 1.0:
            sdoh_data = self._apply_differential_privacy(sdoh_data)
        
        # Compute composite risk scores
        sdoh_data['food_insecurity_risk'] = self._compute_food_insecurity_score(sdoh_data)
        sdoh_data['financial_constraint_risk'] = self._compute_financial_constraint_score(sdoh_data)
        sdoh_data['mobility_limitation_risk'] = self._compute_mobility_score(sdoh_data)
        sdoh_data['health_risk'] = self._compute_health_risk_score(sdoh_data)
        
        return sdoh_data
    
    def _geocode_to_census_tract(self, address: Dict[str, str]) -> Optional[str]:
        """
        Geocode address to census tract
        
        For simulation: sample from census tract distribution
        For production: use geocoding API
        """
        zip_code = address.get('zip_code')
        if zip_code:
            # Map ZIP to census tract (simplified)
            return self.sdoh_loader.zip_to_tract(zip_code)
        
        # If census_tract provided directly (for simulation)
        return address.get('census_tract')
    
    def _apply_differential_privacy(self, sdoh_data: Dict[str, float]) -> Dict[str, float]:
        """
        Apply differential privacy to SDOH signals
        
        Adds Laplace noise: Lap(sensitivity / epsilon)
        """
        epsilon = self.config.differential_privacy_epsilon
        
        # Add noise to numeric fields
        noisy_data = sdoh_data.copy()
        for key, value in sdoh_data.items():
            if isinstance(value, (int, float)):
                sensitivity = 1.0  # Assume normalized [0, 1]
                noise = np.random.laplace(0, sensitivity / epsilon)
                noisy_data[key] = np.clip(value + noise, 0, 1)
        
        return noisy_data
    
    def _compute_food_insecurity_score(self, sdoh_data: Dict[str, Any]) -> float:
        """
        Compute food insecurity risk score from SDOH indices
        
        Combines:
        - SVI (Social Vulnerability Index)
        - Food access metrics
        - Income indicators
        """
        score = 0.0
        
        # SVI component (0-1)
        svi = sdoh_data.get('svi', 0.0)
        score += svi * 0.4
        
        # Food access component
        food_access = sdoh_data.get('food_access_score', 0.0)
        score += (1 - food_access) * 0.3  # Lower access = higher risk
        
        # Income component
        median_income = sdoh_data.get('median_income', 50000)
        income_risk = 1.0 - min(median_income / 75000, 1.0)  # Normalize
        score += income_risk * 0.3
        
        return np.clip(score, 0, 1)
    
    def _compute_financial_constraint_score(self, sdoh_data: Dict[str, Any]) -> float:
        """Compute financial constraint risk score"""
        score = 0.0
        
        # Income
        median_income = sdoh_data.get('median_income', 50000)
        score += (1.0 - min(median_income / 75000, 1.0)) * 0.5
        
        # Poverty rate
        poverty_rate = sdoh_data.get('poverty_rate', 0.0)
        score += poverty_rate * 0.3
        
        # Unemployment
        unemployment = sdoh_data.get('unemployment_rate', 0.0)
        score += unemployment * 0.2
        
        return np.clip(score, 0, 1)
    
    def _compute_mobility_score(self, sdoh_data: Dict[str, Any]) -> float:
        """Compute mobility limitation risk score"""
        score = 0.0
        
        # Transit availability (lower = higher risk)
        transit_score = sdoh_data.get('transit_score', 0.5)
        score += (1 - transit_score) * 0.5
        
        # Elderly population (proxy for mobility issues)
        elderly_pct = sdoh_data.get('elderly_percentage', 0.0)
        score += elderly_pct * 0.3
        
        # Disability rate
        disability_rate = sdoh_data.get('disability_rate', 0.0)
        score += disability_rate * 0.2
        
        return np.clip(score, 0, 1)
    
    def _compute_health_risk_score(self, sdoh_data: Dict[str, Any]) -> float:
        """Compute health risk score"""
        score = 0.0
        
        # Chronic disease prevalence
        chronic_disease = sdoh_data.get('chronic_disease_rate', 0.0)
        score += chronic_disease * 0.4
        
        # Healthcare access
        healthcare_access = sdoh_data.get('healthcare_access_score', 0.5)
        score += (1 - healthcare_access) * 0.3
        
        # Health insurance coverage
        uninsured_rate = sdoh_data.get('uninsured_rate', 0.0)
        score += uninsured_rate * 0.3
        
        return np.clip(score, 0, 1)
    
    def _extract_behavioral_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract behavioral features from cart and user history
        """
        cart = context['cart']
        
        return {
            'cart_size': cart['total_items'],
            'cart_value': cart['total_price'],
            'avg_item_price': cart['avg_price_per_item'],
            'category_diversity': len(cart['categories']),
            'snap_eligible_ratio': cart['snap_eligible_ratio'],
            'has_snap_payment': 'SNAP_EBT' in context['payment_methods'],
            'has_fsa_hsa': any(pm in ['FSA', 'HSA'] for pm in context['payment_methods']),
            'time_of_day': self._get_time_of_day(context['timestamp']),
            'day_of_week': self._get_day_of_week(context['timestamp'])
        }
    
    def _detect_constraints(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect user constraints from context
        """
        cart = context['cart']
        behavioral = context['behavioral']
        sdoh = context.get('sdoh', {})
        
        return {
            'budget_constrained': (
                behavioral['avg_item_price'] < 5.0 or
                sdoh.get('financial_constraint_risk', 0) > 0.7
            ),
            'nutrition_concerned': (
                any(item.get('glycemic_index') for item in cart['items'])
            ),
            'mobility_limited': sdoh.get('mobility_limitation_risk', 0) > 0.6,
            'food_insecure': sdoh.get('food_insecurity_risk', 0) > 0.7,
            'health_at_risk': sdoh.get('health_risk', 0) > 0.6
        }
    
    def _build_feature_vector(self, context: Dict[str, Any]) -> np.ndarray:
        """
        Build feature vector for contextual bandit
        
        Combines:
        - Cart features
        - SDOH signals
        - Behavioral features
        - Constraints
        """
        features = []
        
        # Cart features (10 dims)
        cart = context['cart']
        features.extend([
            cart['total_items'] / 50.0,  # Normalize
            cart['total_price'] / 200.0,
            cart['unique_items'] / 30.0,
            cart['snap_eligible_ratio'],
            cart['avg_price_per_item'] / 10.0,
            len(cart['categories']) / 10.0,
            float('SNAP_EBT' in context['payment_methods']),
            float('FSA' in context['payment_methods'] or 'HSA' in context['payment_methods']),
            0.0,  # Reserved
            0.0   # Reserved
        ])
        
        # SDOH features (8 dims)
        sdoh = context.get('sdoh', {})
        features.extend([
            sdoh.get('food_insecurity_risk', 0.0),
            sdoh.get('financial_constraint_risk', 0.0),
            sdoh.get('mobility_limitation_risk', 0.0),
            sdoh.get('health_risk', 0.0),
            sdoh.get('svi', 0.0),
            sdoh.get('adi', 0.0),
            sdoh.get('food_access_score', 0.5),
            sdoh.get('transit_score', 0.5)
        ])
        
        # Behavioral features (6 dims)
        behavioral = context['behavioral']
        features.extend([
            behavioral['time_of_day'] / 24.0,
            behavioral['day_of_week'] / 7.0,
            behavioral['has_snap_payment'],
            behavioral['has_fsa_hsa'],
            0.0,  # Reserved
            0.0   # Reserved
        ])
        
        # Constraint flags (5 dims)
        constraints = context['constraints']
        features.extend([
            float(constraints['budget_constrained']),
            float(constraints['nutrition_concerned']),
            float(constraints['mobility_limited']),
            float(constraints['food_insecure']),
            float(constraints['health_at_risk'])
        ])
        
        # Pad to context_dim
        while len(features) < self.config.bandit_context_dim:
            features.append(0.0)
        
        return np.array(features[:self.config.bandit_context_dim], dtype=np.float32)
    
    @staticmethod
    def _get_time_of_day(timestamp: float) -> int:
        """Get hour of day from timestamp"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).hour
    
    @staticmethod
    def _get_day_of_week(timestamp: float) -> int:
        """Get day of week from timestamp (0=Monday)"""
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).weekday()
