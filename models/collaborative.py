"""
Collaborative Filtering for User Similarity

Matrix factorization and neural collaborative filtering
Trained on Instacart/dunnhumby purchase histories
"""
import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
import torch
import torch.nn as nn
import joblib


class CollaborativeFilter:
    """
    Collaborative filtering for personalized recommendations
    
    Methods:
    1. Matrix Factorization (NMF) for user-product interactions
    2. Neural Collaborative Filtering for deep patterns
    """
    
    def __init__(
        self,
        n_factors: int = 50,
        method: str = "nmf",  # "nmf" or "neural"
        model_path: Optional[str] = None
    ):
        self.logger = logging.getLogger("EAC.Models.CollaborativeFilter")
        self.n_factors = n_factors
        self.method = method
        self.device=""

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            #self.model = self.model.to(self.device)
            self.logger.info("Using Apple Metal (MPS) for GPU acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            #self.model = self.model.to(self.device)
            self.logger.info("Using CUDA for GPU acceleration")
        else:
            self.device = torch.device("cpu")
    

        if model_path:
            self.load(model_path)
        else:
            if method == "nmf":
                self.model = NMF(
                    n_components=n_factors,
                    init='nndsvda',
                    max_iter=500,
                    random_state=42
                )
            elif method == "neural":
                self.model = NeuralCF(n_users=5,n_items=5,n_factors=n_factors)
            
            self.user_factors = None
            self.item_factors = None
            self.user_to_idx = {}
            self.item_to_idx = {}
        
        self.logger.info(f"Collaborative Filter initialized: {method}")
    
    def train(self, transactions: pd.DataFrame, min_user_purchases: int = 1, min_product_users: int = 1):
        """
        Train on transaction data
        
        Args:
            transactions: DataFrame with columns:
                - user_id
                - product_id
                - quantity (or implicit feedback)
            min_user_purchases: Minimum purchases per user to include (default: 10)
            min_product_users: Minimum users per product to include (default: 5)
        """
        self.logger.info(f"Training on {len(transactions)} transactions")
        original_transactions = len(transactions)
        original_users = transactions['user_id'].nunique()
        original_products = transactions['product_id'].nunique()
        
        # Filter: Keep only active users
        user_counts = transactions.groupby('user_id').size()
        active_users = user_counts[user_counts >= min_user_purchases].index
        transactions = transactions[transactions['user_id'].isin(active_users)]
        self.logger.info(f"Filtered to {len(active_users):,} active users (min {min_user_purchases} purchases)")
        
        # Filter: Keep only popular products
        product_counts = transactions.groupby('product_id').size()
        popular_products = product_counts[product_counts >= min_product_users].index
        transactions = transactions[transactions['product_id'].isin(popular_products)]
        self.logger.info(f"Filtered to {len(popular_products):,} popular products (min {min_product_users} users)")
        
        self.logger.info(f"Filtered data: {len(transactions):,} transactions ({len(transactions)/original_transactions:.1%} of original)")
        print(transactions.describe())
        # Build user-item matrix
        user_item_matrix, self.user_to_idx, self.item_to_idx = self._build_matrix(
            transactions
        )
        
        if self.method == "nmf":
            # Matrix factorization
            self.user_factors = self.model.fit_transform(user_item_matrix)
            self.item_factors = self.model.components_.T
            
            reconstruction_error = self.model.reconstruction_err_
            self.logger.info(f"Training complete: reconstruction_error={reconstruction_error:.4f}")
        
        elif self.method == "neural":
            # Neural collaborative filtering
            self._train_neural(user_item_matrix)
        
        return {
            'n_users': len(self.user_to_idx),
            'n_items': len(self.item_to_idx),
            'n_interactions': user_item_matrix.nnz,
            'original_users': original_users,
            'original_products': original_products,
            'original_transactions': original_transactions,
            'avg_purchases_per_user': user_item_matrix.nnz / len(self.user_to_idx),
            'avg_users_per_product': user_item_matrix.nnz / len(self.item_to_idx),
            'sparsity': 1 - (user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])),
            'density': user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
        }
    
    def _build_matrix(self, transactions: pd.DataFrame) -> tuple:
        """Build sparse user-item interaction matrix"""
        # Create mappings
        users = transactions['user_id'].unique()
        items = transactions['product_id'].unique()
        
        user_to_idx = {user: idx for idx, user in enumerate(users)}
        item_to_idx = {item: idx for idx, item in enumerate(items)}
        
        # Build matrix
        rows = transactions['user_id'].map(user_to_idx)
        cols = transactions['product_id'].map(item_to_idx)
        data = transactions.get('quantity', 1)  # Use quantity or implicit 1
        
        matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(len(users), len(items))
        )
        
        return matrix, user_to_idx, item_to_idx
    

    def _train_neural(self, user_item_matrix: csr_matrix):
        """Train neural collaborative filtering"""
    #     if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
        # Convert to dense for training (or use batch sampling for large datasets)
        X_dense = user_item_matrix.toarray()
        
        # Training loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        #X_tensor = torch.FloatTensor(X_dense)
        X_tensor = torch.tensor(X_dense, dtype=torch.float32, device=self.device)
        self.model = self.model.to(self.device)
        for epoch in range(50):
            self.model.train()
            
            # Forward
            predictions = self.model(
                torch.arange(X_tensor.shape[0]),
                torch.arange(X_tensor.shape[1])
            )
            
            # Loss
            loss = criterion(predictions, X_tensor)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
        
        self.model.eval()
    
    def get_similar_users(self, user_id: str, top_k: int = 10) -> List[tuple]:
        """
        Find similar users based on purchase patterns
        
        Args:
            user_id: User ID
            top_k: Number of similar users to return
            
        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user_to_idx:
            self.logger.warning(f"User {user_id} not in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Compute similarities with all users
        similarities = np.dot(self.user_factors, user_vector)
        similarities = similarities / (
            np.linalg.norm(self.user_factors, axis=1) * np.linalg.norm(user_vector)
        )
        
        # Get top-k (excluding self)
        similar_indices = np.argsort(similarities)[::-1][1:top_k+1]
        
        # Map back to user IDs
        idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        similar_users = [
            (idx_to_user[idx], float(similarities[idx]))
            for idx in similar_indices
        ]
        
        return similar_users
    
    def recommend_products(
        self, 
        user_id: str, 
        exclude_purchased: bool = True,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Recommend products for a user
        
        Args:
            user_id: User ID
            exclude_purchased: Exclude already purchased items
            top_k: Number of recommendations
            
        Returns:
            List of (product_id, score) tuples
        """
        
        user_id = np.int64(user_id)

        if user_id not in self.user_to_idx:
            self.logger.warning(f"User {user_id} not in training data")
            return []
        
        user_idx = self.user_to_idx[user_id]
        user_vector = self.user_factors[user_idx]
        
        # Compute scores for all items
        scores = np.dot(self.item_factors, user_vector)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Map back to product IDs
        idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        recommendations = [
            (idx_to_item[idx], float(scores[idx]))
            for idx in top_indices
        ]
        
        return recommendations
    
    def predict_interaction(self, user_id: str, product_id: str) -> float:
        """
        Predict interaction strength between user and product
        
        Returns:
            Predicted score
        """
        if user_id not in self.user_to_idx or product_id not in self.item_to_idx:
            return 0.0
        
        user_idx = self.user_to_idx[user_id]
        item_idx = self.item_to_idx[product_id]
        
        score = np.dot(
            self.user_factors[user_idx],
            self.item_factors[item_idx]
        )
        
        return float(score)
    
    def save(self, path: str):
        """Save model to disk"""
        joblib.dump({
            'model': self.model,
            'user_factors': self.user_factors,
            'item_factors': self.item_factors,
            'user_to_idx': self.user_to_idx,
            'item_to_idx': self.item_to_idx,
            'method': self.method
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        data = joblib.load(path)
        self.model = data['model']
        self.user_factors = data['user_factors']
        self.item_factors = data['item_factors']
        self.user_to_idx = data['user_to_idx']
        self.item_to_idx = data['item_to_idx']
        self.method = data['method']
        self.logger.info(f"Model loaded from {path}")


class NeuralCF(nn.Module):
    """Neural Collaborative Filtering model"""
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 50):
        self.logger = logging.getLogger(__name__)
        super().__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.device=""
        # Detect and use MPS if available
        #self.logger = logging.getLogger(__name__)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            #self.model = self.model.to(self.device)
            self.logger.info("Using Apple Metal (MPS) for GPU acceleration")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            #self.model = self.model.to(self.device)
            self.logger.info("Using CUDA for GPU acceleration")
        else:
            self.device = torch.device("cpu")
            self.logger.info("Using CPU")
        # MLP layers
        self.fc_layers = nn.Sequential(
            nn.Linear(n_factors * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Concatenate embeddings
        x = torch.cat([user_emb, item_emb], dim=-1)
        
        # MLP
        output = self.fc_layers(x)
        
        return output.squeeze()
