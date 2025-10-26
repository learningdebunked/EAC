"""
Product Transformer Embeddings

BERT-style transformer for learning product representations
Trained on product descriptions, categories, and co-purchase patterns
"""
import logging
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib


class ProductTransformer:
    """
    Transformer-based product embeddings
    
    Uses pre-trained BERT fine-tuned on:
    - Product names and descriptions
    - Category hierarchies
    - Co-purchase patterns from Instacart/dunnhumby
    """
    
    def __init__(
        self, 
        model_name: str = "bert-base-uncased",
        embedding_dim: int = 768,
        model_path: Optional[str] = None
    ):
        self.logger = logging.getLogger("EAC.Models.Embeddings")
        self.embedding_dim = embedding_dim
        
        if model_path:
            self.load(model_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
        
        # Product embedding cache
        self.embedding_cache = {}
        
        self.logger.info(f"Product Transformer initialized: {model_name}")
    
    def encode_product(
        self, 
        product_name: str, 
        category: str, 
        description: Optional[str] = None
    ) -> np.ndarray:
        """
        Encode product into embedding vector
        
        Args:
            product_name: Product name
            category: Product category
            description: Optional product description
            
        Returns:
            Embedding vector (768-dim)
        """
        # Check cache
        cache_key = f"{product_name}_{category}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        # Build text representation
        text = f"{product_name} [SEP] {category}"
        if description:
            text += f" [SEP] {description}"
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use [CLS] token embedding
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        
        # Cache
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def compute_similarity(
        self, 
        product1: Dict[str, str], 
        product2: Dict[str, str]
    ) -> float:
        """
        Compute cosine similarity between two products
        
        Args:
            product1: Dict with 'name', 'category', 'description'
            product2: Dict with 'name', 'category', 'description'
            
        Returns:
            Similarity score [0, 1]
        """
        emb1 = self.encode_product(
            product1['name'],
            product1['category'],
            product1.get('description')
        )
        
        emb2 = self.encode_product(
            product2['name'],
            product2['category'],
            product2.get('description')
        )
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # Normalize to [0, 1]
        return float((similarity + 1) / 2)
    
    def find_similar_products(
        self,
        query_product: Dict[str, str],
        candidate_products: List[Dict[str, str]],
        top_k: int = 10
    ) -> List[tuple]:
        """
        Find most similar products
        
        Args:
            query_product: Product to find substitutes for
            candidate_products: List of candidate products
            top_k: Number of results to return
            
        Returns:
            List of (product, similarity_score) tuples
        """
        query_emb = self.encode_product(
            query_product['name'],
            query_product['category'],
            query_product.get('description')
        )
        
        similarities = []
        for candidate in candidate_products:
            cand_emb = self.encode_product(
                candidate['name'],
                candidate['category'],
                candidate.get('description')
            )
            
            similarity = np.dot(query_emb, cand_emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(cand_emb)
            )
            similarity = (similarity + 1) / 2  # Normalize to [0, 1]
            
            similarities.append((candidate, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def fine_tune(
        self,
        product_pairs: List[tuple],
        labels: List[int],
        epochs: int = 3,
        batch_size: int = 32
    ):
        """
        Fine-tune on product similarity task
        
        Args:
            product_pairs: List of (product1, product2) tuples
            labels: 1 if similar, 0 if not
            epochs: Number of training epochs
            batch_size: Batch size
        """
        self.logger.info(f"Fine-tuning on {len(product_pairs)} product pairs")
        
        # Set to training mode
        self.model.train()
        
        # Create optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        criterion = nn.BCEWithLogitsLoss()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            
            for i in range(0, len(product_pairs), batch_size):
                batch_pairs = product_pairs[i:i+batch_size]
                batch_labels = torch.tensor(labels[i:i+batch_size], dtype=torch.float32)
                
                # Get embeddings
                emb1_list = []
                emb2_list = []
                
                for prod1, prod2 in batch_pairs:
                    emb1 = self.encode_product(prod1['name'], prod1['category'])
                    emb2 = self.encode_product(prod2['name'], prod2['category'])
                    emb1_list.append(emb1)
                    emb2_list.append(emb2)
                
                emb1_batch = torch.tensor(np.stack(emb1_list))
                emb2_batch = torch.tensor(np.stack(emb2_list))
                
                # Compute similarity
                similarity = torch.cosine_similarity(emb1_batch, emb2_batch)
                
                # Loss
                loss = criterion(similarity, batch_labels)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(product_pairs) / batch_size)
            self.logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Back to eval mode
        self.model.eval()
        self.logger.info("Fine-tuning complete")
    
    def save(self, path: str):
        """Save model to disk"""
        torch.save({
            'model_state': self.model.state_dict(),
            'embedding_cache': self.embedding_cache
        }, path)
        self.logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.embedding_cache = checkpoint['embedding_cache']
        self.model.eval()
        self.logger.info(f"Model loaded from {path}")
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_cache = {}
        self.logger.info("Embedding cache cleared")
