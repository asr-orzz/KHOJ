#!/usr/bin/env python3
"""
CADENCE Backend API Server
Production-ready FastAPI backend for query autocomplete and product search
"""
import os
import json
import pickle
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import structlog

from real_cadence_training import RealCADENCEModel

logger = structlog.get_logger()

# Request/Response models
class AutocompleteRequest(BaseModel):
    query: str
    max_suggestions: int = 10
    category: Optional[str] = None

class AutocompleteResponse(BaseModel):
    suggestions: List[str]
    query: str
    category: Optional[str]
    processing_time_ms: float

class SearchRequest(BaseModel):
    query: str
    max_results: int = 20
    category_filter: Optional[str] = None
    sort_by: str = "relevance"  # relevance, rating, price

class ProductResult(BaseModel):
    product_id: str
    title: str
    processed_title: str
    category: str
    cluster_description: str
    price: Optional[float] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    relevance_score: float

class SearchResponse(BaseModel):
    results: List[ProductResult]
    query: str
    total_results: int
    processing_time_ms: float
    category_filter: Optional[str]

class CADENCEModelManager:
    """Manages loaded CADENCE models and provides inference capabilities"""
    
    def __init__(self):
        self.model = None
        self.vocab = None
        self.reverse_vocab = None
        self.cluster_info = None
        self.config = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Cached data
        self.query_data = None
        self.product_data = None
        
        logger.info(f"🔧 Model manager initialized on {self.device}")
    
    def load_models(self, model_dir: str = "trained_models", data_dir: str = "processed_data"):
        """Load all trained models and data"""
        logger.info("📂 Loading CADENCE models and data...")
        
        model_path = Path(model_dir)
        data_path = Path(data_dir)
        
        # Load model configuration
        config_file = model_path / "real_cadence_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Model config not found: {config_file}")
        
        with open(config_file, 'r') as f:
            self.config = json.load(f)
        
        # Load vocabulary
        vocab_file = model_path / "real_cadence_vocab.pkl"
        if not vocab_file.exists():
            raise FileNotFoundError(f"Vocabulary not found: {vocab_file}")
            
        with open(vocab_file, 'rb') as f:
            self.vocab = pickle.load(f)
        
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # Load cluster information
        cluster_file = data_path / "cluster_mappings.json"
        if cluster_file.exists():
            with open(cluster_file, 'r') as f:
                self.cluster_info = json.load(f)
        
        # Create and load model
        vocab_size = len(self.vocab)
        num_categories = self.config['num_categories']
        
        self.model = RealCADENCEModel(vocab_size, num_categories, self.config['model_config'])
        
        # Load model weights
        model_file = model_path / "real_cadence.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model weights not found: {model_file}")
        
        self.model.load_state_dict(torch.load(model_file, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Load processed data for search
        if (data_path / "queries.parquet").exists():
            self.query_data = pd.read_parquet(data_path / "queries.parquet")
            logger.info(f"   Loaded {len(self.query_data):,} queries")
        
        if (data_path / "products.parquet").exists():
            self.product_data = pd.read_parquet(data_path / "products.parquet")
            logger.info(f"   Loaded {len(self.product_data):,} products")
        
        logger.info("✅ All models and data loaded successfully")
        logger.info(f"   Model parameters: {self.model.count_parameters()/1e6:.1f}M")
        logger.info(f"   Vocabulary size: {vocab_size:,}")
        logger.info(f"   Categories: {num_categories}")
    
    def preprocess_query(self, query: str) -> str:
        """Preprocess query text (same as training)"""
        import re
        from nltk.tokenize import word_tokenize
        from nltk.stem import WordNetLemmatizer
        from nltk.corpus import stopwords
        
        if not query:
            return ""
        
        # Basic preprocessing
        query = query.lower().strip()
        query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
        
        # Tokenize and clean
        try:
            tokens = word_tokenize(query)
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
            important_words = {'for', 'with', 'to', 'under', 'below', 'above', 'over', 'best', 'top'}
            tokens = [lemmatizer.lemmatize(token) for token in tokens 
                     if len(token) > 1 and (token not in stop_words or token in important_words)]
            
            return ' '.join(tokens)
        except:
            # Fallback if NLTK fails
            return ' '.join([token for token in query.split() if len(token) > 1])
    
    def tokenize_query(self, query: str) -> List[int]:
        """Convert query to token IDs"""
        processed = self.preprocess_query(query)
        tokens = processed.split()
        
        token_ids = [self.vocab.get('<s>', 3)]  # BOS
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab.get('<UNK>', 1)))
        
        return token_ids
    
    def generate_autocomplete(self, query: str, max_suggestions: int = 10, 
                            category: Optional[str] = None) -> List[str]:
        """Generate autocomplete suggestions using Query LM"""
        if not self.model:
            raise RuntimeError("Model not loaded")
        
        with torch.no_grad():
            # Tokenize input
            token_ids = self.tokenize_query(query)
            
            # Determine category ID
            category_id = 0  # Default category
            if category and self.cluster_info:
                # Find category in cluster mappings
                for cid, desc in self.cluster_info.get('query_clusters', {}).items():
                    if category.lower() in desc.lower():
                        category_id = int(cid)
                        break
            
            # Prepare input tensors
            input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
            category_ids = torch.full_like(input_ids, category_id, device=self.device)
            
            # Generate suggestions
            suggestions = []
            
            for _ in range(max_suggestions * 2):  # Generate more to filter
                # Forward pass
                outputs = self.model(input_ids, category_ids, model_type='query')
                logits = outputs['logits']
                
                # Get next token probabilities
                next_token_logits = logits[0, -1, :]
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token (with temperature for diversity)
                temperature = 0.8
                next_token_probs = next_token_probs / temperature
                next_token_probs = F.softmax(next_token_probs, dim=-1)
                
                # Get top candidates
                top_k = min(50, len(next_token_probs))
                top_probs, top_indices = torch.topk(next_token_probs, top_k)
                
                # Sample from top candidates
                sampled_idx = torch.multinomial(top_probs, 1).item()
                next_token_id = top_indices[sampled_idx].item()
                
                # Convert to word
                next_word = self.reverse_vocab.get(next_token_id, '<UNK>')
                
                # Check for end of sequence
                if next_word in ['</s>', '<PAD>']:
                    break
                
                # Add to current sequence
                if next_word != '<UNK>':
                    current_tokens = [self.reverse_vocab.get(tid, '') for tid in token_ids[1:]]  # Skip BOS
                    current_tokens.append(next_word)
                    suggestion = ' '.join([t for t in current_tokens if t and t not in ['<s>', '</s>']])
                    
                    if suggestion not in suggestions and len(suggestion.split()) > len(query.split()):
                        suggestions.append(suggestion)
                
                # Update input for next token
                token_ids.append(next_token_id)
                input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
                category_ids = torch.full_like(input_ids, category_id, device=self.device)
                
                if len(suggestions) >= max_suggestions:
                    break
            
            return suggestions[:max_suggestions]
    
    def search_products(self, query: str, max_results: int = 20, 
                       category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search products using processed query and similarity matching"""
        if self.product_data is None:
            raise RuntimeError("Product data not loaded")
        
        processed_query = self.preprocess_query(query)
        query_tokens = set(processed_query.lower().split())
        
        # Calculate relevance scores
        results = []
        
        for _, product in self.product_data.iterrows():
            # Skip if category filter doesn't match
            if category_filter:
                product_category = product.get('cluster_description', '').lower()
                if category_filter.lower() not in product_category:
                    continue
            
            # Calculate relevance score
            product_tokens = set(product['processed_title'].lower().split())
            
            # Jaccard similarity
            intersection = len(query_tokens.intersection(product_tokens))
            union = len(query_tokens.union(product_tokens))
            jaccard_score = intersection / union if union > 0 else 0
            
            # Boost score if query tokens appear in title
            exact_matches = sum(1 for token in query_tokens if token in product['processed_title'].lower())
            exact_match_score = exact_matches / len(query_tokens) if query_tokens else 0
            
            # Combined relevance score
            relevance_score = (jaccard_score * 0.6) + (exact_match_score * 0.4)
            
            if relevance_score > 0.1:  # Minimum relevance threshold
                results.append({
                    'product_id': product['product_id'],
                    'title': product['original_title'],
                    'processed_title': product['processed_title'],
                    'category': product.get('main_category', 'Unknown'),
                    'cluster_description': product.get('cluster_description', ''),
                    'price': product.get('price'),
                    'rating': product.get('rating'),
                    'rating_count': product.get('rating_count'),
                    'relevance_score': relevance_score
                })
        
        # Sort by relevance score
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return results[:max_results]

# Initialize model manager
model_manager = CADENCEModelManager()

# Create FastAPI app
app = FastAPI(
    title="CADENCE API",
    description="Real-time Query Autocomplete and Product Search API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    try:
        model_manager.load_models()
        logger.info("🚀 CADENCE API server ready!")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CADENCE API Server",
        "version": "1.0.0",
        "status": "ready",
        "endpoints": {
            "autocomplete": "/api/v1/autocomplete",
            "search": "/api/v1/search",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.model is not None,
        "device": str(model_manager.device),
        "vocab_size": len(model_manager.vocab) if model_manager.vocab else 0
    }

@app.post("/api/v1/autocomplete", response_model=AutocompleteResponse)
async def get_autocomplete(request: AutocompleteRequest):
    """Get autocomplete suggestions for a query"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not request.query or len(request.query.strip()) < 2:
            raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
        
        suggestions = model_manager.generate_autocomplete(
            request.query, 
            request.max_suggestions,
            request.category
        )
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return AutocompleteResponse(
            suggestions=suggestions,
            query=request.query,
            category=request.category,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Autocomplete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/search", response_model=SearchResponse)
async def search_products(request: SearchRequest):
    """Search products based on query"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        if not request.query or len(request.query.strip()) < 2:
            raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
        
        results = model_manager.search_products(
            request.query,
            request.max_results,
            request.category_filter
        )
        
        # Convert to ProductResult objects
        product_results = [
            ProductResult(**result) for result in results
        ]
        
        processing_time = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return SearchResponse(
            results=product_results,
            query=request.query,
            total_results=len(product_results),
            processing_time_ms=processing_time,
            category_filter=request.category_filter
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/categories")
async def get_categories():
    """Get available product categories"""
    try:
        categories = []
        
        if model_manager.cluster_info:
            # Get query categories
            query_categories = [
                {"id": cid, "name": desc, "type": "query"}
                for cid, desc in model_manager.cluster_info.get('query_clusters', {}).items()
            ]
            
            # Get product categories
            product_categories = [
                {"id": cid, "name": desc, "type": "product"}
                for cid, desc in model_manager.cluster_info.get('product_clusters', {}).items()
            ]
            
            categories = {
                "query_categories": query_categories,
                "product_categories": product_categories
            }
        
        return categories
        
    except Exception as e:
        logger.error(f"Categories error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "model_info": {
                "parameters": model_manager.model.count_parameters() if model_manager.model else 0,
                "vocab_size": len(model_manager.vocab) if model_manager.vocab else 0,
                "device": str(model_manager.device)
            },
            "data_info": {
                "queries": len(model_manager.query_data) if model_manager.query_data is not None else 0,
                "products": len(model_manager.product_data) if model_manager.product_data is not None else 0
            },
            "cluster_info": model_manager.cluster_info if model_manager.cluster_info else {}
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Run the server"""
    logger.info("🚀 Starting CADENCE API server...")
    
    # Check if models exist
    if not Path("trained_models/real_cadence.pt").exists():
        logger.error("❌ Trained models not found!")
        logger.info("Please run the training pipeline first:")
        logger.info("1. python real_cadence_training.py")
        logger.info("2. python real_model_training.py")
        return
    
    # Run server
    uvicorn.run(
        "cadence_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()