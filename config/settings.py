"""
Configuration settings for Enhanced CADENCE System
"""
import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///./cadence.db", env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Gemini API
    GEMINI_API_KEY: str = "AIzaSyCpQLaZJwQwYTAB7UFDPxxCnx_ABwDXez0"

    
    # Model Configuration
    MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384
    MAX_QUERY_LENGTH: int = 50
    BEAM_WIDTH: int = 5
    
    # CADENCE Specific
    VOCAB_SIZE: int = 50000
    HIDDEN_DIMS: List[int] = [2000, 1500, 1000]
    ATTENTION_DIMS: List[int] = [1000, 750, 500]
    DROPOUT_RATE: float = 0.8
    LEARNING_RATE: float = 1e-5
    BATCH_SIZE: int = 128
    
    # Clustering Configuration
    N_CLUSTERS: int = 50  # Pseudo-categories
    MIN_CLUSTER_SIZE: int = 15
    MIN_SAMPLES: int = 5
    
    # Personalization
    MAX_USER_HISTORY: int = 1000
    ENGAGEMENT_WEIGHTS: Dict[str, float] = {
        "view": 1.0,
        "click": 2.0,
        "add_to_cart": 5.0,
        "wishlist": 3.0,
        "purchase": 10.0,
        "review_view": 1.5,
        "specs_view": 2.0,
        "scroll": 0.5
    }
    
    # Performance Settings
    MAX_SUGGESTIONS: int = 10
    CACHE_TTL: int = 300  # 5 minutes
    MAX_LATENCY_MS: int = 100  # 100ms max response time
    
    # A/B Testing
    ENABLE_AB_TESTING: bool = True
    DEFAULT_TREATMENT: str = "enhanced_cadence"
    
    # Synthetic Data Generation
    SYNTHETIC_BATCH_SIZE: int = 1000
    MAX_SYNTHETIC_QUERIES: int = 100000
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra environment variables

# Global settings instance
settings = Settings()

# Category mappings for e-commerce
ECOMMERCE_CATEGORIES = {
    0: "Electronics",
    1: "Clothing & Fashion", 
    2: "Home & Kitchen",
    3: "Sports & Outdoors",
    4: "Books & Media",
    5: "Beauty & Personal Care",
    6: "Automotive",
    7: "Toys & Games",
    8: "Health & Household",
    9: "Tools & Home Improvement",
    10: "Pet Supplies",
    11: "Baby Products",
    12: "Grocery & Gourmet Food",
    13: "Arts & Crafts",
    14: "Office Products",
    15: "Jewelry & Accessories"
}

# Engagement action types
ENGAGEMENT_ACTIONS = [
    "view", "click", "add_to_cart", "remove_from_cart",
    "wishlist", "remove_wishlist", "purchase", "return",
    "review_view", "specs_view", "image_view", "scroll",
    "search", "filter_apply", "sort_change"
]

# Feature flags
FEATURE_FLAGS = {
    "enable_clustering_categories": True,
    "enable_collaborative_filtering": True,
    "enable_location_based_ranking": True,
    "enable_time_decay": True,
    "enable_cold_start_handling": True,
    "enable_real_time_updates": True
} 