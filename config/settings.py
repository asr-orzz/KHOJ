"""
Configuration settings for KHOJ+ System
Bharat-first Search & Recommendations for Meesho
"""
import os
from typing import List, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///./khoj.db", env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # Gemini API (for enhanced processing)
    GEMINI_API_KEY: str = "AIzaSyCpQLaZJwQwYTAB7UFDPxxCnx_ABwDXez0"
    
    # KHOJ+ Specific Configuration
    
    # Vernacular Processing
    ENABLE_VERNACULAR_PROCESSING: bool = True
    SUPPORTED_LANGUAGES: List[str] = ["en", "hi", "hinglish"]
    DEFAULT_LANGUAGE: str = "hinglish"
    
    # Trust & Quality Thresholds
    MIN_RATING_THRESHOLD: float = 3.5
    MAX_RETURN_RATE_THRESHOLD: float = 0.15
    MIN_REVIEW_COUNT: int = 5
    QUALITY_BADGE_BOOST: float = 1.2
    
    # Price Intent Bands (for Bharat market)
    PRICE_BANDS: Dict[str, Dict[str, float]] = {
        "budget": {"min": 0, "max": 500},
        "mid": {"min": 500, "max": 2000},
        "premium": {"min": 2000, "max": 10000}
    }
    
    # Ranking Weights
    RANKING_WEIGHTS: Dict[str, float] = {
        "relevance": 0.35,
        "trust": 0.25,
        "affordability": 0.15,
        "deliverability": 0.15,
        "personalization": 0.10
    }
    
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
    
    # Performance Settings (Meesho scale)
    MAX_SUGGESTIONS: int = 10
    CACHE_TTL: int = 300  # 5 minutes
    MAX_LATENCY_MS: int = 120  # 120ms max for autosuggest P95
    MAX_SEARCH_LATENCY_MS: int = 500  # 500ms max for search
    
    # A/B Testing & Experiments
    ENABLE_AB_TESTING: bool = True
    DEFAULT_TREATMENT: str = "khoj_plus"
    EXPERIMENT_TRAFFIC_SPLIT: float = 0.1  # 10% for experiments
    
    # SRP Strips Configuration
    ENABLE_SRP_STRIPS: bool = True
    MAX_STRIPS: int = 5
    PRODUCTS_PER_STRIP: int = 5
    
    # Seller Diversity
    MAX_RESULTS_PER_SELLER: int = 3
    MIN_SELLER_DIVERSITY_RATIO: float = 0.7
    
    # Deliverability (Bharat-specific)
    DEFAULT_DELIVERY_PROMISE_DAYS: int = 7
    FAST_DELIVERY_THRESHOLD_DAYS: int = 3
    COD_PREFERENCE_WEIGHT: float = 0.1
    
    # Synthetic Data Generation
    SYNTHETIC_BATCH_SIZE: int = 1000
    MAX_SYNTHETIC_QUERIES: int = 100000
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

# Global settings instance
settings = Settings()

# Enhanced Category mappings for Indian e-commerce
MEESHO_CATEGORIES = {
    0: "Women Clothing",
    1: "Men Clothing", 
    2: "Kids & Baby",
    3: "Home & Kitchen",
    4: "Electronics",
    5: "Beauty & Health",
    6: "Jewelry & Accessories",
    7: "Bags & Footwear",
    8: "Sports & Fitness",
    9: "Books & Education",
    10: "Automotive",
    11: "Pet Supplies",
    12: "Grocery & Gourmet",
    13: "Wedding & Festive",
    14: "Handicrafts",
    15: "Regional Specialties"
}

# Bharat-specific engagement actions
ENGAGEMENT_ACTIONS = [
    "view", "click", "add_to_cart", "remove_from_cart",
    "wishlist", "remove_wishlist", "purchase", "return",
    "review_view", "specs_view", "image_view", "scroll",
    "search", "filter_apply", "sort_change", "share",
    "price_comparison", "seller_view", "delivery_check",
    "cod_preference", "return_policy_check"
]

# Intent chip types for UI
INTENT_CHIP_TYPES = {
    "price": ["under_299", "under_499", "under_999", "budget_friendly"],
    "quality": ["4star_plus", "high_rated", "quality_badge"],
    "delivery": ["fast_delivery", "same_day", "cod"],
    "source": ["meesho_mall", "verified_seller"],
    "occasion": ["wedding", "festive", "casual", "formal", "party"],
    "policy": ["returnable", "exchange", "warranty"]
}

# Feature flags for KHOJ+
FEATURE_FLAGS = {
    "vernacular_processing": True,
    "trust_ranking": True,
    "intent_chips": True,
    "srp_strips": True,
    "personalization": True,
    "ab_testing": True,
    "price_intent_detection": True,
    "deliverability_ranking": True,
    "seller_diversity": True,
    "quality_guardrails": True,
    "clustering_categories": True,
    "collaborative_filtering": True,
    "location_based_ranking": True,
    "time_decay": True,
    "cold_start_handling": True,
    "real_time_updates": True
} 