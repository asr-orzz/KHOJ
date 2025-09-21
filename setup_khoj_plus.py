#!/usr/bin/env python3
"""
KHOJ+ Quick Setup Script
Sets up the complete KHOJ+ system for Meesho hackathon demo
"""
import os
import sys
import subprocess
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()

def setup_khoj_plus():
    """Setup KHOJ+ system for demo"""
    print("🚀 Setting up KHOJ+ - Bharat-first Search & Recommendations")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 9):
        print("❌ Python 3.9+ required. Current version:", sys.version)
        return False
    
    print("✅ Python version check passed")
    
    # Install dependencies
    print("\n📦 Installing KHOJ+ dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    
    # Download NLTK data
    print("\n📚 Downloading NLTK data for vernacular processing...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('averaged_perceptron_tagger', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK download warning: {e}")
    
    # Create necessary directories
    print("\n📁 Creating directory structure...")
    directories = [
        "trained_models",
        "processed_data", 
        "logs",
        "cache",
        "experiments"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Directory structure created")
    
    # Create environment file
    print("\n⚙️  Creating environment configuration...")
    env_content = """# KHOJ+ Environment Configuration
# Database
DATABASE_URL=sqlite:///./khoj.db
REDIS_URL=redis://localhost:6379

# API Configuration
API_HOST=localhost
API_PORT=8000
API_WORKERS=1

# Model Configuration
MODEL_CACHE_DIR=./trained_models
DATA_CACHE_DIR=./processed_data

# Feature Flags
ENABLE_VERNACULAR_PROCESSING=true
ENABLE_TRUST_RANKING=true
ENABLE_INTENT_CHIPS=true
ENABLE_SRP_STRIPS=true
ENABLE_AB_TESTING=true

# Performance
MAX_LATENCY_MS=120
MAX_SEARCH_LATENCY_MS=500
CACHE_TTL=300

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Meesho Specific
DEFAULT_PINCODE=560001
DEFAULT_LANGUAGE=hinglish
PRICE_CURRENCY=INR
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Environment configuration created")
    
    # Initialize database
    print("\n🗄️  Initializing database...")
    try:
        from database.connection import initialize_database
        initialize_database()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
    
    # Generate sample data
    print("\n🎲 Generating sample data for demo...")
    try:
        generate_sample_data()
        print("✅ Sample data generated")
    except Exception as e:
        print(f"⚠️  Sample data generation warning: {e}")
    
    print("\n🎯 KHOJ+ Setup Complete!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. Start the backend: python cadence_backend.py")
    print("2. Test autosuggest: curl -X POST http://localhost:8000/api/v2/autosuggest \\")
    print("   -H 'Content-Type: application/json' \\")
    print("   -d '{\"query_prefix\": \"sadi under 300\"}'")
    print("3. Open API docs: http://localhost:8000/docs")
    print("\n🌟 Demo Queries to Try:")
    print("   • 'sadi under 300' (vernacular + price intent)")
    print("   • 'kurti set wedding' (colloquial + occasion)")
    print("   • 'chasma' (transliteration)")
    print("   • 'cheap mobile phone' (price intent)")
    
    return True

def generate_sample_data():
    """Generate sample data for KHOJ+ demo"""
    
    # Sample Meesho products for demo
    sample_products = [
        {
            "product_id": "MSH001",
            "title": "Cotton Saree for Women - Wedding Collection",
            "brand": "MeeshoStyle",
            "category": "Women Clothing",
            "price": 299.0,
            "rating": 4.2,
            "review_count": 156,
            "return_rate": 0.08,
            "seller_id": "SELLER001",
            "seller_name": "Fashion Hub",
            "is_meesho_mall": True,
            "cod_available": True,
            "pincode_eta": 2,
            "main_category": "saree"
        },
        {
            "product_id": "MSH002", 
            "title": "Kurti Set with Dupatta - Festive Wear",
            "brand": "EthnicWear",
            "category": "Women Clothing",
            "price": 499.0,
            "rating": 4.1,
            "review_count": 89,
            "return_rate": 0.12,
            "seller_id": "SELLER002",
            "seller_name": "Ethnic Collection",
            "is_meesho_mall": False,
            "cod_available": True,
            "pincode_eta": 4,
            "main_category": "kurti"
        },
        {
            "product_id": "MSH003",
            "title": "Sunglasses for Men and Women - UV Protection",
            "brand": "StyleShades", 
            "category": "Accessories",
            "price": 199.0,
            "rating": 3.9,
            "review_count": 234,
            "return_rate": 0.15,
            "seller_id": "SELLER003",
            "seller_name": "Accessory World",
            "is_meesho_mall": True,
            "cod_available": True,
            "pincode_eta": 3,
            "main_category": "sunglasses"
        },
        {
            "product_id": "MSH004",
            "title": "Smartphone Case Cover - Clear Design",
            "brand": "TechGuard",
            "category": "Electronics",
            "price": 149.0,
            "rating": 4.0,
            "review_count": 67,
            "return_rate": 0.10,
            "seller_id": "SELLER004", 
            "seller_name": "Mobile Accessories",
            "is_meesho_mall": False,
            "cod_available": True,
            "pincode_eta": 5,
            "main_category": "mobile"
        },
        {
            "product_id": "MSH005",
            "title": "Bedsheet Double Bed Cotton - Floral Print",
            "brand": "HomeComfort",
            "category": "Home & Kitchen",
            "price": 399.0,
            "rating": 4.3,
            "review_count": 145,
            "return_rate": 0.06,
            "seller_id": "SELLER005",
            "seller_name": "Home Essentials",
            "is_meesho_mall": True,
            "cod_available": True,
            "pincode_eta": 3,
            "main_category": "bedsheet"
        }
    ]
    
    # Sample queries for training
    sample_queries = [
        "sadi under 300",
        "saree under 300", 
        "kurti set",
        "kurti set wedding",
        "chasma",
        "sunglasses",
        "mobile cover",
        "phone case",
        "bedsheet double",
        "bedsheet double bed",
        "cheap mobile",
        "affordable phone",
        "party wear",
        "wedding dress",
        "cotton saree",
        "silk kurti"
    ]
    
    # Save sample data
    os.makedirs("processed_data", exist_ok=True)
    
    # Save products
    with open("processed_data/sample_products.json", "w") as f:
        json.dump(sample_products, f, indent=2)
    
    # Save queries
    with open("processed_data/sample_queries.json", "w") as f:
        json.dump(sample_queries, f, indent=2)
    
    # Create basic vocabulary for demo
    vocab = {}
    for i, query in enumerate(sample_queries):
        for word in query.split():
            if word not in vocab:
                vocab[word] = len(vocab)
    
    # Add special tokens
    special_tokens = ["<PAD>", "<UNK>", "<s>", "</s>"]
    for token in special_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    
    with open("processed_data/vocab.json", "w") as f:
        json.dump(vocab, f, indent=2)
    
    # Create cluster mappings
    cluster_mappings = {
        "query_clusters": {
            "0": "clothing_women",
            "1": "accessories", 
            "2": "electronics",
            "3": "home_kitchen",
            "4": "general"
        },
        "product_clusters": {
            "0": "women_clothing",
            "1": "accessories",
            "2": "electronics", 
            "3": "home_decor",
            "4": "miscellaneous"
        }
    }
    
    with open("processed_data/cluster_mappings.json", "w") as f:
        json.dump(cluster_mappings, f, indent=2)

if __name__ == "__main__":
    setup_khoj_plus()