"""
Main FastAPI Application for Enhanced CADENCE System
Provides real-time hyper-personalized autosuggest and product recommendations
"""
import asyncio
import json
import pickle
import time
import uuid
import torch
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import structlog

from config.settings import settings, ENGAGEMENT_ACTIONS, ECOMMERCE_CATEGORIES
from database.connection import db_manager, initialize_database
from core.data_processor import DataProcessor
from core.cadence_model import CADENCEModel, DynamicBeamSearch, create_cadence_model
from core.personalization import PersonalizationEngine, UserEmbeddingModel, ProductReranker
from core.ecommerce_autocomplete import ECommerceAutocompleteEngine, ProductSpecificQueryGenerator
from training.train_models import CADENCETrainer

logger = structlog.get_logger()

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced CADENCE API",
    description="Hyper-personalized search autosuggest and product recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# NOTE: The static React build should be mounted *after* API routes are registered
from pathlib import Path
build_dir = Path(__file__).parent.parent / "frontend" / "build"

# Global variables for models
cadence_model: Optional[CADENCEModel] = None
personalization_engine: Optional[PersonalizationEngine] = None
product_reranker: Optional[ProductReranker] = None
beam_search: Optional[DynamicBeamSearch] = None
data_processor: Optional[DataProcessor] = None
ecommerce_autocomplete: Optional[ECommerceAutocompleteEngine] = None
product_database: List[Dict[str, Any]] = []

# Request/Response Models
class AutosuggestRequest(BaseModel):
    query_prefix: str = Field(..., description="The prefix user has typed")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Current session ID")
    max_suggestions: int = Field(10, description="Maximum number of suggestions")
    include_personalization: bool = Field(True, description="Whether to apply personalization")

class AutosuggestResponse(BaseModel):
    suggestions: List[str] = Field(..., description="List of query suggestions")
    personalized: bool = Field(..., description="Whether personalization was applied")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    session_id: str = Field(..., description="Session ID for tracking")

class ProductSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    user_id: str = Field(..., description="Unique user identifier")
    session_id: Optional[str] = Field(None, description="Current session ID")
    max_results: int = Field(20, description="Maximum number of products")
    include_personalization: bool = Field(True, description="Whether to apply personalization")
    filters: Dict[str, Any] = Field(default_factory=dict, description="Search filters")

class Product(BaseModel):
    product_id: str
    title: str
    description: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    brand: Optional[str] = None
    category: Optional[str] = None
    image_url: Optional[str] = None

class ProductSearchResponse(BaseModel):
    products: List[Product] = Field(..., description="List of products")
    total_results: int = Field(..., description="Total number of results")
    personalized: bool = Field(..., description="Whether personalization was applied")
    response_time_ms: float = Field(..., description="Response time in milliseconds")
    session_id: str = Field(..., description="Session ID for tracking")

class EngagementEvent(BaseModel):
    user_id: str = Field(..., description="User who performed the action")
    session_id: str = Field(..., description="Current session")
    action_type: str = Field(..., description="Type of engagement action")
    item_id: Optional[str] = Field(None, description="Product/query ID if applicable")
    item_rank: Optional[int] = Field(None, description="Position in list")
    duration_seconds: Optional[float] = Field(None, description="Time spent")
    additional_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

class EngagementResponse(BaseModel):
    success: bool = Field(..., description="Whether the event was logged successfully")
    message: str = Field(..., description="Response message")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize models and database connections"""
    global cadence_model, personalization_engine, product_reranker, beam_search, data_processor, ecommerce_autocomplete, product_database
    
    logger.info("Starting Enhanced CADENCE API")
    # Record start time for uptime metric
    app.state.start_time = datetime.utcnow()
    
    try:
        # Initialize database
        await initialize_database()
        
        # Initialize data processor
        data_processor = DataProcessor()
        
        # Initialize trainer
        trainer = CADENCETrainer()
        
        # Try to load existing trained models - check multiple locations
        cadence_model = None
        vocab = {}
        config = {}
        num_categories = 10
        
        # Try legendary models first
        legendary_model_path = Path("legendary_models/legendary_cadence_complete.pt")
        if legendary_model_path.exists():
            try:
                logger.info("Loading LEGENDARY CADENCE models...")
                # Load the legendary complete model
                checkpoint = torch.load(legendary_model_path, map_location='cpu', weights_only=False)
                
                # Extract model config and vocab from checkpoint
                model_config = checkpoint.get('model_config', {})
                vocab = checkpoint.get('vocab', {})
                cluster_info = checkpoint.get('cluster_info', {})
                
                # Get dimensions from config
                vocab_size = len(vocab) if vocab else model_config.get('vocab_size', 50000)
                num_categories = model_config.get('num_categories', len(cluster_info.get('query_clusters', {})) + len(cluster_info.get('product_clusters', {})))
                if num_categories == 0:
                    num_categories = 50  # Default fallback
                
                # Create model with proper config
                filtered_config = {k: v for k, v in model_config.items() if k not in ('vocab_size', 'num_categories')}
                cadence_model = create_cadence_model(vocab_size, num_categories, **filtered_config)
                # Load weights from checkpoint, allowing for keys that may be missing or extra
                cadence_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                cadence_model.eval()
                # Attach vocabulary to model for prefix lookups
                setattr(cadence_model, 'vocab', vocab)
                
                config = {
                    'num_categories': num_categories, 
                    'vocab_size': vocab_size,
                    'model_config': model_config
                }
                
                logger.info(f"Loaded LEGENDARY CADENCE model ({vocab_size:,} vocab, {num_categories} categories)")
                logger.info(f"   Training date: {checkpoint.get('training_date', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"Failed to load legendary models: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                cadence_model = None
        
        # Fallback to optimized models
        if cadence_model is None:
            optimized_model_path = Path("models/optimized_cadence.pt")
            optimized_config_path = Path("models/optimized_cadence_config.json")
            optimized_vocab_path = Path("models/optimized_cadence_vocab.pkl")
            
            if optimized_model_path.exists() and optimized_config_path.exists() and optimized_vocab_path.exists():
                try:
                    logger.info("Loading optimized CADENCE models...")
                    
                    # Load config
                    with open(optimized_config_path, 'r') as f:
                        config = json.load(f)
                    
                    # Load vocab
                    with open(optimized_vocab_path, 'rb') as f:
                        vocab = pickle.load(f)
                    
                    # Load model
                    vocab_size = len(vocab)
                    num_categories = config['num_categories']
                    
                    filtered_config = {k: v for k, v in config.items() if k not in ('vocab_size', 'num_categories')}
                    cadence_model = create_cadence_model(vocab_size, num_categories, **filtered_config)
                    # Load weights with non-strict flag to ignore keys that do not match exactly
                    cadence_model.load_state_dict(torch.load(optimized_model_path, map_location='cpu'), strict=False)
                    cadence_model.eval()
                    # Attach vocabulary to model for prefix lookups
                    setattr(cadence_model, 'vocab', vocab)
                    
                    logger.info(f"Loaded optimized CADENCE models ({vocab_size:,} vocab, {num_categories} categories)")
                    
                except Exception as e:
                    logger.error(f"Failed to load optimized models: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    cadence_model = None
        
        # Final fallback
        if cadence_model is None:
            try:
                logger.info("Trying standard trained models...")
                cadence_model, vocab, config = trainer.load_model_and_vocab('cadence_trained')
                # Attach vocabulary from trainer to the model
                setattr(cadence_model, 'vocab', vocab)
                num_categories = config['num_categories']
                logger.info("Loaded standard CADENCE models")
            except FileNotFoundError:
                logger.warning("No pre-trained models found. Running in fallback mode.")
                cadence_model = None
                vocab = {}
                num_categories = 10  # Default categories
        
        # Initialize personalization components
        user_embedding_model = UserEmbeddingModel(
            num_categories=num_categories,
            num_actions=len(ENGAGEMENT_ACTIONS),
            embedding_dim=128
        )
        
        personalization_engine = PersonalizationEngine(user_embedding_model)
        product_reranker = ProductReranker(personalization_engine)
        
        # Initialize beam search
        beam_search = DynamicBeamSearch(cadence_model, vocab)
        
        # Load Amazon products dataset for real product database
        logger.info("Loading Amazon products for real product database...")
        try:
            product_df = data_processor.load_and_process_amazon_products(max_samples=10000)
            product_database = product_df.to_dict('records')
            logger.info(f"Loaded {len(product_database)} products from Amazon dataset")
        except Exception as e:
            logger.error(f"Failed to load Amazon products: {e}")
            product_database = []
        
        # Initialize enhanced e-commerce autocomplete
        ecommerce_autocomplete = ECommerceAutocompleteEngine(
            cadence_model=cadence_model,
            vocab=vocab,
            product_data=product_database
        )
        
        logger.info("Enhanced CADENCE API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Enhanced CADENCE API")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": cadence_model is not None
    }

# Core API endpoints
@app.post("/autosuggest", response_model=AutosuggestResponse)
async def get_autosuggest(request: AutosuggestRequest, background_tasks: BackgroundTasks):
    """
    Get hyper-personalized autosuggest suggestions
    """
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Step 1: Get base suggestions from CADENCE model
        base_suggestions = await _get_base_suggestions(request.query_prefix)
        
        # Step 2: Apply personalization if requested and user exists
        if request.include_personalization:
            personalized_suggestions = await personalization_engine.personalize_query_suggestions(
                user_id=request.user_id,
                query_prefix=request.query_prefix,
                base_suggestions=base_suggestions
            )
        else:
            personalized_suggestions = base_suggestions
        
        # Step 3: Limit results
        final_suggestions = personalized_suggestions[:request.max_suggestions]
        
        # Step 4: Log the query for future personalization
        background_tasks.add_task(
            _log_autosuggest_query,
            request.user_id,
            session_id,
            request.query_prefix,
            final_suggestions
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return AutosuggestResponse(
            suggestions=final_suggestions,
            personalized=request.include_personalization,
            response_time_ms=response_time,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error in autosuggest: {e}")
        # Fallback to basic suggestions
        response_time = (time.time() - start_time) * 1000
        return AutosuggestResponse(
            suggestions=[f"{request.query_prefix} {suffix}" for suffix in ["", "online", "cheap", "best", "reviews"]],
            personalized=False,
            response_time_ms=response_time,
            session_id=request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        )

@app.post("/search", response_model=ProductSearchResponse)
async def search_products(request: ProductSearchRequest, background_tasks: BackgroundTasks):
    """
    Search for products with hyper-personalized ranking
    """
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Step 1: Get base search results (would integrate with product search engine)
        base_products = await _get_base_search_results(request.query, request.max_results)
        
        # Step 2: Apply personalized reranking if requested
        if request.include_personalization and product_reranker:
            personalized_products = await product_reranker.rerank_products(
                user_id=request.user_id,
                query=request.query,
                products=base_products
            )
        else:
            personalized_products = base_products
        
        # Step 3: Convert to response format
        product_results = [
            Product(
                product_id=p.get('product_id', ''),
                title=p.get('title', ''),
                description=p.get('description'),
                price=p.get('price'),
                rating=p.get('rating'),
                brand=p.get('brand'),
                category=p.get('main_category'),
                image_url=p.get('image_url')
            )
            for p in personalized_products
        ]
        
        # Step 4: Log the search for future personalization
        background_tasks.add_task(
            _log_product_search,
            request.user_id,
            session_id,
            request.query,
            len(product_results)
        )
        
        response_time = (time.time() - start_time) * 1000
        
        return ProductSearchResponse(
            products=product_results,
            total_results=len(product_results),
            personalized=request.include_personalization,
            response_time_ms=response_time,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Error in product search: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/engagement", response_model=EngagementResponse)
async def log_engagement(event: EngagementEvent):
    """
    Log user engagement event for personalization
    """
    try:
        # Create engagement record
        engagement_data = {
            "engagement_id": f"eng_{uuid.uuid4().hex[:12]}",
            "user_id": event.user_id,
            "session_id": event.session_id,
            "action_type": event.action_type,
            "product_id": event.item_id,
            "item_rank": event.item_rank,
            "timestamp": datetime.utcnow().isoformat(),
            "duration_seconds": event.duration_seconds
        }
        
        # Save to database
        success = await db_manager.log_engagement(engagement_data)
        
        if success:
            return EngagementResponse(
                success=True,
                message="Engagement logged successfully"
            )
        else:
            return EngagementResponse(
                success=False,
                message="Failed to log engagement"
            )
            
    except Exception as e:
        logger.error(f"Error logging engagement: {e}")
        return EngagementResponse(
            success=False,
            message=f"Error: {str(e)}"
        )

@app.post("/user/session/start")
async def start_user_session(user_id: str, device_type: str = "web"):
    """
    Start a new user session
    """
    try:
        session_id = f"session_{uuid.uuid4().hex[:12]}"
        
        session_data = {
            "session_id": session_id,
            "user_id": user_id,
            "device_type": device_type,
            "start_time": datetime.utcnow().isoformat()
        }
        
        success = await db_manager.create_session(session_data)
        
        if success:
            return {"session_id": session_id, "message": "Session started successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to start session")
            
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/analytics/user/{user_id}")
async def get_user_analytics(user_id: str):
    """
    Get user behavior analytics
    """
    try:
        # Get user profile
        profile = await personalization_engine.get_user_profile(user_id)
        
        # Get recent engagement data
        engagements = await db_manager.get_user_engagements(user_id, limit=100)
        
        # Calculate analytics
        analytics = {
            "user_id": user_id,
            "profile": profile,
            "recent_activity": {
                "total_engagements": len(engagements),
                "engagement_breakdown": _calculate_engagement_breakdown(engagements),
                "last_active": engagements[0]['timestamp'] if engagements else None
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error getting user analytics: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Administrative endpoints
@app.post("/admin/retrain-models")
async def retrain_models(max_samples: int = 5000, epochs: int = 1):
    """
    Retrain CADENCE models with new data (admin only)
    """
    try:
        global cadence_model, beam_search
        
        trainer = CADENCETrainer()
        
        # Train new models
        new_model, new_vocab, cluster_mappings = trainer.train_full_pipeline(
            max_samples=max_samples,
            epochs=epochs
        )
        
        # Update global models
        cadence_model = new_model
        beam_search = DynamicBeamSearch(cadence_model, new_vocab)
        
        return {
            "message": f"Successfully retrained models with {max_samples} samples",
            "vocab_size": len(new_vocab),
            "num_categories": len(cluster_mappings.get('query_clusters', {})) + len(cluster_mappings.get('product_clusters', {}))
        }
            
    except Exception as e:
        logger.error(f"Error retraining models: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/admin/stats")
async def get_system_stats():
    """
    Get system statistics (used by the frontend dashboard).
    The shape is aligned with the TS/JS expectations:
    {
        "model_info": {"parameters": int, "vocab_size": int, "device": str},
        "data_info" : {"products": int, "queries": int},
        "uptime_secs": float
    }
    """
    try:
        # ---- model statistics ----
        if cadence_model is not None:
            param_count = sum(p.numel() for p in cadence_model.parameters())
            vocab_size = getattr(cadence_model, "vocab_size", 0)
            device = next(cadence_model.parameters()).device.type
        else:
            param_count = 0
            vocab_size = 0
            device = "cpu"

        model_info = {
            "parameters": int(param_count),
            "vocab_size": int(vocab_size),
            "device": device,
        }

        # ---- data statistics ----
        # Get real data statistics
        data_info = {
            "products": len(product_database),
            "queries": 0,  # TODO: Implement query count from database
        }

        stats = {
            "model_info": model_info,
            "data_info": data_info,
            "uptime_secs": (datetime.utcnow() - app.state.start_time).total_seconds() if hasattr(app.state, "start_time") else 0,
        }
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Helper functions
async def _get_base_suggestions(query_prefix: str) -> List[str]:
    """
    Get enhanced e-commerce specific autocomplete suggestions
    """
    prefix = query_prefix.strip().lower()
    if not prefix:
        return []

    # Use enhanced e-commerce autocomplete if available
    if ecommerce_autocomplete is not None:
        try:
            suggestions_data = await ecommerce_autocomplete.get_suggestions(
                query_prefix=prefix,
                max_suggestions=10
            )
            
            # Extract text suggestions
            suggestions = [s['text'] for s in suggestions_data if s.get('text')]
            if suggestions:
                logger.info(f"Generated {len(suggestions)} e-commerce suggestions for '{prefix}'")
                return suggestions
        except Exception as e:
            logger.error(f"E-commerce autocomplete failed: {e}")

    # Fallback: Neural generation
    try:
        if cadence_model is not None and beam_search is not None:
            tokens = prefix.split()
            generated = beam_search.search(tokens, category_id=0, model_type="query")
            suggestions = [s for s in generated if s]
            if suggestions:
                return suggestions[:10]
    except Exception as gen_err:
        logger.warning(f"BeamSearch generation failed: {gen_err}")

    # Final fallback: Vocabulary prefix match
    try:
        vocab_map = getattr(cadence_model, 'vocab', {})
        if vocab_map:
            prefix_matches = [tok for tok in vocab_map.keys() if tok.isalpha() and tok.startswith(prefix)]
            prefix_matches.sort(key=lambda x: (len(x), x))
            if prefix_matches:
                return prefix_matches[:10]
    except Exception as vocab_err:
        logger.warning(f"Vocab prefix match failed: {vocab_err}")

    return []

async def _get_base_search_results(query: str, max_results: int) -> List[Dict[str, Any]]:
    """
    Get base search results from real Amazon product database
    """
    try:
        if not product_database:
            logger.error("❌ CRITICAL: No product database loaded!")
            logger.error("The system must have real Amazon products loaded.")
            logger.error("Run: python run_enhanced_cadence_system.py to load real data")
            raise HTTPException(status_code=500, detail="Product database not initialized. Real Amazon products required.")
        
        # Search in real product database
        query_lower = query.lower()
        matching_products = []
        
        for product in product_database:
            title = product.get('title', '').lower()
            description = product.get('description', '').lower()
            brand = product.get('brand', '').lower()
            category = product.get('main_category', '').lower()
            
            # Calculate relevance score
            score = 0.0
            if query_lower in title:
                score += 2.0
            if query_lower in description:
                score += 1.0
            if query_lower in brand:
                score += 1.5
            if query_lower in category:
                score += 0.5
            
            # Add word-level matching
            query_words = query_lower.split()
            for word in query_words:
                if word in title:
                    score += 0.5
                if word in brand:
                    score += 0.3
            
            if score > 0:
                matching_products.append((product, score))
        
        # Sort by relevance score
        matching_products.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results
        results = []
        for product, score in matching_products[:max_results]:
            # Ensure all required fields are present
            result = {
                'product_id': product.get('product_id', ''),
                'title': product.get('title', ''),
                'description': product.get('description', ''),
                'price': product.get('price'),
                'rating': product.get('rating'),
                'brand': product.get('brand', ''),
                'main_category': product.get('main_category', ''),
                'image_url': product.get('image_url', f"https://images.unsplash.com/300x300/?text={product.get('title', 'Product')[:20].replace(' ', '+')}")
            }
            results.append(result)
        
        logger.info(f"Found {len(results)} products for query '{query}'")
        return results
        
    except Exception as e:
        logger.error(f"Error getting search results: {e}")
        return []

async def _log_autosuggest_query(user_id: str, session_id: str, query_prefix: str, suggestions: List[str]):
    """
    Log autosuggest query for analytics
    """
    try:
        query_data = {
            "query_id": f"query_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "user_id": user_id,
            "query_text": query_prefix,
            "suggested_completions": suggestions,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await db_manager.log_search_query(query_data)
        
    except Exception as e:
        logger.error(f"Error logging autosuggest query: {e}")

async def _log_product_search(user_id: str, session_id: str, query: str, result_count: int):
    """
    Log product search for analytics
    """
    try:
        query_data = {
            "query_id": f"query_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "user_id": user_id,
            "query_text": query,
            "results_shown": result_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await db_manager.log_search_query(query_data)
        
    except Exception as e:
        logger.error(f"Error logging product search: {e}")

def _calculate_engagement_breakdown(engagements: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Calculate breakdown of engagement types
    """
    breakdown = {}
    for engagement in engagements:
        action_type = engagement.get('action_type', 'unknown')
        breakdown[action_type] = breakdown.get(action_type, 0) + 1
    
    return breakdown

# ---------------------------
# v1 Compatibility Router
# ---------------------------
router_v1 = APIRouter(prefix="/api/v1")

class AutocompleteV1Request(BaseModel):
    query: str
    max_suggestions: int = 10
    category: Optional[int] = None

@router_v1.post("/autocomplete")
async def autocomplete_v1(payload: AutocompleteV1Request):
    # Minimal wrapper around existing _get_base_suggestions; production
    # implementation would call personalization, etc.
    start = datetime.utcnow()
    suggestions = await _get_base_suggestions(payload.query)
    return {
        "suggestions": suggestions[: payload.max_suggestions],
        "processing_time_ms": (datetime.utcnow() - start).total_seconds() * 1000,
    }

class SearchV1Request(BaseModel):
    query: str
    max_results: int = 20
    category_filter: Optional[int] = None
    sort_by: Optional[str] = "relevance"

@router_v1.post("/search")
async def search_v1(payload: SearchV1Request):
    start = datetime.utcnow()
    results = await _get_base_search_results(payload.query, payload.max_results)
    return {
        "results": results,
        "total_results": len(results),
        "processing_time_ms": (datetime.utcnow() - start).total_seconds() * 1000,
    }

@router_v1.get("/categories")
async def categories_v1():
    # Convert the mapping into an array of objects expected by the React UI
    product_categories = [
        {"id": cat_id, "name": name}
        for cat_id, name in ECOMMERCE_CATEGORIES.items()
    ]
    return {"product_categories": product_categories}

@router_v1.get("/stats")
async def stats_v1():
    return await get_system_stats()

# Register compatibility router
app.include_router(router_v1)

# Serve React build ONLY if the production build directory exists. Mounting at the
# end ensures it does not shadow API routes like /health or /api/v1/*.
if build_dir.exists():
    app.mount("/", StaticFiles(directory=str(build_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 