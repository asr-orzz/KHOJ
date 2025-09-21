"""
KHOJ+ API Endpoints
New endpoints for Bharat-first search and recommendations
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from typing import List, Dict, Any, Optional
import time
import uuid
import structlog

from api.main import (
    AutosuggestRequest, AutosuggestResponse, SearchRequest, SearchResponse,
    UserBehaviorEvent, ExperimentRequest, ExperimentResponse, IntentChip,
    vernacular_processor, trust_ranker, ecommerce_autocomplete, personalization_engine
)

logger = structlog.get_logger()

# Create router
khoj_router = APIRouter(prefix="/api/v2", tags=["KHOJ+ Endpoints"])

@khoj_router.post("/autosuggest", response_model=AutosuggestResponse)
async def get_vernacular_autosuggest(
    request: AutosuggestRequest,
    background_tasks: BackgroundTasks
):
    """
    Get Bharat-first autosuggest with vernacular processing and intent chips
    """
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Step 1: Process vernacular query and extract intent
        normalized_query, intent_tags = vernacular_processor.process_query(request.query_prefix)
        
        # Step 2: Generate intent chips
        intent_chips = vernacular_processor.generate_intent_chips(request.query_prefix, intent_tags)
        
        # Step 3: Get base suggestions from CADENCE
        base_suggestions = await ecommerce_autocomplete.get_suggestions(
            normalized_query,
            max_suggestions=request.max_suggestions
        )
        
        # Step 4: Apply personalization if requested
        if request.include_personalization and request.user_id:
            personalized_suggestions = await personalization_engine.personalize_query_suggestions(
                user_id=request.user_id,
                query_prefix=normalized_query,
                base_suggestions=[s.get('suggestion', '') for s in base_suggestions],
                user_pincode=request.user_pincode
            )
        else:
            personalized_suggestions = [s.get('suggestion', '') for s in base_suggestions]
        
        # Step 5: Log behavior for learning
        background_tasks.add_task(
            _log_autosuggest_event,
            user_id=request.user_id,
            session_id=session_id,
            query=request.query_prefix,
            normalized_query=normalized_query,
            suggestions=personalized_suggestions[:request.max_suggestions],
            intent_tags=intent_tags
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return AutosuggestResponse(
            suggestions=personalized_suggestions[:request.max_suggestions],
            intent_chips=intent_chips,
            original_query=request.query_prefix,
            normalized_query=normalized_query,
            session_id=session_id,
            processing_time_ms=processing_time,
            has_vernacular_processing=(normalized_query != request.query_prefix.lower().strip())
        )
        
    except Exception as e:
        logger.error(f"Autosuggest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@khoj_router.post("/search", response_model=SearchResponse)
async def search_with_trust_ranking(
    request: SearchRequest,
    background_tasks: BackgroundTasks
):
    """
    Search with trust-aware ranking and SRP strips
    """
    start_time = time.time()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or f"session_{uuid.uuid4().hex[:12]}"
        
        # Step 1: Process vernacular query
        normalized_query, intent_tags = vernacular_processor.process_query(request.query)
        
        # Step 2: Get base search results
        base_results = await ecommerce_autocomplete.search_products(
            normalized_query,
            max_results=request.max_results * 2,  # Get more for ranking
            category_filter=request.category_filter
        )
        
        # Step 3: Apply trust-aware ranking
        user_signals = None
        if request.user_id:
            user_signals = await _get_user_personalization_signals(request.user_id)
        
        ranked_results = trust_ranker.rank_results(
            base_results,
            user_signals=user_signals,
            user_pincode=request.user_pincode
        )
        
        # Step 4: Generate SRP strips if requested
        strips = []
        if request.include_strips:
            strip_data = trust_ranker.generate_srp_strips(ranked_results, user_signals)
            strips = [
                {"title": title, "products": products}
                for title, products in strip_data.items()
            ]
        
        # Step 5: Convert to response format
        final_results = ranked_results[:request.max_results]
        product_results = [_convert_to_product_result(result) for result in final_results]
        
        # Step 6: Log search event
        background_tasks.add_task(
            _log_search_event,
            user_id=request.user_id,
            session_id=session_id,
            query=request.query,
            normalized_query=normalized_query,
            results=product_results,
            intent_tags=intent_tags
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return SearchResponse(
            results=product_results,
            strips=strips,
            query=request.query,
            normalized_query=normalized_query,
            total_results=len(product_results),
            processing_time_ms=processing_time,
            intent_detected=intent_tags.__dict__ if intent_tags else None,
            applied_filters=request.filters or {},
            session_id=session_id
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@khoj_router.post("/behavior/log")
async def log_user_behavior(event: UserBehaviorEvent):
    """
    Log user behavior for personalization and analytics
    """
    try:
        # Store in database for personalization
        await db_manager.log_user_behavior(
            user_id=event.user_id,
            session_id=event.session_id,
            event_type=event.event_type,
            query=event.query,
            product_id=event.product_id,
            timestamp=event.timestamp,
            metadata=event.metadata
        )
        
        return {"status": "logged", "event_id": str(uuid.uuid4())}
        
    except Exception as e:
        logger.error(f"Behavior logging error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@khoj_router.post("/experiments/assign", response_model=ExperimentResponse)
async def assign_experiment(request: ExperimentRequest):
    """
    Assign user to experiment treatment (A/B testing)
    """
    try:
        # Simple hash-based assignment for demo
        # In production, use proper experiment management platform
        user_hash = hash(f"{request.session_id}{request.experiment_name}") % 100
        
        # Example experiment: suggestion diversity
        if request.experiment_name == "suggestion_diversity":
            treatment = "high_diversity" if user_hash < 50 else "balanced_diversity"
        else:
            treatment = "control"
        
        return ExperimentResponse(
            experiment_name=request.experiment_name,
            treatment=treatment,
            session_id=request.session_id,
            metadata={"assignment_hash": user_hash}
        )
        
    except Exception as e:
        logger.error(f"Experiment assignment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@khoj_router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check for monitoring
    """
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Check model availability
    health_status["components"]["vernacular_processor"] = {
        "status": "healthy" if vernacular_processor else "unavailable"
    }
    health_status["components"]["trust_ranker"] = {
        "status": "healthy" if trust_ranker else "unavailable"
    }
    health_status["components"]["ecommerce_autocomplete"] = {
        "status": "healthy" if ecommerce_autocomplete else "unavailable"
    }
    
    # Check database connectivity
    try:
        await db_manager.ping()
        health_status["components"]["database"] = {"status": "healthy"}
    except:
        health_status["components"]["database"] = {"status": "unhealthy"}
        health_status["status"] = "degraded"
    
    return health_status

# Helper functions

async def _log_autosuggest_event(user_id: Optional[str], session_id: str, 
                                query: str, normalized_query: str,
                                suggestions: List[str], intent_tags):
    """Log autosuggest interaction for learning"""
    try:
        await db_manager.log_user_behavior(
            user_id=user_id,
            session_id=session_id,
            event_type="autosuggest_query",
            query=query,
            metadata={
                "normalized_query": normalized_query,
                "suggestions": suggestions,
                "intent_tags": intent_tags.__dict__ if intent_tags else None
            }
        )
    except Exception as e:
        logger.error(f"Error logging autosuggest event: {e}")

async def _log_search_event(user_id: Optional[str], session_id: str,
                           query: str, normalized_query: str,
                           results: List[Dict], intent_tags):
    """Log search interaction for learning"""
    try:
        await db_manager.log_user_behavior(
            user_id=user_id,
            session_id=session_id,
            event_type="search_query",
            query=query,
            metadata={
                "normalized_query": normalized_query,
                "result_count": len(results),
                "intent_tags": intent_tags.__dict__ if intent_tags else None
            }
        )
    except Exception as e:
        logger.error(f"Error logging search event: {e}")

async def _get_user_personalization_signals(user_id: str):
    """Get user personalization signals from database"""
    try:
        # This would query user's historical behavior
        # For demo, return basic signals
        return {
            "user_price_affinity": "budget",  # budget, mid, premium
            "user_category_preference": "clothing",
            "historical_spend_avg": 500.0,
            "cod_tendency": 0.8,
            "return_tendency": 0.1,
            "brand_affinity": ["meesho", "local_brand"]
        }
    except Exception as e:
        logger.error(f"Error getting user signals: {e}")
        return None

def _convert_to_product_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert internal result format to API response format"""
    return {
        "product_id": result.get("product_id", ""),
        "title": result.get("title", ""),
        "brand": result.get("brand"),
        "category": result.get("category", ""),
        "price": result.get("price"),
        "rating": result.get("rating"),
        "review_count": result.get("review_count"),
        "relevance_score": result.get("relevance_score", 0.0),
        "ranking_score": result.get("ranking_score", 0.0),
        "trust_signals": result.get("trust_signals"),
        "deliverability": result.get("deliverability"),
        "image_url": result.get("image_url"),
        "seller_id": result.get("seller_id"),
        "seller_name": result.get("seller_name"),
        "is_meesho_mall": result.get("is_meesho_mall", False),
        "cod_available": result.get("cod_available", True),
        "return_policy": result.get("return_policy"),
        "pincode_eta": result.get("pincode_eta")
    }