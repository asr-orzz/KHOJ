"""
Database connection and initialization for Enhanced CADENCE System using SQLite
"""
import asyncio
import json
import redis
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from config.settings import settings
from database.models import Base, User, SearchSession, SearchQuery, UserEngagement, ProductCatalog, QueryCandidate, UserBehaviorProfile, ABTestAssignment
import structlog

logger = structlog.get_logger()

class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.engine = None
        self.async_engine = None
        self.SessionLocal = None
        self.AsyncSessionLocal = None
        self.redis_client: Optional[redis.Redis] = None
        self._initialize_connections()
    
    def _initialize_connections(self):
        """Initialize database connections"""
        try:
            # Initialize SQLite engine
            database_url = settings.DATABASE_URL
            
            # For async operations
            if "sqlite" in database_url:
                async_database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
            else:
                async_database_url = database_url
            
            self.engine = create_engine(database_url, echo=False)
            self.async_engine = create_async_engine(async_database_url, echo=False)
            
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.AsyncSessionLocal = async_sessionmaker(
                bind=self.async_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info("SQLite database connection initialized")
            
            # Initialize Redis
            try:
                self.redis_client = redis.from_url(
                    settings.REDIS_URL,
                    decode_responses=True
                )
                logger.info("Redis connection initialized")
            except Exception as e:
                logger.warning(f"Redis connection failed, continuing without caching: {e}")
                self.redis_client = None
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def setup_database(self):
        """Set up database schema"""
        try:
            # Create all tables
            if self.async_engine:
                async with self.async_engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
            else:
                Base.metadata.create_all(bind=self.engine)
            
            logger.info("Database schema created successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to create database schema: {e}")
            return False
    
    def get_session(self) -> Session:
        """Get synchronous database session"""
        return self.SessionLocal()
    
    async def get_async_session(self) -> AsyncSession:
        """Get asynchronous database session"""
        return self.AsyncSessionLocal()
    
    # User Management
    async def create_user(self, user_data: Dict[str, Any]) -> bool:
        """Create a new user"""
        try:
            async with self.get_async_session() as session:
                user = User(**user_data)
                session.add(user)
                await session.commit()
                logger.info(f"Created user: {user_data['user_id']}")
                return True
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            return False
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID"""
        try:
            async with self.get_async_session() as session:
                result = await session.get(User, user_id)
                if result:
                    return {
                        'user_id': result.user_id,
                        'location': result.location,
                        'age_group': result.age_group,
                        'gender': result.gender,
                        'preferred_categories': result.preferred_categories or [],
                        'created_at': result.created_at.isoformat() if result.created_at else None,
                        'updated_at': result.updated_at.isoformat() if result.updated_at else None
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    # Session Management
    async def create_session(self, session_data: Dict[str, Any]) -> bool:
        """Create a new search session"""
        try:
            async with self.get_async_session() as session:
                search_session = SearchSession(**session_data)
                session.add(search_session)
                await session.commit()
                logger.info(f"Created session: {session_data['session_id']}")
                return True
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return False
    
    async def update_session(self, session_id: str, updates: Dict[str, Any]) -> bool:
        """Update session data"""
        try:
            async with self.get_async_session() as session:
                search_session = await session.get(SearchSession, session_id)
                if search_session:
                    for key, value in updates.items():
                        setattr(search_session, key, value)
                    await session.commit()
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to update session {session_id}: {e}")
            return False
    
    # Query Management
    async def log_search_query(self, query_data: Dict[str, Any]) -> bool:
        """Log a search query"""
        try:
            async with self.get_async_session() as session:
                query = SearchQuery(**query_data)
                session.add(query)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to log search query: {e}")
            return False
    
    async def get_user_query_history(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get user's query history"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text("SELECT * FROM search_queries WHERE user_id = :user_id ORDER BY timestamp DESC LIMIT :limit"),
                    {"user_id": user_id, "limit": limit}
                )
                queries = []
                for row in result:
                    queries.append({
                        'query_id': row.query_id,
                        'query_text': row.query_text,
                        'timestamp': row.timestamp.isoformat() if row.timestamp else None,
                        'suggested_completions': json.loads(row.suggested_completions) if row.suggested_completions else [],
                        'selected_completion': row.selected_completion,
                        'selected_completion_rank': row.selected_completion_rank
                    })
                return queries
        except Exception as e:
            logger.error(f"Failed to get query history for {user_id}: {e}")
            return []
    
    # Engagement Tracking
    async def log_engagement(self, engagement_data: Dict[str, Any]) -> bool:
        """Log user engagement"""
        try:
            async with self.get_async_session() as session:
                engagement = UserEngagement(**engagement_data)
                session.add(engagement)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to log engagement: {e}")
            return False
    
    async def get_user_engagements(self, user_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get user engagements"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text("SELECT * FROM user_engagements WHERE user_id = :user_id ORDER BY timestamp DESC LIMIT :limit"),
                    {"user_id": user_id, "limit": limit}
                )
                engagements = []
                for row in result:
                    engagements.append({
                        'engagement_id': row.engagement_id,
                        'action_type': row.action_type,
                        'product_id': row.product_id,
                        'item_rank': row.item_rank,
                        'timestamp': row.timestamp.isoformat() if row.timestamp else None,
                        'duration_seconds': row.duration_seconds
                    })
                return engagements
        except Exception as e:
            logger.error(f"Failed to get engagements for {user_id}: {e}")
            return []
    
    # Product Catalog
    async def insert_products(self, products: List[Dict[str, Any]]) -> bool:
        """Batch insert products"""
        try:
            async with self.get_async_session() as session:
                for product_data in products:
                    product = ProductCatalog(**product_data)
                    session.add(product)
                await session.commit()
                logger.info(f"Inserted {len(products)} products")
                return True
        except Exception as e:
            logger.error(f"Failed to insert products: {e}")
            return False
    
    async def get_products_by_cluster(self, cluster_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Get products by cluster"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text("SELECT * FROM product_catalog WHERE cluster_id = :cluster_id LIMIT :limit"),
                    {"cluster_id": cluster_id, "limit": limit}
                )
                products = []
                for row in result:
                    products.append({
                        'product_id': row.product_id,
                        'title': row.title,
                        'description': row.description,
                        'main_category': row.main_category,
                        'brand': row.brand,
                        'price': row.price,
                        'rating': row.rating,
                        'cluster_id': row.cluster_id
                    })
                return products
        except Exception as e:
            logger.error(f"Failed to get products for cluster {cluster_id}: {e}")
            return []
    
    # Query Candidates
    async def insert_query_candidates(self, candidates: List[Dict[str, Any]]) -> bool:
        """Insert generated query candidates"""
        try:
            async with self.get_async_session() as session:
                for candidate_data in candidates:
                    candidate = QueryCandidate(**candidate_data)
                    session.add(candidate)
                await session.commit()
                logger.info(f"Inserted {len(candidates)} query candidates")
                return True
        except Exception as e:
            logger.error(f"Failed to insert query candidates: {e}")
            return False
    
    async def search_query_candidates(self, query_prefix: str, cluster_id: Optional[int] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search query candidates by prefix"""
        try:
            async with self.get_async_session() as session:
                if cluster_id is not None:
                    result = await session.execute(
                        text("SELECT * FROM query_candidates WHERE query_text LIKE :prefix AND cluster_id = :cluster_id ORDER BY confidence_score DESC LIMIT :limit"),
                        {"prefix": f"{query_prefix}%", "cluster_id": cluster_id, "limit": limit}
                    )
                else:
                    result = await session.execute(
                        text("SELECT * FROM query_candidates WHERE query_text LIKE :prefix ORDER BY confidence_score DESC LIMIT :limit"),
                        {"prefix": f"{query_prefix}%", "limit": limit}
                    )
                
                candidates = []
                for row in result:
                    candidates.append({
                        'candidate_id': row.candidate_id,
                        'query_text': row.query_text,
                        'confidence_score': row.confidence_score,
                        'source_model': row.source_model,
                        'cluster_id': row.cluster_id
                    })
                return candidates
        except Exception as e:
            logger.error(f"Failed to search query candidates: {e}")
            return []
    
    # User Behavior Profiles
    async def update_user_behavior_profile(self, user_id: str, profile_data: Dict[str, Any]) -> bool:
        """Update user behavior profile"""
        try:
            async with self.get_async_session() as session:
                # Try to get existing profile
                profile = await session.get(UserBehaviorProfile, user_id)
                
                if profile:
                    # Update existing
                    for key, value in profile_data.items():
                        if key != 'user_id':  # Don't update the primary key
                            setattr(profile, key, value)
                else:
                    # Create new
                    profile_data['user_id'] = user_id
                    profile = UserBehaviorProfile(**profile_data)
                    session.add(profile)
                
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to update behavior profile for {user_id}: {e}")
            return False
    
    async def get_user_behavior_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user behavior profile"""
        try:
            async with self.get_async_session() as session:
                profile = await session.get(UserBehaviorProfile, user_id)
                if profile:
                    return {
                        'user_id': profile.user_id,
                        'category_preferences': profile.category_preferences or {},
                        'engagement_patterns': profile.engagement_patterns or {},
                        'time_patterns': profile.time_patterns or {},
                        'price_sensitivity': profile.price_sensitivity,
                        'brand_preferences': profile.brand_preferences or {},
                        'avg_session_length': profile.avg_session_length,
                        'purchase_frequency': profile.purchase_frequency,
                        'last_updated': profile.last_updated.isoformat() if profile.last_updated else None
                    }
                return None
        except Exception as e:
            logger.error(f"Failed to get behavior profile for {user_id}: {e}")
            return None
    
    # A/B Testing
    async def assign_ab_test(self, user_id: str, experiment_name: str, variant: str, context: Dict[str, Any] = None) -> bool:
        """Assign user to A/B test variant"""
        try:
            async with self.get_async_session() as session:
                assignment = ABTestAssignment(
                    user_id=user_id,
                    experiment_name=experiment_name,
                    variant=variant,
                    context=context or {}
                )
                session.add(assignment)
                await session.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to assign A/B test: {e}")
            return False
    
    async def get_ab_test_assignment(self, user_id: str, experiment_name: str) -> Optional[str]:
        """Get user's A/B test assignment"""
        try:
            async with self.get_async_session() as session:
                result = await session.execute(
                    text("SELECT variant FROM ab_test_assignments WHERE user_id = :user_id AND experiment_name = :experiment_name"),
                    {"user_id": user_id, "experiment_name": experiment_name}
                )
                row = result.first()
                return row.variant if row else None
        except Exception as e:
            logger.error(f"Failed to get A/B test assignment: {e}")
            return None
    
    # Redis Cache Operations
    async def cache_set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set cache value"""
        if not self.redis_client:
            return False
        
        try:
            ttl = ttl or settings.CACHE_TTL
            self.redis_client.setex(key, ttl, json.dumps(value))
            return True
        except Exception as e:
            logger.error(f"Failed to set cache for key {key}: {e}")
            return False
    
    async def cache_get(self, key: str) -> Optional[Any]:
        """Get cache value"""
        if not self.redis_client:
            return None
        
        try:
            value = self.redis_client.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Failed to get cache for key {key}: {e}")
            return None
    
    async def cache_delete(self, key: str) -> bool:
        """Delete cache value"""
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to delete cache for key {key}: {e}")
            return False
    
    # Real-time user data caching
    async def cache_user_session_data(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Cache user session data for real-time access"""
        cache_key = f"user_session:{user_id}"
        return await self.cache_set(cache_key, session_data, ttl=3600)  # 1 hour
    
    async def get_cached_user_session_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user session data"""
        cache_key = f"user_session:{user_id}"
        return await self.cache_get(cache_key)

# Global database manager instance
db_manager = DatabaseManager()

# Async initialization function
async def initialize_database():
    """Initialize database schema and connections"""
    try:
        await db_manager.setup_database()
        logger.info("Database initialization completed")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False 