"""
Database models for Enhanced CADENCE System using SQLAlchemy and SQLite
"""
from datetime import datetime
from typing import List, Dict, Optional, Any
from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field
from enum import Enum

Base = declarative_base()

class EngagementType(str, Enum):
    """User engagement action types"""
    VIEW = "view"
    CLICK = "click"
    ADD_TO_CART = "add_to_cart"
    REMOVE_FROM_CART = "remove_from_cart"
    WISHLIST = "wishlist"
    REMOVE_WISHLIST = "remove_wishlist"
    PURCHASE = "purchase"
    RETURN = "return"
    REVIEW_VIEW = "review_view"
    SPECS_VIEW = "specs_view"
    IMAGE_VIEW = "image_view"
    SCROLL = "scroll"
    SEARCH = "search"
    FILTER_APPLY = "filter_apply"
    SORT_CHANGE = "sort_change"

# SQLAlchemy Models
class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True)
    location = Column(String)
    age_group = Column(String)
    gender = Column(String)
    preferred_categories = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship("SearchSession", back_populates="user")
    queries = relationship("SearchQuery", back_populates="user")
    engagements = relationship("UserEngagement", back_populates="user")

class SearchSession(Base):
    __tablename__ = "search_sessions"
    
    session_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime)
    device_type = Column(String)
    location = Column(String)
    total_searches = Column(Integer, default=0)
    total_engagement_score = Column(Float, default=0.0)
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    queries = relationship("SearchQuery", back_populates="session")
    engagements = relationship("UserEngagement", back_populates="session")

class SearchQuery(Base):
    __tablename__ = "search_queries"
    
    query_id = Column(String, primary_key=True)
    session_id = Column(String, ForeignKey("search_sessions.session_id"))
    user_id = Column(String, ForeignKey("users.user_id"))
    query_text = Column(Text)
    suggested_completions = Column(JSON)
    selected_completion = Column(String)
    selected_completion_rank = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    results_shown = Column(Integer, default=0)
    category_cluster = Column(Integer)
    
    # Relationships
    user = relationship("User", back_populates="queries")
    session = relationship("SearchSession", back_populates="queries")
    engagements = relationship("UserEngagement", back_populates="query")

class UserEngagement(Base):
    __tablename__ = "user_engagements"
    
    engagement_id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    session_id = Column(String, ForeignKey("search_sessions.session_id"))
    query_id = Column(String, ForeignKey("search_queries.query_id"))
    product_id = Column(String)
    action_type = Column(String)
    action_value = Column(String)
    item_rank = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float)
    page_section = Column(String)
    
    # Relationships
    user = relationship("User", back_populates="engagements")
    session = relationship("SearchSession", back_populates="engagements")
    query = relationship("SearchQuery", back_populates="engagements")

class ProductCatalog(Base):
    __tablename__ = "product_catalog"
    
    product_id = Column(String, primary_key=True)
    title = Column(Text)
    description = Column(Text)
    main_category = Column(String)
    categories = Column(JSON)
    brand = Column(String)
    price = Column(Float)
    rating = Column(Float)
    rating_count = Column(Integer)
    features = Column(JSON)
    attributes = Column(JSON)
    embedding = Column(JSON)
    cluster_id = Column(Integer)
    date_added = Column(DateTime, default=datetime.utcnow)

class QueryCandidate(Base):
    __tablename__ = "query_candidates"
    
    candidate_id = Column(String, primary_key=True)
    query_text = Column(Text)
    source_model = Column(String)
    confidence_score = Column(Float)
    cluster_id = Column(Integer)
    embedding = Column(JSON)
    popularity_score = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserBehaviorProfile(Base):
    __tablename__ = "user_behavior_profiles"
    
    user_id = Column(String, ForeignKey("users.user_id"), primary_key=True)
    category_preferences = Column(JSON)
    engagement_patterns = Column(JSON)
    time_patterns = Column(JSON)
    price_sensitivity = Column(Float)
    brand_preferences = Column(JSON)
    avg_session_length = Column(Float, default=0.0)
    purchase_frequency = Column(Float, default=0.0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class ABTestAssignment(Base):
    __tablename__ = "ab_test_assignments"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.user_id"))
    experiment_name = Column(String)
    variant = Column(String)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    context = Column(JSON)

# Pydantic Models for API
class UserProfile(BaseModel):
    """User profile model"""
    user_id: str = Field(..., description="Unique user identifier")
    location: Optional[str] = None
    age_group: Optional[str] = None
    gender: Optional[str] = None
    preferred_categories: List[int] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

class SearchSessionModel(BaseModel):
    """Search session model"""
    session_id: str = Field(..., description="Unique session identifier")
    user_id: str = Field(..., description="User who owns this session")
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    device_type: Optional[str] = None
    location: Optional[str] = None
    total_searches: int = 0
    total_engagement_score: float = 0.0

    class Config:
        from_attributes = True

class SearchQueryModel(BaseModel):
    """Individual search query within a session"""
    query_id: str = Field(..., description="Unique query identifier")
    session_id: str = Field(..., description="Parent session")
    user_id: str = Field(..., description="User who made the query")
    query_text: str = Field(..., description="The search query")
    suggested_completions: List[str] = Field(default_factory=list)
    selected_completion: Optional[str] = None
    selected_completion_rank: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    results_shown: int = 0
    category_cluster: Optional[int] = None

    class Config:
        from_attributes = True

class UserEngagementModel(BaseModel):
    """User engagement tracking"""
    engagement_id: str = Field(..., description="Unique engagement identifier")
    user_id: str = Field(..., description="User who performed the action")
    session_id: str = Field(..., description="Session context")
    query_id: Optional[str] = None
    product_id: Optional[str] = None
    action_type: EngagementType = Field(..., description="Type of engagement")
    action_value: Optional[str] = None
    item_rank: Optional[int] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    duration_seconds: Optional[float] = None
    page_section: Optional[str] = None

    class Config:
        from_attributes = True

class ProductCatalogModel(BaseModel):
    """Product catalog model"""
    product_id: str = Field(..., description="Unique product identifier")
    title: str = Field(..., description="Product title")
    description: Optional[str] = None
    main_category: Optional[str] = None
    categories: List[str] = Field(default_factory=list)
    brand: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    rating_count: Optional[int] = None
    features: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    embedding: List[float] = Field(default_factory=list)
    cluster_id: Optional[int] = None
    date_added: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

class QueryCandidateModel(BaseModel):
    """Generated query candidates from CADENCE"""
    candidate_id: str = Field(..., description="Unique candidate identifier")
    query_text: str = Field(..., description="Generated query")
    source_model: str = Field(..., description="query_lm or catalog_lm")
    confidence_score: float = Field(..., description="Model confidence")
    cluster_id: Optional[int] = None
    embedding: List[float] = Field(default_factory=list)
    popularity_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True

class UserBehaviorProfileModel(BaseModel):
    """Aggregated user behavior for personalization"""
    user_id: str = Field(..., description="Unique user identifier")
    category_preferences: Dict[int, float] = Field(default_factory=dict)
    engagement_patterns: Dict[str, float] = Field(default_factory=dict)
    time_patterns: Dict[str, float] = Field(default_factory=dict)
    price_sensitivity: Optional[float] = None
    brand_preferences: Dict[str, float] = Field(default_factory=dict)
    avg_session_length: float = 0.0
    purchase_frequency: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True 