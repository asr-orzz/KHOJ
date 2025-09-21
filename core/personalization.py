"""
Personalization and Reranking System for Enhanced CADENCE
Implements hyper-personalized search suggestions and product recommendations
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict, Counter
import structlog
from database.connection import db_manager
from config.settings import settings

logger = structlog.get_logger()

class UserEmbeddingModel(nn.Module):
    """
    Neural model to generate user embeddings based on behavior patterns
    """
    
    def __init__(self, num_categories: int, num_actions: int, embedding_dim: int = 128):
        super().__init__()
        
        self.num_categories = num_categories
        self.num_actions = num_actions
        self.embedding_dim = embedding_dim
        
        # Category preference embeddings
        self.category_embeddings = nn.Embedding(num_categories, embedding_dim)
        
        # Action type embeddings
        self.action_embeddings = nn.Embedding(num_actions, embedding_dim)
        
        # Time decay neural network
        self.time_decay_net = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # User behavior encoder
        self.behavior_encoder = nn.Sequential(
            nn.Linear(embedding_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
    def forward(self, category_prefs: torch.Tensor, action_patterns: torch.Tensor, 
                time_weights: torch.Tensor) -> torch.Tensor:
        """
        Generate user embedding from behavior data
        """
        # Category preference embedding
        category_embed = torch.sum(self.category_embeddings.weight * category_prefs.unsqueeze(-1), dim=0)
        
        # Action pattern embedding
        action_embed = torch.sum(self.action_embeddings.weight * action_patterns.unsqueeze(-1), dim=0)
        
        # Time decay factors
        time_decay = self.time_decay_net(time_weights.unsqueeze(-1)).squeeze(-1)
        time_embed = torch.sum(time_decay.unsqueeze(-1) * category_embed.unsqueeze(0), dim=0)
        
        # Combine all embeddings
        combined = torch.cat([category_embed, action_embed, time_embed], dim=0)
        
        # Generate final user embedding
        user_embedding = self.behavior_encoder(combined)
        
        return user_embedding

class PersonalizationEngine:
    """
    Core personalization engine for reranking search suggestions and products
    """
    
    def __init__(self, embedding_model: UserEmbeddingModel):
        self.embedding_model = embedding_model
        self.action_types = {
            'view': 0, 'click': 1, 'add_to_cart': 2, 'wishlist': 3,
            'purchase': 4, 'review_view': 5, 'specs_view': 6, 'scroll': 7
        }
        self.user_profiles_cache = {}
        
    async def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Get or create user behavior profile
        """
        # Check cache first
        if user_id in self.user_profiles_cache:
            return self.user_profiles_cache[user_id]
        
        # Get from database
        profile = await db_manager.get_user_behavior_profile(user_id)
        
        if not profile:
            # Create new profile
            profile = await self._create_user_profile(user_id)
        else:
            # Update if stale
            last_updated = datetime.fromisoformat(profile['last_updated'])
            if datetime.utcnow() - last_updated > timedelta(hours=1):
                profile = await self._update_user_profile(user_id)
        
        # Cache the profile
        self.user_profiles_cache[user_id] = profile
        return profile
    
    async def _create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Create initial user profile from engagement data
        """
        # Get user engagement history
        engagements = await db_manager.get_user_engagements(user_id, limit=1000)
        
        if not engagements:
            # Return default profile for new users
            return {
                'user_id': user_id,
                'category_preferences': {},
                'engagement_patterns': {action: 0.0 for action in self.action_types.keys()},
                'time_patterns': {str(hour): 0.0 for hour in range(24)},
                'price_sensitivity': 0.5,
                'brand_preferences': {},
                'avg_session_length': 0.0,
                'purchase_frequency': 0.0
            }
        
        # Analyze engagements
        category_prefs = defaultdict(float)
        action_patterns = defaultdict(float)
        time_patterns = defaultdict(float)
        brand_prefs = defaultdict(float)
        
        total_engagements = len(engagements)
        purchase_count = 0
        
        for engagement in engagements:
            action_type = engagement['action_type']
            timestamp = datetime.fromisoformat(engagement['timestamp'])
            
            # Weight by recency (decay over time)
            days_ago = (datetime.utcnow() - timestamp).days
            time_weight = np.exp(-days_ago / 30.0)  # 30-day decay
            
            # Update action patterns
            engagement_weight = settings.ENGAGEMENT_WEIGHTS.get(action_type, 1.0)
            action_patterns[action_type] += engagement_weight * time_weight
            
            # Update time patterns
            hour = timestamp.hour
            time_patterns[str(hour)] += time_weight
            
            # Track purchases
            if action_type == 'purchase':
                purchase_count += 1
        
        # Normalize patterns
        total_action_weight = sum(action_patterns.values())
        if total_action_weight > 0:
            action_patterns = {k: v / total_action_weight for k, v in action_patterns.items()}
        
        total_time_weight = sum(time_patterns.values())
        if total_time_weight > 0:
            time_patterns = {k: v / total_time_weight for k, v in time_patterns.items()}
        
        # Calculate purchase frequency (purchases per month)
        if engagements:
            days_span = max(1, (datetime.utcnow() - datetime.fromisoformat(engagements[-1]['timestamp'])).days)
            purchase_frequency = (purchase_count * 30.0) / days_span
        else:
            purchase_frequency = 0.0
        
        profile = {
            'user_id': user_id,
            'category_preferences': dict(category_prefs),
            'engagement_patterns': dict(action_patterns),
            'time_patterns': dict(time_patterns),
            'price_sensitivity': 0.5,  # Will be calculated separately
            'brand_preferences': dict(brand_prefs),
            'avg_session_length': 0.0,  # Will be calculated from sessions
            'purchase_frequency': purchase_frequency
        }
        
        # Save to database
        await db_manager.update_user_behavior_profile(user_id, profile)
        
        return profile
    
    async def _update_user_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Update existing user profile with recent activity
        """
        return await self._create_user_profile(user_id)  # Full recalculation for now
    
    async def personalize_query_suggestions(self, user_id: str, query_prefix: str, 
                                          base_suggestions: List[str], 
                                          session_context: Dict[str, Any] = None) -> List[str]:
        """
        Personalize and rerank query suggestions based on user behavior
        """
        try:
            # Get user profile
            user_profile = await self.get_user_profile(user_id)
            
            # Get session context
            if not session_context:
                session_context = await db_manager.get_cached_user_session_data(user_id) or {}
            
            # Score each suggestion
            scored_suggestions = []
            
            for suggestion in base_suggestions:
                score = await self._score_query_suggestion(
                    user_id, suggestion, user_profile, session_context
                )
                scored_suggestions.append((suggestion, score))
            
            # Sort by score (descending)
            scored_suggestions.sort(key=lambda x: x[1], reverse=True)
            
            # Apply diversity to prevent filter bubbles
            reranked_suggestions = self._apply_diversity_constraints(
                scored_suggestions, user_profile
            )
            
            return [suggestion for suggestion, _ in reranked_suggestions]
            
        except Exception as e:
            logger.error(f"Error personalizing suggestions for user {user_id}: {e}")
            return base_suggestions  # Fallback to base suggestions
    
    async def _score_query_suggestion(self, user_id: str, suggestion: str, 
                                    user_profile: Dict[str, Any], 
                                    session_context: Dict[str, Any]) -> float:
        """
        Score a query suggestion based on user behavior and context
        """
        base_score = 1.0
        
        # Personal preference scoring
        personal_score = self._calculate_personal_preference_score(suggestion, user_profile)
        
        # Contextual scoring (current session)
        contextual_score = self._calculate_contextual_score(suggestion, session_context)
        
        # Collaborative filtering score
        collaborative_score = await self._calculate_collaborative_score(user_id, suggestion)
        
        # Time-based scoring
        temporal_score = self._calculate_temporal_score(user_profile)
        
        # Combine scores with weights
        final_score = (
            0.4 * personal_score +
            0.3 * contextual_score +
            0.2 * collaborative_score +
            0.1 * temporal_score
        )
        
        return final_score
    
    def _calculate_personal_preference_score(self, suggestion: str, 
                                          user_profile: Dict[str, Any]) -> float:
        """
        Calculate score based on user's historical preferences
        """
        score = 0.0
        suggestion_lower = suggestion.lower()
        
        # Category preference matching
        category_prefs = user_profile.get('category_preferences', {})
        for category, preference in category_prefs.items():
            if any(keyword in suggestion_lower for keyword in self._get_category_keywords(category)):
                score += preference * 0.5
        
        # Brand preference matching  
        brand_prefs = user_profile.get('brand_preferences', {})
        for brand, preference in brand_prefs.items():
            if brand.lower() in suggestion_lower:
                score += preference * 0.3
        
        # Engagement pattern influence
        engagement_patterns = user_profile.get('engagement_patterns', {})
        high_engagement_actions = ['purchase', 'add_to_cart', 'wishlist']
        
        high_engagement_score = sum(
            engagement_patterns.get(action, 0) for action in high_engagement_actions
        )
        score += high_engagement_score * 0.2
        
        return min(score, 2.0)  # Cap the score
    
    def _calculate_contextual_score(self, suggestion: str, 
                                  session_context: Dict[str, Any]) -> float:
        """
        Calculate score based on current session context
        """
        score = 1.0
        
        # Recent search history
        recent_searches = session_context.get('recent_searches', [])
        if recent_searches:
            # Boost similar queries
            for recent_search in recent_searches[-3:]:  # Last 3 searches
                similarity = self._calculate_text_similarity(suggestion, recent_search)
                score += similarity * 0.3
        
        # Current time context
        current_hour = datetime.utcnow().hour
        time_patterns = session_context.get('time_patterns', {})
        time_preference = time_patterns.get(str(current_hour), 0.5)
        score *= (0.5 + time_preference)
        
        return score
    
    async def _calculate_collaborative_score(self, user_id: str, suggestion: str) -> float:
        """
        Calculate score based on similar users' behavior
        """
        # Simplified collaborative filtering
        # In production, this would use more sophisticated CF algorithms
        
        try:
            # Get users with similar behavior patterns
            similar_users = await self._find_similar_users(user_id, limit=50)
            
            if not similar_users:
                return 1.0
            
            # Check how many similar users engaged with similar queries
            engagement_score = 0.0
            total_similar_users = len(similar_users)
            
            for similar_user_id in similar_users:
                # Check if similar user has engaged with this type of query
                user_queries = await db_manager.get_user_query_history(similar_user_id, limit=100)
                
                for query_data in user_queries:
                    query_text = query_data.get('query_text', '')
                    similarity = self._calculate_text_similarity(suggestion, query_text)
                    
                    if similarity > 0.5:  # Threshold for similarity
                        engagement_score += similarity
                        break
            
            # Normalize by number of similar users
            if total_similar_users > 0:
                collaborative_score = engagement_score / total_similar_users
            else:
                collaborative_score = 1.0
            
            return collaborative_score
            
        except Exception as e:
            logger.error(f"Error calculating collaborative score: {e}")
            return 1.0
    
    def _calculate_temporal_score(self, user_profile: Dict[str, Any]) -> float:
        """
        Calculate score based on temporal patterns
        """
        current_hour = datetime.utcnow().hour
        time_patterns = user_profile.get('time_patterns', {})
        
        # Get user's activity level at current time
        current_time_activity = time_patterns.get(str(current_hour), 0.1)
        
        # Boost score if user is typically active at this time
        temporal_score = 0.5 + current_time_activity
        
        return temporal_score
    
    async def _find_similar_users(self, user_id: str, limit: int = 50) -> List[str]:
        """
        Find users with similar behavior patterns
        """
        # This is a simplified version. In production, use more sophisticated clustering
        
        try:
            current_user_profile = await self.get_user_profile(user_id)
            
            # Get sample of other users (would be more efficient with proper indexing)
            # For now, return empty list to avoid performance issues
            return []
            
        except Exception as e:
            logger.error(f"Error finding similar users: {e}")
            return []
    
    def _apply_diversity_constraints(self, scored_suggestions: List[Tuple[str, float]], 
                                   user_profile: Dict[str, Any]) -> List[Tuple[str, float]]:
        """
        Apply diversity constraints to prevent filter bubbles
        """
        if len(scored_suggestions) <= 5:
            return scored_suggestions
        
        # Ensure at least 20% of suggestions are from different categories
        diversified = []
        seen_categories = set()
        
        # First, add top-scored items
        for suggestion, score in scored_suggestions[:3]:
            diversified.append((suggestion, score))
            category = self._infer_category_from_query(suggestion)
            seen_categories.add(category)
        
        # Then add diverse items
        for suggestion, score in scored_suggestions[3:]:
            category = self._infer_category_from_query(suggestion)
            
            if category not in seen_categories or len(diversified) >= 8:
                diversified.append((suggestion, score))
                seen_categories.add(category)
            
            if len(diversified) >= 10:
                break
        
        return diversified
    
    def _get_category_keywords(self, category: str) -> List[str]:
        """
        Get keywords associated with a category
        """
        # Simplified keyword mapping
        category_keywords = {
            'electronics': ['phone', 'laptop', 'computer', 'tv', 'camera'],
            'clothing': ['shirt', 'pants', 'dress', 'shoes', 'jacket'],
            'home': ['furniture', 'kitchen', 'bed', 'table', 'chair'],
            'sports': ['fitness', 'exercise', 'outdoor', 'sports', 'gym']
        }
        
        return category_keywords.get(category.lower(), [])
    
    def _infer_category_from_query(self, query: str) -> str:
        """
        Infer category from query text
        """
        query_lower = query.lower()
        
        category_keywords = {
            'electronics': ['phone', 'laptop', 'computer', 'tv', 'camera', 'headphone'],
            'clothing': ['shirt', 'pants', 'dress', 'shoes', 'jacket', 'clothes'],
            'home': ['furniture', 'kitchen', 'bed', 'table', 'chair', 'home'],
            'sports': ['fitness', 'exercise', 'outdoor', 'sports', 'gym', 'bike']
        }
        
        for category, keywords in category_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return category
        
        return 'general'
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using simple word overlap
        """
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class ProductReranker:
    """
    Personalized product reranking for search results
    """
    
    def __init__(self, personalization_engine: PersonalizationEngine):
        self.personalization_engine = personalization_engine
    
    async def rerank_products(self, user_id: str, query: str, products: List[Dict[str, Any]], 
                            session_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Rerank products based on user preferences and behavior
        """
        try:
            user_profile = await self.personalization_engine.get_user_profile(user_id)
            
            # Score each product
            scored_products = []
            
            for product in products:
                score = await self._score_product(user_id, product, user_profile, session_context)
                scored_products.append((product, score))
            
            # Sort by score
            scored_products.sort(key=lambda x: x[1], reverse=True)
            
            # Apply diversity
            reranked_products = self._apply_product_diversity(scored_products, user_profile)
            
            return [product for product, _ in reranked_products]
            
        except Exception as e:
            logger.error(f"Error reranking products for user {user_id}: {e}")
            return products
    
    async def _score_product(self, user_id: str, product: Dict[str, Any], 
                           user_profile: Dict[str, Any], 
                           session_context: Dict[str, Any]) -> float:
        """
        Score a product based on user preferences
        """
        base_score = product.get('rating', 3.0) / 5.0  # Normalize rating
        
        # Price preference scoring
        price_score = self._calculate_price_preference_score(product, user_profile)
        
        # Brand preference scoring
        brand_score = self._calculate_brand_preference_score(product, user_profile)
        
        # Category preference scoring
        category_score = self._calculate_category_preference_score(product, user_profile)
        
        # Historical engagement scoring
        engagement_score = await self._calculate_product_engagement_score(user_id, product)
        
        # Combine scores
        final_score = (
            0.2 * base_score +
            0.2 * price_score +
            0.2 * brand_score +
            0.2 * category_score +
            0.2 * engagement_score
        )
        
        return final_score
    
    def _calculate_price_preference_score(self, product: Dict[str, Any], 
                                        user_profile: Dict[str, Any]) -> float:
        """
        Score based on price sensitivity
        """
        price = product.get('price')
        if not price:
            return 0.5
        
        price_sensitivity = user_profile.get('price_sensitivity', 0.5)
        
        # Normalize price (would need market data for proper normalization)
        # For now, use simple heuristic
        if price < 50:
            price_category = 'low'
        elif price < 200:
            price_category = 'medium'
        else:
            price_category = 'high'
        
        # Score based on user's price sensitivity
        if price_sensitivity < 0.3:  # Price-sensitive user
            price_scores = {'low': 1.0, 'medium': 0.6, 'high': 0.2}
        elif price_sensitivity > 0.7:  # Not price-sensitive
            price_scores = {'low': 0.5, 'medium': 0.8, 'high': 1.0}
        else:  # Moderate
            price_scores = {'low': 0.7, 'medium': 1.0, 'high': 0.7}
        
        return price_scores.get(price_category, 0.5)
    
    def _calculate_brand_preference_score(self, product: Dict[str, Any], 
                                        user_profile: Dict[str, Any]) -> float:
        """
        Score based on brand preferences
        """
        brand = product.get('brand')
        if not brand:
            return 0.5
        
        brand_preferences = user_profile.get('brand_preferences', {})
        return brand_preferences.get(brand, 0.5)
    
    def _calculate_category_preference_score(self, product: Dict[str, Any], 
                                           user_profile: Dict[str, Any]) -> float:
        """
        Score based on category preferences
        """
        category = product.get('main_category')
        if not category:
            return 0.5
        
        category_preferences = user_profile.get('category_preferences', {})
        return category_preferences.get(category, 0.5)
    
    async def _calculate_product_engagement_score(self, user_id: str, 
                                                product: Dict[str, Any]) -> float:
        """
        Score based on historical engagement with similar products
        """
        # Simplified: check if user has engaged with this specific product
        product_id = product.get('product_id')
        if not product_id:
            return 0.5
        
        try:
            # Check user engagements for this product
            engagements = await db_manager.get_user_engagements(user_id, limit=1000)
            
            for engagement in engagements:
                if engagement.get('product_id') == product_id:
                    action_type = engagement.get('action_type')
                    if action_type in ['purchase', 'add_to_cart']:
                        return 1.0
                    elif action_type in ['wishlist', 'click']:
                        return 0.8
                    elif action_type in ['view']:
                        return 0.6
            
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating engagement score: {e}")
            return 0.5
    
    def _apply_product_diversity(self, scored_products: List[Tuple[Dict[str, Any], float]], 
                               user_profile: Dict[str, Any]) -> List[Tuple[Dict[str, Any], float]]:
        """
        Apply diversity to product rankings
        """
        if len(scored_products) <= 10:
            return scored_products
        
        diversified = []
        seen_brands = set()
        seen_categories = set()
        
        # Add top products with diversity constraints
        for product, score in scored_products:
            brand = product.get('brand', 'unknown')
            category = product.get('main_category', 'general')
            
            # Add if we haven't seen too many from this brand/category
            if (len([p for p, _ in diversified if p.get('brand') == brand]) < 3 and
                len([p for p, _ in diversified if p.get('main_category') == category]) < 5):
                diversified.append((product, score))
                seen_brands.add(brand)
                seen_categories.add(category)
            
            if len(diversified) >= 20:
                break
        
        return diversified 