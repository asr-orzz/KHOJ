"""
Synthetic Data Generation Pipeline using Gemini LLM
Generates realistic user engagement patterns, search behaviors, and session data
"""
import os
import json
import asyncio
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import uuid
import google.generativeai as genai
import structlog
from config.settings import settings
from database.connection import db_manager

logger = structlog.get_logger()

class GeminiSyntheticDataGenerator:
    """
    Generates synthetic data using Gemini LLM for training and testing
    """
    
    def __init__(self):
        # Configure Gemini with API key
        api_key = settings.GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be set in settings or environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        
        logger.info(f"Initialized Gemini LLM with model: gemini-1.5-flash")
        
        # Data templates and patterns
        self.user_personas = [
            "tech_enthusiast", "fashion_lover", "home_decorator", "fitness_enthusiast",
            "book_reader", "gaming_enthusiast", "cooking_enthusiast", "outdoor_adventurer",
            "music_lover", "art_collector", "pet_owner", "parent", "student", "professional"
        ]
        
        self.engagement_patterns = {
            "browser": ["view", "scroll", "click"],
            "researcher": ["view", "specs_view", "review_view", "compare"],
            "impulse_buyer": ["view", "add_to_cart", "purchase"],
            "careful_shopper": ["view", "wishlist", "review_view", "specs_view", "purchase"],
            "window_shopper": ["view", "scroll", "wishlist"]
        }
    
    async def generate_synthetic_users(self, num_users: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate synthetic user profiles with diverse characteristics
        """
        logger.info(f"Generating {num_users} synthetic users")
        
        users = []
        for i in range(num_users):
            user_id = f"synthetic_user_{uuid.uuid4().hex[:8]}"
            
            # Select persona and characteristics
            persona = random.choice(self.user_personas)
            age_group = random.choice(["18-25", "26-35", "36-45", "46-55", "56+"])
            location = random.choice(["urban", "suburban", "rural"])
            
            # Generate user with Gemini
            user_data = await self._generate_user_with_gemini(user_id, persona, age_group, location)
            users.append(user_data)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1} users")
        
        return users
    
    async def _generate_user_with_gemini(self, user_id: str, persona: str, 
                                       age_group: str, location: str) -> Dict[str, Any]:
        """
        Use Gemini to generate realistic user characteristics
        """
        prompt = f"""
        Generate a realistic e-commerce user profile for a {persona} aged {age_group} from a {location} area.
        
        Return a JSON object with the following structure:
        {{
            "user_id": "{user_id}",
            "persona": "{persona}",
            "age_group": "{age_group}",
            "location": "{location}",
            "interests": ["list of 5-7 interests"],
            "preferred_categories": ["list of 3-5 product categories they shop for"],
            "shopping_behavior": "description of how they typically shop online",
            "price_sensitivity": "low/medium/high",
            "device_preference": "mobile/desktop/both",
            "shopping_frequency": "daily/weekly/monthly/occasionally",
            "preferred_brands": ["list of 2-4 brands they might prefer"]
        }}
        
        Make it realistic and consistent with the persona.
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Parse JSON response
            response_text = response.text.strip()
            # Clean up any markdown code blocks
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            user_data = json.loads(response_text)
            return user_data
            
        except Exception as e:
            logger.error(f"Error generating user with Gemini: {e}")
            # Fallback to basic user
            return {
                "user_id": user_id,
                "persona": persona,
                "age_group": age_group,
                "location": location,
                "interests": ["general"],
                "preferred_categories": ["electronics"],
                "shopping_behavior": "typical online shopper",
                "price_sensitivity": "medium",
                "device_preference": "both",
                "shopping_frequency": "weekly",
                "preferred_brands": []
            }
    
    async def generate_search_sessions(self, users: List[Dict[str, Any]], 
                                     num_sessions_per_user: int = 50) -> List[Dict[str, Any]]:
        """
        Generate realistic search sessions for users
        """
        logger.info(f"Generating search sessions for {len(users)} users")
        
        all_sessions = []
        
        for user in users:
            user_sessions = await self._generate_user_sessions(user, num_sessions_per_user)
            all_sessions.extend(user_sessions)
        
        return all_sessions
    
    async def _generate_user_sessions(self, user: Dict[str, Any], 
                                    num_sessions: int) -> List[Dict[str, Any]]:
        """
        Generate search sessions for a specific user
        """
        sessions = []
        persona = user['persona']
        interests = user['interests']
        preferred_categories = user['preferred_categories']
        
        # Generate session patterns based on persona
        for i in range(num_sessions):
            session_id = f"session_{uuid.uuid4().hex[:12]}"
            
            # Random session timing (past 90 days)
            days_ago = random.randint(0, 90)
            session_start = datetime.utcnow() - timedelta(days=days_ago)
            session_start += timedelta(
                hours=random.randint(8, 22),
                minutes=random.randint(0, 59)
            )
            
            # Generate session with Gemini
            session_data = await self._generate_session_with_gemini(
                session_id, user, session_start, interests, preferred_categories
            )
            
            sessions.append(session_data)
        
        return sessions
    
    async def _generate_session_with_gemini(self, session_id: str, user: Dict[str, Any],
                                          session_start: datetime, interests: List[str],
                                          categories: List[str]) -> Dict[str, Any]:
        """
        Generate a realistic search session using Gemini
        """
        prompt = f"""
        Generate a realistic e-commerce search session for a user with these characteristics:
        - Persona: {user['persona']}
        - Age: {user['age_group']}
        - Interests: {', '.join(interests)}
        - Preferred categories: {', '.join(categories)}
        - Shopping behavior: {user.get('shopping_behavior', 'typical')}
        
        Create a JSON object representing a search session with 3-8 search queries.
        Each query should show the natural progression of a shopping session.
        
        Format:
        {{
            "session_id": "{session_id}",
            "user_id": "{user['user_id']}",
            "start_time": "{session_start.isoformat()}",
            "device_type": "mobile/desktop",
            "queries": [
                {{
                    "query_text": "search query",
                    "timestamp": "timestamp",
                    "intent": "browsing/researching/buying",
                    "results_clicked": 2,
                    "time_spent": 45
                }}
            ]
        }}
        
        Make the queries realistic and progressive (broad to specific).
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=2048,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text.strip()
            # Clean up any markdown code blocks
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            session_data = json.loads(response_text)
            return session_data
            
        except Exception as e:
            logger.error(f"Error generating session with Gemini: {e}")
            # Fallback session
            return {
                "session_id": session_id,
                "user_id": user['user_id'],
                "start_time": session_start.isoformat(),
                "device_type": random.choice(["mobile", "desktop"]),
                "queries": [
                    {
                        "query_text": f"{random.choice(categories)} products",
                        "timestamp": session_start.isoformat(),
                        "intent": "browsing",
                        "results_clicked": random.randint(1, 5),
                        "time_spent": random.randint(30, 180)
                    }
                ]
            }
    
    async def generate_user_engagements(self, sessions: List[Dict[str, Any]], 
                                      products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate realistic user engagement data for sessions
        """
        logger.info(f"Generating engagements for {len(sessions)} sessions")
        
        all_engagements = []
        
        for session in sessions:
            session_engagements = await self._generate_session_engagements(session, products)
            all_engagements.extend(session_engagements)
        
        return all_engagements
    
    async def _generate_session_engagements(self, session: Dict[str, Any], 
                                          products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate engagements for a specific session
        """
        engagements = []
        user_id = session['user_id']
        session_id = session['session_id']
        
        # Get user behavior pattern
        behavior_pattern = random.choice(list(self.engagement_patterns.keys()))
        possible_actions = self.engagement_patterns[behavior_pattern]
        
        for query in session.get('queries', []):
            query_timestamp = datetime.fromisoformat(query['timestamp'])
            
            # Generate engagements for this query
            num_products_viewed = query.get('results_clicked', 3)
            query_products = random.sample(products, min(num_products_viewed, len(products)))
            
            for rank, product in enumerate(query_products):
                # Generate engagement sequence
                engagement_sequence = await self._generate_engagement_sequence_with_gemini(
                    user_id, session_id, product, rank + 1, behavior_pattern, query_timestamp
                )
                
                engagements.extend(engagement_sequence)
        
        return engagements
    
    async def _generate_engagement_sequence_with_gemini(self, user_id: str, session_id: str,
                                                      product: Dict[str, Any], rank: int,
                                                      behavior_pattern: str, 
                                                      start_time: datetime) -> List[Dict[str, Any]]:
        """
        Generate a realistic sequence of engagements with a product
        """
        prompt = f"""
        Generate a realistic sequence of user engagements with an e-commerce product.
        
        Product: {product.get('title', 'Product')}
        Category: {product.get('main_category', 'General')}
        Price: ${product.get('price', 50)}
        Rating: {product.get('rating', 4.0)}/5
        User behavior pattern: {behavior_pattern}
        Product rank in search results: {rank}
        
        Generate 1-5 engagement actions that a user might take with this product.
        Higher ranked products (rank 1-3) get more engagement.
        
        Available actions: view, click, add_to_cart, remove_from_cart, wishlist, 
        remove_wishlist, purchase, review_view, specs_view, image_view, scroll
        
        Return JSON array:
        [
            {{
                "action_type": "view",
                "duration_seconds": 30,
                "timestamp_offset_seconds": 0
            }}
        ]
        
        Make it realistic - people view first, then might click, then add to cart, etc.
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=1024,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text.strip()
            # Clean up any markdown code blocks
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            engagement_actions = json.loads(response_text)
            
            # Convert to full engagement objects
            engagements = []
            for action in engagement_actions:
                engagement_time = start_time + timedelta(seconds=action.get('timestamp_offset_seconds', 0))
                
                engagement = {
                    "engagement_id": f"eng_{uuid.uuid4().hex[:12]}",
                    "user_id": user_id,
                    "session_id": session_id,
                    "product_id": product.get('product_id'),
                    "action_type": action['action_type'],
                    "item_rank": rank,
                    "timestamp": engagement_time.isoformat(),
                    "duration_seconds": action.get('duration_seconds'),
                    "page_section": "product_list"
                }
                
                engagements.append(engagement)
            
            return engagements
            
        except Exception as e:
            logger.error(f"Error generating engagement sequence: {e}")
            # Fallback engagement
            return [{
                "engagement_id": f"eng_{uuid.uuid4().hex[:12]}",
                "user_id": user_id,
                "session_id": session_id,
                "product_id": product.get('product_id'),
                "action_type": "view",
                "item_rank": rank,
                "timestamp": start_time.isoformat(),
                "duration_seconds": random.randint(10, 60),
                "page_section": "product_list"
            }]
    
    async def generate_synthetic_queries(self, categories: List[str], 
                                       num_queries_per_category: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate synthetic search queries for different categories with e-commerce focus
        """
        logger.info(f"Generating synthetic queries for {len(categories)} categories")
        
        all_queries = []
        
        for category in categories:
            category_queries = await self._generate_category_queries(category, num_queries_per_category)
            all_queries.extend(category_queries)
        
        # Also generate cross-category and trending queries
        trending_queries = await self._generate_trending_ecommerce_queries(num_queries_per_category // 2)
        all_queries.extend(trending_queries)
        
        return all_queries
    
    async def _generate_trending_ecommerce_queries(self, num_queries: int) -> List[Dict[str, Any]]:
        """
        Generate trending e-commerce queries that are popular across platforms
        """
        prompt = f"""
        Generate {min(num_queries, 100)} trending e-commerce search queries that are currently popular 
    on platforms like Meesho, etc. Include:
        
        - Seasonal queries (holidays, weather-based)
        - Technology trends (latest gadgets, smartphones)
        - Fashion trends (current styles, popular brands)
        - Home essentials (work from home, fitness)
        - Gift ideas (occasions, demographics)
        - Deal-seeking queries (discounts, offers)
        
        Return as JSON array:
        [
            {{
                "query_text": "search query",
                "intent": "browsing/research/purchase",
                "specificity": "broad/medium/specific",
                "estimated_results": 1000,
                "trend_category": "seasonal/tech/fashion/home/gifts/deals"
            }}
        ]
        
        Make them realistic and diverse.
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=4096,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            queries = json.loads(response_text)
            
            # Add metadata
            for query in queries:
                query['category'] = 'trending'
                query['generated_at'] = datetime.utcnow().isoformat()
                query['source'] = 'gemini_synthetic_trending'
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating trending queries: {e}")
            return []
    
    async def _generate_category_queries(self, category: str, num_queries: int) -> List[Dict[str, Any]]:
        """
        Generate queries for a specific category using Gemini
        """
        prompt = f"""
        Generate {min(num_queries, 100)} realistic search queries for the e-commerce category: {category}
        
        Include various types of queries:
        - Brand-specific queries (e.g., "nike shoes")
        - Feature-specific queries (e.g., "wireless headphones")
        - Price-focused queries (e.g., "cheap laptops under 500")
        - Use case queries (e.g., "running shoes for marathon")
        - Comparison queries (e.g., "iphone vs samsung")
        
        Return as JSON array:
        [
            {{
                "query_text": "search query",
                "intent": "browsing/research/purchase",
                "specificity": "broad/medium/specific",
                "estimated_results": 1000
            }}
        ]
        
        Make queries diverse and realistic for {category}.
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=4096,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text.strip()
            # Clean up any markdown code blocks
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            queries = json.loads(response_text)
            
            # Add metadata
            for query in queries:
                query['category'] = category
                query['generated_at'] = datetime.utcnow().isoformat()
                query['source'] = 'gemini_synthetic'
            
            return queries
            
        except Exception as e:
            logger.error(f"Error generating queries for {category}: {e}")
            return []
    
    async def generate_realistic_products(self, num_products: int = 1000) -> List[Dict[str, Any]]:
        """
        Generate realistic product data for engagement simulation
        """
        logger.info(f"Generating {num_products} realistic products")
        
        # Generate products in batches
        batch_size = 50  # Generate 50 products at a time
        all_products = []
        
        categories = [
            "Electronics", "Clothing & Fashion", "Home & Kitchen", "Sports & Outdoors",
            "Books & Media", "Beauty & Personal Care", "Automotive", "Toys & Games",
            "Health & Household", "Tools & Home Improvement"
        ]
        
        for i in range(0, num_products, batch_size):
            batch_products = await self._generate_product_batch(
                min(batch_size, num_products - i), categories
            )
            all_products.extend(batch_products)
            
            if (i + batch_size) % 200 == 0:
                logger.info(f"Generated {min(i + batch_size, num_products)} products")
        
        return all_products
    
    async def _generate_product_batch(self, batch_size: int, categories: List[str]) -> List[Dict[str, Any]]:
        """
        Generate a batch of realistic products
        """
        selected_categories = random.sample(categories, min(5, len(categories)))
        
        prompt = f"""
        Generate {batch_size} realistic e-commerce products for these categories: {', '.join(selected_categories)}
        
    Create diverse products with realistic details that would be found on Meesho:
        
        Return as JSON array:
        [
            {{
                "product_id": "unique_id",
                "title": "realistic product title",
                "description": "detailed description",
                "main_category": "category from the list",
                "price": 25.99,
                "rating": 4.2,
                "rating_count": 150,
                "brand": "brand name",
                "features": ["feature1", "feature2"],
                "keywords": ["keyword1", "keyword2"]
            }}
        ]
        
        Make titles and descriptions realistic for actual products you'd find online.
        Vary prices realistically within each category.
        """
        
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=4096,
            )
            
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            response_text = response.text.strip()
            if response_text.startswith('```json'):
                response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            products = json.loads(response_text)
            
            # Add unique product IDs and timestamps
            for i, product in enumerate(products):
                if 'product_id' not in product or not product['product_id']:
                    product['product_id'] = f"prod_{uuid.uuid4().hex[:12]}"
                product['generated_at'] = datetime.utcnow().isoformat()
                product['source'] = 'gemini_synthetic'
            
            return products
            
        except Exception as e:
            logger.error(f"Error generating product batch: {e}")
            # Critical failure - cannot proceed without real data
            logger.error("❌ CRITICAL: Cannot generate synthetic products without Gemini")
            logger.error("Check your GEMINI_API_KEY and internet connection")
            raise RuntimeError("Synthetic product generation failed - Gemini LLM required")
    
    async def save_synthetic_data_to_database(self, users: List[Dict[str, Any]], 
                                            sessions: List[Dict[str, Any]],
                                            engagements: List[Dict[str, Any]]) -> bool:
        """
        Save all synthetic data to the database
        """
        logger.info("Saving synthetic data to database")
        
        try:
            # Save users
            for user in users:
                await db_manager.create_user({
                    'user_id': user['user_id'],
                    'location': user.get('location'),
                    'age_group': user.get('age_group'),
                    'preferred_categories': []  # Will be populated from behavior
                })
            
            # Save sessions
            for session in sessions:
                await db_manager.create_session({
                    'session_id': session['session_id'],
                    'user_id': session['user_id'],
                    'start_time': session['start_time'],
                    'device_type': session.get('device_type'),
                    'total_searches': len(session.get('queries', []))
                })
                
                # Save queries
                for query in session.get('queries', []):
                    await db_manager.log_search_query({
                        'query_id': f"query_{uuid.uuid4().hex[:12]}",
                        'session_id': session['session_id'],
                        'user_id': session['user_id'],
                        'query_text': query['query_text'],
                        'timestamp': query['timestamp']
                    })
            
            # Save engagements
            for engagement in engagements:
                await db_manager.log_engagement(engagement)
            
            logger.info("Successfully saved synthetic data to database")
            return True
            
        except Exception as e:
            logger.error(f"Error saving synthetic data: {e}")
            return False

async def generate_full_synthetic_dataset(num_users: int = 1000) -> bool:
    """
    Generate a complete synthetic dataset for testing and training
    """
    generator = GeminiSyntheticDataGenerator()
    
    try:
        # Generate users
        logger.info("Step 1: Generating synthetic users")
        users = await generator.generate_synthetic_users(num_users)
        
        # Generate sessions
        logger.info("Step 2: Generating search sessions")
        sessions = await generator.generate_search_sessions(users, num_sessions_per_user=30)
        
        # Load some product data for engagement generation
        logger.info("Step 3: Loading products for engagement generation")
        # Generate realistic products using Gemini
        products = await generator.generate_realistic_products(num_products=1000)
        
        # Generate engagements
        logger.info("Step 4: Generating user engagements")
        engagements = await generator.generate_user_engagements(sessions, products)
        
        # Save to database
        logger.info("Step 5: Saving to database")
        success = await generator.save_synthetic_data_to_database(users, sessions, engagements)
        
        if success:
            logger.info(f"Successfully generated synthetic dataset with {num_users} users")
            return True
        else:
            logger.error("Failed to save synthetic dataset")
            return False
            
    except Exception as e:
        logger.error(f"Error generating synthetic dataset: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(generate_full_synthetic_dataset(num_users=100)) 