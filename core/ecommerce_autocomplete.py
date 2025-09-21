"""
E-commerce Specific Autocomplete Engine
Implements real product-specific autocomplete suggestions using CADENCE + real data
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import defaultdict, Counter
import structlog
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from core.cadence_model import CADENCEModel, DynamicBeamSearch
from core.data_processor import DataProcessor
from config.settings import settings

logger = structlog.get_logger()

class ECommerceAutocompleteEngine:
    """
    Advanced e-commerce autocomplete engine with product-specific suggestions
    """
    
    def __init__(self, cadence_model: CADENCEModel, vocab: Dict[str, int], 
                 product_data: Optional[List[Dict[str, Any]]] = None):
        self.cadence_model = cadence_model
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.product_data = product_data or []
        
        # Initialize components
        self.beam_search = DynamicBeamSearch(cadence_model, vocab)
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.data_processor = DataProcessor()
        
        # Build product-specific indices
        self._build_product_indices()
        
        # E-commerce specific patterns
        self.query_patterns = {
            'brand_product': r'(\w+)\s+(phone|laptop|shoes|watch|headphones|tablet)',
            'product_features': r'(wireless|bluetooth|gaming|running|smart|portable)\s+(\w+)',
            'price_queries': r'(cheap|affordable|under|below)\s+(\d+|\w+)',
            'comparison': r'(\w+)\s+(vs|versus)\s+(\w+)',
            'specifications': r'(\d+gb|gb|inch|size|color|material)'
        }
        
        logger.info(f"Initialized E-commerce Autocomplete Engine with {len(self.product_data)} products")
    
    def _build_product_indices(self):
        """Build efficient indices for product search"""
        self.brand_index = defaultdict(list)
        self.category_index = defaultdict(list)
        self.keyword_index = defaultdict(list)
        self.title_embeddings = []
        
        if not self.product_data:
            logger.warning("No product data provided for building indices")
            return
        
        logger.info("Building product indices...")
        
        # Process each product
        for i, product in enumerate(self.product_data):
            title = product.get('title', '').lower()
            brand = product.get('brand', '').lower()
            category = product.get('main_category', '').lower()
            
            # Build brand index
            if brand:
                self.brand_index[brand].append(i)
            
            # Build category index
            if category:
                self.category_index[category].append(i)
            
            # Build keyword index
            keywords = self._extract_keywords(title)
            for keyword in keywords:
                self.keyword_index[keyword].append(i)
        
        # Create embeddings for semantic similarity
        if self.product_data:
            titles = [product.get('title', '') for product in self.product_data]
            try:
                self.title_embeddings = self.embedding_model.encode(titles[:1000])  # Limit for memory
                logger.info(f"Created embeddings for {len(self.title_embeddings)} products")
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                self.title_embeddings = []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from product text"""
        # Remove special characters and split
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words and short words
        stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if len(word) > 2 and word not in stop_words]
        
        return keywords
    
    async def get_suggestions(self, query_prefix: str, max_suggestions: int = 10,
                            user_context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Get comprehensive e-commerce autocomplete suggestions
        """
        query_prefix = query_prefix.strip().lower()
        if len(query_prefix) < 2:
            return []
        
        suggestions = []
        
        # 1. Neural CADENCE suggestions
        neural_suggestions = await self._get_neural_suggestions(query_prefix, max_suggestions // 2)
        suggestions.extend(neural_suggestions)
        
        # 2. Product-based suggestions
        product_suggestions = await self._get_product_based_suggestions(query_prefix, max_suggestions // 2)
        suggestions.extend(product_suggestions)
        
        # 3. Pattern-based suggestions
        pattern_suggestions = await self._get_pattern_based_suggestions(query_prefix, max_suggestions // 4)
        suggestions.extend(pattern_suggestions)
        
        # 4. Trending/popular suggestions
        trending_suggestions = await self._get_trending_suggestions(query_prefix, max_suggestions // 4)
        suggestions.extend(trending_suggestions)
        
        # Deduplicate and rank
        final_suggestions = self._rank_and_deduplicate(suggestions, query_prefix, max_suggestions)
        
        return final_suggestions
    
    async def _get_neural_suggestions(self, query_prefix: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Get suggestions from CADENCE neural model"""
        try:
            # Tokenize prefix
            tokens = query_prefix.split()
            if not tokens:
                return []
            
            # Generate with CADENCE
            generated = self.beam_search.search(tokens, category_id=0, model_type="query")
            
            suggestions = []
            for suggestion in generated[:max_suggestions]:
                if suggestion and suggestion != query_prefix:
                    suggestions.append({
                        'text': suggestion,
                        'type': 'neural',
                        'score': 0.8,
                        'source': 'CADENCE'
                    })
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting neural suggestions: {e}")
            return []
    
    async def _get_product_based_suggestions(self, query_prefix: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Get suggestions based on actual product data"""
        suggestions = []
        
        try:
            # Search in product titles directly
            matching_products = []
            
            for i, product in enumerate(self.product_data[:1000]):  # Limit search for performance
                title = product.get('title', '').lower()
                if query_prefix in title:
                    # Calculate relevance score
                    score = self._calculate_product_relevance(query_prefix, product)
                    matching_products.append((product, score, i))
            
            # Sort by relevance and extract suggestions
            matching_products.sort(key=lambda x: x[1], reverse=True)
            
            seen_suggestions = set()
            for product, score, idx in matching_products[:max_suggestions * 2]:
                # Generate suggestion from product title
                title = product.get('title', '')
                suggestion_text = self._extract_suggestion_from_title(query_prefix, title)
                
                if suggestion_text and suggestion_text not in seen_suggestions:
                    suggestions.append({
                        'text': suggestion_text,
                        'type': 'product',
                        'score': score,
                        'source': 'product_match',
                        'product_id': product.get('product_id'),
                        'category': product.get('main_category'),
                        'brand': product.get('brand')
                    })
                    seen_suggestions.add(suggestion_text)
                    
                    if len(suggestions) >= max_suggestions:
                        break
            
            return suggestions
            
        except Exception as e:
            logger.error(f"Error getting product-based suggestions: {e}")
            return []
    
    def _calculate_product_relevance(self, query_prefix: str, product: Dict[str, Any]) -> float:
        """Calculate how relevant a product is to the query prefix"""
        title = product.get('title', '').lower()
        brand = product.get('brand', '').lower()
        category = product.get('main_category', '').lower()
        rating = product.get('rating', 0) or 0
        rating_count = product.get('rating_count', 0) or 0
        
        score = 0.0
        
        # Exact prefix match bonus
        if title.startswith(query_prefix):
            score += 1.0
        elif query_prefix in title:
            score += 0.6
        
        # Brand match bonus
        if query_prefix in brand:
            score += 0.4
        
        # Category relevance
        if query_prefix in category:
            score += 0.3
        
        # Popularity score (rating * log(rating_count))
        if rating > 0 and rating_count > 0:
            popularity = rating * np.log(1 + rating_count) / 10.0
            score += min(popularity, 0.3)
        
        return score
    
    def _extract_suggestion_from_title(self, query_prefix: str, title: str) -> str:
        """Extract a good autocomplete suggestion from a product title"""
        title_lower = title.lower()
        words = title_lower.split()
        
        # Find the position of the prefix
        prefix_words = query_prefix.split()
        
        # Simple approach: find where prefix matches and extend
        if len(prefix_words) == 1:
            prefix_word = prefix_words[0]
            for i, word in enumerate(words):
                if word.startswith(prefix_word):
                    # Take next 2-4 words as suggestion
                    end_idx = min(i + 4, len(words))
                    suggestion = ' '.join(words[i:end_idx])
                    return suggestion
        
        # Multi-word prefix
        for i in range(len(words) - len(prefix_words) + 1):
            if all(words[i + j].startswith(prefix_words[j]) for j in range(len(prefix_words))):
                end_idx = min(i + len(prefix_words) + 2, len(words))
                suggestion = ' '.join(words[i:end_idx])
                return suggestion
        
        # Fallback: if prefix is contained, return relevant part
        if query_prefix in title_lower:
            return title_lower[:50]  # First 50 chars
        
        return ""
    
    async def _get_pattern_based_suggestions(self, query_prefix: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Get suggestions based on common e-commerce query patterns"""
        suggestions = []
        
        try:
            # Check for different query patterns
            for pattern_name, pattern in self.query_patterns.items():
                if re.search(pattern, query_prefix):
                    pattern_suggestions = self._generate_pattern_suggestions(
                        query_prefix, pattern_name, max_suggestions // 2
                    )
                    suggestions.extend(pattern_suggestions)
            
            # Brand + product type suggestions
            brand_suggestions = self._generate_brand_suggestions(query_prefix, max_suggestions // 2)
            suggestions.extend(brand_suggestions)
            
            return suggestions[:max_suggestions]
            
        except Exception as e:
            logger.error(f"Error getting pattern-based suggestions: {e}")
            return []
    
    def _generate_pattern_suggestions(self, query_prefix: str, pattern_name: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Generate suggestions based on specific patterns"""
        suggestions = []
        
        if pattern_name == 'brand_product':
            # Suggest popular brand + product combinations
            brand_products = [
                "apple iphone", "samsung galaxy", "nike shoes", "adidas sneakers",
                "sony headphones", "bose speakers", "dell laptop", "hp printer"
            ]
            for combo in brand_products:
                if combo.startswith(query_prefix):
                    suggestions.append({
                        'text': combo,
                        'type': 'pattern',
                        'score': 0.7,
                        'source': f'pattern_{pattern_name}'
                    })
        
        elif pattern_name == 'price_queries':
            # Suggest price-based completions
            price_completions = [
                f"{query_prefix} 100", f"{query_prefix} 500", f"{query_prefix} 1000",
                f"{query_prefix} smartphone", f"{query_prefix} laptop"
            ]
            for completion in price_completions:
                suggestions.append({
                    'text': completion,
                    'type': 'pattern',
                    'score': 0.6,
                    'source': f'pattern_{pattern_name}'
                })
        
        return suggestions[:max_suggestions]
    
    def _generate_brand_suggestions(self, query_prefix: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Generate brand-based suggestions"""
        suggestions = []
        
        # Popular brands in different categories
        popular_brands = {
            'electronics': ['apple', 'samsung', 'sony', 'lg', 'panasonic'],
            'clothing': ['nike', 'adidas', 'puma', 'levis', 'calvin klein'],
            'home': ['ikea', 'philips', 'black decker', 'dyson']
        }
        
        for category, brands in popular_brands.items():
            for brand in brands:
                if brand.startswith(query_prefix.lower()):
                    suggestions.append({
                        'text': brand,
                        'type': 'brand',
                        'score': 0.5,
                        'source': 'brand_suggestion',
                        'category': category
                    })
        
        return suggestions[:max_suggestions]
    
    async def _get_trending_suggestions(self, query_prefix: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Get trending/popular suggestions"""
        suggestions = []
        
        # Simulated trending terms (in real system, these would come from analytics)
        trending_terms = [
            "iphone 15", "macbook pro", "air fryer", "smart watch", "wireless earbuds",
            "gaming laptop", "running shoes", "bluetooth speaker", "tablet", "home security"
        ]
        
        for term in trending_terms:
            if term.startswith(query_prefix):
                suggestions.append({
                    'text': term,
                    'type': 'trending',
                    'score': 0.4,
                    'source': 'trending'
                })
        
        return suggestions[:max_suggestions]
    
    def _rank_and_deduplicate(self, suggestions: List[Dict[str, Any]], 
                            query_prefix: str, max_suggestions: int) -> List[Dict[str, Any]]:
        """Rank and deduplicate suggestions"""
        # Remove duplicates
        seen = set()
        unique_suggestions = []
        
        for suggestion in suggestions:
            text = suggestion['text'].lower().strip()
            if text not in seen and text != query_prefix.lower():
                seen.add(text)
                unique_suggestions.append(suggestion)
        
        # Sort by score and relevance
        unique_suggestions.sort(key=lambda x: (
            x['score'],
            len(x['text']),  # Prefer shorter suggestions
            -ord(x['text'][0]) if x['text'] else 0  # Alphabetical tie-breaking
        ), reverse=True)
        
        return unique_suggestions[:max_suggestions]


class ProductSpecificQueryGenerator:
    """
    Generates product-specific autocomplete suggestions using real product data
    """
    
    def __init__(self, product_data: List[Dict[str, Any]]):
        self.product_data = product_data
        self.category_products = defaultdict(list)
        self.brand_products = defaultdict(list)
        
        # Build indices
        for product in product_data:
            category = product.get('main_category', '').lower()
            brand = product.get('brand', '').lower()
            
            if category:
                self.category_products[category].append(product)
            if brand:
                self.brand_products[brand].append(product)
    
    def generate_category_queries(self, category: str, num_queries: int = 20) -> List[str]:
        """Generate queries specific to a product category"""
        if category.lower() not in self.category_products:
            return []
        
        products = self.category_products[category.lower()]
        queries = []
        
        # Extract common terms from product titles
        all_words = []
        for product in products[:100]:  # Limit for performance
            title = product.get('title', '').lower()
            words = re.findall(r'\b\w+\b', title)
            all_words.extend(words)
        
        # Get most common terms
        word_freq = Counter(all_words)
        common_words = [word for word, freq in word_freq.most_common(50) if len(word) > 3]
        
        # Generate queries
        for word in common_words[:num_queries]:
            queries.append(f"{word} {category}")
            if len(queries) >= num_queries:
                break
        
        return queries
    
    def generate_brand_queries(self, brand: str, num_queries: int = 20) -> List[str]:
        """Generate queries specific to a brand"""
        if brand.lower() not in self.brand_products:
            return []
        
        products = self.brand_products[brand.lower()]
        queries = set()
        
        for product in products[:50]:
            title = product.get('title', '').lower()
            
            # Extract product types after brand name
            if brand.lower() in title:
                parts = title.split(brand.lower())
                if len(parts) > 1:
                    after_brand = parts[1].strip()
                    words = after_brand.split()[:3]  # Take first 3 words
                    if words:
                        query = f"{brand} {' '.join(words)}"
                        queries.add(query)
        
        return list(queries)[:num_queries]