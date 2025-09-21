"""
Vernacular and Code-Mix Query Processing for KHOJ+
Handles Roman-script Hindi, phonetic corrections, and Bharat-specific language patterns
"""
import re
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import structlog

logger = structlog.get_logger()

@dataclass
class IntentTags:
    price_band: Optional[str] = None  # "under_299", "under_499", etc.
    occasion: Optional[str] = None    # "wedding", "festive", "casual"
    quality_intent: Optional[str] = None  # "4star_plus", "high_rated"
    constraints: List[str] = None     # ["cod", "returnable", "fast_delivery"]

class VernacularProcessor:
    """Handles vernacular queries and intent understanding for Bharat users"""
    
    def __init__(self):
        self.transliteration_map = self._build_transliteration_map()
        self.colloquial_dictionary = self._build_colloquial_dictionary()
        self.price_patterns = self._build_price_patterns()
        self.occasion_patterns = self._build_occasion_patterns()
        
    def _build_transliteration_map(self) -> Dict[str, str]:
        """Build Roman-script Hindi to English transliteration mapping"""
        return {
            # Clothing
            'sadi': 'saree', 'saadi': 'saree', 'saree': 'saree',
            'kurti': 'kurti', 'kurta': 'kurta', 'kurtis': 'kurti',
            'dupatta': 'dupatta', 'duppatta': 'dupatta',
            'lehenga': 'lehenga', 'lehnga': 'lehenga',
            'salwar': 'salwar', 'shalwar': 'salwar',
            'churidar': 'churidar', 'chudi': 'churidar',
            'anarkali': 'anarkali', 'anarkali': 'anarkali',
            
            # Accessories
            'chasma': 'sunglasses', 'chashma': 'sunglasses',
            'chappal': 'sandals', 'chappals': 'sandals',
            'jutti': 'shoes', 'juttis': 'shoes',
            'mojari': 'shoes', 'mojaris': 'shoes',
            
            # Home & Kitchen
            'bartan': 'utensils', 'bartans': 'utensils',
            'tiffin': 'lunch box', 'dabba': 'box',
            'chawal': 'rice', 'dal': 'lentils',
            'masala': 'spice', 'masalas': 'spices',
            
            # Electronics
            'mobile': 'phone', 'mobail': 'phone',
            'charger': 'charger', 'charjar': 'charger',
            'earphone': 'earphone', 'earfone': 'earphone',
            
            # Common price terms
            'sasta': 'cheap', 'saste': 'cheap',
            'accha': 'good', 'ache': 'good',
            'badiya': 'good', 'badhiya': 'good',
        }
    
    def _build_colloquial_dictionary(self) -> Dict[str, str]:
        """Build colloquial phrase to standard term mapping"""
        return {
            'kurti set': 'kurti with dupatta',
            'bedsheet double': 'double bed sheet',
            'bedsheet single': 'single bed sheet',
            'night dress': 'nightwear',
            'track pant': 'track pants',
            'sports shoes': 'sports shoes',
            'running shoes': 'running shoes',
            'formal shoes': 'formal shoes',
            'casual shoes': 'casual shoes',
            'party wear': 'party dress',
            'office wear': 'formal wear',
            'gym wear': 'sportswear',
            'winter wear': 'winter clothes',
        }
    
    def _build_price_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Build price intent detection patterns"""
        patterns = [
            (re.compile(r'\bunder\s*(\d+)\b', re.IGNORECASE), r'under_\1'),
            (re.compile(r'\bbelow\s*(\d+)\b', re.IGNORECASE), r'under_\1'),
            (re.compile(r'\bunder\s*rs?\s*(\d+)\b', re.IGNORECASE), r'under_\1'),
            (re.compile(r'\bunder\s*₹\s*(\d+)\b', re.IGNORECASE), r'under_\1'),
            (re.compile(r'\bless\s*than\s*(\d+)\b', re.IGNORECASE), r'under_\1'),
            (re.compile(r'\b(cheap|sasta)\b', re.IGNORECASE), 'budget_friendly'),
            (re.compile(r'\b(affordable|reasonable)\b', re.IGNORECASE), 'budget_friendly'),
        ]
        return patterns
    
    def _build_occasion_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Build occasion intent detection patterns"""
        patterns = [
            (re.compile(r'\b(wedding|shadi|marriage)\b', re.IGNORECASE), 'wedding'),
            (re.compile(r'\b(party|celebration)\b', re.IGNORECASE), 'party'),
            (re.compile(r'\b(office|work|formal)\b', re.IGNORECASE), 'formal'),
            (re.compile(r'\b(casual|daily|everyday)\b', re.IGNORECASE), 'casual'),
            (re.compile(r'\b(festival|festive|diwali|holi|eid)\b', re.IGNORECASE), 'festive'),
            (re.compile(r'\b(gym|workout|exercise|sports)\b', re.IGNORECASE), 'sports'),
            (re.compile(r'\b(hostel|college|student)\b', re.IGNORECASE), 'student'),
        ]
        return patterns
    
    def process_query(self, query: str) -> Tuple[str, IntentTags]:
        """
        Process vernacular query and extract intent tags
        Returns: (normalized_query, intent_tags)
        """
        original_query = query
        
        # Step 1: Normalize and clean
        query = query.lower().strip()
        
        # Step 2: Apply transliteration
        query = self._apply_transliteration(query)
        
        # Step 3: Apply colloquial mapping
        query = self._apply_colloquial_mapping(query)
        
        # Step 4: Extract intent tags
        intent_tags = self._extract_intent_tags(original_query)
        
        # Step 5: Phonetic spell correction
        query = self._phonetic_spell_correct(query)
        
        return query, intent_tags
    
    def _apply_transliteration(self, query: str) -> str:
        """Apply transliteration mapping"""
        words = query.split()
        translated_words = []
        
        for word in words:
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.transliteration_map:
                translated_words.append(self.transliteration_map[clean_word])
            else:
                translated_words.append(word)
        
        return ' '.join(translated_words)
    
    def _apply_colloquial_mapping(self, query: str) -> str:
        """Apply colloquial phrase mapping"""
        for phrase, replacement in self.colloquial_dictionary.items():
            query = re.sub(rf'\b{re.escape(phrase)}\b', replacement, query, flags=re.IGNORECASE)
        return query
    
    def _extract_intent_tags(self, query: str) -> IntentTags:
        """Extract structured intent tags from query"""
        intent = IntentTags(constraints=[])
        
        # Extract price intent
        for pattern, replacement in self.price_patterns:
            match = pattern.search(query)
            if match:
                if replacement.startswith('under_'):
                    intent.price_band = replacement
                else:
                    intent.price_band = replacement
                break
        
        # Extract occasion intent
        for pattern, occasion in self.occasion_patterns:
            if pattern.search(query):
                intent.occasion = occasion
                break
        
        # Extract quality intent
        if re.search(r'\b(4\s*star|4★|high\s*rated|good\s*quality|accha)\b', query, re.IGNORECASE):
            intent.quality_intent = "4star_plus"
        
        # Extract constraints
        if re.search(r'\b(cod|cash\s*on\s*delivery)\b', query, re.IGNORECASE):
            intent.constraints.append("cod")
        if re.search(r'\b(return|returnable|exchange)\b', query, re.IGNORECASE):
            intent.constraints.append("returnable")
        if re.search(r'\b(fast\s*delivery|quick\s*delivery|same\s*day)\b', query, re.IGNORECASE):
            intent.constraints.append("fast_delivery")
        if re.search(r'\b(meesho\s*mall|mall)\b', query, re.IGNORECASE):
            intent.constraints.append("meesho_mall")
        
        return intent
    
    def _phonetic_spell_correct(self, query: str) -> str:
        """Apply phonetic spell correction for common misspellings"""
        corrections = {
            'fone': 'phone', 'phon': 'phone',
            'shos': 'shoes', 'shoos': 'shoes',
            'tshrt': 'tshirt', 'tshirt': 't-shirt',
            'jens': 'jeans', 'jeens': 'jeans',
            'dres': 'dress', 'drees': 'dress',
            'wach': 'watch', 'watc': 'watch',
        }
        
        words = query.split()
        corrected_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in corrections:
                corrected_words.append(corrections[clean_word])
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def generate_intent_chips(self, query: str, intent_tags: IntentTags) -> List[Dict[str, str]]:
        """Generate intent chips for UI"""
        chips = []
        
        # Price chips
        if intent_tags.price_band:
            if intent_tags.price_band.startswith('under_'):
                amount = intent_tags.price_band.split('_')[1]
                chips.append({
                    'type': 'price',
                    'label': f'Under ₹{amount}',
                    'value': intent_tags.price_band
                })
            elif intent_tags.price_band == 'budget_friendly':
                chips.append({
                    'type': 'price',
                    'label': 'Budget Friendly',
                    'value': 'budget_friendly'
                })
        
        # Quality chips
        if intent_tags.quality_intent == "4star_plus":
            chips.append({
                'type': 'quality',
                'label': '4★+',
                'value': '4star_plus'
            })
        
        # Constraint chips
        for constraint in intent_tags.constraints:
            if constraint == "cod":
                chips.append({'type': 'payment', 'label': 'COD', 'value': 'cod'})
            elif constraint == "fast_delivery":
                chips.append({'type': 'delivery', 'label': 'Fast Delivery', 'value': 'fast_delivery'})
            elif constraint == "meesho_mall":
                chips.append({'type': 'source', 'label': 'Meesho Mall', 'value': 'meesho_mall'})
            elif constraint == "returnable":
                chips.append({'type': 'policy', 'label': 'Returnable', 'value': 'returnable'})
        
        # Occasion chips
        if intent_tags.occasion:
            occasion_labels = {
                'wedding': 'Wedding',
                'party': 'Party',
                'formal': 'Formal',
                'casual': 'Casual',
                'festive': 'Festive',
                'sports': 'Sports',
                'student': 'Student'
            }
            chips.append({
                'type': 'occasion',
                'label': occasion_labels.get(intent_tags.occasion, intent_tags.occasion.title()),
                'value': intent_tags.occasion
            })
        
        return chips