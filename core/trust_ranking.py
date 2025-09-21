"""
Trust-Aware Ranking System for KHOJ+
Implements multi-objective ranking with trust, quality, and deliverability features
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import math

@dataclass
class TrustSignals:
    rating: Optional[float] = None
    review_count: Optional[int] = None
    return_rate: Optional[float] = None
    seller_cancel_rate: Optional[float] = None
    dispute_ratio: Optional[float] = None
    quality_badge: bool = False
    seller_rating: Optional[float] = None
    delivery_promise_adherence: Optional[float] = None

@dataclass
class DeliverabilitySignals:
    pincode_eta: Optional[int] = None  # days
    stock_availability: bool = True
    seller_location_distance: Optional[float] = None
    delivery_success_rate: Optional[float] = None
    cod_available: bool = True

@dataclass
class PersonalizationSignals:
    user_price_affinity: Optional[str] = None  # "budget", "mid", "premium"
    user_category_preference: Optional[str] = None
    historical_spend_avg: Optional[float] = None
    cod_tendency: Optional[float] = None  # 0-1
    return_tendency: Optional[float] = None  # 0-1
    brand_affinity: List[str] = None

class RankingObjective(Enum):
    RELEVANCE = "relevance"
    TRUST = "trust"
    AFFORDABILITY = "affordability"
    DELIVERABILITY = "deliverability"
    PERSONALIZATION = "personalization"

class TrustAwareRanker:
    """
    Multi-objective ranker that balances relevance, trust, and user expectations
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self.objective_weights = self.config.get('objective_weights', {
            RankingObjective.RELEVANCE: 0.35,
            RankingObjective.TRUST: 0.25,
            RankingObjective.AFFORDABILITY: 0.15,
            RankingObjective.DELIVERABILITY: 0.15,
            RankingObjective.PERSONALIZATION: 0.10
        })
        
        # Trust thresholds
        self.min_rating_threshold = self.config.get('min_rating_threshold', 3.5)
        self.max_return_rate_threshold = self.config.get('max_return_rate_threshold', 0.15)
        self.min_review_count = self.config.get('min_review_count', 5)
        
        # Seller diversity constraints
        self.max_results_per_seller = self.config.get('max_results_per_seller', 3)
        self.min_seller_diversity_ratio = self.config.get('min_seller_diversity_ratio', 0.7)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default ranking configuration"""
        return {
            'boost_factors': {
                'quality_badge': 1.2,
                'four_star_plus': 1.15,
                'low_return_rate': 1.1,
                'fast_delivery': 1.1,
                'meesho_mall': 1.05
            },
            'penalty_factors': {
                'high_return_rate': 0.7,
                'high_seller_cancel': 0.8,
                'low_rating': 0.6,
                'high_dispute_ratio': 0.5
            },
            'deliverability_weights': {
                'eta_score': 0.4,
                'availability_score': 0.3,
                'delivery_success_rate': 0.3
            }
        }
    
    def rank_results(self, results: List[Dict[str, Any]], 
                    user_signals: Optional[PersonalizationSignals] = None,
                    user_pincode: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Rank search results using multi-objective scoring
        """
        if not results:
            return results
        
        # Score each result
        scored_results = []
        seller_counts = {}
        
        for result in results:
            scores = self._calculate_scores(result, user_signals, user_pincode)
            final_score = self._combine_scores(scores)
            
            result_copy = result.copy()
            result_copy['ranking_score'] = final_score
            result_copy['score_breakdown'] = scores
            
            scored_results.append(result_copy)
        
        # Sort by score
        scored_results.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Apply seller diversity constraints
        diverse_results = self._apply_seller_diversity(scored_results)
        
        # Apply trust guardrails
        filtered_results = self._apply_trust_guardrails(diverse_results)
        
        return filtered_results
    
    def _calculate_scores(self, result: Dict[str, Any], 
                         user_signals: Optional[PersonalizationSignals],
                         user_pincode: Optional[str]) -> Dict[str, float]:
        """Calculate individual objective scores"""
        scores = {}
        
        # 1. Relevance Score (from search engine)
        scores[RankingObjective.RELEVANCE] = result.get('relevance_score', 0.5)
        
        # 2. Trust Score
        scores[RankingObjective.TRUST] = self._calculate_trust_score(result)
        
        # 3. Affordability Score
        scores[RankingObjective.AFFORDABILITY] = self._calculate_affordability_score(
            result, user_signals)
        
        # 4. Deliverability Score
        scores[RankingObjective.DELIVERABILITY] = self._calculate_deliverability_score(
            result, user_pincode)
        
        # 5. Personalization Score
        scores[RankingObjective.PERSONALIZATION] = self._calculate_personalization_score(
            result, user_signals)
        
        return scores
    
    def _calculate_trust_score(self, result: Dict[str, Any]) -> float:
        """Calculate trust score based on product and seller signals"""
        trust_score = 0.5  # baseline
        
        # Rating component
        rating = result.get('rating', 0)
        if rating > 0:
            rating_score = min(rating / 5.0, 1.0)
            trust_score += rating_score * 0.3
        
        # Review count component (logarithmic scaling)
        review_count = result.get('review_count', 0)
        if review_count > 0:
            review_score = min(math.log(review_count + 1) / math.log(101), 1.0)  # log(101) ≈ 2
            trust_score += review_score * 0.2
        
        # Return rate component (inverted)
        return_rate = result.get('return_rate', 0.1)
        return_score = max(0, 1 - (return_rate / 0.3))  # penalty after 30% return rate
        trust_score += return_score * 0.2
        
        # Seller signals
        seller_cancel_rate = result.get('seller_cancel_rate', 0.05)
        seller_score = max(0, 1 - (seller_cancel_rate / 0.2))  # penalty after 20%
        trust_score += seller_score * 0.15
        
        # Quality badges
        if result.get('quality_badge', False):
            trust_score *= self.config['boost_factors']['quality_badge']
        
        # Dispute ratio
        dispute_ratio = result.get('dispute_ratio', 0.02)
        if dispute_ratio > 0.1:  # 10% threshold
            trust_score *= self.config['penalty_factors']['high_dispute_ratio']
        
        return min(trust_score, 1.0)
    
    def _calculate_affordability_score(self, result: Dict[str, Any], 
                                     user_signals: Optional[PersonalizationSignals]) -> float:
        """Calculate affordability score based on price and user preferences"""
        price = result.get('price', 0)
        if price <= 0:
            return 0.5  # neutral if no price
        
        affordability_score = 0.5
        
        # User price affinity matching
        if user_signals and user_signals.user_price_affinity:
            affinity = user_signals.user_price_affinity
            if affinity == "budget" and price <= 500:
                affordability_score += 0.3
            elif affinity == "mid" and 500 < price <= 2000:
                affordability_score += 0.3
            elif affinity == "premium" and price > 2000:
                affordability_score += 0.3
        
        # Historical spend matching
        if user_signals and user_signals.historical_spend_avg:
            avg_spend = user_signals.historical_spend_avg
            price_ratio = price / avg_spend
            
            if 0.5 <= price_ratio <= 1.5:  # within user's typical range
                affordability_score += 0.2
            elif price_ratio < 0.5:  # cheaper than usual
                affordability_score += 0.1
        
        return min(affordability_score, 1.0)
    
    def _calculate_deliverability_score(self, result: Dict[str, Any], 
                                      user_pincode: Optional[str]) -> float:
        """Calculate deliverability score based on logistics"""
        deliverability_score = 0.5
        
        # ETA component
        eta_days = result.get('pincode_eta', 7)  # default 7 days
        if eta_days <= 2:
            eta_score = 1.0
        elif eta_days <= 5:
            eta_score = 0.8
        elif eta_days <= 7:
            eta_score = 0.6
        else:
            eta_score = 0.3
        
        deliverability_score += eta_score * self.config['deliverability_weights']['eta_score']
        
        # Stock availability
        if result.get('stock_availability', True):
            availability_score = 1.0
        else:
            availability_score = 0.0
        
        deliverability_score += availability_score * self.config['deliverability_weights']['availability_score']
        
        # Delivery success rate
        delivery_success_rate = result.get('delivery_success_rate', 0.9)
        deliverability_score += delivery_success_rate * self.config['deliverability_weights']['delivery_success_rate']
        
        # COD availability bonus
        if result.get('cod_available', True):
            deliverability_score += 0.1
        
        return min(deliverability_score, 1.0)
    
    def _calculate_personalization_score(self, result: Dict[str, Any],
                                       user_signals: Optional[PersonalizationSignals]) -> float:
        """Calculate personalization score based on user preferences"""
        if not user_signals:
            return 0.5
        
        personalization_score = 0.5
        
        # Category preference matching
        if (user_signals.user_category_preference and 
            result.get('category') == user_signals.user_category_preference):
            personalization_score += 0.2
        
        # Brand affinity
        if (user_signals.brand_affinity and 
            result.get('brand') in user_signals.brand_affinity):
            personalization_score += 0.2
        
        # COD preference matching
        if (user_signals.cod_tendency and user_signals.cod_tendency > 0.7 and
            result.get('cod_available', False)):
            personalization_score += 0.1
        
        return min(personalization_score, 1.0)
    
    def _combine_scores(self, scores: Dict[RankingObjective, float]) -> float:
        """Combine individual objective scores using weights"""
        final_score = 0.0
        
        for objective, weight in self.objective_weights.items():
            final_score += scores.get(objective, 0.5) * weight
        
        return final_score
    
    def _apply_seller_diversity(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply seller diversity constraints"""
        diverse_results = []
        seller_counts = {}
        
        for result in results:
            seller_id = result.get('seller_id', 'unknown')
            current_count = seller_counts.get(seller_id, 0)
            
            if current_count < self.max_results_per_seller:
                diverse_results.append(result)
                seller_counts[seller_id] = current_count + 1
        
        return diverse_results
    
    def _apply_trust_guardrails(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply trust guardrails to filter out unreliable products"""
        filtered_results = []
        
        for result in results:
            # Rating guardrail
            rating = result.get('rating', 0)
            if rating > 0 and rating < self.min_rating_threshold:
                continue
            
            # Return rate guardrail
            return_rate = result.get('return_rate', 0)
            if return_rate > self.max_return_rate_threshold:
                continue
            
            # Review count guardrail (for rated products)
            if rating > 0:
                review_count = result.get('review_count', 0)
                if review_count < self.min_review_count:
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def generate_srp_strips(self, results: List[Dict[str, Any]], 
                           user_signals: Optional[PersonalizationSignals] = None) -> Dict[str, List[Dict[str, Any]]]:
        """Generate SRP strips for better user experience"""
        strips = {}
        
        # "Better under ₹X" strip
        if user_signals and user_signals.historical_spend_avg:
            avg_spend = user_signals.historical_spend_avg
            budget_results = [r for r in results if r.get('price', 0) < avg_spend and r.get('rating', 0) >= 4.0]
            if budget_results:
                strips[f"Better under ₹{int(avg_spend)}"] = budget_results[:5]
        
        # "4★+ in your pincode" strip
        high_rated_results = [r for r in results if r.get('rating', 0) >= 4.0 and r.get('pincode_eta', 10) <= 3]
        if high_rated_results:
            strips["4★+ with fast delivery"] = high_rated_results[:5]
        
        # "Similar but faster delivery" strip
        fast_delivery_results = [r for r in results if r.get('pincode_eta', 10) <= 2]
        if fast_delivery_results:
            strips["Similar but faster delivery"] = fast_delivery_results[:5]
        
        # "Meesho Mall" strip
        mall_results = [r for r in results if r.get('is_meesho_mall', False)]
        if mall_results:
            strips["Meesho Mall"] = mall_results[:5]
        
        return strips