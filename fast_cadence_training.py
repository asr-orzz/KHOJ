#!/usr/bin/env python3
"""
FAST CADENCE Training System - Optimized for Quick Setup
- Reduced dataset sizes for fast processing
- Simplified clustering for speed
- Memory optimized for RTX 3050 4GB
"""
import os
import gc
import json
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import structlog
from tqdm import tqdm
from real_cadence_training import RealDataProcessor, RealCADENCEModel

logger = structlog.get_logger()

class FastCADENCEProcessor(RealDataProcessor):
    """Fast processor with reduced dataset sizes"""
    
    def load_amazon_qac_dataset(self, max_samples: int = 20000) -> pd.DataFrame:
        """Load smaller Amazon QAC dataset for fast processing"""
        logger.info(f"📥 Loading Amazon QAC dataset ({max_samples:,} samples - FAST MODE)...")
        return super().load_amazon_qac_dataset(max_samples)
    
    def _create_products_from_queries(self, query_df: pd.DataFrame = None, max_samples: int = 10000) -> pd.DataFrame:
        """Create smaller synthetic product dataset"""
        logger.info(f"🔄 Creating {max_samples:,} synthetic products from queries (FAST MODE)...")
        return super()._create_products_from_queries(query_df, max_samples)
    
    def cluster_queries_fast(self, query_df: pd.DataFrame, n_clusters: int = 20) -> pd.DataFrame:
        """Ultra-fast clustering using simple K-means"""
        logger.info(f"⚡ Fast clustering {len(query_df):,} queries into {n_clusters} categories...")
        
        texts = query_df['processed_query'].tolist()
        
        # Simple TF-IDF features
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.query_vectorizer = TfidfVectorizer(
            max_features=1000,  # Reduced for speed
            stop_words='english',
            ngram_range=(1, 1),  # Only unigrams for speed
            max_df=0.8,
            min_df=2
        )
        
        tfidf_matrix = self.query_vectorizer.fit_transform(texts)
        logger.info(f"   TF-IDF matrix: {tfidf_matrix.shape}")
        
        # Fast K-means clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=5,  # Reduced for speed
            max_iter=50  # Reduced for speed
        )
        
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        
        # Create simple cluster descriptions
        feature_names = self.query_vectorizer.get_feature_names_out()
        cluster_descriptions = {}
        
        for label in range(n_clusters):
            cluster_center = kmeans.cluster_centers_[label]
            top_indices = np.argsort(cluster_center)[-3:][::-1]
            top_terms = [feature_names[i] for i in top_indices if cluster_center[i] > 0]
            
            if not top_terms:
                cluster_descriptions[label] = f"category_{label}"
            else:
                cluster_descriptions[label] = "_".join(top_terms[:2])
        
        # Add cluster information
        query_df = query_df.copy()
        query_df['cluster_id'] = cluster_labels
        query_df['cluster_description'] = query_df['cluster_id'].map(cluster_descriptions)
        
        self.query_clusters = cluster_descriptions
        
        logger.info(f"✅ Created {len(cluster_descriptions)} query clusters")
        for i, (cid, desc) in enumerate(cluster_descriptions.items()):
            count = np.sum(cluster_labels == cid)
            logger.info(f"   Cluster {cid}: {desc} ({count:,} queries)")
            if i >= 5:  # Show only first 5
                logger.info(f"   ... and {len(cluster_descriptions)-5} more clusters")
                break
        
        return query_df
    
    def cluster_products_fast(self, product_df: pd.DataFrame, n_clusters: int = 15) -> pd.DataFrame:
        """Ultra-fast product clustering"""
        logger.info(f"⚡ Fast clustering {len(product_df):,} products into {n_clusters} categories...")
        
        # Use existing categories when possible
        existing_categories = product_df['main_category'].value_counts().head(n_clusters).index.tolist()
        
        if len(existing_categories) >= n_clusters:
            # Use existing categories
            category_map = {cat: i for i, cat in enumerate(existing_categories)}
            cluster_labels = product_df['main_category'].map(category_map).fillna(0).astype(int)
            cluster_descriptions = {i: cat.lower().replace(' ', '_') for i, cat in enumerate(existing_categories)}
        else:
            # Fall back to clustering
            texts = product_df['processed_title'].tolist()
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.product_vectorizer = TfidfVectorizer(
                max_features=500,
                stop_words='english',
                ngram_range=(1, 1),
                max_df=0.9,
                min_df=2
            )
            
            tfidf_matrix = self.product_vectorizer.fit_transform(texts)
            
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=3, max_iter=30)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Create descriptions
            feature_names = self.product_vectorizer.get_feature_names_out()
            cluster_descriptions = {}
            for label in range(n_clusters):
                cluster_center = kmeans.cluster_centers_[label]
                top_indices = np.argsort(cluster_center)[-2:][::-1]
                top_terms = [feature_names[i] for i in top_indices if cluster_center[i] > 0]
                cluster_descriptions[label] = "_".join(top_terms) if top_terms else f"products_{label}"
        
        # Add cluster information
        product_df = product_df.copy()
        product_df['cluster_id'] = cluster_labels
        product_df['cluster_description'] = product_df['cluster_id'].map(cluster_descriptions)
        
        self.product_clusters = cluster_descriptions
        
        logger.info(f"✅ Created {len(cluster_descriptions)} product clusters")
        return product_df

def main():
    """Fast training function"""
    logger.info("=" * 80)
    logger.info("⚡ FAST CADENCE TRAINING SYSTEM")
    logger.info("=" * 80)
    logger.info("Features:")
    logger.info("• Real Amazon QAC dataset (20K queries for speed)")
    logger.info("• Synthetic products (10K products)")
    logger.info("• Fast clustering algorithms")
    logger.info("• Memory optimized for RTX 3050 4GB")
    logger.info("• Quick setup in <5 minutes")
    logger.info("")
    
    # Initialize processor
    processor = FastCADENCEProcessor()
    
    try:
        # Step 1: Load Amazon QAC dataset (reduced size)
        logger.info("STEP 1/6: Loading Amazon QAC Dataset (FAST MODE)")
        logger.info("-" * 50)
        query_df = processor.load_amazon_qac_dataset(max_samples=20000)
        
        # Step 2: Create synthetic products (reduced size)
        logger.info("\nSTEP 2/6: Creating Synthetic Products (FAST MODE)")
        logger.info("-" * 50)
        processor._query_df_for_products = query_df
        product_df = processor._create_products_from_queries(query_df, max_samples=10000)
        
        # Step 3: Fast query clustering
        logger.info("\nSTEP 3/6: Fast Query Clustering")
        logger.info("-" * 50)
        query_df = processor.cluster_queries_fast(query_df, n_clusters=20)
        
        # Step 4: Fast product clustering
        logger.info("\nSTEP 4/6: Fast Product Clustering")
        logger.info("-" * 50)
        product_df = processor.cluster_products_fast(product_df, n_clusters=15)
        
        # Step 5: Create vocabulary
        logger.info("\nSTEP 5/6: Creating Vocabulary")
        logger.info("-" * 50)
        vocab = processor.create_vocabulary(query_df, product_df, max_vocab=15000)  # Reduced vocab
        
        # Step 6: Save processed data
        logger.info("\nSTEP 6/6: Saving Processed Data")
        logger.info("-" * 50)
        processor.save_processed_data(query_df, product_df, vocab)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ FAST DATA PROCESSING COMPLETED!")
        logger.info("=" * 80)
        logger.info(f"📊 Results:")
        logger.info(f"   Queries: {len(query_df):,} with {len(processor.query_clusters)} categories")
        logger.info(f"   Products: {len(product_df):,} with {len(processor.product_clusters)} categories")
        logger.info(f"   Vocabulary: {len(vocab):,} words")
        logger.info(f"   Data saved to: processed_data/")
        logger.info("")
        logger.info("🚀 Ready for fast model training!")
        logger.info("   Run: python real_model_training.py")
        
    except Exception as e:
        logger.error(f"❌ Fast training failed: {e}")
        logger.exception("Full error traceback:")
        raise

if __name__ == "__main__":
    main()