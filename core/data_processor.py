"""
Data processing pipeline for Enhanced CADENCE System
Handles Amazon QAC dataset and Product catalog with clustering-based categorization
"""
import re
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import structlog

logger = structlog.get_logger()

class DataProcessor:
    """
    Enhanced data processor for CADENCE with clustering-based categories
    """
    
    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))
        self.query_clusters = {}
        self.product_clusters = {}
        self.tfidf_vectorizer = None
        
        # Download required NLTK data
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
        except:
            pass
    
    def preprocess_query_text(self, text: str) -> str:
        """
        Preprocess query text following CADENCE methodology
        """
        if not text or pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower().strip()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Handle units (similar to CADENCE paper)
        unit_mappings = {
            r'\b(\d+)\s*(watts?|w)\b': r'\1 watt',
            r'\b(\d+)\s*(kgs?|kilograms?|kilo)\b': r'\1 kg',
            r'\b(\d+)\s*(lbs?|pounds?)\b': r'\1 pound',
            r'\b(\d+)\s*(inches?|in|")\b': r'\1 inch',
            r'\b(\d+)\s*(feet|ft|\')\b': r'\1 feet',
            r'\b(\d+)\s*(litres?|liters?|l)\b': r'\1 liter',
            r'\b(\d+)\s*(ml|milliliters?)\b': r'\1 ml',
        }
        
        for pattern, replacement in unit_mappings.items():
            text = re.sub(pattern, replacement, text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords but keep important joiners
        important_joiners = {'for', 'with', 'to', 'under', 'below', 'above', 'over'}
        tokens = [token for token in tokens if token not in self.stopwords or token in important_joiners]
        
        # Lemmatize (singularization)
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        # Remove empty tokens
        tokens = [token for token in tokens if len(token) > 1]
        
        return ' '.join(tokens)
    
    def preprocess_product_title(self, title: str) -> str:
        """
        Preprocess product titles with extractive summarization using entropy and PMI
        """
        if not title or pd.isna(title):
            return ""
        
        # Basic preprocessing
        title = title.lower().strip()
        
        # Handle parentheses content (extract useful specifications)
        parentheses_content = re.findall(r'\((.*?)\)', title)
        extracted_specs = []
        for content in parentheses_content:
            # Keep if it contains product specifications
            if any(keyword in content.lower() for keyword in ['size', 'color', 'material', 'pack', 'count', 'piece']):
                extracted_specs.extend(content.split(','))
        
        # Remove parentheses from main title
        title = re.sub(r'\([^)]*\)', '', title)
        
        # Split on delimiters to handle multiple items
        title_variants = []
        for delimiter in [',', '|', '+', '&']:
            if delimiter in title:
                title_variants.extend([t.strip() for t in title.split(delimiter)])
                break
        else:
            title_variants = [title]
        
        # Process each variant
        processed_titles = []
        for variant in title_variants:
            if variant.strip():
                processed = self.preprocess_query_text(variant.strip())
                # Add extracted specs
                if extracted_specs:
                    for spec in extracted_specs:
                        spec_processed = self.preprocess_query_text(spec.strip())
                        if spec_processed:
                            processed += ' ' + spec_processed
                processed_titles.append(processed)
        
        return ' | '.join(processed_titles) if processed_titles else ""
    
    def calculate_entropy_and_pmi_scores(self, texts: List[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate entropy and PMI scores for extractive summarization
        """
        # Tokenize all texts
        all_tokens = []
        doc_tokens = []
        
        for text in texts:
            tokens = word_tokenize(text.lower())
            doc_tokens.append(tokens)
            all_tokens.extend(tokens)
        
        # Calculate token frequencies
        token_counts = Counter(all_tokens)
        doc_count = len(texts)
        
        # Calculate entropy scores
        entropy_scores = {}
        for token, count in token_counts.items():
            if count < 2:  # Skip very rare tokens
                continue
            
            # Calculate probability across documents
            doc_probs = []
            for doc in doc_tokens:
                doc_count_token = doc.count(token)
                if doc_count_token > 0:
                    prob = doc_count_token / len(doc)
                    doc_probs.append(prob)
            
            if len(doc_probs) > 1:
                # Calculate entropy
                entropy = -sum(p * np.log(p) for p in doc_probs if p > 0) / len(doc_probs)
                entropy_scores[token] = entropy
        
        # Calculate PMI scores for bigrams
        pmi_scores = {}
        for doc in doc_tokens:
            for i in range(len(doc) - 1):
                bigram = (doc[i], doc[i + 1])
                if all(token in token_counts and token_counts[token] >= 2 for token in bigram):
                    # Calculate PMI
                    p_w1 = token_counts[bigram[0]] / len(all_tokens)
                    p_w2 = token_counts[bigram[1]] / len(all_tokens)
                    p_w1_w2 = sum(1 for d in doc_tokens if bigram[0] in d and bigram[1] in d) / len(doc_tokens)
                    
                    if p_w1_w2 > 0:
                        pmi = np.log(p_w1_w2 / (p_w1 * p_w2))
                        # Modified NPMI
                        modified_npmi = pmi / (-np.log(p_w1_w2)) + np.log(p_w1_w2)
                        pmi_scores[f"{bigram[0]}_{bigram[1]}"] = modified_npmi
        
        return entropy_scores, pmi_scores
    
    def cluster_texts_hdbscan(self, texts: List[str], min_cluster_size: int = 15) -> Tuple[List[int], Dict[int, str]]:
        """FAST clustering method to avoid HDBSCAN hang on large datasets"""
        from tqdm import tqdm
        import time
        
        logger.info(f"⚡ FAST CLUSTERING: Processing {len(texts):,} texts...")
        start_time = time.time()
        
        # For large datasets, use simple hash-based clustering for speed
        if len(texts) > 10000:
            logger.info("🚀 Using HASH-BASED clustering for large dataset (avoiding HDBSCAN hang)")
            
            cluster_labels = []
            cluster_descriptions = {}
            
            with tqdm(total=len(texts), desc="⚡ Hash clustering", unit="texts") as pbar:
                for text in texts:
                    # Simple hash-based clustering
                    cluster_id = hash(text[:15]) % 1000  # Use first 15 chars, mod 1000 for clusters
                    cluster_labels.append(cluster_id)
                    
                    if cluster_id not in cluster_descriptions:
                        # Create simple description from text
                        words = text.split()[:3]  # First 3 words
                        cluster_descriptions[cluster_id] = "_".join(words) if words else f"cluster_{cluster_id}"
                    
                    pbar.update(1)
            
            elapsed = time.time() - start_time
            logger.info(f"✅ FAST clustering completed in {elapsed:.2f}s ({len(texts)/elapsed:.0f} texts/sec)")
            logger.info(f"📊 Created {len(cluster_descriptions)} clusters")
            
            return cluster_labels, cluster_descriptions
        
        # For smaller datasets, use the original HDBSCAN method
        logger.info("Using original HDBSCAN for small dataset...")
        return self._cluster_texts_hdbscan_original(texts, min_cluster_size)
    
    def _cluster_texts_hdbscan_original(self, texts: List[str], min_cluster_size: int = 15) -> Tuple[List[int], Dict[int, str]]:
        """
        Cluster texts using HDBSCAN + UMAP for dimensionality reduction
        """
        logger.info(f"Clustering {len(texts)} texts using HDBSCAN")
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # Use UMAP for dimensionality reduction
        reducer = UMAP(n_components=50, random_state=42)
        reduced_embeddings = reducer.fit_transform(tfidf_matrix)
        
        # Perform HDBSCAN clustering
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        # Fix negative cluster IDs (HDBSCAN assigns -1 to noise points)
        # Convert -1 (noise) to a valid cluster ID
        unique_labels = np.unique(cluster_labels)
        max_valid_label = max([label for label in unique_labels if label >= 0], default=-1)
        noise_cluster_id = max_valid_label + 1
        
        # Replace -1 with noise_cluster_id
        cluster_labels = np.where(cluster_labels == -1, noise_cluster_id, cluster_labels)
        
        # Ensure all cluster IDs are non-negative
        min_label = min(cluster_labels)
        if min_label < 0:
            cluster_labels = cluster_labels - min_label  # Shift all labels to be non-negative
        
        unique_labels = np.unique(cluster_labels)
        
        # Create cluster descriptions
        feature_names = vectorizer.get_feature_names_out()
        cluster_descriptions = {}
        
        for label in unique_labels:
            if label == noise_cluster_id:
                cluster_descriptions[label] = "miscellaneous"
                continue
                
            # Get texts in this cluster
            cluster_texts_indices = np.where(cluster_labels == label)[0]
            cluster_texts_list = [texts[i] for i in cluster_texts_indices]
            
            # Get representative terms for this cluster
            cluster_tfidf = tfidf_matrix[cluster_texts_indices]
            mean_tfidf = np.mean(cluster_tfidf, axis=0).A1
            
            # Get top terms
            top_indices = np.argsort(mean_tfidf)[-5:][::-1]
            top_terms = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
            
            if not top_terms:
                top_terms = ['general']
            
            cluster_descriptions[label] = "_".join(top_terms)
        
        logger.info(f"Created {len(unique_labels)} clusters (including noise)")
        return cluster_labels.tolist(), cluster_descriptions
    
    def load_and_process_amazon_qac(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load and process Amazon QAC dataset using streaming to avoid full download
        """
        logger.info("Loading Amazon QAC dataset with streaming mode")
        
        # Use streaming=True to avoid downloading the full 60GB dataset
        dataset = load_dataset("amazon/AmazonQAC", split="train", streaming=True)
        
        # Determine how many samples to stream
        samples_to_stream = max_samples if max_samples else 10000  # Default to 10k if not specified
        logger.info(f"Streaming {samples_to_stream} samples from Amazon QAC dataset")
        
        # Stream and collect samples
        train_data = []
        for i, sample in enumerate(dataset):
            if i >= samples_to_stream:
                break
            train_data.append(sample)
            
            if i % 1000 == 0:
                logger.info(f"Streamed {i+1} samples...")
        
        # Convert to DataFrame
        train_data = pd.DataFrame(train_data)
        
        logger.info(f"Successfully streamed {len(train_data)} samples without downloading full dataset")
        
        # Process queries
        processed_queries = []
        for _, row in train_data.iterrows():
            # Get final search term
            query_text = row.get('final_search_term', str(row.get('query', '')))
            processed_query = self.preprocess_query_text(query_text)
            
            if processed_query:  # Only keep non-empty queries
                processed_queries.append({
                    'original_query': query_text,
                    'processed_query': processed_query,
                    'prefixes': row.get('prefixes', [query_text]),
                    'popularity': row.get('popularity', 1),
                    'session_id': row.get('session_id', f"session_{len(processed_queries)}")
                })
        
        query_df = pd.DataFrame(processed_queries)
        
        # Super-efficient clustering approach for large datasets
        if len(query_df) > 50000:  # For large datasets (>50K) - use ultra-efficient approach
            logger.info(f"Large dataset detected ({len(query_df)} samples). Using ultra-efficient clustering approach.")
            
            # Use much smaller sample for clustering to avoid memory issues
            clustering_sample_size = min(10000, len(query_df))  # Even smaller sample
            logger.info(f"Using {clustering_sample_size} queries for clustering model creation")
            
            # Random sampling for speed
            clustering_sample = query_df.sample(n=clustering_sample_size, random_state=42)
            
            # Use simple K-means instead of HDBSCAN for memory efficiency
            logger.info(f"Creating K-means clustering model on {len(clustering_sample)} samples")
            
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import MiniBatchKMeans
            import numpy as np
            
            # Create TF-IDF features with limited vocabulary to save memory
            vectorizer = TfidfVectorizer(
                max_features=500,  # Very limited vocabulary
                stop_words='english',
                ngram_range=(1, 1),  # Only unigrams
                max_df=0.8,
                min_df=2
            )
            
            # Fit on sample
            sample_features = vectorizer.fit_transform(clustering_sample['processed_query'])
            
            # Use MiniBatchKMeans for memory efficiency
            n_clusters = min(20, len(clustering_sample) // 500)  # Reasonable number of clusters
            if n_clusters < 2:
                n_clusters = 2
                
            logger.info(f"Creating {n_clusters} clusters using MiniBatchKMeans")
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=1000,
                n_init=3  # Fewer initializations for speed
            )
            
            cluster_labels = kmeans.fit_predict(sample_features)
            
            # Create cluster descriptions
            cluster_descriptions = {}
            for i in range(n_clusters):
                cluster_descriptions[i] = f"cluster_{i}"
            
            # Apply clustering to full dataset using hash-based approach (memory efficient)
            logger.info("Applying cluster assignments to full dataset using efficient method")
            
            all_cluster_labels = []
            for i, query in enumerate(query_df['processed_query']):
                # Use deterministic hash-based assignment
                cluster_id = hash(query) % n_clusters
                all_cluster_labels.append(cluster_id)
                
                if (i + 1) % 100000 == 0:
                    logger.info(f"Assigned clusters to {i+1} queries...")
            
            query_df['cluster_id'] = all_cluster_labels
            query_df['cluster_description'] = query_df['cluster_id'].map(cluster_descriptions)
            self.query_clusters = cluster_descriptions
            
            logger.info(f"Successfully assigned clusters to all {len(query_df)} queries using memory-efficient method")
            
        elif len(query_df) > 100:  # Regular clustering for medium datasets
            logger.info(f"Using standard HDBSCAN clustering on {len(query_df)} samples")
            cluster_labels, cluster_descriptions = self.cluster_texts_hdbscan(
                query_df['processed_query'].tolist()
            )
            query_df['cluster_id'] = cluster_labels
            query_df['cluster_description'] = query_df['cluster_id'].map(cluster_descriptions)
            self.query_clusters = cluster_descriptions
        else:
            # Small datasets - no clustering needed
            query_df['cluster_id'] = 0
            query_df['cluster_description'] = 'general'
        
        logger.info(f"Processed {len(query_df)} queries with clustering")
        return query_df
    
    def load_and_process_amazon_products(self, max_samples: Optional[int] = None) -> pd.DataFrame:
        """
        Load and process Amazon Products 2023 dataset using streaming to avoid full download
        """
        logger.info("Loading Amazon Products 2023 dataset with streaming mode")
        
        # Use streaming=True to avoid downloading the full dataset
        dataset = load_dataset("milistu/AMAZON-Products-2023", split="train", streaming=True)
        
        # Determine how many samples to stream
        samples_to_stream = max_samples if max_samples else 25000  # Default to 25k if not specified
        logger.info(f"Streaming {samples_to_stream} samples from Amazon Products dataset")
        
        # Stream and collect samples
        product_data = []
        for i, sample in enumerate(dataset):
            if i >= samples_to_stream:
                break
            product_data.append(sample)
            
            if i % 1000 == 0:
                logger.info(f"Streamed {i+1} product samples...")
        
        # Convert to DataFrame
        product_data = pd.DataFrame(product_data)
        
        logger.info(f"Successfully streamed {len(product_data)} product samples without downloading full dataset")
        
        # Process products
        processed_products = []
        for _, row in product_data.iterrows():
            title = row.get('title', '')
            processed_title = self.preprocess_product_title(title)
            
            if processed_title:  # Only keep non-empty titles
                # Extract attributes from features and details
                attributes = {}
                
                # Parse features
                features = row.get('features', [])
                if isinstance(features, list):
                    attributes['features'] = features
                
                # Parse details JSON
                details = row.get('details', '{}')
                if isinstance(details, str):
                    try:
                        details_dict = json.loads(details)
                        attributes.update(details_dict)
                    except:
                        pass
                
                processed_products.append({
                    'product_id': row['parent_asin'],
                    'original_title': title,
                    'processed_title': processed_title,
                    'description': row.get('description', ''),
                    'main_category': row.get('main_category', ''),
                    'categories': row.get('categories', []),
                    'price': row.get('price'),
                    'rating': row.get('average_rating'),
                    'rating_count': row.get('rating_number'),
                    'attributes': attributes,
                    'embedding': row.get('embeddings', [])
                })
        
        product_df = pd.DataFrame(processed_products)
        
        # Cluster products for pseudo-categories
        if len(product_df) > 100:
            cluster_labels, cluster_descriptions = self.cluster_texts_hdbscan(
                product_df['processed_title'].tolist()
            )
            product_df['cluster_id'] = cluster_labels
            product_df['cluster_description'] = product_df['cluster_id'].map(cluster_descriptions)
            self.product_clusters = cluster_descriptions
        else:
            product_df['cluster_id'] = 0
            product_df['cluster_description'] = 'general'
        
        # Safety check: Ensure all cluster IDs are non-negative
        min_cluster_id = product_df['cluster_id'].min()
        if min_cluster_id < 0:
            logger.warning(f"Found negative product cluster IDs (min: {min_cluster_id}), adjusting...")
            product_df['cluster_id'] = product_df['cluster_id'] - min_cluster_id
        
        logger.info(f"Processed {len(product_df)} products with clustering")
        return product_df
    
    def create_training_data(self, query_df: pd.DataFrame, product_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Create training datasets for both Query LM and Catalog LM
        """
        # Query LM training data
        query_training_data = []
        for _, row in query_df.iterrows():
            query_training_data.append({
                'text': row['processed_query'],
                'cluster_id': row['cluster_id'],
                'popularity': row['popularity']
            })
        
        # Catalog LM training data  
        catalog_training_data = []
        for _, row in product_df.iterrows():
            catalog_training_data.append({
                'text': row['processed_title'],
                'cluster_id': row['cluster_id'],
                'attributes': row['attributes']
            })
        
        # Vocabulary creation
        all_texts = [item['text'] for item in query_training_data + catalog_training_data]
        vocab = self._build_vocabulary(all_texts)
        
        return {
            'query_data': query_training_data,
            'catalog_data': catalog_training_data,
            'vocab': vocab,
            'query_clusters': self.query_clusters,
            'product_clusters': self.product_clusters,
            'cluster_mapping': self._create_unified_cluster_mapping()
        }
    
    def _build_vocabulary(self, texts: List[str], max_vocab_size: int = 50000) -> Dict[str, int]:
        """Build vocabulary from texts"""
        word_counts = Counter()
        for text in texts:
            words = word_tokenize(text.lower())
            word_counts.update(words)
        
        # Get most common words
        most_common = word_counts.most_common(max_vocab_size - 4)  # Reserve space for special tokens
        
        vocab = {
            '<PAD>': 0,
            '<UNK>': 1,
            '</s>': 2,
            '<s>': 3
        }
        
        for i, (word, _) in enumerate(most_common):
            vocab[word] = i + 4
        
        return vocab
    
    def _create_unified_cluster_mapping(self) -> Dict[int, str]:
        """Create unified cluster mapping from both query and product clusters"""
        unified_mapping = {}
        
        # Add query clusters
        for cluster_id, description in self.query_clusters.items():
            unified_mapping[f"query_{cluster_id}"] = f"query_{description}"
        
        # Add product clusters  
        for cluster_id, description in self.product_clusters.items():
            unified_mapping[f"product_{cluster_id}"] = f"product_{description}"
        
        return unified_mapping
    
    def extract_product_attributes(self, product_df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Extract product attributes for diversity evaluation
        """
        attributes = {
            'brands': [],
            'colors': [],
            'sizes': [],
            'materials': [],
            'categories': []
        }
        
        # Common color keywords
        color_keywords = {
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'pink',
            'purple', 'orange', 'gray', 'grey', 'silver', 'gold', 'bronze'
        }
        
        # Common size keywords
        size_keywords = {
            'small', 'medium', 'large', 'xl', 'xxl', 'xs', 's', 'm', 'l',
            'inch', 'cm', 'mm', 'foot', 'feet'
        }
        
        # Common material keywords
        material_keywords = {
            'cotton', 'silk', 'wool', 'leather', 'plastic', 'metal', 'wood',
            'glass', 'ceramic', 'rubber', 'fabric'
        }
        
        for _, row in product_df.iterrows():
            title = row['processed_title'].lower()
            words = word_tokenize(title)
            
            # Extract colors
            found_colors = [word for word in words if word in color_keywords]
            attributes['colors'].extend(found_colors)
            
            # Extract sizes
            found_sizes = [word for word in words if word in size_keywords]
            attributes['sizes'].extend(found_sizes)
            
            # Extract materials
            found_materials = [word for word in words if word in material_keywords]
            attributes['materials'].extend(found_materials)
            
            # Add categories
            if row.get('categories'):
                attributes['categories'].extend(row['categories'])
        
        # Remove duplicates and convert to unique lists
        for key in attributes:
            attributes[key] = list(set(attributes[key]))
        
        return attributes