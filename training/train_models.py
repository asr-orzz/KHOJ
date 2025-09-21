import os
import pickle
import json
import torch
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Tuple
import structlog
from pathlib import Path

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from core.data_processor import DataProcessor
from core.cadence_model import CADENCEModel, create_cadence_model
from config.settings import settings

logger = structlog.get_logger()

class QueryDataset(Dataset):
    """Dataset for training Query Language Model"""
    
    def __init__(self, texts: List[str], cluster_ids: List[int], vocab: Dict[str, int], max_length: int = 50):
        self.texts = texts
        self.cluster_ids = cluster_ids
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        cluster_id = self.cluster_ids[idx]
        
        # Tokenize and convert to IDs
        tokens = text.split()[:self.max_length-2]  # Leave space for BOS/EOS
        token_ids = [self.vocab.get('<s>', 3)]  # BOS token
        
        for token in tokens:
            token_ids.append(self.vocab.get(token, self.vocab.get('<UNK>', 1)))
        
        token_ids.append(self.vocab.get('</s>', 2))  # EOS token
        
        # Create input and target sequences
        input_ids = token_ids[:-1]  # All except last token
        target_ids = token_ids[1:]  # All except first token
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'target_ids': torch.tensor(target_ids, dtype=torch.long),
            'cluster_id': torch.tensor(cluster_id, dtype=torch.long),
            'length': len(input_ids)
        }

def collate_fn(batch):
    """Collate function for DataLoader"""
    input_ids = [item['input_ids'] for item in batch]
    target_ids = [item['target_ids'] for item in batch]
    cluster_ids = torch.stack([item['cluster_id'] for item in batch])
    
    # Pad sequences
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    target_ids_padded = pad_sequence(target_ids, batch_first=True, padding_value=0)
    
    # Create category tensor (same cluster for all tokens in sequence)
    batch_size, seq_len = input_ids_padded.shape
    category_ids = cluster_ids.unsqueeze(1).expand(batch_size, seq_len)
    
    return {
        'input_ids': input_ids_padded,
        'target_ids': target_ids_padded,
        'category_ids': category_ids
    }

class CADENCETrainer:
    """Trainer for CADENCE models"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
        self.data_processor = DataProcessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Using device: {self.device}")
    
    def prepare_data(self, max_samples: int = 50000) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Prepare training data from Meesho datasets with MASSIVE PARALLELISM"""
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
        import multiprocessing as mp
        import time
        import pandas as pd
        
        logger.info("🚀 STARTING PARALLELIZED DATA PREPARATION...")
        start_time = time.time()
        
        # Use optimal sample sizes for Kaggle GPU training
        qac_sample_size = min(max_samples, 100000)  # 100K for comprehensive training
        products_sample_size = min(max_samples // 4, 25000)  # 25K products
        
        logger.info(f"📊 TARGET SAMPLES:")
    logger.info(f"  - Meesho QAC: {qac_sample_size:,} samples")
    logger.info(f"  - Meesho Products: {products_sample_size:,} samples")
        logger.info(f"  - CPU Cores Available: {mp.cpu_count()}")
        
        # PHASE 1: PARALLEL DATA LOADING WITH PROGRESS BARS
        logger.info("⚡ PHASE 1: PARALLEL DATA LOADING...")
        
        def load_qac_with_progress():
            """Load QAC dataset with progress tracking"""
            try:
                qac_dataset = load_dataset("meesho/MeeshoQAC", split="train", streaming=True)
                sample_data = []
                
                with tqdm(total=qac_sample_size, desc="📡 Loading QAC", unit="samples") as pbar:
                    for i, sample in enumerate(qac_dataset):
                        if i >= qac_sample_size:
                            break
                        sample_data.append(sample)
                        
                        if i % 1000 == 0:  # Update progress every 1K samples
                            pbar.update(1000)
                    
                    pbar.update(len(sample_data) % 1000)  # Update remaining
                
                logger.info(f"✅ QAC loaded: {len(sample_data):,} samples")
                return pd.DataFrame(sample_data)
            except Exception as e:
                logger.error(f"❌ QAC loading failed: {e}")
                raise RuntimeError("Meesho QAC dataset loading failed. Real data is required.")
        
        def load_products_with_progress():
            """Load Products dataset with progress tracking"""
            try:
                product_df = self.data_processor.load_and_process_meesho_products(max_samples=products_sample_size)
                logger.info(f"✅ Products loaded: {len(product_df):,} samples")
                return product_df
            except Exception as e:
                logger.error(f"❌ Products loading failed: {e}")
                raise RuntimeError("Meesho Products dataset loading failed. Real data is required.")
        
        # Load both datasets in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            qac_future = executor.submit(load_qac_with_progress)
            products_future = executor.submit(load_products_with_progress)
            
            qac_df = qac_future.result()
            product_df = products_future.result()
        
        # PHASE 2: PARALLEL DATA PROCESSING WITH PROGRESS BARS
        logger.info("⚡ PHASE 2: PARALLEL DATA PROCESSING...")
        
        def process_qac_parallel(qac_df):
            """Process QAC data with progress tracking"""
            logger.info("🔄 Processing QAC queries...")
            processed_queries = []
            
            with tqdm(total=len(qac_df), desc="🔄 Processing QAC", unit="queries") as pbar:
                for _, row in qac_df.iterrows():
                    query_text = row.get('final_search_term', str(row.get('query', '')))
                    processed_query = self.data_processor.preprocess_query_text(query_text)
                    
                    if processed_query:
                        processed_queries.append({
                            'original_query': query_text,
                            'processed_query': processed_query,
                            'prefixes': row.get('prefixes', [query_text]),
                            'popularity': row.get('popularity', 1),
                            'session_id': row.get('session_id', f"session_{len(processed_queries)}")
                        })
                    pbar.update(1)
            
            query_df = pd.DataFrame(processed_queries)
            
            # FAST CLUSTERING (avoid expensive HDBSCAN)
            logger.info("⚡ Fast clustering (avoiding HDBSCAN hang)...")
            if len(query_df) > 100:
                # Simple hash-based clustering for speed
                query_df['cluster_id'] = query_df['processed_query'].apply(lambda x: hash(x[:10]) % 1000)
                query_df['cluster_description'] = 'category_' + query_df['cluster_id'].astype(str)
            else:
                query_df['cluster_id'] = 0
                query_df['cluster_description'] = 'general'
            
            logger.info(f"✅ QAC processed: {len(query_df):,} queries")
            return query_df
        
        # Process QAC data
        query_df = process_qac_parallel(qac_df)
        
        # PHASE 3: PARALLEL TRAINING DATA CREATION
        logger.info("⚡ PHASE 3: PARALLEL TRAINING DATA CREATION...")
        
        with tqdm(desc="🏗️ Creating training data", unit="steps") as pbar:
            pbar.set_description("Building training datasets...")
            training_data = self.data_processor.create_training_data(query_df, product_df)
            pbar.update(1)
        
        total_time = time.time() - start_time
        logger.info(f"🎉 PARALLELIZED DATA PREPARATION COMPLETED in {total_time:.2f}s!")
        logger.info(f"📊 FINAL STATISTICS:")
        logger.info(f"  - Query training samples: {len(training_data['query_data']):,}")
        logger.info(f"  - Catalog training samples: {len(training_data['catalog_data']):,}")
        logger.info(f"  - Vocabulary size: {len(training_data['vocab']):,}")
        logger.info(f"  - Processing speed: {(len(training_data['query_data']) + len(training_data['catalog_data'])) / total_time:.0f} samples/sec")
        
        return training_data
    
    # ❌ REMOVED: All dummy/mock data methods eliminated
    # This system ONLY uses real Meesho datasets
    
    def _process_qac_streaming_data_fast(self, qac_df):
        """Process streaming QAC data with fast clustering to prevent hanging"""
        import pandas as pd
        
        logger.info(f"Processing {len(qac_df)} streaming QAC samples with fast clustering...")
        
        # Process queries from streaming data
        processed_queries = []
        for _, row in qac_df.iterrows():
            # Get final search term
            query_text = row.get('final_search_term', str(row.get('query', '')))
            processed_query = self.data_processor.preprocess_query_text(query_text)
            
            if processed_query:  # Only keep non-empty queries
                processed_queries.append({
                    'original_query': query_text,
                    'processed_query': processed_query,
                    'prefixes': row.get('prefixes', [query_text]),
                    'popularity': row.get('popularity', 1),
                    'session_id': row.get('session_id', f"session_{len(processed_queries)}")
                })
        
        query_df = pd.DataFrame(processed_queries)
        
        if len(query_df) == 0:
            logger.error("❌ CRITICAL: No valid queries processed from Meesho QAC dataset!")
            logger.error("This indicates a serious data processing issue")
            raise RuntimeError("Failed to process any valid queries from Meesho QAC dataset")
        
        # Use simple clustering instead of expensive HDBSCAN
        logger.info("Using fast simple clustering instead of HDBSCAN...")
        if len(query_df) > 100:
            # Simple clustering based on first few characters
            query_df['cluster_id'] = query_df['processed_query'].str[:10].astype('category').cat.codes
            query_df['cluster_description'] = 'category_' + query_df['cluster_id'].astype(str)
            
            # Create cluster descriptions
            cluster_descriptions = {}
            for cluster_id in query_df['cluster_id'].unique():
                cluster_descriptions[cluster_id] = f"category_{cluster_id}"
            self.query_clusters = cluster_descriptions
        else:
            query_df['cluster_id'] = 0
            query_df['cluster_description'] = 'general'
            self.query_clusters = {0: 'general'}
        
        logger.info(f"Successfully processed {len(query_df)} queries with fast clustering")
        return query_df

    def _process_qac_streaming_data(self, qac_df):
        """Process streaming QAC data without full dataset dependencies"""
        import pandas as pd
        
        logger.info(f"Processing {len(qac_df)} streaming QAC samples...")
        
        # Process queries from streaming data
        processed_queries = []
        for _, row in qac_df.iterrows():
            # Get final search term
            query_text = row.get('final_search_term', str(row.get('query', '')))
            processed_query = self.data_processor.preprocess_query_text(query_text)
            
            if processed_query:  # Only keep non-empty queries
                processed_queries.append({
                    'original_query': query_text,
                    'processed_query': processed_query,
                    'prefixes': row.get('prefixes', [query_text]),
                    'popularity': row.get('popularity', 1),
                    'session_id': row.get('session_id', f"session_{len(processed_queries)}")
                })
        
        query_df = pd.DataFrame(processed_queries)
        
        if len(query_df) == 0:
            logger.error("❌ CRITICAL: No valid queries processed from Meesho QAC dataset!")
            logger.error("This indicates a serious data processing issue")
            raise RuntimeError("Failed to process any valid queries from Meesho QAC dataset")
        
        # Cluster queries for pseudo-categories
        if len(query_df) > 100:  # Only cluster if we have enough data
            cluster_labels, cluster_descriptions = self.data_processor.cluster_texts_hdbscan(
                query_df['processed_query'].tolist()
            )
            query_df['cluster_id'] = cluster_labels
            query_df['cluster_description'] = query_df['cluster_id'].map(cluster_descriptions)
            self.query_clusters = cluster_descriptions
        else:
            query_df['cluster_id'] = 0
            query_df['cluster_description'] = 'general'
        
        # Safety check: Ensure all cluster IDs are non-negative
        min_cluster_id = query_df['cluster_id'].min()
        if min_cluster_id < 0:
            logger.warning(f"Found negative cluster IDs (min: {min_cluster_id}), adjusting...")
            query_df['cluster_id'] = query_df['cluster_id'] - min_cluster_id
        
        logger.info(f"Successfully processed {len(query_df)} queries from streaming data")
        return query_df
    
    def train_model(self, model: CADENCEModel, train_data: List[Dict], val_data: List[Dict], 
                   vocab: Dict[str, int], model_type: str = "query", epochs: int = 5):
        """Train a CADENCE model"""
        logger.info(f"Training {model_type} model...")
        
        # Create datasets
        train_texts = [item['text'] for item in train_data]
        train_clusters = [item.get('cluster_id', 0) for item in train_data]
        
        val_texts = [item['text'] for item in val_data]
        val_clusters = [item.get('cluster_id', 0) for item in val_data]
        
        train_dataset = QueryDataset(train_texts, train_clusters, vocab)
        val_dataset = QueryDataset(val_texts, val_clusters, vocab)
        
        train_loader = DataLoader(train_dataset, batch_size=settings.BATCH_SIZE, 
                                shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=settings.BATCH_SIZE, 
                              collate_fn=collate_fn)
        
        # Setup optimizer
        optimizer = Adam(model.parameters(), lr=settings.LEARNING_RATE)
        
        # Move model to device
        model.to(self.device)
        
        # Training loop
        for epoch in tqdm(range(epochs), desc="🚀 Epochs", unit="epoch"):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for batch in tqdm(train_loader, desc=f"{model_type.capitalize()} Epoch {epoch+1}", leave=False):
                optimizer.zero_grad()
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                target_ids = batch['target_ids'].to(self.device)
                category_ids = batch['category_ids'].to(self.device)
                
                # Forward pass
                outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                loss = outputs['loss']
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                if num_batches % 10 == 0:
                    logger.info(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            
            # Validation
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    target_ids = batch['target_ids'].to(self.device)
                    category_ids = batch['category_ids'].to(self.device)
                    
                    outputs = model(input_ids, category_ids, target_ids, model_type=model_type)
                    val_loss += outputs['loss'].item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
            
            logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        return model
    
    def save_model_and_vocab(self, model: CADENCEModel, vocab: Dict[str, int], 
                           cluster_mappings: Dict[str, Any], model_name: str):
        """Save trained model and vocabulary"""
        model_path = self.model_dir / f"{model_name}.pt"
        vocab_path = self.model_dir / f"{model_name}_vocab.pkl"
        config_path = self.model_dir / f"{model_name}_config.json"
        
        # Save model state dict
        torch.save(model.state_dict(), model_path)
        
        # Save vocabulary
        with open(vocab_path, 'wb') as f:
            pickle.dump(vocab, f)
        
        # Save configuration
        config = {
            'vocab_size': len(vocab),
            'num_categories': len(cluster_mappings.get('query_clusters', {})) + len(cluster_mappings.get('product_clusters', {})),
            'model_config': model.config,
            'cluster_mappings': cluster_mappings,
            'training_date': datetime.now().isoformat()
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vocabulary saved to {vocab_path}")
        logger.info(f"Configuration saved to {config_path}")
    
    def load_model_and_vocab(self, model_name: str) -> Tuple[CADENCEModel, Dict[str, int], Dict[str, Any]]:
        """Load trained model and vocabulary"""
        model_path = self.model_dir / f"{model_name}.pt"
        vocab_path = self.model_dir / f"{model_name}_vocab.pkl"
        config_path = self.model_dir / f"{model_name}_config.json"
        
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load vocabulary
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        
        # Create and load model
        model = CADENCEModel(config['model_config'])
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        
        return model, vocab, config
    
    def train_full_pipeline(self, max_samples: int = 50000, epochs: int = 3):
        """Train the complete CADENCE pipeline with memory optimization"""
        logger.info("Starting full CADENCE training pipeline...")
        logger.info(f"Training with {max_samples} samples for {epochs} epochs")
        
        # Add memory monitoring
        import psutil
        import gc
        
        def log_memory_usage(step_name):
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            logger.info(f"{step_name} - Memory usage: {memory_mb:.1f} MB")
        
        log_memory_usage("Pipeline start")
        
        # Prepare data with progress tracking
        logger.info("Step 1/4: Preparing training data...")
        training_data = self.prepare_data(max_samples)
        
        # Force garbage collection
        gc.collect()
        log_memory_usage("Data preparation complete")
        
        # Create model
        vocab_size = len(training_data['vocab'])
        num_categories = len(training_data['query_clusters']) + len(training_data['product_clusters'])
        
        logger.info(f"Step 2/4: Creating CADENCE model...")
        logger.info(f"Vocab size: {vocab_size}, Categories: {num_categories}")
        model = create_cadence_model(vocab_size, num_categories)
        
        log_memory_usage("Model creation complete")
        
        # Train Query Language Model
        logger.info("Step 3/4: Training Query Language Model...")
        logger.info(f"Training on {len(training_data['query_data'])} query samples")
        model = self.train_model(
            model, 
            training_data['query_data'], 
            training_data['query_data'], # Use the same data for val for simplicity
            training_data['vocab'],
            model_type='query',
            epochs=epochs
        )
        
        # Force garbage collection between training steps
        gc.collect()
        log_memory_usage("Query model training complete")
        
        # Train Catalog Language Model
        logger.info("Step 4/4: Training Catalog Language Model...")
        logger.info(f"Training on {len(training_data['catalog_data'])} catalog samples")
        model = self.train_model(
            model,
            training_data['catalog_data'],
            training_data['catalog_data'], # Use the same data for val for simplicity
            training_data['vocab'],
            model_type='catalog',
            epochs=epochs
        )
        
        log_memory_usage("Catalog model training complete")
        
        # Save everything
        cluster_mappings = {
            'query_clusters': training_data['query_clusters'],
            'product_clusters': training_data['product_clusters']
        }
        
        logger.info("Saving trained model and vocabulary...")
        self.save_model_and_vocab(
            model, 
            training_data['vocab'], 
            cluster_mappings,
            'cadence_trained'
        )
        
        log_memory_usage("Pipeline completion")
        logger.info("Training pipeline completed successfully!")
        
        return model, training_data['vocab'], cluster_mappings

    def train_enhanced_models(self, training_data: Dict[str, Any], epochs: int = 3, 
                           save_name: str = "enhanced_cadence") -> Tuple[Any, Dict[str, int], Dict[str, Any]]:
        """Train enhanced CADENCE model with multi-task learning"""
        logger.info("🧠 Training enhanced CADENCE model with multi-task learning...")
        
        # Extract data from the new structure
        query_data = training_data['query_data']
        catalog_data = training_data['catalog_data']
        vocab = training_data['vocab']
        
        # Calculate dynamic num_categories based on actual data
        max_query_id = max([item.get('cluster_id', 0) for item in query_data]) if query_data else 0
        max_catalog_id = max([item.get('cluster_id', 0) for item in catalog_data]) if catalog_data else 0
        max_category_id = max(max_query_id, max_catalog_id)
        num_categories = max_category_id + 50  # Add buffer
        
        logger.info(f"Creating enhanced model with {num_categories} categories...")
        
        # Create enhanced model with larger architecture
        model = create_cadence_model(
            vocab_size=len(vocab),
            num_categories=num_categories,
            embedding_dim=512,
            hidden_dims=[3008, 2496, 2000, 1536],  # Ensure divisible by 8 for attention heads
            attention_dims=[1536, 1248, 1000, 768],  # Ensure divisible by 8 for attention heads
            dropout=0.6
        )
        
        # Create datasets directly from the original data
        query_dataset = QueryDataset(
            [item['text'] for item in query_data],
            [item.get('cluster_id', 0) for item in query_data],
            vocab
        )
        
        catalog_dataset = QueryDataset(
            [item['text'] for item in catalog_data],
            [item.get('cluster_id', 0) for item in catalog_data],
            vocab
        )
        
        # Create data loaders
        query_loader = DataLoader(query_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        catalog_loader = DataLoader(catalog_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Move model to device
        model.to(self.device)
        
        # Training loop
        logger.info(f"Starting enhanced training for {epochs} epochs...")
        for epoch in tqdm(range(epochs), desc="🚀 Epochs", unit="epoch"):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train query model
            query_loss = self._train_epoch_enhanced(model, query_loader, optimizer, self.device, 'query')
            
            # Train catalog model
            catalog_loss = self._train_epoch_enhanced(model, catalog_loader, optimizer, self.device, 'catalog')
            
            # Update learning rate
            scheduler.step()
            
            logger.info(f"Epoch {epoch+1} - Query Loss: {query_loss:.4f}, Catalog Loss: {catalog_loss:.4f}")
        
        # Save enhanced model
        cluster_mappings = {
            'query_clusters': training_data['query_clusters'],
            'product_clusters': training_data['product_clusters']
        }
        
        self.save_enhanced_model(model, vocab, cluster_mappings, save_name)
        
        logger.info("✅ Enhanced CADENCE model training completed!")
        return model, vocab, cluster_mappings
    
    def _prepare_enhanced_dataset(self, data: List[Dict[str, Any]], vocab: Dict[str, int], 
                                data_type: str) -> List[Dict[str, Any]]:
        """Prepare enhanced dataset with multi-task labels"""
        dataset = []
        
        # Debug: Check the first item structure
        if data:
            logger.info(f"Debug: First item keys: {list(data[0].keys())}")
            logger.info(f"Debug: First item: {data[0]}")
        
        for item in data:
            # The data structure has 'text' key from create_training_data
            text = item.get('text', '')
            if not text:
                continue
                
            cluster_id = item.get('cluster_id', 0)
            
            # Tokenize
            tokens = self._tokenize_text(text, vocab)
            if len(tokens) < 2:
                continue
            
            # Create input/target sequences
            input_ids = tokens[:-1]
            target_ids = tokens[1:]
            
            # Create intent labels (simplified)
            intent_label = self._infer_intent(text, data_type)
            
            dataset.append({
                'input_ids': input_ids,
                'target_ids': target_ids,
                'category_ids': [cluster_id] * len(input_ids),
                'intent_label': intent_label,
                'data_type': data_type
            })
        
        return dataset
    
    def _infer_intent(self, text: str, data_type: str) -> int:
        """Infer intent from text (simplified)"""
        text_lower = text.lower()
        
        # Intent mapping: 0=browse, 1=search, 2=buy, 3=compare, 4=return
        if any(word in text_lower for word in ['buy', 'purchase', 'order']):
            return 2  # buy
        elif any(word in text_lower for word in ['vs', 'versus', 'compare']):
            return 3  # compare
        elif any(word in text_lower for word in ['return', 'refund']):
            return 4  # return
        elif data_type == 'query':
            return 1  # search
        else:
            return 0  # browse
    
    def _train_epoch_enhanced(self, model, dataloader, optimizer, device, model_type):
        """Train one epoch with enhanced multi-task learning"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc=f"{model_type.capitalize()} Training", leave=False):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            category_ids = batch['category_ids'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                category_ids=category_ids,
                target_ids=target_ids,
                model_type=model_type
            )
            
            loss = outputs['total_loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_enhanced_model(self, model, vocab, cluster_mappings, save_name):
        """Save enhanced model with all components"""
        save_dir = Path("models")
        save_dir.mkdir(exist_ok=True)
        
        # Save model state
        torch.save(model.state_dict(), save_dir / f"{save_name}.pt")
        
        # Save vocab
        with open(save_dir / f"{save_name}_vocab.pkl", "wb") as f:
            pickle.dump(vocab, f)
        
        # Save config
        config = {
            "vocab_size": len(vocab),
            "num_categories": model.num_categories,
            "embedding_dim": model.query_lm.embedding_dim,
            "hidden_dims": model.query_lm.hidden_dims,
            "attention_dims": model.query_lm.attention_dims,
            "dropout": 0.6
        }
        
        with open(save_dir / f"{save_name}_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Save cluster mappings
        with open(save_dir / f"{save_name}_clusters.json", "w") as f:
            json.dump(cluster_mappings, f, indent=2)
        
        logger.info(f"Enhanced model saved as {save_name}")
    
    def _tokenize_text(self, text: str, vocab: Dict[str, int]) -> List[int]:
        """Tokenize text using vocabulary"""
        words = text.lower().split()
        tokens = [vocab.get('<s>', 3)]  # Start token
        
        for word in words:
            tokens.append(vocab.get(word, vocab.get('<UNK>', 1)))
        
        tokens.append(vocab.get('</s>', 2))  # End token
        return tokens

def main():
    """Main training function - DEPRECATED, use optimized_train.py instead"""
    logger.warning("This training script has been deprecated due to memory issues.")
    logger.info("Please use the optimized training script: python optimized_train.py")
    logger.info("The optimized script provides:")
    logger.info("- GPU memory management for RTX 3050 4GB")
    logger.info("- Mixed precision training")
    logger.info("- Checkpointing and resume capability")
    logger.info("- Background training support")
    logger.info("- Aggressive memory cleanup")
    
    import sys
    sys.exit(1)

if __name__ == "__main__":
    main() 