"""
Enhanced CADENCE Model Implementation
GRU-MN with Category Constraints and Self-Attention for Query Generation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math
import structlog

logger = structlog.get_logger()

class GRUMemoryNetworkCell(nn.Module):
    """
    GRU Memory Network Cell with Self-Attention
    Based on CADENCE paper implementation
    """
    
    def __init__(self, input_size: int, hidden_size: int, attention_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        
        # Standard GRU gates
        self.weight_ih = nn.Linear(input_size, 3 * hidden_size)
        self.weight_hh = nn.Linear(hidden_size, 3 * hidden_size)
        
        # Attention mechanism weights
        self.W_h = nn.Linear(hidden_size, attention_size)
        self.W_x = nn.Linear(input_size, attention_size)
        self.W_tilde_h = nn.Linear(hidden_size, attention_size)
        self.v = nn.Linear(attention_size, 1)
        
    def forward(self, input_tensor: torch.Tensor, hidden: torch.Tensor, 
                memory_tape: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through GRU-MN cell
        """
        batch_size = input_tensor.size(0)
        
        # Compute attention if memory tape exists
        if memory_tape is not None and memory_tape.size(1) > 0:
            # Compute attention scores
            seq_len = memory_tape.size(1)
            
            # Expand inputs for attention computation
            h_expanded = self.W_h(memory_tape)  # [batch, seq_len, attention_size]
            x_expanded = self.W_x(input_tensor).unsqueeze(1).expand(-1, seq_len, -1)
            tilde_h_expanded = self.W_tilde_h(hidden).unsqueeze(1).expand(-1, seq_len, -1)
            
            # Compute attention scores
            attention_input = torch.tanh(h_expanded + x_expanded + tilde_h_expanded)
            attention_scores = self.v(attention_input).squeeze(-1)  # [batch, seq_len]
            attention_weights = F.softmax(attention_scores, dim=1)
            
            # Compute aggregated hidden state
            aggregated_hidden = torch.sum(attention_weights.unsqueeze(-1) * memory_tape, dim=1)
        else:
            aggregated_hidden = hidden
        
        # Standard GRU computation with aggregated hidden state
        gi = self.weight_ih(input_tensor)
        gh = self.weight_hh(aggregated_hidden)
        
        i_reset, i_update, i_new = gi.chunk(3, 1)
        h_reset, h_update, h_new = gh.chunk(3, 1)
        
        reset_gate = torch.sigmoid(i_reset + h_reset)
        update_gate = torch.sigmoid(i_update + h_update)
        new_gate = torch.tanh(i_new + reset_gate * h_new)
        
        new_hidden = (1 - update_gate) * new_gate + update_gate * aggregated_hidden
        
        return new_hidden, aggregated_hidden

class CategoryConstrainedGRUMN(nn.Module):
    """
    Enhanced Multi-layer GRU-MN with Category Constraints and E-commerce Features
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dims: List[int], 
                 attention_dims: List[int], num_categories: int, dropout: float = 0.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.attention_dims = attention_dims
        self.num_categories = num_categories
        self.num_layers = len(hidden_dims)
        
        # Enhanced embedding layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Position embedding for better sequence understanding
        self.position_embedding = nn.Embedding(100, embedding_dim)  # Max sequence length 100
        
        # E-commerce specific embeddings
        self.brand_embedding = nn.Embedding(1000, embedding_dim // 4)  # Brand embeddings
        self.price_range_embedding = nn.Embedding(10, embedding_dim // 4)  # Price range embeddings
        
        # Enhanced GRU-MN layers
        self.gru_cells = nn.ModuleList()
        input_size = embedding_dim * 2  # word + category + position
        
        # Layer normalization for better training stability
        self.layer_norms = nn.ModuleList()
        
        for i, (hidden_dim, attention_dim) in enumerate(zip(hidden_dims, attention_dims)):
            self.gru_cells.append(GRUMemoryNetworkCell(input_size, hidden_dim, attention_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            input_size = hidden_dim
        
        # Multi-head self-attention for better context understanding
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dims[-1], 
            num_heads=8, 
            dropout=dropout,
            batch_first=True
        )
        
        # Enhanced intermediate context layers
        self.intermediate_context = nn.Sequential(
            nn.Linear(hidden_dims[-1] + embedding_dim, 1024),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # E-commerce specific prediction heads
        self.query_head = nn.Linear(512, vocab_size)  # Query completion
        self.intent_head = nn.Linear(512, 5)  # Intent classification (browse, search, buy, compare, return)
        self.category_head = nn.Linear(512, num_categories)  # Category prediction
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        self.attention_dropout = nn.Dropout(dropout * 0.5)
        
        # Memory tapes for each layer
        self.memory_tapes = [None] * self.num_layers
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights properly"""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, 0.0, 0.1)
        nn.init.normal_(self.category_embedding.weight, 0.0, 0.1)
        nn.init.normal_(self.position_embedding.weight, 0.0, 0.1)
        nn.init.normal_(self.brand_embedding.weight, 0.0, 0.1)
        nn.init.normal_(self.price_range_embedding.weight, 0.0, 0.1)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def reset_memory(self):
        """Reset memory tapes (call between batches to avoid gradient issues)"""
        self.memory_tapes = [None] * self.num_layers
        
    def forward(self, input_ids: torch.Tensor, category_ids: torch.Tensor, 
                hidden_states: Optional[List[torch.Tensor]] = None,
                brand_ids: Optional[torch.Tensor] = None,
                price_range_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with e-commerce specific features.
        The memory tapes are **reset at the beginning of every forward** call so that
        they do not carry hidden states across independent mini-batches. Carrying
        memory across batches can lead to a mismatch in the batch dimension when
        the last batch of an epoch has a different size (e.g. 31 instead of 32),
        which resulted in the runtime error:
            RuntimeError: The size of tensor a (32) must match the size of tensor b (31)
        Resetting the memory for each new batch keeps the temporal memory within a
        sequence while preventing cross-batch leakage.
        """
        # Reset memory tapes at the start of each new batch/sequence
        self.reset_memory()

        batch_size, seq_len = input_ids.size()
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [torch.zeros(batch_size, dim, device=input_ids.device) 
                           for dim in self.hidden_dims]
        
        # Embed inputs
        word_embeds = self.embedding(input_ids)  # [batch, seq_len, embed_dim]
        category_embeds = self.category_embedding(category_ids)  # [batch, seq_len, embed_dim]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embedding(positions)
        
        # Enhanced input embeddings with positional encoding
        input_embeds = word_embeds + category_embeds + position_embeds
        input_embeds = torch.cat([word_embeds, category_embeds], dim=-1)
        
        # Process sequence through enhanced layers
        all_hidden_states = []
        current_input = input_embeds
        
        for t in range(seq_len):
            x_t = current_input[:, t, :]  # [batch, embed_dim * 2]
            
            # Pass through each layer with enhancements
            layer_input = x_t
            new_hidden_states = []
            
            for layer_idx, (gru_cell, layer_norm) in enumerate(zip(self.gru_cells, self.layer_norms)):
                # Add category constraint to input
                category_t = category_embeds[:, t, :]
                
                # Handle dimension mismatch
                if layer_input.size(-1) != category_t.size(-1):
                    if not hasattr(self, f'category_projection_{layer_idx}'):
                        projection = nn.Linear(category_t.size(-1), layer_input.size(-1), bias=False)
                        setattr(self, f'category_projection_{layer_idx}', projection)
                        projection = projection.to(layer_input.device)
                    else:
                        projection = getattr(self, f'category_projection_{layer_idx}')
                    
                    category_t_projected = projection(category_t)
                else:
                    category_t_projected = category_t
                
                layer_input_with_category = F.relu(layer_input + category_t_projected)
                
                # Forward through GRU-MN cell
                new_hidden, _ = gru_cell(
                    layer_input_with_category, 
                    hidden_states[layer_idx],
                    self.memory_tapes[layer_idx]
                )
                
                # Apply layer normalization
                new_hidden = layer_norm(new_hidden)
                
                # Update memory tape
                if self.memory_tapes[layer_idx] is None:
                    self.memory_tapes[layer_idx] = new_hidden.unsqueeze(1).detach()
                else:
                    self.memory_tapes[layer_idx] = torch.cat([
                        self.memory_tapes[layer_idx], 
                        new_hidden.unsqueeze(1).detach()
                    ], dim=1)
                    
                    max_memory_length = getattr(self, 'max_memory_length', 128)
                    if self.memory_tapes[layer_idx].size(1) > max_memory_length:
                        self.memory_tapes[layer_idx] = self.memory_tapes[layer_idx][:, -max_memory_length:]
                
                new_hidden_states.append(new_hidden)
                layer_input = self.dropout(new_hidden)
            
            hidden_states = new_hidden_states
            all_hidden_states.append(hidden_states[-1])
        
        # Stack all hidden states for attention
        sequence_hidden = torch.stack(all_hidden_states, dim=1)  # [batch, seq_len, hidden_dim]
        
        # Apply multi-head self-attention
        attended_output, attention_weights = self.self_attention(
            sequence_hidden, sequence_hidden, sequence_hidden
        )
        attended_output = self.attention_dropout(attended_output)
        
        # Enhanced context processing
        final_outputs = []
        query_outputs = []
        intent_outputs = []
        category_outputs = []
        
        for t in range(seq_len):
            # Combine attended output with category information
            context_input = torch.cat([attended_output[:, t, :], category_embeds[:, t, :]], dim=-1)
            context_features = self.intermediate_context(context_input)
            
            # Multi-head predictions
            query_logits = self.query_head(context_features)
            intent_logits = self.intent_head(context_features)
            category_logits = self.category_head(context_features)
            
            query_outputs.append(query_logits)
            intent_outputs.append(intent_logits)
            category_outputs.append(category_logits)
        
        return {
            'query_logits': torch.stack(query_outputs, dim=1),
            'intent_logits': torch.stack(intent_outputs, dim=1),
            'category_logits': torch.stack(category_outputs, dim=1),
            'attention_weights': attention_weights,
            'hidden_states': hidden_states
        }
    
    def reset_memory(self):
        """Reset memory tapes for new sequence"""
        self.memory_tapes = [None] * self.num_layers

class CADENCEModel(nn.Module):
    """
    Complete CADENCE Model for Query Generation
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.vocab_size = config['vocab_size']
        self.num_categories = config['num_categories']
        
        # Query Language Model
        self.query_lm = CategoryConstrainedGRUMN(
            vocab_size=self.vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims'],
            attention_dims=config['attention_dims'],
            num_categories=self.num_categories,
            dropout=config['dropout']
        )
        
        # Catalog Language Model (same architecture)
        self.catalog_lm = CategoryConstrainedGRUMN(
            vocab_size=self.vocab_size,
            embedding_dim=config['embedding_dim'],
            hidden_dims=config['hidden_dims'],
            attention_dims=config['attention_dims'],
            num_categories=self.num_categories,
            dropout=config['dropout']
        )
        
        # Enhanced loss functions for multi-task learning
        self.query_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        self.intent_criterion = nn.CrossEntropyLoss()
        self.category_criterion = nn.CrossEntropyLoss()
        
        # Loss weights for multi-task learning
        self.loss_weights = {
            'query': 0.7,
            'intent': 0.2,
            'category': 0.1
        }
        
    def forward(self, input_ids: torch.Tensor, category_ids: torch.Tensor, 
                target_ids: torch.Tensor, model_type: str = 'query',
                intent_targets: Optional[torch.Tensor] = None,
                category_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass for multi-task training
        """
        if model_type == 'query':
            outputs = self.query_lm(input_ids, category_ids)
        else:
            outputs = self.catalog_lm(input_ids, category_ids)
        
        # Calculate multi-task losses
        query_loss = self.query_criterion(
            outputs['query_logits'].view(-1, self.vocab_size), 
            target_ids.view(-1)
        )
        
        total_loss = self.loss_weights['query'] * query_loss
        loss_dict = {'query_loss': query_loss}
        
        # Add intent loss if targets provided
        if intent_targets is not None:
            # Use only the last timestep for intent classification (per-sequence prediction)
            intent_logits_last = outputs['intent_logits'][:, -1, :]  # [batch_size, 5]
            intent_loss = self.intent_criterion(
                intent_logits_last,
                intent_targets
            )
            total_loss += self.loss_weights['intent'] * intent_loss
            loss_dict['intent_loss'] = intent_loss
        
        # Add category loss if targets provided
        if category_targets is not None:
            category_loss = self.category_criterion(
                outputs['category_logits'].view(-1, self.num_categories),
                category_targets.view(-1)
            )
            total_loss += self.loss_weights['category'] * category_loss
            loss_dict['category_loss'] = category_loss
        
        return {
            **outputs,
            'total_loss': total_loss,
            'loss_breakdown': loss_dict
        }
    
    def generate(self, start_tokens: List[int], category_id: int, max_length: int = 50, 
                 model_type: str = 'query', temperature: float = 1.0,
                 return_intent: bool = False) -> Dict[str, Any]:
        """
        Enhanced generation with multi-task outputs
        """
        self.eval()
        
        if model_type == 'query':
            model = self.query_lm
        else:
            model = self.catalog_lm
        
        model.reset_memory()
        
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # Initialize
            current_tokens = start_tokens.copy()
            hidden_states = None
            intent_predictions = []
            category_predictions = []
            
            for step in range(max_length):
                # Prepare input
                input_tensor = torch.tensor([current_tokens], device=device)
                category_tensor = torch.full_like(input_tensor, category_id, device=device)
                
                # Forward pass
                outputs = model(input_tensor, category_tensor, hidden_states)
                hidden_states = outputs['hidden_states']
                
                # Get next token probabilities
                next_token_logits = outputs['query_logits'][0, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(next_token_probs, 1).item()
                
                # Store predictions
                if return_intent:
                    intent_probs = F.softmax(outputs['intent_logits'][0, -1, :], dim=-1)
                    intent_predictions.append(intent_probs.cpu().numpy())
                    
                    category_probs = F.softmax(outputs['category_logits'][0, -1, :], dim=-1)
                    category_predictions.append(category_probs.cpu().numpy())
                
                # Check for end token
                if next_token == 2:  # </s> token
                    break
                
                current_tokens.append(next_token)
            
            result = {'tokens': current_tokens}
            
            if return_intent:
                result['intent_predictions'] = intent_predictions
                result['category_predictions'] = category_predictions
            
            return result

class DynamicBeamSearch:
    """
    Dynamic Beam Search for Diverse Query Generation
    """
    
    def __init__(self, model: CADENCEModel, vocab: Dict[str, int], 
                 beam_width: int = 5, max_length: int = 50):
        self.model = model
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.beam_width = beam_width
        self.max_length = max_length
    
    def search(self, start_tokens: List[str], category_id: int, model_type: str = 'query',
               p1_threshold: float = 0.3, p2_threshold: float = 0.4) -> List[str]:
        """
        Perform dynamic beam search with diversity
        """
        self.model.eval()
        
        # Convert start tokens to IDs
        start_token_ids = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in start_tokens]
        
        # First level: high diversity
        first_level_candidates = self._get_diverse_next_tokens(
            start_token_ids, category_id, model_type, p1_threshold
        )
        
        # Second level: moderate diversity
        second_level_candidates = []
        for candidate in first_level_candidates:
            next_candidates = self._get_diverse_next_tokens(
                candidate, category_id, model_type, p2_threshold
            )
            second_level_candidates.extend(next_candidates)
        
        # Final level: beam search
        final_candidates = []
        for candidate in second_level_candidates:
            beam_results = self._beam_search_from_prefix(
                candidate, category_id, model_type
            )
            final_candidates.extend(beam_results)
        
        # Convert back to strings and remove duplicates
        result_strings = []
        for candidate in final_candidates:
            tokens = [self.inv_vocab.get(token_id, '<UNK>') for token_id in candidate]
            query_string = ' '.join(tokens).replace('<s>', '').replace('</s>', '').strip()
            if query_string and query_string not in result_strings:
                result_strings.append(query_string)
        
        return result_strings[:10]  # Return top 10
    
    def _get_diverse_next_tokens(self, prefix: List[int], category_id: int, 
                                model_type: str, threshold: float) -> List[List[int]]:
        """Get diverse next tokens above threshold"""
        with torch.no_grad():
            device = next(self.model.parameters()).device
            
            # Prepare input
            input_tensor = torch.tensor([prefix], device=device)
            category_tensor = torch.full_like(input_tensor, category_id, device=device)
            
            # Get model
            if model_type == 'query':
                model = self.model.query_lm
            else:
                model = self.model.catalog_lm
            
            model.reset_memory()
            outputs = model(input_tensor, category_tensor)
            
            # Get probabilities for next token (handle new output format)
            if isinstance(outputs, dict):
                query_logits = outputs['query_logits'][0, -1, :]
            else:
                query_logits = outputs[0, -1, :]
            
            next_token_probs = F.softmax(query_logits, dim=-1)
            
            # Get tokens above threshold
            candidates = []
            for token_id, prob in enumerate(next_token_probs):
                if prob.item() > threshold:
                    candidates.append(prefix + [token_id])
            
            return candidates[:20]  # Limit for efficiency
    
    def _beam_search_from_prefix(self, prefix: List[int], category_id: int, 
                                model_type: str) -> List[List[int]]:
        """Standard beam search from given prefix"""
        with torch.no_grad():
            device = next(self.model.parameters()).device
            
            # Initialize beam
            beam = [(prefix, 0.0)]  # (sequence, score)
            
            for _ in range(self.max_length - len(prefix)):
                candidates = []
                
                for sequence, score in beam:
                    if sequence[-1] == 2:  # End token
                        candidates.append((sequence, score))
                        continue
                    
                    # Get next token probabilities
                    input_tensor = torch.tensor([sequence], device=device)
                    category_tensor = torch.full_like(input_tensor, category_id, device=device)
                    
                    if model_type == 'query':
                        model = self.model.query_lm
                    else:
                        model = self.model.catalog_lm
                    
                    model.reset_memory()
                    outputs = model(input_tensor, category_tensor)
                    
                    # Handle new output format
                    if isinstance(outputs, dict):
                        query_logits = outputs['query_logits'][0, -1, :]
                    else:
                        query_logits = outputs[0, -1, :]
                    
                    next_token_probs = F.softmax(query_logits, dim=-1)
                    
                    # Add top tokens to candidates
                    top_tokens = torch.topk(next_token_probs, self.beam_width)
                    
                    for token_id, prob in zip(top_tokens.indices, top_tokens.values):
                        new_sequence = sequence + [token_id.item()]
                        new_score = score + torch.log(prob).item()
                        candidates.append((new_sequence, new_score))
                
                # Keep top candidates
                candidates.sort(key=lambda x: x[1], reverse=True)
                beam = candidates[:self.beam_width]
                
                # Check if all sequences ended
                if all(seq[-1] == 2 for seq, _ in beam):
                    break
            
            return [seq for seq, _ in beam]

def create_cadence_model(vocab_size: int, num_categories: int, **overrides) -> CADENCEModel:
    """
    Factory function to create a CADENCE model.

    Parameters
    ----------
    vocab_size : int
        Size of the vocabulary.
    num_categories : int
        Number of pseudo-category clusters.
    **overrides : Any
        Optional keyword arguments that override the default hyper-parameters.  This
        makes the helper backwards-compatible with older checkpoints whose
        architecture parameters (e.g. embedding_dim, hidden_dims) differ from the
        current defaults.
    """
    # Enhanced hyper-parameters for larger, more sophisticated models
    config = {
        'vocab_size': vocab_size,
        'num_categories': num_categories,
        'embedding_dim': 512,  # Increased from 256
        'hidden_dims': [3000, 2500, 2000, 1500],  # Added more layers
        'attention_dims': [1500, 1250, 1000, 750],  # Enhanced attention
        'dropout': 0.6  # Reduced dropout for larger model
    }

    # Apply any overrides coming from an existing checkpoint configuration
    for k, v in overrides.items():
        # These keys are already set explicitly and should not be overridden via **kwargs
        if k in ('vocab_size', 'num_categories'):
            continue
        # Skip None values so that defaults remain intact when a key is missing
        if v is not None:
            config[k] = v

    return CADENCEModel(config) 