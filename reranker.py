"""
Cross-encoder reranker for HMS v2.4.

Takes (query, candidate_text) pairs and produces relevance scores.
Uses cross-encoder/ms-marco-MiniLM-L-6-v2 — same model Hindsight uses.
Runs 100% locally, no API calls.

Typical latency: 30-80ms for 20 candidates on CPU.
"""
from typing import List, Dict, Optional
import os
import time


class Reranker:
    """Cross-encoder reranker using sentence-transformers."""
    
    _instance: Optional['Reranker'] = None
    _model = None
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.environ.get(
            'HMS_RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        )
        self._load_model()
    
    @classmethod
    def get_instance(cls, model_name: str = None) -> 'Reranker':
        """Singleton — load model once, reuse across queries."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance
    
    def _load_model(self):
        """Load the cross-encoder model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers required for reranking. "
                "Install with: pip install sentence-transformers"
            )
    
    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        text_key: str = 'text',
        top_k: int = None,
        blend_weight: float = 0.4,
    ) -> List[Dict]:
        """
        Rerank candidates using cross-encoder scores.
        
        Args:
            query: The search query
            candidates: List of result dicts (must have text_key field)
            text_key: Key to extract text from candidate dicts
            top_k: Number of results to return (None = all)
            blend_weight: How much to weight original score vs cross-encoder.
                         0.0 = pure cross-encoder, 1.0 = pure original score.
                         Default 0.4 means 60% cross-encoder, 40% original.
        
        Returns:
            Reranked list of candidates with 'rerank_score' and 'blended_score' added.
        """
        if not candidates or self._model is None:
            return candidates
        
        # Build (query, text) pairs
        pairs = []
        for c in candidates:
            text = c.get(text_key, '')
            if not text:
                text = ''
            pairs.append((query, text))
        
        # Score all pairs at once (batched)
        start = time.time()
        scores = self._model.predict(pairs)
        elapsed_ms = (time.time() - start) * 1000
        
        # Normalize cross-encoder scores to [0, 1] range using sigmoid
        # ms-marco model outputs logits, not probabilities
        import math
        def sigmoid(x):
            try:
                return 1 / (1 + math.exp(-x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0
        
        normalized = [sigmoid(float(s)) for s in scores]
        
        # Get original score range for blending
        orig_scores = [c.get('combined_score', 0) for c in candidates]
        max_orig = max(orig_scores) if orig_scores else 1
        min_orig = min(orig_scores) if orig_scores else 0
        orig_range = max_orig - min_orig if max_orig != min_orig else 1
        
        # Attach scores and blend
        for i, c in enumerate(candidates):
            c['rerank_score'] = normalized[i]
            c['rerank_raw'] = float(scores[i])
            c['rerank_latency_ms'] = round(elapsed_ms, 1)
            
            # Normalize original score to [0, 1]
            orig_norm = (c.get('combined_score', 0) - min_orig) / orig_range
            
            # Blend: higher blend_weight = more weight on original ranking
            c['blended_score'] = (
                blend_weight * orig_norm + 
                (1 - blend_weight) * normalized[i]
            )
        
        # Sort by blended score
        candidates.sort(key=lambda x: x['blended_score'], reverse=True)
        
        if top_k:
            return candidates[:top_k]
        return candidates
