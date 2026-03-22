"""
Hybrid search v2.4: HNSW vectors + FTS5 keywords + entity lookup + temporal decay + 
query expansion + adjacency boost + type awareness + query intent routing +
date-aware routing + tuned transcript suppression + TEMPORAL SEARCH +
CROSS-ENCODER RERANKING.

v2.4 additions:
- Cross-encoder reranking: after initial retrieval, re-score top candidates
  using cross-encoder/ms-marco-MiniLM-L-6-v2 for much better precision
- Blended scoring: 40% original hybrid score + 60% cross-encoder relevance
- Wider initial retrieval (top-20) to give reranker more candidates
- Reranker is optional: degrades gracefully if model not available

v2.3 additions:
- Temporal query detection ("when", "last time", "most recent", "first time")
- Two-pass temporal search: semantic pass → rerank by date
- Strong date filtering: explicit dates in queries → filter to matching files
- Recency bias for "last/latest/most recent" queries
- Date context in chunks (prepended by chunker) improves embedding matches

Architecture:
- Two-tier HNSW: primary (curated files) vs secondary (transcripts)
- FTS5 keyword matching with BM25 scoring
- Entity lookup with per-word fallback for proper nouns
- Query intent detection (13 categories) for smart boosting
- Temporal search mode for time-related queries
- Date-aware routing for temporal queries
- Cross-encoder reranking (top-20 → best top-N)
- Configurable synonym expansion
"""
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from embedder import Embedder
from database import MemoryDatabase
import math
import re
import json
import os


def _load_synonyms() -> dict:
    """Load synonym expansions. Override via HMS_SYNONYMS_PATH env var or synonyms.json."""
    default_synonyms = {
        # Tech
        'ip': ['ip', 'ip address', 'network address', 'host'],
        'ssh': ['ssh', 'remote access', 'secure shell', 'login'],
        'password': ['password', 'credential', 'secret', 'passphrase'],
        'token': ['token', 'api key', 'secret', 'credential'],
        'vm': ['vm', 'virtual machine', 'server', 'instance'],
        'gpu': ['gpu', 'graphics card', 'video card'],
        'dns': ['dns', 'domain name', 'nameserver'],
        'seo': ['seo', 'search engine optimization', 'ranking'],
        
        # Relationships
        'brother': ['brother', 'sibling', 'family'],
        'sister': ['sister', 'sibling', 'family'],
        'daughter': ['daughter', 'child', 'kids', 'family'],
        'son': ['son', 'child', 'kids', 'family'],
        'kids': ['kids', 'children', 'son', 'daughter'],
        'children': ['children', 'kids', 'son', 'daughter'],
        'family': ['family', 'kids', 'children', 'wife', 'brother', 'sister'],
        
        # Common expansions
        'book': ['book', 'novel', 'story', 'writing', 'fiction'],
        'novel': ['novel', 'book', 'story', 'fiction'],
        'veteran': ['veteran', 'military', 'army', 'service'],
        'military': ['military', 'army', 'veteran', 'service'],
        'weight': ['weight', 'pounds', 'lbs', 'weight loss'],
        'boat': ['boat', 'jet boat', 'boating', 'watercraft'],
        'football': ['football', 'nfl', 'sports'],
    }
    
    # Allow user override via JSON file
    custom_path = os.environ.get('HMS_SYNONYMS_PATH', 
                                  os.path.join(os.path.dirname(__file__), 'synonyms.json'))
    if os.path.exists(custom_path):
        try:
            with open(custom_path) as f:
                custom = json.load(f)
            default_synonyms.update(custom)
        except Exception:
            pass
    
    return default_synonyms


SYNONYMS = _load_synonyms()


# Query intent patterns — what kind of answer is the user looking for?
INTENT_PATTERNS = {
    'age': {
        'patterns': [
            re.compile(r'\bhow old\b', re.I),
            re.compile(r'\bage\b', re.I),
            re.compile(r'\byears? old\b', re.I),
            re.compile(r'\bbirthday\b', re.I),
            re.compile(r'\bborn\b', re.I),
        ],
        'boost_keywords': ['old', 'age', 'years', 'born', 'birthday'],
        'boost_patterns': [re.compile(r'\d{2,3}\s*years?\s*old', re.I), re.compile(r'at\s+\d{2,3}', re.I), re.compile(r'age\s*\d{2,3}', re.I)],
    },
    'location': {
        'patterns': [
            re.compile(r'\bwhere\s+(?:does|is|do)\b.*\bliv', re.I),
            re.compile(r'\blocation\b', re.I),
            re.compile(r'\baddress\b', re.I),
            re.compile(r'\bcity\b', re.I),
            re.compile(r'\bstate\b', re.I),
            re.compile(r'\bwhere.*from\b', re.I),
        ],
        'boost_keywords': ['living', 'located', 'lives', 'address', 'city', 'state', 'near'],
        'boost_patterns': [re.compile(r'living in|located in|lives in|near\s+\w+', re.I)],
    },
    'relationship': {
        'patterns': [
            re.compile(r'\bmarried|divorced|spouse|wife|husband|relationship\b', re.I),
            re.compile(r'\bsingle\b', re.I),
        ],
        'boost_keywords': ['married', 'divorced', 'spouse', 'wife', 'husband', 'single'],
        'boost_patterns': [re.compile(r'married|divorced|spouse|wife|husband', re.I)],
    },
    'name': {
        'patterns': [
            re.compile(r"(?:what(?:'s| is).*(?:full )?name|who is)", re.I),
            re.compile(r'\bfull name\b', re.I),
        ],
        'boost_keywords': ['name', 'called', 'known as'],
        'boost_patterns': [re.compile(r'[A-Z][a-z]+\s+[A-Z][a-z]+', re.I)],
    },
    'children': {
        'patterns': [
            re.compile(r"\b(?:son|daughter|kid|child|children)(?:'s)?(?:\s+name)?\b", re.I),
            re.compile(r'\bfamily\b', re.I),
        ],
        'boost_keywords': ['son', 'daughter', 'children', 'kids', 'child', 'family'],
        'boost_patterns': [re.compile(r'son|daughter|children|kids', re.I)],
    },
    'credential': {
        'patterns': [
            re.compile(r'\b(?:password|token|key|secret|credential|login|ssh)\b', re.I),
        ],
        'boost_keywords': ['password', 'token', 'key', 'secret', 'credential', 'ssh', 'login'],
        'boost_patterns': [re.compile(r'password|token|key|secret|credential', re.I)],
    },
    'config': {
        'patterns': [
            re.compile(r'\b(?:ip|port|host|server|config|setting|url)\b', re.I),
        ],
        'boost_keywords': ['ip', 'port', 'host', 'server', 'config', 'address'],
        'boost_patterns': [re.compile(r'\d+\.\d+\.\d+\.\d+|port\s*\d+', re.I)],
    },
    'health': {
        'patterns': [
            re.compile(r'\b(?:weight|health|medication|diet|exercise|medical|doctor)\b', re.I),
        ],
        'boost_keywords': ['weight', 'health', 'medication', 'diet', 'pounds', 'doctor'],
        'boost_patterns': [re.compile(r'\d+\s*(?:pounds|lbs|kg)', re.I)],
    },
    'hobby': {
        'patterns': [
            re.compile(r'\b(?:hobby|hobbies|interest|leisure|fun|enjoy)\b', re.I),
        ],
        'boost_keywords': ['hobby', 'hobbies', 'interest', 'enjoy'],
        'boost_patterns': [re.compile(r'hobby|hobbies|interest|enjoy|leisure', re.I)],
    },
    'writing': {
        'patterns': [
            re.compile(r'\b(?:book|novel|writing|fiction|story|character|author)\b', re.I),
        ],
        'boost_keywords': ['book', 'novel', 'writing', 'fiction', 'story', 'character'],
        'boost_patterns': [re.compile(r'novel|fiction|book|writing', re.I)],
    },
    'financial': {
        'patterns': [
            re.compile(r'\b(?:money|cost|price|revenue|salary|income|budget|investment)\b', re.I),
        ],
        'boost_keywords': ['money', 'cost', 'price', 'revenue', 'salary', 'budget'],
        'boost_patterns': [re.compile(r'\$[\d,]+|\d+\s*(?:dollars|USD)', re.I)],
    },
    'timeline': {
        'patterns': [
            re.compile(r'\bwhat happened\b', re.I),
            re.compile(r'\bwhen did\b', re.I),
            re.compile(r'\btimeline\b', re.I),
        ],
        'boost_keywords': ['happened', 'event', 'timeline', 'date'],
        'boost_patterns': [re.compile(r'\d{4}-\d{2}-\d{2}', re.I)],
    },
}

# v2.3: Temporal query patterns
TEMPORAL_RECENCY_PATTERNS = [
    re.compile(r'\blast\s+time\b', re.I),
    re.compile(r'\bmost\s+recent\b', re.I),
    re.compile(r'\blatest\b', re.I),
    re.compile(r'\blast\b.*\b(?:update|change|email|sent|created|published|deployed|fixed|broke|issue|rotated)\b', re.I),
    re.compile(r'\brecently\b', re.I),
    re.compile(r'\blast\s+(?:week|month|day)\b', re.I),
    re.compile(r'\bcurrent(?:ly)?\b', re.I),
    re.compile(r'\bnow\b', re.I),
    re.compile(r'\btoday\b', re.I),
    re.compile(r'\bpresent\b', re.I),
]

TEMPORAL_WHEN_PATTERNS = [
    re.compile(r'\bwhen\s+(?:did|was|were|is|has)\b', re.I),
    re.compile(r'\bwhat\s+date\b', re.I),
    re.compile(r'\bwhat\s+day\b', re.I),
    re.compile(r'\bhow\s+long\s+ago\b', re.I),
    re.compile(r'\bfirst\s+time\b', re.I),
]

TEMPORAL_SPECIFIC_DATE_PATTERNS = [
    re.compile(r'\bon\s+(?:January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}', re.I),
    re.compile(r'\bon\s+\d{4}-\d{2}-\d{2}\b', re.I),
    re.compile(r'\bon\s+\d{1,2}/\d{1,2}', re.I),
    re.compile(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?(?:[,\s]+\d{4})?\b', re.I),
]


class HybridSearch:
    """Hybrid search with query intent detection, expansion, adjacency, and reranking."""
    
    def __init__(self, db: MemoryDatabase, embedder: Embedder, enable_reranker: bool = True):
        self.db = db
        self.embedder = embedder
        self.reranker = None
        
        if enable_reranker:
            try:
                from reranker import Reranker
                self.reranker = Reranker.get_instance()
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Reranker not available: {e}. Running without reranking.")
    
    # Date extraction patterns
    _date_patterns = [
        re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(?:[,\s]+(\d{4}))?\b', re.I),
        re.compile(r'\b(\d{4})-(\d{2})-(\d{2})\b'),
        re.compile(r'\b(\d{1,2})/(\d{1,2})(?:/(\d{2,4}))?\b'),
    ]
    
    _month_map = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12',
        'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04',
        'jun': '06', 'jul': '07', 'aug': '08', 'sep': '09',
        'oct': '10', 'nov': '11', 'dec': '12',
    }
    
    def _extract_date_patterns(self, query: str) -> List[str]:
        """Extract date patterns from query to match against file paths."""
        patterns = []
        
        m = self._date_patterns[0].search(query)
        if m:
            month = self._month_map.get(m.group(1).lower(), '')
            day = m.group(2).zfill(2)
            year = m.group(3) or '2026'
            if month:
                patterns.append(f"{year}-{month}-{day}")
                patterns.append(f"{month}-{day}")
        
        m = self._date_patterns[1].search(query)
        if m:
            patterns.append(m.group(0))
        
        m = self._date_patterns[2].search(query)
        if m:
            month = m.group(1).zfill(2)
            day = m.group(2).zfill(2)
            year = m.group(3) or '2026'
            if len(year) == 2:
                year = '20' + year
            patterns.append(f"{year}-{month}-{day}")
            patterns.append(f"{month}-{day}")
        
        return patterns
    
    def _date_file_boost(self, file_path: str, date_patterns: List[str]) -> float:
        """Boost chunks from files matching query date patterns."""
        if not date_patterns:
            return 0.0
        for dp in date_patterns:
            if dp in file_path:
                return 0.3
        return 0.0
    
    def _detect_temporal_mode(self, query: str) -> Optional[str]:
        """Detect if query needs temporal search. Returns mode or None.
        
        Modes:
        - 'recency': "last time", "most recent" → sort by newest
        - 'when': "when did X happen" → find date of event
        - 'specific_date': "on March 15th" → filter to that date
        - None: not a temporal query
        """
        # Check specific date first (most restrictive)
        for pattern in TEMPORAL_SPECIFIC_DATE_PATTERNS:
            if pattern.search(query):
                return 'specific_date'
        
        # Check for recency queries
        for pattern in TEMPORAL_RECENCY_PATTERNS:
            if pattern.search(query):
                return 'recency'
        
        # Check for "when" queries
        for pattern in TEMPORAL_WHEN_PATTERNS:
            if pattern.search(query):
                return 'when'
        
        return None
    
    def _get_chunks_by_date(self, date_str: str) -> List[Dict]:
        """Get all chunks from files matching a specific date."""
        cursor = self.db.conn.cursor()
        rows = cursor.execute("""
            SELECT id, file_path, line_start, line_end, text, indexed_at, metadata
            FROM chunks WHERE file_path LIKE ?
        """, (f'%{date_str}%',)).fetchall()
        
        results = []
        for row in rows:
            results.append({
                'id': row[0], 'file_path': row[1], 'line_start': row[2],
                'line_end': row[3], 'text': row[4], 'indexed_at': row[5],
                'metadata': row[6], 'score': 0.5,
            })
        return results
    
    def _extract_file_date(self, file_path: str) -> Optional[str]:
        """Extract YYYY-MM-DD from a file path."""
        m = re.search(r'(\d{4}-\d{2}-\d{2})', file_path)
        return m.group(1) if m else None
    
    def _temporal_rerank(self, results: List[Dict], mode: str) -> List[Dict]:
        """Rerank results based on temporal mode.
        
        - recency: newest first (by file date or indexed_at)
        - when: sort by date (newest first, since user wants to know when)
        """
        def get_date_key(r):
            # Try file date first
            file_date = self._extract_file_date(r.get('file_path', ''))
            if file_date:
                return file_date
            # Fall back to indexed_at
            return r.get('indexed_at', '1970-01-01')
        
        if mode == 'recency':
            # Sort by date descending (newest first), but keep some semantic relevance
            # Top semantic results that are also recent should win
            for r in results:
                date_str = get_date_key(r)
                # Boost recent chunks significantly
                try:
                    file_dt = datetime.strptime(date_str[:10], '%Y-%m-%d')
                    now = datetime.now()
                    days_ago = max(0, (now - file_dt).days)
                    # Aggressive recency boost: recent chunks get up to +0.5
                    recency_boost = max(0, 0.5 * (1 - days_ago / 365))
                    r['combined_score'] = r.get('combined_score', 0) + recency_boost
                    r['temporal_boost'] = recency_boost
                except (ValueError, TypeError):
                    r['temporal_boost'] = 0
            
            results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        elif mode == 'when':
            # For "when" queries, date is the answer — boost chunks with clear dates
            for r in results:
                text = r.get('text', '')
                # Boost chunks that contain date information
                date_mentions = len(re.findall(r'\d{4}-\d{2}-\d{2}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}', text, re.I))
                date_boost = min(0.3, date_mentions * 0.1)
                r['combined_score'] = r.get('combined_score', 0) + date_boost
                r['temporal_boost'] = date_boost
            
            results.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        return results
    
    def _detect_intents(self, query: str) -> List[str]:
        """Detect query intents."""
        intents = []
        for intent_name, config in INTENT_PATTERNS.items():
            for pattern in config['patterns']:
                if pattern.search(query):
                    intents.append(intent_name)
                    break
        return intents
    
    def _intent_boost(self, text: str, intents: List[str]) -> float:
        """Calculate intent-based boost for a chunk."""
        if not intents:
            return 0.0
        
        total_boost = 0.0
        for intent in intents:
            config = INTENT_PATTERNS.get(intent, {})
            
            pattern_match = False
            for pattern in config.get('boost_patterns', []):
                if pattern.search(text):
                    pattern_match = True
                    break
            
            if pattern_match:
                total_boost += 0.25
            else:
                text_lower = text.lower()
                keyword_hits = sum(1 for kw in config.get('boost_keywords', []) 
                                  if kw in text_lower)
                if keyword_hits >= 2:
                    total_boost += 0.15
                elif keyword_hits >= 1:
                    total_boost += 0.08
        
        return min(total_boost, 0.35)
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms for better recall."""
        query_lower = query.lower()
        expansions = set()
        
        for term, synonyms in SYNONYMS.items():
            if re.search(r'\b' + re.escape(term) + r'\b', query_lower):
                for syn in synonyms:
                    if syn.lower() != term:
                        expansions.add(syn)
        
        if expansions:
            return f"{query} {' '.join(list(expansions)[:5])}"
        return query
    
    def _get_adjacent_chunks(self, chunk_id: int, file_path: str) -> List[Dict]:
        """Get chunks immediately before and after a given chunk in the same file."""
        cursor = self.db.conn.cursor()
        rows = cursor.execute("""
            SELECT id, file_path, line_start, line_end, text, indexed_at, metadata
            FROM chunks WHERE file_path = ? ORDER BY line_start
        """, (file_path,)).fetchall()
        
        adjacent = []
        for i, row in enumerate(rows):
            if row[0] == chunk_id:
                if i > 0:
                    prev = rows[i-1]
                    adjacent.append({
                        'id': prev[0], 'file_path': prev[1], 'line_start': prev[2],
                        'line_end': prev[3], 'text': prev[4], 'indexed_at': prev[5],
                        'metadata': prev[6], 'score': 0.0, '_adjacent': True,
                    })
                if i < len(rows) - 1:
                    nxt = rows[i+1]
                    adjacent.append({
                        'id': nxt[0], 'file_path': nxt[1], 'line_start': nxt[2],
                        'line_end': nxt[3], 'text': nxt[4], 'indexed_at': nxt[5],
                        'metadata': nxt[6], 'score': 0.0, '_adjacent': True,
                    })
                break
        
        return adjacent
    
    def search(
        self,
        query: str,
        max_results: int = 10,
        min_score: float = 0.0,
        prefer_type: str = None,
        **kwargs
    ) -> List[Dict]:
        """Full hybrid search with intent detection."""
        
        cleaned = re.sub(r'[^\w\s]', '', query or '').strip()
        if not cleaned:
            return []
        
        # 0. Detect query intent + date patterns + temporal mode
        intents = self._detect_intents(query)
        date_patterns = self._extract_date_patterns(query)
        temporal_mode = self._detect_temporal_mode(query)
        
        # v2.3: For specific_date queries with extracted dates, get date-filtered chunks
        date_filtered_chunks = []
        if temporal_mode == 'specific_date' and date_patterns:
            for dp in date_patterns:
                date_filtered_chunks.extend(self._get_chunks_by_date(dp))
        
        # v2.3: Widen the search for temporal queries (need more candidates to rerank)
        temporal_multiplier = 3 if temporal_mode else 1
        
        # 1. Entity lookup — try full query first, then proper noun pairs, then singles
        entity_results = self.db.search_entities(cleaned, limit=5)
        if not entity_results:
            common_words = {'what', 'who', 'where', 'when', 'how', 'why', 'which', 'is', 'are', 
                        'was', 'were', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of',
                        'and', 'or', 'but', 'do', 'does', 'did', 'has', 'have', 'had', 'be',
                        'been', 'being', 'with', 'about', 'tell', 'me', 'give', 'show', 'find',
                        'search', 'look', 'up', 'can', 'you', 'i', 'my', 'his', 'her', 'their',
                        'what', 'current', 'status', 'update', 'info', 'data', 'details',
                        'latest', 'report', 'summary', 'metrics', 'please', 'question',
                        'does', 'have', 'any', 'food', 'last', 'team', 'work', 'person'}
            original_words = query.split()
            # v2.3: Extract proper nouns (capitalized, non-common)
            proper_nouns = []
            for word in original_words:
                clean_word = re.sub(r"[^\w']", '', word)
                if (clean_word and len(clean_word) > 1 and 
                    clean_word[0].isupper() and clean_word.lower() not in common_words):
                    proper_nouns.append(clean_word)
            
            # v2.3: Try full name pairs first (e.g., "Kai Osei")
            if len(proper_nouns) >= 2:
                for i in range(len(proper_nouns) - 1):
                    full_name = f"{proper_nouns[i]} {proper_nouns[i+1]}"
                    name_results = self.db.search_entities(full_name, limit=5)
                    if name_results:
                        # Filter: only keep results where BOTH name parts appear
                        filtered = [e for e in name_results 
                                   if proper_nouns[i].lower() in (e['entity_value'] + ' ' + e['entity_name']).lower()
                                   and proper_nouns[i+1].lower() in (e['entity_value'] + ' ' + e['entity_name']).lower()]
                        if filtered:
                            entity_results.extend(filtered)
                        else:
                            entity_results.extend(name_results[:3])
            
            # v2.3: Fall back to single proper nouns only if no pair matches found
            if not entity_results:
                for pn in proper_nouns:
                    word_results = self.db.search_entities(pn, limit=3)
                    if word_results:
                        entity_results.extend(word_results)
        entity_chunk_ids = {e['chunk_id'] for e in entity_results}
        
        # 2. Keyword search (wider for temporal queries)
        keyword_results = self.db.search_keyword(query, limit=max_results * 5 * temporal_multiplier)
        
        # 3. Semantic search with expanded query (wider for temporal queries)
        expanded_query = self._expand_query(query)
        query_embedding = self.embedder.embed(expanded_query)
        semantic_results = self.db.search_semantic(
            query_embedding.tolist(), limit=max_results * 5 * temporal_multiplier
        )
        
        # v2.3: Inject date-filtered chunks for specific_date queries
        if date_filtered_chunks:
            existing_ids = {r['id'] for r in semantic_results}
            for chunk in date_filtered_chunks:
                if chunk['id'] not in existing_ids:
                    existing_ids.add(chunk['id'])
                    semantic_results.append(chunk)
        
        # 4. Inject entity parent chunks
        if entity_chunk_ids:
            semantic_ids = {r['id'] for r in semantic_results}
            cursor = self.db.conn.cursor()
            for chunk_id in entity_chunk_ids:
                if chunk_id not in semantic_ids:
                    row = cursor.execute("""
                        SELECT id, file_path, line_start, line_end, text, indexed_at, metadata
                        FROM chunks WHERE id = ?
                    """, (chunk_id,)).fetchone()
                    if row:
                        semantic_results.append({
                            'id': row[0], 'file_path': row[1], 'line_start': row[2],
                            'line_end': row[3], 'text': row[4], 'indexed_at': row[5],
                            'metadata': row[6], 'score': 0.5,
                        })
        
        # Build keyword score map
        keyword_map = {r['id']: r['score'] for r in keyword_results}
        max_kw_score = max((r['score'] for r in keyword_results), default=1) or 1
        
        # 5. Score and rank with intent boosting
        results = []
        now = datetime.now()
        seen_ids = set()
        top_chunk_info = []
        
        for result in semantic_results:
            chunk_id = result['id']
            seen_ids.add(chunk_id)
            semantic_score = result['score']
            text = result.get('text', '')
            
            # Keyword boost
            kw_score = keyword_map.get(chunk_id, 0)
            if kw_score > 0:
                normalized_kw = kw_score / max_kw_score
                keyword_boost = min(0.4, normalized_kw * 0.4)
            else:
                keyword_boost = 0
            
            # Temporal decay
            indexed_at = datetime.fromisoformat(result['indexed_at'])
            days_old = max(0, (now - indexed_at).days)
            temporal_factor = math.pow(0.5, days_old / 60)
            
            # Metadata
            metadata = result.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            chunk_type = metadata.get('type', 'narrative')
            is_transcript = metadata.get('source') == 'transcript'
            
            # Type boost
            type_boost = 0
            if prefer_type and chunk_type == prefer_type:
                type_boost = 0.15
            elif not prefer_type:
                query_lower = query.lower()
                if any(w in query_lower for w in ['ip', 'password', 'token', 'ssh', 'login', 'credential']):
                    if chunk_type == 'config':
                        type_boost = 0.2
                elif any(w in query_lower for w in ['birthday', 'name', 'address', 'phone', 'email', 'who is']):
                    if chunk_type == 'fact':
                        type_boost = 0.15
            
            # Source boost/penalty (curated memory > transcripts)
            source_boost = -0.18 if is_transcript else 0.10
            
            # Entity boost
            entity_boost = 0
            if entity_results:
                for e in entity_results:
                    if e['chunk_id'] == chunk_id:
                        entity_boost = 0.4
                        break
                if entity_boost == 0:
                    entity_files = {e['file_path'] for e in entity_results}
                    if result.get('file_path', '') in entity_files:
                        entity_boost = 0.1
            
            # Intent boost
            intent_boost = self._intent_boost(text, intents)
            
            # Date routing (v2.3: much stronger for temporal queries)
            date_boost = self._date_file_boost(result.get('file_path', ''), date_patterns)
            if temporal_mode == 'specific_date' and date_boost > 0:
                date_boost = 0.8  # Very strong boost — date match is highly relevant
            
            combined = (semantic_score + keyword_boost + type_boost + source_boost + 
                       entity_boost + intent_boost + date_boost) * temporal_factor
            
            scored = {
                **result,
                'semantic_score': semantic_score,
                'keyword_score': kw_score,
                'keyword_boost': keyword_boost,
                'type_boost': type_boost,
                'entity_boost': entity_boost,
                'intent_boost': intent_boost,
                'temporal_factor': temporal_factor,
                'chunk_type': chunk_type,
                'combined_score': combined,
            }
            results.append(scored)
            top_chunk_info.append((combined, chunk_id, result.get('file_path', '')))
        
        # Add keyword-only results
        for result in keyword_results:
            if result['id'] not in seen_ids:
                seen_ids.add(result['id'])
                kw_score = result['score']
                normalized_kw = kw_score / max_kw_score
                text = result.get('text', '')
                
                indexed_at = datetime.fromisoformat(result['indexed_at'])
                days_old = max(0, (now - indexed_at).days)
                temporal_factor = math.pow(0.5, days_old / 60)
                
                metadata = result.get('metadata', {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except:
                        metadata = {}
                chunk_type = metadata.get('type', 'unknown')
                is_transcript = metadata.get('source') == 'transcript'
                
                intent_boost = self._intent_boost(text, intents)
                date_boost = self._date_file_boost(result.get('file_path', ''), date_patterns)
                source_penalty = -0.15 if is_transcript else 0
                combined = (0.25 + normalized_kw * 0.5 + source_penalty + intent_boost + date_boost) * temporal_factor
                
                results.append({
                    **result,
                    'semantic_score': 0.0,
                    'keyword_score': kw_score,
                    'keyword_boost': normalized_kw * 0.5,
                    'type_boost': 0,
                    'entity_boost': 0,
                    'intent_boost': intent_boost,
                    'temporal_factor': temporal_factor,
                    'chunk_type': chunk_type,
                    'combined_score': combined,
                })
        
        # 6. Adjacency boost for top chunks
        top_chunk_info.sort(reverse=True)
        for score, chunk_id, file_path in top_chunk_info[:3]:
            if score < 0.3:
                break
            adjacent = self._get_adjacent_chunks(chunk_id, file_path)
            for adj in adjacent:
                if adj['id'] not in seen_ids:
                    seen_ids.add(adj['id'])
                    text = adj.get('text', '')
                    
                    metadata = adj.get('metadata', {})
                    if isinstance(metadata, str):
                        try:
                            metadata = json.loads(metadata)
                        except:
                            metadata = {}
                    chunk_type = metadata.get('type', 'narrative')
                    
                    intent_boost = self._intent_boost(text, intents)
                    adj_score = score * 0.5 + intent_boost
                    
                    results.append({
                        **adj,
                        'semantic_score': 0.0,
                        'keyword_score': 0.0,
                        'keyword_boost': 0.0,
                        'type_boost': 0.0,
                        'entity_boost': 0.0,
                        'intent_boost': intent_boost,
                        'temporal_factor': 1.0,
                        'chunk_type': chunk_type,
                        'combined_score': adj_score,
                        '_adjacent_to': chunk_id,
                    })
        
        # v2.3: Apply temporal reranking before final sort
        if temporal_mode in ('recency', 'when'):
            results = self._temporal_rerank(results, temporal_mode)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Deduplicate by content hash
        seen_hashes = set()
        deduped = []
        for r in results:
            metadata = r.get('metadata', {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            content_hash = metadata.get('hash', '')
            if content_hash and content_hash in seen_hashes:
                continue
            if content_hash:
                seen_hashes.add(content_hash)
            deduped.append(r)
        
        # Inject entity results as metadata
        if entity_results:
            entity_info = [{
                'entity_type': e['entity_type'],
                'entity_name': e['entity_name'],
                'entity_value': e['entity_value'],
                'file_path': e['file_path'],
            } for e in entity_results[:3]]
            if deduped:
                deduped[0]['entities'] = entity_info
        
        filtered = [r for r in deduped if r['combined_score'] >= min_score]
        
        # v2.4: Cross-encoder reranking
        # Take top-20 candidates and rerank with cross-encoder for better precision
        if self.reranker and filtered:
            rerank_pool = filtered[:20]  # Wide pool for reranker
            reranked = self.reranker.rerank(
                query=query,
                candidates=rerank_pool,
                text_key='text',
                top_k=max_results,
                blend_weight=0.4,  # 40% original, 60% cross-encoder
            )
            return reranked
        
        return filtered[:max_results]
