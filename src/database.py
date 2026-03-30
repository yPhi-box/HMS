"""
SQLite + HNSW database: FTS5 for keyword search, hnswlib for fast vector search.
Eidetic memory — everything gets indexed, nothing is thrown away.
"""
import sqlite3
import json
import hnswlib
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


class MemoryDatabase:
    """SQLite for metadata/FTS + hnswlib for vector search."""
    
    VECTOR_DIM = 384
    HNSW_EF_CONSTRUCTION = 200
    HNSW_M = 16
    HNSW_EF_SEARCH = 100
    INITIAL_MAX_ELEMENTS = 50000  # grows dynamically
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.hnsw_primary_path = db_path.with_suffix('.hnsw')  # memory files
        self.hnsw_secondary_path = db_path.with_suffix('.hnsw2')  # transcripts
        
        # SQLite for metadata + FTS
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._init_schema()
        
        # Two-tier HNSW: primary (memory) + secondary (transcripts)
        self.primary_index = hnswlib.Index(space='cosine', dim=self.VECTOR_DIM)
        self.secondary_index = hnswlib.Index(space='cosine', dim=self.VECTOR_DIM)
        self._init_hnsw()
    
    def _init_schema(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                chars INTEGER NOT NULL,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_modified_at TIMESTAMP,
                metadata TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_file_path ON chunks(file_path);
            
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content=chunks,
                content_rowid=id
            );
            
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
            END;
            
            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                DELETE FROM chunks_fts WHERE rowid = old.id;
            END;
            
            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                UPDATE chunks_fts SET text = new.text WHERE rowid = new.id;
            END;
            
            -- Entity extraction table
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_id INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_name TEXT NOT NULL,
                entity_value TEXT NOT NULL,
                file_path TEXT NOT NULL,
                FOREIGN KEY (chunk_id) REFERENCES chunks(id) ON DELETE CASCADE
            );
            
            CREATE INDEX IF NOT EXISTS idx_entity_type ON entities(entity_type);
            CREATE INDEX IF NOT EXISTS idx_entity_name ON entities(entity_name);
            
            -- FTS on entities for fast text search
            CREATE VIRTUAL TABLE IF NOT EXISTS entities_fts USING fts5(
                entity_name,
                entity_value,
                content=entities,
                content_rowid=id
            );
            
            CREATE TRIGGER IF NOT EXISTS entities_ai AFTER INSERT ON entities BEGIN
                INSERT INTO entities_fts(rowid, entity_name, entity_value) 
                VALUES (new.id, new.entity_name, new.entity_value);
            END;
            
            CREATE TRIGGER IF NOT EXISTS entities_ad AFTER DELETE ON entities BEGIN
                DELETE FROM entities_fts WHERE rowid = old.id;
            END;
        """)
        self.conn.commit()
    
    def _init_hnsw(self):
        """Load or create HNSW indexes."""
        for idx, path in [(self.primary_index, self.hnsw_primary_path), 
                          (self.secondary_index, self.hnsw_secondary_path)]:
            if path.exists():
                idx.load_index(str(path))
                idx.set_ef(self.HNSW_EF_SEARCH)
            else:
                idx.init_index(
                    max_elements=self.INITIAL_MAX_ELEMENTS,
                    ef_construction=self.HNSW_EF_CONSTRUCTION,
                    M=self.HNSW_M
                )
                idx.set_ef(self.HNSW_EF_SEARCH)
    
    def _ensure_capacity(self, index, needed: int):
        """Grow HNSW index if needed."""
        current = index.get_max_elements()
        if index.get_current_count() + needed > current:
            new_max = max(current * 2, index.get_current_count() + needed + 10000)
            index.resize_index(new_max)
    
    def _get_index_for_chunk(self, chunk: Dict):
        """Route chunk to primary (memory) or secondary (transcript) index."""
        metadata = chunk.get('metadata', {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        is_transcript = metadata.get('source') == 'transcript'
        return self.secondary_index if is_transcript else self.primary_index
    
    def add_chunk(self, chunk: Dict, embedding: List[float]) -> int:
        """Add a chunk with its embedding to SQLite and appropriate HNSW index."""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO chunks (file_path, line_start, line_end, text, chars, file_modified_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk['file_path'],
            chunk['line_start'],
            chunk['line_end'],
            chunk['text'],
            chunk['chars'],
            datetime.now().isoformat(),
            json.dumps(chunk.get('metadata', {}))
        ))
        
        chunk_id = cursor.lastrowid
        self.conn.commit()
        
        # Add to appropriate HNSW index
        index = self._get_index_for_chunk(chunk)
        self._ensure_capacity(index, 1)
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        index.add_items(vec, np.array([chunk_id]))
        
        return chunk_id
    
    def add_chunks_batch(self, chunks: List[Dict], embeddings: List[List[float]]):
        """Add multiple chunks efficiently."""
        cursor = self.conn.cursor()
        chunk_ids = []
        
        for chunk in chunks:
            cursor.execute("""
                INSERT INTO chunks (file_path, line_start, line_end, text, chars, file_modified_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk['file_path'],
                chunk['line_start'],
                chunk['line_end'],
                chunk['text'],
                chunk['chars'],
                datetime.now().isoformat(),
                json.dumps(chunk.get('metadata', {}))
            ))
            chunk_ids.append(cursor.lastrowid)
        
        self.conn.commit()
        
        # Batch add to HNSW — route to correct index per chunk
        if embeddings:
            primary_vecs, primary_ids = [], []
            secondary_vecs, secondary_ids = [], []
            
            for chunk, emb, cid in zip(chunks, embeddings, chunk_ids):
                idx = self._get_index_for_chunk(chunk)
                if idx is self.secondary_index:
                    secondary_vecs.append(emb)
                    secondary_ids.append(cid)
                else:
                    primary_vecs.append(emb)
                    primary_ids.append(cid)
            
            if primary_vecs:
                self._ensure_capacity(self.primary_index, len(primary_vecs))
                self.primary_index.add_items(np.array(primary_vecs, dtype=np.float32), np.array(primary_ids, dtype=np.int64))
            if secondary_vecs:
                self._ensure_capacity(self.secondary_index, len(secondary_vecs))
                self.secondary_index.add_items(np.array(secondary_vecs, dtype=np.float32), np.array(secondary_ids, dtype=np.int64))
    
    def _search_index(self, index, query_embedding: List[float], limit: int) -> List[Dict]:
        """Search a single HNSW index."""
        if index.get_current_count() == 0:
            return []
        
        vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        k = min(limit, index.get_current_count())
        labels, distances = index.knn_query(vec, k=k)
        
        results = []
        cursor = self.conn.cursor()
        
        for label, distance in zip(labels[0], distances[0]):
            row = cursor.execute("""
                SELECT id, file_path, line_start, line_end, text, indexed_at, metadata
                FROM chunks WHERE id = ?
            """, (int(label),)).fetchone()
            
            if row:
                score = 1.0 - distance
                results.append({
                    'id': row[0], 'file_path': row[1], 'line_start': row[2],
                    'line_end': row[3], 'text': row[4], 'indexed_at': row[5],
                    'metadata': row[6], 'score': max(0, score),
                })
        
        return results
    
    def search_semantic(self, query_embedding: List[float], limit: int = 10) -> List[Dict]:
        """Two-tier vector search: primary (memory) first, then secondary (transcripts)."""
        # Search primary index (memory files) — these are the important ones
        primary_results = self._search_index(self.primary_index, query_embedding, limit)
        
        # Search secondary index (transcripts) — supplementary
        secondary_results = self._search_index(self.secondary_index, query_embedding, limit)
        
        # Tag results with their tier
        for r in primary_results:
            r['_tier'] = 'primary'
        for r in secondary_results:
            r['_tier'] = 'secondary'
        
        # Merge: primary results first, then secondary
        return primary_results + secondary_results
    
    def search_keyword(self, query: str, limit: int = 10) -> List[Dict]:
        """FTS5 keyword search."""
        import re
        sanitized = re.sub(r'[?()"\'\-*:.,;!@#$%^&+=\[\]{}|\\/<>~`]', ' ', query)
        sanitized = ' '.join(sanitized.split())
        
        if not sanitized.strip():
            return []
        
        cursor = self.conn.cursor()
        try:
            rows = cursor.execute("""
                SELECT c.id, c.file_path, c.line_start, c.line_end, c.text, 
                       c.indexed_at, c.metadata, 
                       bm25(chunks_fts) * -1 as score
                FROM chunks_fts f
                JOIN chunks c ON c.id = f.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY score DESC
                LIMIT ?
            """, (sanitized, limit)).fetchall()
            
            return [{
                'id': r[0], 'file_path': r[1], 'line_start': r[2],
                'line_end': r[3], 'text': r[4], 'indexed_at': r[5],
                'metadata': r[6], 'score': r[7],
            } for r in rows]
        except Exception:
            return []
    
    def search_entities(self, query: str, entity_type: str = None, limit: int = 10) -> List[Dict]:
        """Search extracted entities via FTS5."""
        import re
        sanitized = re.sub(r'[?()"\'\-*:.,;!@#$%^&+=\[\]{}|\\/<>~`]', ' ', query)
        sanitized = ' '.join(sanitized.split())
        if not sanitized.strip():
            return []
        
        cursor = self.conn.cursor()
        try:
            if entity_type:
                rows = cursor.execute("""
                    SELECT e.entity_type, e.entity_name, e.entity_value, e.file_path, e.chunk_id
                    FROM entities_fts f
                    JOIN entities e ON e.id = f.rowid
                    WHERE entities_fts MATCH ? AND e.entity_type = ?
                    ORDER BY rank
                    LIMIT ?
                """, (sanitized, entity_type, limit)).fetchall()
            else:
                rows = cursor.execute("""
                    SELECT e.entity_type, e.entity_name, e.entity_value, e.file_path, e.chunk_id
                    FROM entities_fts f
                    JOIN entities e ON e.id = f.rowid
                    WHERE entities_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (sanitized, limit)).fetchall()
        except Exception:
            # Fallback to LIKE if FTS fails
            rows = cursor.execute("""
                SELECT entity_type, entity_name, entity_value, file_path, chunk_id
                FROM entities 
                WHERE entity_name LIKE ? OR entity_value LIKE ?
                LIMIT ?
            """, (f'%{query}%', f'%{query}%', limit)).fetchall()
        
        return [{
            'entity_type': r[0], 'entity_name': r[1], 
            'entity_value': r[2], 'file_path': r[3], 'chunk_id': r[4]
        } for r in rows]
    
    def add_entity(self, chunk_id: int, entity_type: str, name: str, value: str, file_path: str):
        """Add an extracted entity."""
        self.conn.execute("""
            INSERT INTO entities (chunk_id, entity_type, entity_name, entity_value, file_path)
            VALUES (?, ?, ?, ?, ?)
        """, (chunk_id, entity_type, name, value, file_path))
        self.conn.commit()
    
    def clear_file(self, file_path: str):
        """Remove all chunks and entities for a file."""
        cursor = self.conn.cursor()
        ids = [r[0] for r in cursor.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()]
        
        # Remove from both HNSW indexes
        for chunk_id in ids:
            for idx in [self.primary_index, self.secondary_index]:
                try:
                    idx.mark_deleted(chunk_id)
                except Exception:
                    pass
        
        # Remove from SQLite
        cursor.execute("DELETE FROM entities WHERE file_path = ?", (file_path,))
        cursor.execute("DELETE FROM chunks WHERE file_path = ?", (file_path,))
        self.conn.commit()
    
    def get_stats(self) -> Dict:
        """Get database statistics."""
        cursor = self.conn.cursor()
        total_chunks = cursor.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        total_files = cursor.execute("SELECT COUNT(DISTINCT file_path) FROM chunks").fetchone()[0]
        total_entities = cursor.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        
        db_size = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
        p_size = self.hnsw_primary_path.stat().st_size / (1024 * 1024) if self.hnsw_primary_path.exists() else 0
        s_size = self.hnsw_secondary_path.stat().st_size / (1024 * 1024) if self.hnsw_secondary_path.exists() else 0
        
        return {
            'total_chunks': total_chunks,
            'total_files': total_files,
            'total_entities': total_entities,
            'db_size_mb': round(db_size, 2),
            'hnsw_size_mb': round(p_size + s_size, 2),
            'total_size_mb': round(db_size + p_size + s_size, 2),
            'hnsw_count': self.primary_index.get_current_count() + self.secondary_index.get_current_count(),
            'hnsw_primary': self.primary_index.get_current_count(),
            'hnsw_secondary': self.secondary_index.get_current_count(),
        }
    
    def save(self):
        """Save HNSW indexes to disk."""
        if self.primary_index.get_current_count() > 0:
            self.primary_index.save_index(str(self.hnsw_primary_path))
        if self.secondary_index.get_current_count() > 0:
            self.secondary_index.save_index(str(self.hnsw_secondary_path))
    
    def close(self):
        """Save and close."""
        self.save()
        self.conn.close()
