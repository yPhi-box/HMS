"""
Main indexer — eidetic mode. Everything gets chunked, embedded, and entity-extracted.
"""
from pathlib import Path
from typing import List
from chunker import Chunker
from embedder import Embedder
from database import MemoryDatabase
from entities import EntityExtractor
from tqdm import tqdm


class MemoryIndexer:
    """Coordinate indexing operations."""
    
    def __init__(self, db_path: Path, model_name: str = None):
        self.db = MemoryDatabase(db_path)
        self.embedder = Embedder(model_name)
        self.chunker = Chunker(max_chunk_size=800, min_chunk_size=100)
        self.extractor = EntityExtractor()
    
    def index_file(self, file_path: Path, force: bool = False):
        """Index a single file — full embeddings + entity extraction."""
        print(f"Indexing: {file_path}")
        
        if force:
            self.db.clear_file(str(file_path))
        
        chunks = self.chunker.chunk_file(file_path)
        
        if not chunks:
            print(f"  No chunks generated")
            return
        
        # ALL chunks get embeddings (eidetic — nothing skipped)
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_batch(texts)
        self.db.add_chunks_batch(chunks, embeddings)
        
        # Entity extraction on every chunk
        entity_count = 0
        cursor = self.db.conn.cursor()
        # Get the IDs of chunks we just inserted
        recent_ids = cursor.execute(
            "SELECT id, text FROM chunks WHERE file_path = ? ORDER BY id DESC LIMIT ?",
            (str(file_path), len(chunks))
        ).fetchall()
        
        for chunk_id, text in recent_ids:
            entities = self.extractor.extract(text, str(file_path))
            for e in entities:
                self.db.add_entity(chunk_id, e['entity_type'], e['entity_name'], e['entity_value'], str(file_path))
                entity_count += 1
        
        # Save HNSW index periodically
        self.db.save()
        
        print(f"  Indexed {len(chunks)} chunks, {entity_count} entities")
    
    def index_directory(self, dir_path: Path, pattern: str = "**/*.md", force: bool = False):
        """
        Index all files in a directory.
        
        Args:
            dir_path: Directory to scan
            pattern: Glob pattern for files
            force: Force reindex all files
        """
        files = list(dir_path.glob(pattern))
        print(f"Found {len(files)} files to index")
        
        for file_path in tqdm(files, desc="Indexing"):
            try:
                self.index_file(file_path, force=force)
            except Exception as e:
                print(f"Error indexing {file_path}: {e}")
    
    def reindex_file(self, file_path: Path):
        """
        Reindex a file (clear + index).
        
        Args:
            file_path: Path to file
        """
        print(f"Reindexing: {file_path}")
        self.db.clear_file(str(file_path))
        self.index_file(file_path, force=False)
    
    def remove_file(self, file_path: Path):
        """
        Remove a file from the index.
        
        Args:
            file_path: Path to file
        """
        print(f"Removing: {file_path}")
        self.db.clear_file(str(file_path))
    
    def get_stats(self):
        """Get indexing statistics."""
        return self.db.get_stats()
    
    def close(self):
        """Close database connection."""
        self.db.close()
