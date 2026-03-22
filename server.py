"""
HMS v2.2 — Hybrid Memory Server
FastAPI server for OpenClaw integration.
"""
import warnings
import os
import logging

# Suppress noisy warnings before any imports
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
from typing import Optional
from database import MemoryDatabase
from embedder import Embedder
from search import HybridSearch
from indexer import MemoryIndexer
import uvicorn
import threading

__version__ = "2.4.0"

app = FastAPI(title="HMS - Hybrid Memory Server", version=__version__)

# Global instances
db_path = Path(__file__).parent / "memory.db"
db = None
embedder = None
search_engine = None
indexer = None
watcher_thread = None


def _reload_search_engine():
    """Reload the search engine with fresh DB connection (picks up new HNSW data)."""
    global db, search_engine
    if db:
        db.close()
    db = MemoryDatabase(db_path)
    search_engine = HybridSearch(db, embedder)


@app.on_event("startup")
async def startup():
    """Initialize on startup."""
    global db, embedder, search_engine, indexer, watcher_thread
    
    print(f"Starting HMS v{__version__}...")
    db = MemoryDatabase(db_path)
    embedder = Embedder()
    search_engine = HybridSearch(db, embedder)
    indexer = MemoryIndexer(db_path)
    
    # Warmup: run a dummy query to pre-load model weights into cache
    try:
        _ = embedder.embed("warmup query")
    except Exception:
        pass
    
    print(f"HMS v{__version__} ready — {db.get_stats()['total_chunks']} chunks indexed")
    
    # Auto-start watcher if WATCH_PATHS is set
    watch_paths = os.environ.get("HMS_WATCH_PATHS", "")
    if watch_paths:
        paths = [p.strip() for p in watch_paths.split(",") if p.strip()]
        if paths:
            from watcher import start_watcher_background
            watcher_thread = start_watcher_background(paths, db_path=str(db_path))
            print(f"Watcher started for: {paths}")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global db
    if db:
        db.close()
    print("Memory System stopped")


class SearchRequest(BaseModel):
    """Search request."""
    query: str
    max_results: Optional[int] = 10
    min_score: Optional[float] = 0.0
    semantic_weight: Optional[float] = 0.6
    keyword_weight: Optional[float] = 0.4


class SearchResult(BaseModel):
    """Search result."""
    file_path: str
    line_start: int
    line_end: int
    text: str
    score: float
    semantic_score: float
    keyword_score: float
    chunk_type: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response."""
    query: str
    results: list[SearchResult]
    total: int


class IndexRequest(BaseModel):
    """Index request."""
    directory: str
    pattern: Optional[str] = "**/*.md"
    force: Optional[bool] = False


class StatsResponse(BaseModel):
    """Stats response."""
    total_chunks: int
    total_files: int
    total_entities: int = 0
    db_size_mb: float
    hnsw_size_mb: float = 0
    total_size_mb: float = 0
    hnsw_count: int = 0


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search memory.
    
    Args:
        request: Search request
        
    Returns:
        Search results
    """
    try:
        results = search_engine.search(
            query=request.query,
            max_results=request.max_results,
            min_score=request.min_score
        )
        
        # Convert to response format
        search_results = [
            SearchResult(
                file_path=r['file_path'],
                line_start=r['line_start'],
                line_end=r['line_end'],
                text=r['text'],
                score=r['combined_score'],
                semantic_score=r['semantic_score'],
                keyword_score=r['keyword_score'],
                chunk_type=r.get('chunk_type', None)
            )
            for r in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=search_results,
            total=len(search_results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index(request: IndexRequest):
    """
    Index a directory.
    
    Args:
        request: Index request
        
    Returns:
        Success message with stats
    """
    try:
        dir_path = Path(request.directory)
        
        if not dir_path.exists():
            raise HTTPException(status_code=404, detail=f"Directory not found: {dir_path}")
        
        indexer.index_directory(dir_path, request.pattern, request.force)
        stats = indexer.get_stats()
        
        # Reload search engine to pick up new HNSW data
        _reload_search_engine()
        
        return {
            "message": "Indexing complete",
            "stats": stats
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get database statistics.
    
    Returns:
        Database stats
    """
    try:
        stats = db.get_stats()
        return StatsResponse(**stats)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": __version__}


@app.get("/version")
async def version():
    """Version endpoint."""
    return {"version": __version__, "name": "HMS - Hybrid Memory Server"}


def run_server(host: str = "0.0.0.0", port: int = 8765):
    """
    Run the server.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
