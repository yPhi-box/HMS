"""
HMS v2.4.1 — Transcript Watcher
Watches OpenClaw session transcript directory for new/modified JSONL files.
Indexes conversations in real time using incremental parsing.
"""
import time
import json
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from transcript_parser import TranscriptParser
from indexer import MemoryIndexer


# Track byte positions for incremental parsing
POSITION_FILE = "transcript_positions.json"


class TranscriptHandler(FileSystemEventHandler):
    """Watch JSONL transcript files and index new content."""
    
    def __init__(self, db_path: Path, positions_path: Path):
        self.db_path = db_path
        self.positions_path = positions_path
        self.parser = TranscriptParser()
        self._indexer = None
        self.positions = self._load_positions()
        self.pending = set()
        self.last_process = 0
    
    def _load_positions(self) -> dict:
        """Load saved byte positions for incremental parsing."""
        if self.positions_path.exists():
            try:
                return json.loads(self.positions_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        return {}
    
    def _save_positions(self):
        """Save byte positions to disk."""
        try:
            self.positions_path.write_text(json.dumps(self.positions, indent=2))
        except OSError as e:
            print(f"Warning: Could not save positions: {e}")
    
    def _get_indexer(self):
        """Get or create persistent indexer."""
        if self._indexer is None:
            self._indexer = MemoryIndexer(self.db_path)
        return self._indexer
    
    def on_modified(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_active_transcript(path):
            self.pending.add(path)
            self._process_pending()
    
    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if self._is_active_transcript(path):
            self.pending.add(path)
            self._process_pending()
    
    def _is_active_transcript(self, path: Path) -> bool:
        """Check if file is an active session transcript (not reset/deleted)."""
        name = path.name
        # Only index .jsonl files (active sessions)
        # Skip .reset. and .deleted. files — those are archived
        if not name.endswith('.jsonl'):
            return False
        if '.reset.' in name or '.deleted.' in name:
            return False
        return True
    
    def _process_pending(self):
        """Process pending transcript files with debouncing."""
        now = time.time()
        if now - self.last_process < 3:
            return
        
        if not self.pending:
            return
        
        files = list(self.pending)
        self.pending.clear()
        self.last_process = now
        
        indexer = self._get_indexer()
        
        for file_path in files:
            if not file_path.exists():
                continue
            try:
                self._index_transcript(file_path, indexer)
            except Exception as e:
                print(f"Error indexing transcript {file_path.name}: {e}")
    
    def _index_transcript(self, file_path: Path, indexer: MemoryIndexer):
        """Index new content from a transcript file."""
        key = str(file_path)
        from_byte = self.positions.get(key, 0)
        
        chunks, new_position = self.parser.parse_incremental(file_path, from_byte)
        
        if not chunks:
            self.positions[key] = new_position
            return
        
        print(f"Transcript update: {file_path.name} — {len(chunks)} new conversation chunks")
        
        # Embed and index
        texts = [c['text'] for c in chunks]
        embeddings = indexer.embedder.embed_batch(texts)
        indexer.db.add_chunks_batch(chunks, embeddings)
        
        # Entity extraction
        entity_count = 0
        cursor = indexer.db.conn.cursor()
        recent_ids = cursor.execute(
            "SELECT id, text FROM chunks WHERE file_path = ? ORDER BY id DESC LIMIT ?",
            (str(file_path), len(chunks))
        ).fetchall()
        
        for chunk_id, text in recent_ids:
            entities = indexer.extractor.extract(text, str(file_path))
            for e in entities:
                indexer.db.add_entity(
                    chunk_id, e['entity_type'], e['entity_name'],
                    e['entity_value'], str(file_path)
                )
                entity_count += 1
        
        indexer.db.save()
        
        # Update position
        self.positions[key] = new_position
        self._save_positions()
        
        print(f"  Indexed {len(chunks)} chunks, {entity_count} entities from transcript")
    
    def index_existing_transcripts(self, sessions_dir: Path):
        """
        Index all existing active transcripts that haven't been fully indexed.
        Called on startup to catch up on anything missed.
        """
        if not sessions_dir.exists():
            return
        
        indexer = self._get_indexer()
        count = 0
        
        for f in sorted(sessions_dir.glob('*.jsonl')):
            if '.reset.' in f.name or '.deleted.' in f.name:
                continue
            
            key = str(f)
            from_byte = self.positions.get(key, 0)
            current_size = f.stat().st_size
            
            if current_size > from_byte:
                try:
                    self._index_transcript(f, indexer)
                    count += 1
                except Exception as e:
                    print(f"Error catching up on {f.name}: {e}")
        
        if count:
            print(f"Caught up on {count} transcript files")
    
    def index_archived_transcripts(self, sessions_dir: Path, days_back: int = 7):
        """
        Index recent archived transcripts (.reset. files) for historical coverage.
        Only indexes files modified within days_back days.
        """
        if not sessions_dir.exists():
            return
        
        import time as _time
        cutoff = _time.time() - (days_back * 86400)
        indexer = self._get_indexer()
        count = 0
        
        for f in sorted(sessions_dir.iterdir()):
            if not (f.name.endswith('.jsonl.reset.' + f.name.split('.reset.')[-1]) if '.reset.' in f.name else False):
                # More robust: check for .reset. in name
                if '.reset.' not in f.name:
                    continue
            
            if f.stat().st_mtime < cutoff:
                continue
            
            key = str(f)
            if key in self.positions:
                continue  # Already indexed
            
            try:
                # Full parse (not incremental) for archived files
                chunks = self.parser.parse_transcript(f)
                if chunks:
                    texts = [c['text'] for c in chunks]
                    embeddings = indexer.embedder.embed_batch(texts)
                    indexer.db.add_chunks_batch(chunks, embeddings)
                    
                    # Entity extraction
                    cursor = indexer.db.conn.cursor()
                    recent_ids = cursor.execute(
                        "SELECT id, text FROM chunks WHERE file_path = ? ORDER BY id DESC LIMIT ?",
                        (str(f), len(chunks))
                    ).fetchall()
                    for chunk_id, text in recent_ids:
                        entities = indexer.extractor.extract(text, str(f))
                        for e in entities:
                            indexer.db.add_entity(
                                chunk_id, e['entity_type'], e['entity_name'],
                                e['entity_value'], str(f)
                            )
                    
                    indexer.db.save()
                    self.positions[key] = f.stat().st_size
                    count += 1
                    print(f"  Archived transcript: {f.name} — {len(chunks)} chunks")
            except Exception as e:
                print(f"Error indexing archived {f.name}: {e}")
        
        if count:
            self._save_positions()
            print(f"Indexed {count} archived transcripts")


def start_transcript_watcher(sessions_dir: str, db_path: str = None, 
                              catch_up: bool = True, archive_days: int = 7):
    """
    Start watching a transcript directory.
    
    Args:
        sessions_dir: Path to OpenClaw sessions directory
        db_path: Path to HMS database
        catch_up: Whether to index existing transcripts on startup
        archive_days: How many days back to index archived transcripts
    """
    sessions_path = Path(sessions_dir)
    if not sessions_path.exists():
        print(f"Sessions directory not found: {sessions_dir}")
        return None
    
    if db_path is None:
        db_path = Path(__file__).parent / 'memory.db'
    else:
        db_path = Path(db_path)
    
    positions_path = db_path.parent / POSITION_FILE
    
    handler = TranscriptHandler(db_path, positions_path)
    
    # Catch up on existing content
    if catch_up:
        print("Catching up on existing transcripts...")
        handler.index_existing_transcripts(sessions_path)
        if archive_days > 0:
            print(f"Indexing archived transcripts (last {archive_days} days)...")
            handler.index_archived_transcripts(sessions_path, archive_days)
    
    # Start watching
    observer = Observer()
    observer.schedule(handler, str(sessions_path), recursive=False)
    observer.start()
    print(f"Transcript watcher started on {sessions_path}")
    
    return observer, handler


def start_transcript_watcher_background(sessions_dir: str, db_path: str = None,
                                         catch_up: bool = True, archive_days: int = 7):
    """Start transcript watcher in background thread."""
    import threading
    
    result = {'observer': None, 'handler': None}
    
    def _run():
        obs_handler = start_transcript_watcher(
            sessions_dir, db_path, catch_up, archive_days
        )
        if obs_handler:
            result['observer'], result['handler'] = obs_handler
            try:
                while True:
                    time.sleep(1)
                    if result['handler'].pending:
                        result['handler']._process_pending()
            except Exception:
                pass
    
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, result
