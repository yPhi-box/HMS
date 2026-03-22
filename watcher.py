"""
File watcher for automatic memory reindexing.
Monitors memory directories and reindexes on changes.
"""

import time
import sys
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from indexer import MemoryIndexer


class MemoryFileHandler(FileSystemEventHandler):
    """Handler for file system events on memory files."""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.pending = set()  # Debounce rapid changes
        self.last_process = 0
        # Keep a persistent indexer to avoid reloading the embedding model on every change
        self._indexer = None
        
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only index markdown files
        if file_path.suffix not in ['.md', '.txt']:
            return
        
        # Skip temp files
        if file_path.name.startswith('.') or file_path.name.startswith('~'):
            return
        
        print(f"File modified: {file_path}")
        self.pending.add(file_path)
        self._process_pending()
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if file_path.suffix not in ['.md', '.txt']:
            return
        
        if file_path.name.startswith('.') or file_path.name.startswith('~'):
            return
        
        print(f"File created: {file_path}")
        self.pending.add(file_path)
        self._process_pending()
    
    def on_deleted(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        if file_path.suffix not in ['.md', '.txt']:
            return
        
        print(f"File deleted: {file_path}")
        indexer = self._get_indexer()
        indexer.db.clear_file(str(file_path))
        stats = indexer.get_stats()
        print(f"  File removed from index. Total chunks: {stats['total_chunks']}")
    
    def _process_pending(self):
        """Process pending files with debouncing."""
        now = time.time()
        
        # Wait 2 seconds after last change before processing
        if now - self.last_process < 2:
            return
        
        if not self.pending:
            return
        
        files_to_process = list(self.pending)
        self.pending.clear()
        self.last_process = now
        
        indexer = self._get_indexer()
        
        for file_path in files_to_process:
            if file_path.exists():
                try:
                    indexer.reindex_file(file_path)
                except Exception as e:
                    print(f"Error reindexing {file_path}: {e}")
    
    def _get_indexer(self):
        """Get or create persistent indexer (avoids reloading model every time)."""
        if self._indexer is None:
            self._indexer = MemoryIndexer(self.db_path)
        return self._indexer


def start_watcher(watch_paths: list, db_path: str = None):
    """
    Start file watcher on specified paths.
    
    Args:
        watch_paths: List of directories to watch
        db_path: Path to database
    """
    print("Starting memory file watcher...")
    print(f"Watching paths:")
    for path in watch_paths:
        print(f"  - {path}")
    
    if db_path is None:
        db_path = Path(__file__).parent / 'memory.db'
    else:
        db_path = Path(db_path)
    
    event_handler = MemoryFileHandler(db_path)
    observer = Observer()
    
    for watch_path in watch_paths:
        path = Path(watch_path)
        if path.exists():
            observer.schedule(event_handler, str(path), recursive=True)
            print(f"Monitoring: {path}")
        else:
            print(f"Warning: Path does not exist: {path}")
    
    observer.start()
    print("Watcher started. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
            # Process any pending files every second
            if event_handler.pending:
                event_handler._process_pending()
    except KeyboardInterrupt:
        print("\nStopping watcher...")
        observer.stop()
    
    observer.join()
    print("Watcher stopped.")


def start_watcher_background(watch_paths: list, db_path: str = None):
    """Start watcher in a background thread (for embedding in server process)."""
    import threading
    
    def _run():
        start_watcher(watch_paths, db_path=db_path)
    
    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


if __name__ == '__main__':
    # Default paths to watch — adjust to your OpenClaw workspace
    default_paths = [
        os.path.expanduser('~/.openclaw/workspace/memory'),
        os.path.expanduser('~/.openclaw/workspace'),
    ]
    
    if len(sys.argv) > 1:
        # Use provided paths
        watch_paths = sys.argv[1:]
    else:
        # Use defaults
        watch_paths = default_paths
    
    start_watcher(watch_paths)
