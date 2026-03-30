"""
HMS v2.4.1 — Transcript Parser
Parses OpenClaw JSONL session transcripts into searchable chunks.
Extracts user and assistant text, skips tool calls/results/system messages.
Groups messages into conversation blocks by time proximity.
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import hashlib


class TranscriptParser:
    """Parse OpenClaw JSONL session transcripts into indexable chunks."""
    
    # Max gap between messages before starting a new conversation block
    CONVERSATION_GAP_MINUTES = 30
    # Max chunk size in characters
    MAX_CHUNK_SIZE = 1200
    # Min chunk size — don't create tiny fragments
    MIN_CHUNK_SIZE = 50
    
    def __init__(self):
        # Pattern to strip OpenClaw metadata envelopes from user messages
        self._metadata_re = re.compile(
            r'Conversation info \(untrusted metadata\):\s*```json\s*\{[^}]*\}\s*```\s*'
            r'(?:Sender \(untrusted metadata\):\s*```json\s*\{[^}]*\}\s*```\s*)?',
            re.DOTALL
        )
        # Extract sender name from metadata
        self._sender_re = re.compile(r'"sender":\s*"([^"]+)"')
        # Extract timestamp
        self._timestamp_re = re.compile(r'"timestamp":\s*"([^"]+)"')
    
    def parse_transcript(self, file_path: Path) -> List[Dict]:
        """
        Parse a JSONL transcript file into conversation chunks.
        
        Returns list of chunk dicts ready for indexing:
        {
            'text': str,
            'file_path': str,
            'line_start': int,
            'line_end': int,
            'chars': int,
            'metadata': {
                'source': 'transcript',
                'session_id': str,
                'timestamp_start': str,
                'timestamp_end': str,
                'type': 'conversation',
            }
        }
        """
        if not file_path.exists():
            return []
        
        messages = self._extract_messages(file_path)
        if not messages:
            return []
        
        # Group into conversation blocks
        blocks = self._group_into_blocks(messages)
        
        # Convert blocks to chunks
        chunks = []
        session_id = file_path.stem.split('.')[0]
        
        for block in blocks:
            block_chunks = self._block_to_chunks(block, str(file_path), session_id)
            chunks.extend(block_chunks)
        
        return chunks
    
    def _extract_messages(self, file_path: Path) -> List[Dict]:
        """Extract user and assistant messages from JSONL."""
        messages = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                msg = obj.get('message', {})
                role = msg.get('role', '')
                timestamp = obj.get('timestamp', '')
                
                # Only user and assistant messages
                if role not in ('user', 'assistant'):
                    continue
                
                # Extract text content
                text = self._extract_text(msg)
                if not text or len(text.strip()) < 10:
                    continue
                
                # For user messages, strip metadata envelope and get clean text
                sender = None
                if role == 'user':
                    sender_match = self._sender_re.search(text)
                    if sender_match:
                        sender = sender_match.group(1)
                    text = self._metadata_re.sub('', text).strip()
                    # Skip if nothing left after stripping metadata
                    if not text or len(text.strip()) < 5:
                        continue
                
                # Skip heartbeat prompts
                if 'Read HEARTBEAT.md' in text and 'HEARTBEAT_OK' in text:
                    continue
                if text.strip() == 'HEARTBEAT_OK':
                    continue
                
                messages.append({
                    'role': role,
                    'text': text,
                    'sender': sender,
                    'timestamp': timestamp,
                    'line_num': line_num,
                })
        
        return messages
    
    def _extract_text(self, msg: dict) -> str:
        """Extract text content from a message, handling string and array formats."""
        content = msg.get('content', '')
        
        if isinstance(content, str):
            return content
        
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get('type') == 'text':
                        texts.append(item.get('text', ''))
                    # Skip toolCall, toolResult, etc.
            return '\n'.join(texts)
        
        return ''
    
    def _group_into_blocks(self, messages: List[Dict]) -> List[List[Dict]]:
        """Group messages into conversation blocks based on time gaps."""
        if not messages:
            return []
        
        blocks = []
        current_block = [messages[0]]
        
        for msg in messages[1:]:
            # Check time gap
            prev_ts = self._parse_timestamp(current_block[-1]['timestamp'])
            curr_ts = self._parse_timestamp(msg['timestamp'])
            
            if prev_ts and curr_ts:
                gap = (curr_ts - prev_ts).total_seconds() / 60
                if gap > self.CONVERSATION_GAP_MINUTES:
                    blocks.append(current_block)
                    current_block = [msg]
                    continue
            
            current_block.append(msg)
        
        if current_block:
            blocks.append(current_block)
        
        return blocks
    
    def _parse_timestamp(self, ts: str) -> Optional[datetime]:
        """Parse ISO timestamp."""
        if not ts:
            return None
        try:
            # Handle various formats
            ts = ts.replace('Z', '+00:00')
            if '.' in ts:
                ts = ts.split('.')[0] + '+00:00' if '+' not in ts.split('.')[-1] else ts
            return datetime.fromisoformat(ts.split('.')[0])
        except (ValueError, IndexError):
            return None
    
    def _block_to_chunks(self, block: List[Dict], file_path: str, session_id: str) -> List[Dict]:
        """Convert a conversation block into one or more chunks."""
        if not block:
            return []
        
        # Format messages as conversation text
        lines = []
        for msg in block:
            role_label = msg['sender'] or msg['role'].capitalize()
            # Truncate very long assistant responses (tool output dumps, etc.)
            text = msg['text']
            if len(text) > 2000:
                text = text[:2000] + '... [truncated]'
            lines.append(f"**{role_label}:** {text}")
        
        full_text = '\n\n'.join(lines)
        
        # Get time range
        ts_start = block[0]['timestamp']
        ts_end = block[-1]['timestamp']
        line_start = block[0]['line_num']
        line_end = block[-1]['line_num']
        
        # Format date context
        dt = self._parse_timestamp(ts_start)
        date_prefix = ''
        if dt:
            date_str = dt.strftime('%Y-%m-%d')
            human_date = dt.strftime('%A, %B %d, %Y')
            time_str = dt.strftime('%H:%M UTC')
            date_prefix = f"Date: {human_date} ({date_str}) at {time_str}\nSession: {session_id[:8]}\n\n"
        
        # Split if too large
        if len(date_prefix + full_text) <= self.MAX_CHUNK_SIZE:
            chunk_text = date_prefix + full_text
            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            return [{
                'text': chunk_text,
                'file_path': file_path,
                'line_start': line_start,
                'line_end': line_end,
                'chars': len(chunk_text),
                'metadata': {
                    'source': 'transcript',
                    'session_id': session_id,
                    'timestamp_start': ts_start,
                    'timestamp_end': ts_end,
                    'type': 'conversation',
                    'hash': content_hash,
                }
            }]
        
        # Split into multiple chunks, keeping messages together
        chunks = []
        current_messages = []
        current_size = len(date_prefix)
        
        for i, msg in enumerate(block):
            role_label = msg['sender'] or msg['role'].capitalize()
            text = msg['text']
            if len(text) > 2000:
                text = text[:2000] + '... [truncated]'
            msg_text = f"**{role_label}:** {text}"
            msg_size = len(msg_text) + 2  # +2 for \n\n
            
            if current_size + msg_size > self.MAX_CHUNK_SIZE and current_messages:
                # Emit chunk
                chunk_text = date_prefix + '\n\n'.join(
                    f"**{m['sender'] or m['role'].capitalize()}:** {m['text'][:2000]}"
                    for m in current_messages
                )
                content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
                chunks.append({
                    'text': chunk_text,
                    'file_path': file_path,
                    'line_start': current_messages[0]['line_num'],
                    'line_end': current_messages[-1]['line_num'],
                    'chars': len(chunk_text),
                    'metadata': {
                        'source': 'transcript',
                        'session_id': session_id,
                        'timestamp_start': current_messages[0]['timestamp'],
                        'timestamp_end': current_messages[-1]['timestamp'],
                        'type': 'conversation',
                        'hash': content_hash,
                    }
                })
                current_messages = []
                current_size = len(date_prefix)
            
            current_messages.append(msg)
            current_size += msg_size
        
        # Final chunk
        if current_messages:
            chunk_text = date_prefix + '\n\n'.join(
                f"**{m['sender'] or m['role'].capitalize()}:** {m['text'][:2000]}"
                for m in current_messages
            )
            content_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:12]
            chunks.append({
                'text': chunk_text,
                'file_path': file_path,
                'line_start': current_messages[0]['line_num'],
                'line_end': current_messages[-1]['line_num'],
                'chars': len(chunk_text),
                'metadata': {
                    'source': 'transcript',
                    'session_id': session_id,
                    'timestamp_start': current_messages[0]['timestamp'],
                    'timestamp_end': current_messages[-1]['timestamp'],
                    'type': 'conversation',
                    'hash': content_hash,
                }
            })
        
        return chunks
    
    def get_file_position(self, file_path: Path) -> int:
        """Get the byte size of a file (for tracking incremental reads)."""
        if file_path.exists():
            return file_path.stat().st_size
        return 0
    
    def parse_incremental(self, file_path: Path, from_byte: int = 0) -> tuple:
        """
        Parse new content from a transcript file starting at byte offset.
        Returns (chunks, new_byte_position).
        Used for real-time indexing of growing session files.
        """
        if not file_path.exists():
            return [], 0
        
        current_size = file_path.stat().st_size
        if current_size <= from_byte:
            return [], from_byte
        
        # Read new lines
        messages = []
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            f.seek(from_byte)
            remaining = f.read()
            line_offset = sum(1 for _ in open(file_path, 'r', encoding='utf-8', errors='ignore')) - remaining.count('\n')
            
            for i, line in enumerate(remaining.split('\n')):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                msg = obj.get('message', {})
                role = msg.get('role', '')
                timestamp = obj.get('timestamp', '')
                
                if role not in ('user', 'assistant'):
                    continue
                
                text = self._extract_text(msg)
                if not text or len(text.strip()) < 10:
                    continue
                
                sender = None
                if role == 'user':
                    sender_match = self._sender_re.search(text)
                    if sender_match:
                        sender = sender_match.group(1)
                    text = self._metadata_re.sub('', text).strip()
                    if not text or len(text.strip()) < 5:
                        continue
                
                if 'Read HEARTBEAT.md' in text:
                    continue
                if text.strip() == 'HEARTBEAT_OK':
                    continue
                
                messages.append({
                    'role': role,
                    'text': text,
                    'sender': sender,
                    'timestamp': timestamp,
                    'line_num': line_offset + i,
                })
        
        if not messages:
            return [], current_size
        
        blocks = self._group_into_blocks(messages)
        session_id = file_path.stem.split('.')[0]
        
        chunks = []
        for block in blocks:
            block_chunks = self._block_to_chunks(block, str(file_path), session_id)
            chunks.extend(block_chunks)
        
        return chunks, current_size
