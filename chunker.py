"""
Markdown-aware text chunker with overlapping windows.
v2: Adds sentence overlap between chunks to prevent fact splitting at boundaries.
"""
from pathlib import Path
from typing import List, Dict, Optional
import re
import hashlib


class Chunker:
    """Smart markdown-aware chunker with overlap and metadata tagging."""
    
    def __init__(self, max_chunk_size: int = 800, min_chunk_size: int = 100, overlap_sentences: int = 2):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_sentences = overlap_sentences  # NEW: sentences to overlap between chunks
        
        self.fact_patterns = [
            re.compile(r'[-•]\s*\w+.*?:\s*.+'),
            re.compile(r'[-•]\s*\w+.*?\bis\b\s+.+'),
            re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
            re.compile(r'https?://\S+'),
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ]
        self.config_patterns = [
            re.compile(r'(?:password|token|key|secret|credential|login|port|host|ip)\s*[:=]', re.I),
            re.compile(r'ssh\s+\w+@', re.I),
            re.compile(r'(?:API|SSH|URL|IP)\s*:', re.I),
        ]
        self.todo_patterns = [
            re.compile(r'[-•]\s*\[[ x]\]', re.I),
            re.compile(r'\b(?:TODO|FIXME|HACK|XXX)\b', re.I),
        ]
        # Sentence boundary detection
        self._sentence_re = re.compile(r'(?<=[.!?])\s+(?=[A-Z])|(?<=\n)\s*(?=[-•*]|\d+\.)')
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (rough but effective)."""
        # Split on sentence boundaries
        parts = self._sentence_re.split(text)
        # Also split on newlines for bullet points
        result = []
        for part in parts:
            lines = part.split('\n')
            for line in lines:
                stripped = line.strip()
                if stripped:
                    result.append(stripped)
        return result if result else [text]
    
    def chunk_file(self, file_path: Path) -> List[Dict]:
        """Chunk a file by markdown sections with overlap."""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = file_path.read_text(encoding='utf-8', errors='ignore')
        if not content.strip():
            return []
        
        is_transcript = 'transcript' in str(file_path).lower()
        if is_transcript:
            old_max = self.max_chunk_size
            self.max_chunk_size = 2000
        
        lines = content.split('\n')
        sections = self._split_by_sections(lines)
        chunks = []
        
        for section in sections:
            section_chunks = self._process_section(section, str(file_path))
            chunks.extend(section_chunks)
        
        if is_transcript:
            self.max_chunk_size = old_max
            for chunk in chunks:
                chunk.setdefault('metadata', {})['source'] = 'transcript'
        
        return chunks
    
    def _split_by_sections(self, lines: List[str]) -> List[Dict]:
        """Split lines into sections based on markdown headings."""
        sections = []
        current_section = {
            'heading': None,
            'heading_level': 0,
            'lines': [],
            'start_line': 1,
        }
        
        for i, line in enumerate(lines, start=1):
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                if current_section['lines'] or current_section['heading']:
                    sections.append(current_section)
                
                current_section = {
                    'heading': heading_match.group(2).strip(),
                    'heading_level': len(heading_match.group(1)),
                    'lines': [line],
                    'start_line': i,
                }
            else:
                current_section['lines'].append(line)
        
        if current_section['lines'] or current_section['heading']:
            sections.append(current_section)
        
        return sections
    
    def _process_section(self, section: Dict, file_path: str) -> List[Dict]:
        """Process a section into one or more chunks with overlap."""
        text = '\n'.join(section['lines']).strip()
        if not text:
            return []
        
        start_line = section['start_line']
        end_line = start_line + len(section['lines']) - 1
        heading = section['heading'] or ''
        
        from pathlib import Path as P
        fname = P(file_path).stem
        if heading and heading != fname:
            context_prefix = f"[{fname} > {heading}] "
        else:
            context_prefix = f"[{fname}] "
        
        text = context_prefix + text
        
        if len(text) <= self.max_chunk_size:
            chunk_type = self._detect_type(text)
            content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            
            return [{
                'text': text,
                'line_start': start_line,
                'line_end': end_line,
                'file_path': file_path,
                'chars': len(text),
                'metadata': {
                    'heading': heading,
                    'type': chunk_type,
                    'hash': content_hash,
                }
            }]
        
        return self._split_large_section_with_overlap(section, file_path)
    
    def _split_large_section_with_overlap(self, section: Dict, file_path: str) -> List[Dict]:
        """Split large section at paragraph boundaries WITH sentence overlap."""
        lines = section['lines']
        heading = section['heading'] or ''
        start_line = section['start_line']
        
        from pathlib import Path as P
        fname = P(file_path).stem
        
        # First, split into raw chunks at paragraph boundaries
        raw_chunks = []
        current_lines = []
        current_chars = 0
        chunk_start = start_line
        
        for i, line in enumerate(lines):
            line_len = len(line) + 1
            is_break = line.strip() == ''
            
            if current_chars + line_len > self.max_chunk_size and is_break and current_lines:
                raw_chunks.append({
                    'lines': current_lines[:],
                    'start': chunk_start,
                    'end': start_line + i - 1,
                })
                current_lines = []
                current_chars = 0
                chunk_start = start_line + i + 1
            else:
                current_lines.append(line)
                current_chars += line_len
        
        if current_lines:
            raw_chunks.append({
                'lines': current_lines[:],
                'start': chunk_start,
                'end': start_line + len(lines) - 1,
            })
        
        # Now build final chunks with overlap
        chunks = []
        prev_tail_sentences = []  # Last N sentences from previous chunk
        
        for idx, raw in enumerate(raw_chunks):
            text = '\n'.join(raw['lines']).strip()
            if not text:
                continue
            
            # Prepend overlap from previous chunk
            if prev_tail_sentences and idx > 0:
                overlap_text = ' '.join(prev_tail_sentences)
                text = f"[...overlap...] {overlap_text}\n{text}"
            
            # Add heading context
            if idx > 0 and heading:
                text = f"[{fname} > {heading}] [Section: {heading}]\n{text}"
            else:
                if heading and heading != fname:
                    text = f"[{fname} > {heading}] {text}"
                else:
                    text = f"[{fname}] {text}"
            
            chunk_type = self._detect_type(text)
            content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
            
            chunks.append({
                'text': text,
                'line_start': raw['start'],
                'line_end': raw['end'],
                'file_path': file_path,
                'chars': len(text),
                'metadata': {
                    'heading': heading,
                    'type': chunk_type,
                    'hash': content_hash,
                    'has_overlap': idx > 0 and bool(prev_tail_sentences),
                }
            })
            
            # Extract tail sentences for next chunk's overlap
            sentences = self._split_sentences('\n'.join(raw['lines']).strip())
            prev_tail_sentences = sentences[-self.overlap_sentences:] if len(sentences) > self.overlap_sentences else sentences
        
        return chunks
    
    def _detect_type(self, text: str) -> str:
        """Detect chunk type: fact, config, todo, or narrative."""
        lines = text.split('\n')
        
        config_score = sum(1 for p in self.config_patterns for l in lines if p.search(l))
        todo_score = sum(1 for p in self.todo_patterns for l in lines if p.search(l))
        fact_score = sum(1 for p in self.fact_patterns for l in lines if p.search(l))
        
        if config_score >= 2:
            return 'config'
        if todo_score >= 1:
            return 'todo'
        if fact_score >= 3 or (fact_score >= 1 and len(text) < 300):
            return 'fact'
        return 'narrative'
    
    def chunk_directory(self, dir_path: Path, pattern: str = "**/*.md") -> List[Dict]:
        """Chunk all files in a directory matching pattern."""
        all_chunks = []
        seen_hashes = set()
        
        for file_path in sorted(dir_path.glob(pattern)):
            if file_path.is_file():
                try:
                    chunks = self.chunk_file(file_path)
                    for chunk in chunks:
                        h = chunk.get('metadata', {}).get('hash', '')
                        if h and h in seen_hashes:
                            continue
                        seen_hashes.add(h)
                        all_chunks.append(chunk)
                except Exception as e:
                    print(f"Error chunking {file_path}: {e}")
        
        return all_chunks
