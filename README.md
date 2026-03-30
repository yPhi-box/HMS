# HMS v2.4.1 — Hybrid Memory Server

**Give your AI agent a perfect memory. Local. Private. Free.**

---

Your OpenClaw agent forgets everything between sessions. Every conversation starts from zero — or you burn through tokens loading entire files into context, hoping the right information is somewhere in there.

HMS fixes this. It's a local search engine that sits between your agent and its memory files, returning *only* the relevant chunks. No cloud APIs. No subscriptions. Sub-second results.

**Before HMS:** Your agent loads 50,000+ tokens of memory files into every conversation. Most of it irrelevant. You pay for all of it.

**After HMS:** Your agent asks "what's the server IP?" and gets back the 3 most relevant chunks — ~500 tokens — in under half a second. For free.

---

## 🆕 What's New in v2.4.1

### Real-Time Transcript Indexing

**The problem:** HMS previously only indexed workspace files (markdown, text). If your agent didn't write a daily memory file, the conversation was lost to search forever — even though OpenClaw saves session transcripts automatically.

**The fix:** HMS 2.4.1 watches your OpenClaw session transcripts directory and indexes conversations in real time. Every message you send and every response your agent gives becomes searchable immediately.

- **Automatic:** Set `HMS_SESSIONS_DIR` and it just works
- **Incremental:** Only parses new content as sessions grow (tracks byte positions)
- **Smart filtering:** Strips metadata envelopes, skips tool calls/results/heartbeats, groups messages into conversation blocks
- **Ranked correctly:** Curated workspace files still rank higher than raw transcripts — your organized notes are the "highlight reel," transcripts are the safety net
- **Catch-up on startup:** Indexes existing transcripts and recent archived sessions when the service starts

### Upgrading from v2.4.0

If you already have HMS installed:

```bash
# 1. Stop HMS
sudo systemctl stop hms

# 2. Back up your database (optional but recommended)
cp ~/hms/memory.db ~/hms/memory.db.bak

# 3. Pull latest code (or re-run installer)
cd ~/hms-release && git pull  # if cloned
# OR: download fresh and copy src/ files to ~/hms/

# 4. Copy new source files
cp hms-release/src/*.py ~/hms/

# 5. Add transcript watcher to service
sudo sed -i '/^Environment=HMS_PORT/a Environment=HMS_SESSIONS_DIR=/home/YOUR_USER/.openclaw/agents/main/sessions\nEnvironment=HMS_TRANSCRIPT_ARCHIVE_DAYS=7' /etc/systemd/system/hms.service

# 6. Restart
sudo systemctl daemon-reload
sudo systemctl restart hms
```

Verify: `curl http://localhost:8765/health` should return `{"status":"ok","version":"2.4.1"}`

### Fresh Install

One command:

```bash
git clone https://github.com/yPhi-box/HMS.git && cd HMS && bash install.sh --auto
```

Or without cloning:

```bash
curl -sSL https://raw.githubusercontent.com/yPhi-box/HMS/main/install.sh | bash
```

---

## What Makes It Different

Most memory systems do one thing: vector search. They embed your query, find similar text, and call it a day. That works for simple lookups but falls apart when questions get real.

HMS uses a **three-stage pipeline** that combines multiple search strategies, then has a second AI model verify the results:

### Stage 1: Triple Search

Every query hits three search engines simultaneously:

| Engine | What It Does | Why It Matters |
|--------|-------------|----------------|
| **Semantic (HNSW)** | Finds text with similar *meaning*, even if different words are used | "What's the network address?" matches "IP: 10.0.0.1" |
| **Keyword (BM25)** | Classic full-text search — exact word matching with relevance scoring | "SSH password" finds the line with those exact words |
| **Entity Lookup** | Pattern-matched extraction of IPs, names, credentials, dates, emails | "server IP" goes straight to the entity index |

No single method handles every query well. Together, they cover each other's blind spots.

### Stage 2: Smart Scoring

Raw results from all three engines get fused into a single score. HMS applies **seven scoring signals**:

1. **Intent Detection** — Classifies queries into 13 categories (credential lookup, relationship question, config query, etc.) and boosts matching chunks
2. **Synonym Expansion** — "kids" → "children/son/daughter". "SSH" → "secure shell/remote access"
3. **Source Tiering** — Curated files outrank transcripts. Best-organized info surfaces first
4. **Type Awareness** — Chunks tagged as `config`, `fact`, `narrative`, `conversation`, or `todo`. Credential queries prefer config chunks
5. **Temporal Decay** — Recent info scores higher. Half-life: 30 days
6. **Date Routing** — "What happened March 15?" targets files with that date
7. **Adjacency Boost** — Top-scoring chunks pull in their neighbors for context

### Stage 3: Cross-Encoder Reranking

Top candidates get re-scored by a cross-encoder model (ms-marco-MiniLM-L-6-v2). This catches false positives and promotes results that are genuinely relevant to the query — not just textually similar.

---

## Architecture

```
                    ┌─────────────────┐
                    │   Your Agent    │
                    └────────┬────────┘
                             │ memory_search("query")
                    ┌────────▼────────┐
                    │  OpenClaw Plugin │
                    └────────┬────────┘
                             │ HTTP POST /search
                    ┌────────▼────────┐
                    │   HMS Server    │
                    │   (port 8765)   │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────▼──────┐ ┌────▼─────┐ ┌──────▼──────┐
     │ Semantic HNSW │ │ FTS5 BM25│ │Entity Index │
     │  (384-dim)    │ │(keywords)│ │ (patterns)  │
     └───────────────┘ └──────────┘ └─────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │  Smart Scoring  │
                    │  + Reranking    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Top-N Results  │
                    └─────────────────┘

     File Watchers (real-time):
     ┌──────────────────┐  ┌────────────────────┐
     │ Workspace Watcher │  │ Transcript Watcher  │
     │ (.md, .txt files) │  │ (.jsonl sessions)   │
     └──────────────────┘  └────────────────────┘
```

---

## Benchmark Results

Tested with IRONMAN benchmark (13,763 messages, 688 queries, 20 categories):

### v2.4.1 Transcript Indexing Tests

| Test Suite | Queries | Pass Rate |
|-----------|---------|-----------|
| Core functionality | 15 | **100%** |
| Edge cases (typos, vague, multi-hop, negatives) | 20 | **100%** |
| **Total** | **35** | **100%** |

Highlights:
- Misspelled queries ("Huntr IP adress") → found correct results
- Vague queries ("the thing about the keys") → found SSH key discussions
- Exact quote search ("bad data bad tests") → **1.000 score**
- Irrelevant queries correctly returned 0 results
- Natural language recall ("we talked about test data not being realistic") → 0.893 score

### Performance

| Metric | Value |
|--------|-------|
| Avg search latency | 35-50ms |
| Embedding model | all-MiniLM-L6-v2 (384-dim) |
| RAM at runtime | ~400MB |
| Disk (model + venv) | ~1.5GB |
| API cost | **$0/month** |

---

## Installation

### Prerequisites

- **OS:** Ubuntu 22.04+ / Debian 12+ (any Linux with systemd)
- **Python:** 3.10+
- **Disk:** ~1.5GB free
- **RAM:** ~400MB at runtime
- **OpenClaw:** 2026.2.1 or later

### One-Liner Install

```bash
git clone https://github.com/yPhi-box/HMS.git && cd HMS && bash install.sh --auto
```

### Install Options

```bash
bash install.sh [options]

--dir PATH       Install directory (default: ~/hms)
--port PORT      Server port (default: 8765)
--watch PATH     Directory to watch (default: ~/.openclaw/workspace)
--sessions PATH  Session transcripts dir (default: auto-detect)
--no-service     Don't install systemd service
--no-plugin      Don't install OpenClaw plugin
--auto           Non-interactive mode
```

### What the Installer Does

1. Checks prerequisites (Python 3.10+, pip, venv, gcc)
2. Copies source files to install directory
3. Creates Python virtual environment
4. Installs dependencies
5. Downloads embedding model (~90MB, one-time)
6. Creates and starts systemd service
7. Enables transcript watcher (if sessions directory exists)
8. Installs OpenClaw plugin
9. Runs initial index of workspace
10. Installs HMS-RULES.md (memory search protocol for your agent)

### Uninstall

```bash
bash uninstall.sh
```

Removes: systemd service, install directory, all data. Does not touch your OpenClaw config.

---

## API Reference

### `GET /health`
Health check. Returns `{"status": "ok", "version": "2.4.1"}`

### `GET /stats`
Database statistics: chunk count, file count, entity count, DB size.

### `GET /version`
Version info.

### `POST /search`
Search memory.

```json
{
  "query": "what's the server IP?",
  "max_results": 10,
  "min_score": 0.0
}
```

Returns ranked results with file path, line numbers, text, and scores.

### `POST /index`
Index a directory on demand.

```json
{
  "directory": "/path/to/files",
  "pattern": "**/*.md",
  "force": false
}
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HMS_WATCH_PATHS` | `~/.openclaw/workspace` | Comma-separated directories to watch for file changes |
| `HMS_SESSIONS_DIR` | *(none)* | OpenClaw sessions directory for transcript indexing |
| `HMS_TRANSCRIPT_ARCHIVE_DAYS` | `7` | How many days of archived transcripts to index on startup |
| `HMS_PORT` | `8765` | Server port |

### OpenClaw Plugin Config

Add to `openclaw.json`:

```json
{
  "plugins": {
    "load": {
      "paths": ["/home/YOUR_USER/hms/openclaw-plugin"]
    },
    "slots": {
      "memory": "hms-memory"
    },
    "entries": {
      "hms-memory": {
        "enabled": true
      }
    }
  }
}
```

---

## Troubleshooting

### HMS not starting
```bash
sudo journalctl -u hms -f  # Check logs
sudo systemctl status hms   # Check service status
```

### Port already in use
```bash
ss -tlnp | grep 8765       # Find what's using the port
kill <PID>                   # Kill stale process
sudo systemctl restart hms
```

### Transcripts not being indexed
- Check `HMS_SESSIONS_DIR` is set in the service file
- Verify the directory exists and contains `.jsonl` files
- Check logs: `sudo journalctl -u hms --since "5 min ago" | grep transcript`

### Search returns no results
- Check stats: `curl http://localhost:8765/stats` — if 0 chunks, reindex
- Force reindex: `curl -X POST http://localhost:8765/index -H 'Content-Type: application/json' -d '{"directory": "~/.openclaw/workspace", "force": true}'`

### High memory usage
HMS uses ~400MB at runtime (embedding model + HNSW index). If it's significantly higher, restart the service to clear accumulated state.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| **2.4.1** | 2026-03-30 | Real-time transcript indexing, conversation chunk parsing, incremental JSONL parsing, archive catch-up on startup |
| 2.4.0 | 2026-03-22 | Cross-encoder reranking (ms-marco-MiniLM), 40/60 blended scoring, wider initial retrieval |
| 2.3.0 | 2026-03-21 | Temporal search, date-aware routing, chunk date context, steeper temporal decay |
| 2.2.0 | 2026-03-20 | Overlapping chunks, query expansion, adjacency boost, entity extraction |
| 2.1.0 | 2026-03-19 | FTS5 keyword search, hybrid scoring |
| 2.0.0 | 2026-03-19 | HNSW vector search, OpenClaw plugin |
| 1.0.0 | 2026-03-18 | Initial release |

---

## FAQ

**Q: Does HMS replace OpenClaw's built-in memory?**
A: Yes, it slots in as a drop-in replacement via the plugin system. OpenClaw's built-in memory requires a paid embedding API key (OpenAI/Google/Voyage). HMS uses local embeddings — zero cost.

**Q: Will uninstalling OpenClaw remove HMS?**
A: No. HMS is a standalone Python application with its own systemd service. It's completely independent. The plugin is just a bridge.

**Q: What if HMS goes down?**
A: Your agent falls back to OpenClaw's built-in memory behavior. HMS is an enhancement, not a dependency. The systemd service auto-restarts on crashes.

**Q: How much data can it handle?**
A: Tested with 13,763 messages / 688 queries. HNSW scales logarithmically. For 10,000+ files, allocate more RAM.

**Q: Does it index my conversations automatically?**
A: Yes, in v2.4.1+. Set `HMS_SESSIONS_DIR` to your OpenClaw sessions directory and every conversation is indexed in real time. Even if your agent never writes a memory file, the data is searchable.

---

## License

MIT
