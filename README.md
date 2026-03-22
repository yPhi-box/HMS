# HMS v2.4 — Hybrid Memory Server

**Give your AI agent a perfect memory. Local. Private. Free.**

---

Your OpenClaw agent forgets everything between sessions. Every conversation starts from zero — or you burn through tokens loading entire files into context, hoping the right information is somewhere in there.

HMS fixes this. It's a local search engine that sits between your agent and its memory files, returning *only* the relevant chunks. No cloud APIs. No subscriptions. Sub-second results.

**Before HMS:** Your agent loads 50,000+ tokens of memory files into every conversation. Most of it irrelevant. You pay for all of it.

**After HMS:** Your agent asks "what's the server IP?" and gets back the 3 most relevant chunks — ~500 tokens — in under half a second. For free.

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

Raw results from all three engines get fused into a single score. But HMS doesn't stop at simple merging — it applies **seven scoring signals**:

1. **Intent Detection** — HMS classifies your query into one of 13 categories (credential lookup, relationship question, health info, config query, etc.) and boosts chunks that match the pattern. Ask "what's the SSH password?" and config-type chunks get a scoring bump.

2. **Synonym Expansion** — "kids" also searches for "children", "son", "daughter". "SSH" also hits "secure shell", "remote access". Customizable via `synonyms.json`.

3. **Source Tiering** — Curated memory files (MEMORY.md, daily notes) outrank transcripts and bulk imports. Your best-organized information surfaces first.

4. **Type Awareness** — Chunks are tagged as `config`, `fact`, `narrative`, or `todo` based on content patterns. Credential queries prefer config chunks. Name queries prefer fact chunks.

5. **Temporal Decay** — Recent information scores slightly higher than old information. Half-life: 60 days.

6. **Date Routing** — "What happened on March 15?" targets files with that date. "Most recent deployment" sorts by recency. HMS detects temporal intent and adjusts.

7. **Adjacency Boosting** — When a chunk scores high, its neighbors get pulled in too. Facts split across chunk boundaries still get found.

### Stage 3: Cross-Encoder Reranking

This is the precision layer. After scoring produces ~20 candidates, HMS runs them through a **cross-encoder model** (ms-marco-MiniLM-L-6-v2) that reads each (query, candidate) pair together and produces a relevance score.

Unlike the embedding model (which encodes query and text *separately*), the cross-encoder sees both at once — catching nuances that vector similarity misses.

Final score: **60% cross-encoder + 40% hybrid score**. The cross-encoder gets the final say, but the hybrid pipeline ensures it has the right candidates to judge.

**Total latency: ~500ms on CPU. Zero API calls.**

---

## Architecture

```
         ┌─────────────────────────────┐
         │          Your Query          │
         └──────────────┬──────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │   Intent Detection (13 types) │
         │   Synonym Expansion           │
         │   Temporal Mode Detection     │
         └──────────────┬───────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌────────────┐┌────────────┐┌────────────┐
   │  Semantic   ││  Keyword   ││  Entity    │
   │  (HNSW)    ││  (BM25)    ││  Lookup    │
   │            ││            ││            │
   │ 384-dim    ││ SQLite     ││ Regex      │
   │ vectors    ││ FTS5       ││ patterns   │
   └─────┬──────┘└─────┬──────┘└─────┬──────┘
         └─────────────┼─────────────┘
                       ▼
         ┌──────────────────────────────┐
         │    Hybrid Score Fusion        │
         │                               │
         │  semantic + keyword + entity  │
         │  + intent + type + temporal   │
         │  + source tier + adjacency    │
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │   Cross-Encoder Reranking     │
         │                               │
         │  Top 20 → rerank → Top K      │
         │  60% reranker / 40% hybrid    │
         └──────────────┬───────────────┘
                        ▼
         ┌──────────────────────────────┐
         │        Best Results           │
         └──────────────────────────────┘
```

### Two-Tier Index

Not all data is equal. HMS maintains two HNSW indexes:

- **Primary** — curated files (MEMORY.md, daily notes, configs) — get a scoring boost
- **Secondary** — transcripts and bulk content — still searchable, but ranked lower

This means your carefully organized notes outrank raw conversation dumps.

### Automatic File Watching

HMS watches your workspace and reindexes when files change. Create, edit, or delete a file — the index stays current in seconds. No cron jobs, no manual rebuilds.

### Entity Extraction

Every chunk gets scanned by pure regex patterns (no LLM) that extract structured entities:

- **Network:** IP addresses (labeled from context), URLs, ports
- **Credentials:** Passwords, tokens, API keys, SSH keys
- **People:** Names, ages, roles, family relationships
- **Places:** Addresses, cities, states, zip codes
- **Financial:** Dollar amounts, percentages
- **Temporal:** ISO dates, named dates
- **Contact:** Emails, phone numbers
- **Config:** Key-value pairs from markdown lists

These feed directly into the entity search engine, enabling instant lookups like "what's the SSH password?" without scanning every chunk.

---

## Benchmarks

Tested on the [IRONMAN benchmark suite](https://github.com/yPhi-box/ironman-benchmark) against comparable memory systems:

### Day Tier (50 messages, 82 queries)
| System | Accuracy | Latency | API Cost |
|--------|----------|---------|----------|
| **HMS v2.4** | **72.0%** | **466ms** | **$0.00** |
| Hindsight | 50.0% | 3,080ms | ~$0.01/query |
| Mem0 | 25.6% | 197ms | ~$0.01/query |

### Month Tier (1,200 messages, 535 queries)
| System | Accuracy | Latency | API Cost |
|--------|----------|---------|----------|
| **HMS v2.4** | **74.8%** | **466ms** | **$0.00** |
| Hindsight | 72.0% | 4,764ms | ~$0.01/query |
| Mem0 | 23.4% | 270ms | ~$0.01/query |

### Year Tier (3,600 messages, 710 queries)
| System | Accuracy | Latency |
|--------|----------|---------|
| **HMS v2.4** | **61.1%** | **~470ms** |

**Key takeaway:** HMS beats cloud-based alternatives while being faster and completely free to run. And unlike API-dependent systems, it works offline.

---

## How It Saves Money

Let's do the math.

A typical OpenClaw agent loads memory files into every conversation turn. With a moderately active agent:

| | Without HMS | With HMS |
|---|-------------|----------|
| **Context per query** | ~50,000 tokens (full files) | ~500 tokens (relevant chunks) |
| **Queries per day** | 50 | 50 |
| **Daily input tokens** | 2,500,000 | 25,000 |
| **Monthly input tokens** | 75,000,000 | 750,000 |
| **Monthly cost (Claude Sonnet @ $3/1M)** | ~$225 | ~$2.25 |
| **HMS cost** | — | $0 |

**~99% reduction in memory-related token costs.** Your agent gets *better* answers from *less* context, and you keep the difference.

Actual savings depend on your usage. The principle: sending 500 tokens of the *right* information beats 50,000 tokens of everything.

---

## Installation

### Prerequisites

- **OS:** Ubuntu 22.04+ / Debian 12+ (any Linux with systemd)
- **Python:** 3.10+
- **Disk:** ~1.5GB free (models + venv + database)
- **RAM:** ~400MB at runtime
- **GPU:** Optional — CPU works great, GPU speeds up initial indexing
- **OpenClaw:** 2026.2.1 or later (HMS is a plugin — tested on 2026.2.1, 2026.2.17, 2026.3.1, 2026.3.11, and 2026.3.13)

### Quick Install

```bash
git clone https://github.com/yPhi-box/HMS.git
cd hms
bash install.sh
```

That's it. The installer:

1. Creates a Python virtual environment (nothing touches your system packages)
2. Detects GPU vs CPU and installs the right PyTorch (~200MB CPU, ~2GB GPU)
3. Downloads the embedding model (all-MiniLM-L6-v2, ~80MB) and reranker model (ms-marco-MiniLM-L-6-v2, ~90MB) — one-time download
4. Indexes your OpenClaw workspace
5. Creates a systemd service (auto-start on boot, auto-restart on crash)
6. Installs the OpenClaw plugin (your agent's `memory_search` calls now route to HMS)
7. Drops `HMS-RULES.md` into your workspace (see [Memory Search Protocol](#memory-search-protocol))

### Installer Options

```bash
bash install.sh --dir ~/my-hms          # Custom install directory (default: ~/hms)
bash install.sh --port 9000             # Custom port (default: 8765)
bash install.sh --watch /path/to/files  # Custom watch directory (default: ~/.openclaw/workspace)
bash install.sh --no-service            # Skip systemd setup (run manually)
bash install.sh --no-plugin             # Skip OpenClaw plugin (use API directly)
bash install.sh --uninstall             # Remove everything cleanly
```

### Verify Installation

After install, confirm everything is working:

```bash
# 1. Check the service is running
sudo systemctl status hms
# Should show: active (running)

# 2. Health check
curl http://127.0.0.1:8765/health
# {"status":"ok","version":"2.4.0"}

# 3. Check your index
curl http://127.0.0.1:8765/stats
# {"total_chunks":1317,"total_files":38,"total_entities":3697,...}

# 4. Test a search
curl -s -X POST http://127.0.0.1:8765/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "test query"}' | python3 -m json.tool
# Should return results from your workspace files
```

If step 1 fails, check the logs:
```bash
sudo journalctl -u hms -n 50
```

---

## Configuration

### Changing the Port

If port 8765 is already in use:

```bash
# Reinstall with a different port
bash install.sh --port 9000

# Or edit the service directly
sudo systemctl edit hms
# Add under [Service]:
# Environment="HMS_PORT=9000"
sudo systemctl restart hms
```

Don't forget to update your OpenClaw plugin config to point to the new port.

### Watch Paths

By default, HMS watches `~/.openclaw/workspace`. To watch additional directories:

```bash
# Set via environment variable (comma-separated)
export HMS_WATCH_PATHS="/home/user/.openclaw/workspace,/home/user/notes"
```

Or edit the systemd service:
```bash
sudo systemctl edit hms
# Add: Environment="HMS_WATCH_PATHS=/path/one,/path/two"
sudo systemctl restart hms
```

### Custom Synonyms

HMS ships with sensible defaults (family terms, tech abbreviations, etc.). To add your own:

Create `synonyms.json` in the HMS directory:
```json
{
  "homelab": ["homelab", "home lab", "server rack", "self-hosted"],
  "plex": ["plex", "media server", "streaming"],
  "proxmox": ["proxmox", "hypervisor", "virtualization"]
}
```

Or set a custom path:
```bash
export HMS_SYNONYMS_PATH=/path/to/my-synonyms.json
```

### Reranker Model

The default cross-encoder (ms-marco-MiniLM-L-6-v2) is optimized for English queries. To use a different model:

```bash
export HMS_RERANKER_MODEL="cross-encoder/ms-marco-TinyBERT-L-2-v2"  # Faster, less accurate
```

---

## Memory Search Protocol

HMS isn't just a search engine — it enforces a **search-first habit** in your agent.

### The Problem

An AI agent with perfect recall is useless if it doesn't bother to search. Agents will confidently guess answers, say "I don't know," or ask you for information that's sitting in their own files. This wastes your time and breaks trust.

### The Solution

HMS installs two enforcement layers automatically:

**1. Tool Description (Plugin Level)**

The `memory_search` tool description tells the LLM the tool is mandatory, not optional:

> *"MANDATORY: Search memory before answering ANY question about machines, people, credentials, configs, IPs, history, or prior decisions. Do NOT guess, do NOT say 'I don't know,' do NOT ask the user — search first."*

The agent reads this description every time it decides which tools to use. It's a direct instruction embedded in the tool itself.

**2. Workspace Rules File (Installer Level)**

The installer drops `HMS-RULES.md` into your OpenClaw workspace. OpenClaw auto-injects all workspace `.md` files into the system prompt, so every session begins with:

```markdown
# HMS Memory Search Protocol

## Rule: Search Before Speaking

Before answering ANY question involving facts, machines, people, credentials,
IPs, configs, history, or prior decisions:

1. Run `memory_search` FIRST. Every time. No exceptions.
2. Do NOT guess. Do not say "I don't know." Do not ask the human for info you could search.
3. Do NOT trust vibes. If you "feel like" you know something, verify it with a search.
4. The search takes 500ms. There is no excuse to skip it.
```

### Why Both Layers?

The tool description travels with the plugin code. The workspace file travels with the workspace. If one gets missed, the other catches it. Belt and suspenders.

### Customizing

You can edit `HMS-RULES.md` in your workspace to add domain-specific search triggers, escalation policies, or anything else your agent should always check before responding. The installer won't overwrite an existing file.

---

## API Reference

HMS exposes a simple REST API on `http://127.0.0.1:8765`.

### `GET /health`

Health check. Returns version and status.

```bash
curl http://127.0.0.1:8765/health
```
```json
{"status": "ok", "version": "2.4.0"}
```

### `POST /search`

Search indexed content. This is the main endpoint your agent uses.

```bash
curl -X POST http://127.0.0.1:8765/search \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "what is the SSH password for the production server?",
    "max_results": 5,
    "min_score": 0.0
  }'
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | string | required | The search query |
| `max_results` | int | 10 | Maximum results to return |
| `min_score` | float | 0.0 | Minimum relevance score (0.0–1.0) |

**Response:** Array of result objects with `text`, `file_path`, `line_start`, `line_end`, `combined_score`, `blended_score` (after reranking), and extracted `entities`.

### `POST /index`

Manually trigger indexing for a directory.

```bash
curl -X POST http://127.0.0.1:8765/index \
  -H 'Content-Type: application/json' \
  -d '{
    "directory": "/path/to/files",
    "pattern": "**/*.md",
    "force": true
  }'
```

**Parameters:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `directory` | string | required | Path to index |
| `pattern` | string | `**/*.md` | Glob pattern for files |
| `force` | bool | false | Re-index even if files haven't changed |

### `GET /stats`

Index statistics.

```bash
curl http://127.0.0.1:8765/stats
```
```json
{
  "total_chunks": 1317,
  "total_files": 38,
  "total_entities": 3697,
  "db_size_mb": 12.4
}
```

---

## Management

### Service Commands

```bash
sudo systemctl status hms        # Check if running
sudo systemctl start hms         # Start
sudo systemctl stop hms          # Stop
sudo systemctl restart hms       # Restart (e.g., after config change)
sudo systemctl enable hms        # Enable auto-start on boot (default)
sudo systemctl disable hms       # Disable auto-start
```

### Logs

```bash
sudo journalctl -u hms -f          # Live tail
sudo journalctl -u hms -n 100      # Last 100 lines
sudo journalctl -u hms --since today  # Today's logs
```

### Re-index Everything

If your index gets out of sync (rare, but possible after a crash):

```bash
curl -X POST http://127.0.0.1:8765/index \
  -H 'Content-Type: application/json' \
  -d '{"directory": "/home/user/.openclaw/workspace", "force": true}'
```

### Uninstall

```bash
bash install.sh --uninstall
```

This removes the systemd service, the HMS directory, and all models. Your OpenClaw plugin config in `openclaw.json` is left intact — remove manually if needed. `HMS-RULES.md` in your workspace is also left intact.

---

## Troubleshooting

### HMS won't start

**Port already in use:**
```bash
sudo lsof -i :8765
# Kill the conflicting process or use a different port
bash install.sh --port 9000
```

**Python version too old:**
```bash
python3 --version
# Needs 3.10+. On Ubuntu: sudo apt install python3.12 python3.12-venv
```

**Missing dependencies:**
```bash
cd ~/hms
source venv/bin/activate
pip install -r requirements.txt
```

### Search returns no results

1. Check that files are indexed: `curl http://127.0.0.1:8765/stats`
2. If `total_chunks: 0`, trigger a manual reindex (see above)
3. Check the watched directory has `.md` files
4. Check logs for indexing errors: `sudo journalctl -u hms -n 50`

### High memory usage

HMS typically uses ~400MB. If it's higher:

- Large indexes (10,000+ chunks) will use more RAM for HNSW
- The cross-encoder loads on first query and stays resident
- To reduce memory: use a smaller reranker model or disable it (`HMS_RERANKER_MODEL=none`)

### Slow first query

The first search after startup takes 2–5 seconds while models warm up. Subsequent queries are ~500ms. HMS runs a warmup query on boot to minimize this.

### Watcher not picking up changes

```bash
# Check inotify limit (Linux caps the number of file watchers)
cat /proc/sys/fs/inotify/max_user_watches
# If low, increase it:
echo 65536 | sudo tee /proc/sys/fs/inotify/max_user_watches
```

### Disk space

The database grows with your content. Typical sizes:

| Files Indexed | DB Size | RAM Usage |
|--------------|---------|-----------|
| 10–50 | 5–15 MB | ~350 MB |
| 50–200 | 15–50 MB | ~400 MB |
| 200–1000 | 50–200 MB | ~500 MB |

---

## Upgrading

### From v2.3 to v2.4

v2.4 adds cross-encoder reranking. To upgrade:

```bash
cd ~/hms
git pull
source venv/bin/activate
pip install sentence-transformers --upgrade  # Includes cross-encoder support
sudo systemctl restart hms
```

The reranker model (~90MB) downloads automatically on first query. No reindexing needed — v2.4 is backward compatible with v2.3 indexes.

**What's new in v2.4:**
- Cross-encoder reranking (60% reranker + 40% hybrid score)
- Wider candidate pool (top-20) before reranking
- Significant accuracy improvement on all benchmark tiers
- Memory Search Protocol (HMS-RULES.md + tool description enforcement)
- Graceful degradation — if reranker fails to load, falls back to v2.3 behavior

### From v2.2 or earlier

A full reinstall is recommended:

```bash
bash install.sh --uninstall
git pull
bash install.sh
```

This rebuilds the index from scratch, which picks up the improved chunker (sentence overlap, date context prepending) and entity extractor.

---

## Under the Hood

For anyone curious about the implementation details.

### Models

| Model | Purpose | Size | Source |
|-------|---------|------|--------|
| all-MiniLM-L6-v2 | Embedding (384-dim vectors) | ~80 MB | sentence-transformers |
| ms-marco-MiniLM-L-6-v2 | Cross-encoder reranking | ~90 MB | sentence-transformers |

Both run 100% locally via PyTorch. No API calls, no telemetry, no data leaves your machine.

### Chunking Strategy

Files are split into chunks of ~800 characters with **2-sentence overlap** between chunks. This overlap prevents facts from being split at boundaries — if a name appears at the end of one chunk, it's also at the start of the next.

Chunks from dated files (e.g., `2025-09-28.md`) get a date prefix prepended: `Date: Sunday, September 28, 2025 (2025-09-28)`. This gets embedded alongside the content, making temporal queries like "what happened in September?" work via semantic similarity — not just keyword matching.

Each chunk is also classified by content type:
- **config** — contains IPs, passwords, ports, key-value pairs
- **fact** — contains bullet-pointed facts, names, relationships
- **todo** — contains checkboxes or TODO markers
- **narrative** — everything else

### Scoring Formula

For each candidate chunk, the hybrid score is:

```
combined = (semantic + keyword_boost + type_boost + source_boost +
            entity_boost + intent_boost + date_boost) × temporal_decay
```

After hybrid scoring, the top 20 candidates are reranked:

```
final = 0.4 × normalized_hybrid + 0.6 × sigmoid(cross_encoder_logit)
```

The cross-encoder gets 60% weight because it sees query and document together — it can judge relevance more precisely than the bi-encoder embedding similarity.

### Codebase

~2,600 lines of Python across 9 files:

| File | Lines | Purpose |
|------|-------|---------|
| `server.py` | 255 | FastAPI server, endpoints, lifecycle |
| `search.py` | 795 | Hybrid search engine, scoring, intent detection |
| `database.py` | 384 | SQLite + HNSW storage, FTS5, entity tables |
| `entities.py` | 338 | Regex-based entity extraction |
| `chunker.py` | 304 | Markdown-aware chunking with overlap |
| `watcher.py` | 177 | Filesystem watcher (inotify) |
| `reranker.py` | 132 | Cross-encoder reranking |
| `indexer.py` | 105 | File discovery and indexing orchestration |
| `embedder.py` | 94 | Sentence-transformer embedding |

No frameworks beyond FastAPI. No ORMs. No unnecessary abstractions.

---

## FAQ

**Q: Does HMS replace OpenClaw's built-in memory?**
A: It enhances it. The built-in `memory_search` tool gets rerouted through HMS via a plugin. Your agent calls the same function — it just gets much better results.

**Q: Can I use HMS without OpenClaw?**
A: Yes. Install with `--no-plugin` and use the REST API directly. Any application that can make HTTP requests can use HMS as a search backend.

**Q: What file types does it index?**
A: Markdown (`.md`) by default. The glob pattern is configurable when indexing.

**Q: Is my data sent anywhere?**
A: No. Everything runs locally — embedding, indexing, searching, reranking. HMS disables all Hugging Face telemetry, progress bars, and implicit token usage. We set `DO_NOT_TRACK=1` at startup.

**Q: How does it compare to RAG with OpenAI embeddings?**
A: HMS uses local embeddings (all-MiniLM-L6-v2) which are smaller than OpenAI's ada-002 but compensates with the hybrid pipeline. The cross-encoder reranking step recovers much of the precision gap. And it's free.

**Q: Can it handle large datasets?**
A: Tested up to 3,600 messages / 710 queries in the Year tier benchmark. For very large datasets (10,000+ files), you may want to increase RAM allocation. The HNSW index scales logarithmically.

**Q: What happens if HMS goes down?**
A: Your agent falls back to OpenClaw's built-in memory behavior. HMS is an enhancement, not a dependency. The systemd service auto-restarts on crashes.

**Q: What about the HMS-RULES.md file?**
A: It's a workspace file that tells your agent to always search memory before answering. You can edit it, add to it, or delete it — the plugin-level enforcement still works independently. See [Memory Search Protocol](#memory-search-protocol).

---

## License

MIT — do whatever you want with it.

---

*Built for [OpenClaw](https://github.com/openclaw/openclaw). Made with stubbornness and SQLite.*
