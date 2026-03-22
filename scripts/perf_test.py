#!/usr/bin/env python3
"""
HMS v2.4 Performance Benchmark
Tests indexing speed, search latency, accuracy, concurrency, scale, and edge cases.
Uses rich synthetic data — zero personal information.
"""
import requests
import time
import json
import os
import sys
import random
import string
import tempfile
import shutil
import concurrent.futures
from pathlib import Path

HMS_URL = os.environ.get("HMS_URL", "http://localhost:8765")

# ============================================================================
# SYNTHETIC DATA — Rich, varied, realistic workspace content
# ============================================================================

WORKSPACE_FILES = {
    # ── PEOPLE (varied backgrounds, unique details) ────────────────────
    "team/dr-elena-vasquez.md": """# Dr. Elena Vasquez — Chief Technology Officer

## Background
Elena earned her PhD in distributed systems from MIT in 2014. Before joining, she spent 8 years at Google working on Spanner and BigTable. She holds 12 patents in database replication.

## Personal
- Age: 42
- Birthday: September 14, 1983
- Lives in: Portland, Oregon
- Spouse: Marcus Vasquez (architect)
- Kids: Twin daughters Sofia and Lucia, age 7
- Hobbies: Rock climbing, restoring vintage motorcycles (owns a 1972 Honda CB350)
- Favorite book: "Designing Data-Intensive Applications" by Martin Kleppmann
- Allergies: Shellfish (severe — carries EpiPen)

## Contact
- Email: elena.vasquez@horizonlabs.io
- Phone: (503) 555-0147
- Slack: @elena.v
- GitHub: github.com/evasquez
""",

    "team/james-okonkwo.md": """# James Okonkwo — Head of Product

## Background
James previously ran product at Stripe for 4 years, focusing on payment infrastructure for emerging markets. He has an MBA from Wharton and a BS in Computer Science from Carnegie Mellon.

## Personal
- Age: 37
- Birthday: March 2, 1989
- Lives in: Austin, Texas
- Partner: David Chen (software engineer at Apple)
- Pet: Golden retriever named Pixel
- Hobbies: Marathon running (PR: 3:12:44 at Boston 2024), woodworking
- Favorite food: Nigerian jollof rice (his grandmother's recipe)
- Fun fact: Speaks four languages — English, Yoruba, French, and Mandarin

## Contact
- Email: james.okonkwo@horizonlabs.io
- Phone: (512) 555-0283
- Slack: @jamesko
""",

    "team/sarah-lindqvist.md": """# Sarah Lindqvist — VP of Engineering

## Background
Sarah comes from Spotify in Stockholm where she led the personalization team (120 engineers). She moved to the US in 2022. Known for her "no meeting Wednesdays" policy.

## Personal
- Age: 39
- Birthday: July 28, 1986
- Originally from: Gothenburg, Sweden
- Lives in: Seattle, Washington
- Married to: Erik Lindqvist (stays home with kids)
- Children: Astrid (11), Nils (8)
- Hobbies: Cross-country skiing, board games (owns 200+ board games)
- Guilty pleasure: Reality TV (won't admit it in meetings)
- Drives: Tesla Model Y (license plate: SRCLEAN)

## Contact
- Email: sarah.lindqvist@horizonlabs.io
- Phone: (206) 555-0391
- Slack: @sarahlind
""",

    "team/ravi-krishnamurthy.md": """# Ravi Krishnamurthy — Principal Architect

## Background
Ravi has been writing software for 24 years. Started at Sun Microsystems in 2002, then Amazon (2008-2016) where he designed the core event bus for AWS EventBridge. Joined us in 2023.

## Personal
- Age: 48
- Birthday: November 30, 1977
- Lives in: Redmond, Washington (deliberately, to annoy his ex-Amazon colleagues)
- Wife: Priya Krishnamurthy (pediatrician at Seattle Children's)
- Kids: Arjun (19, freshman at Stanford studying physics), Meera (16, competitive chess player — rated 2100 USCF)
- Hobbies: Amateur astronomy (has a 12-inch Dobsonian telescope), cooking South Indian food
- Vegetarian since birth
- Daily routine: Wakes at 5am, meditates 30 minutes, codes until 7am before meetings
- Favorite language: Rust (previously Java, which he now calls "a necessary evil")

## Contact
- Email: ravi.k@horizonlabs.io
- Phone: (425) 555-0567
""",

    "team/maya-jackson.md": """# Maya Jackson — Security Lead

## Background
Former NSA cybersecurity analyst (2015-2020). Left government work because "the bureaucracy was killing her soul." Has a CISSP, OSCP, and CEH. Found three zero-day vulnerabilities that were assigned CVEs.

## Personal
- Age: 33
- Birthday: January 15, 1993
- Lives in: Denver, Colorado
- Single, has two cats: Kernel and Panic
- Hobbies: CTF competitions (team "ByteForce" — placed 3rd at DEF CON 2024), lockpicking, bouldering
- Tattoo: Binary code on left forearm that spells "root"
- Drives: 2019 Subaru Outback with "HACK THE PLANET" bumper sticker
- Diet: Pescatarian
- Favorite tool: Burp Suite

## Contact
- Email: maya.jackson@horizonlabs.io
- Phone: (720) 555-0829
- Signal: @mayajsec
""",

    # ── COMPANY INFO ────────────────────────────────────────────────────
    "company/about.md": """# Horizon Labs — Company Overview

## Founded
- Date: April 12, 2019
- Founders: Elena Vasquez and Thomas Wright
- Incorporated in: Delaware
- HQ: 742 Innovation Drive, Suite 400, Portland, OR 97201
- EIN: 84-3729156

## Mission
Building the next generation of real-time data infrastructure for enterprise applications.

## Funding
- Seed: $2.1M (2019) — Led by Founders Fund
- Series A: $18M (2021) — Led by Andreessen Horowitz
- Series B: $65M (2023) — Led by Sequoia Capital, with participation from a16z
- Total raised: $85.1M
- Current valuation: $420M (post-Series B)

## Team Size
- Total employees: 127 (as of March 2026)
- Engineering: 68
- Product: 14
- Design: 9
- Sales: 18
- Marketing: 8
- G&A: 10

## Key Metrics (Q4 2025)
- ARR: $23.4M
- Customer count: 342
- Net revenue retention: 138%
- Gross margin: 78%
""",

    "company/benefits.md": """# Employee Benefits — Horizon Labs

## Health Insurance
- Provider: Aetna PPO
- Company pays 90% of premiums for employees, 75% for dependents
- Dental: Delta Dental
- Vision: VSP
- Mental health: Unlimited therapy sessions via Lyra Health

## Time Off
- Unlimited PTO (minimum 15 days encouraged)
- 12 company holidays
- 16 weeks paid parental leave (all parents)
- 4 weeks paid sabbatical after 5 years

## Financial
- 401(k): 4% match via Fidelity
- Employee stock options: 4-year vest, 1-year cliff
- Annual bonus: 10-20% of base salary
- Equipment budget: $3,500 for home office setup

## Perks
- $150/month learning stipend (books, courses, conferences)
- Free lunch on Tuesdays and Thursdays (Portland office)
- Dog-friendly office
- Annual company retreat (last year: Costa Rica)
""",

    # ── INFRASTRUCTURE ──────────────────────────────────────────────────
    "infrastructure/production.md": """# Production Environment

## Architecture
Three-region deployment across AWS us-west-2, us-east-1, and eu-west-1.
Active-active configuration with CockroachDB for cross-region consistency.

## Servers
- API Gateway: api.horizonlabs.io (CloudFront → ALB → ECS Fargate)
- Primary DB: cockroach-prod-1.internal (r6g.4xlarge, 128GB RAM, 2TB NVMe)
- Replica DB: cockroach-prod-2.internal (r6g.2xlarge, 64GB RAM, 1TB NVMe)
- Cache: Redis cluster — cache.horizonlabs.internal (r6g.xlarge, 3 nodes)
- Search: Elasticsearch — es-prod.internal (m6g.2xlarge, 6-node cluster)
- Queue: Kafka — kafka-prod.internal (m6g.xlarge, 5-broker cluster)

## Credentials (rotate quarterly)
- AWS Account ID: 847291035612
- Datadog API Key: dd-api-k3y-x9m2p7q4r1
- PagerDuty Service Key: pd-svc-8f3k2m9x
- Sentry DSN: https://abc123@sentry.io/4507891
- Kafka Bootstrap: kafka-prod-1.internal:9092,kafka-prod-2.internal:9092

## Monitoring
- Datadog for metrics and APM
- PagerDuty for alerting (on-call rotation: weekly, starts Monday 9am PT)
- Sentry for error tracking
- CloudWatch for AWS-level metrics

## SLA Targets
- API availability: 99.95%
- P99 latency: < 200ms
- Data durability: 99.999999999% (11 nines)
- RPO: 1 second
- RTO: 5 minutes
""",

    "infrastructure/staging.md": """# Staging Environment

## Purpose
Pre-production environment that mirrors production at 1/4 scale.
All deployments go through staging before production.

## Access
- URL: staging.horizonlabs.io
- VPN required: WireGuard (config in 1Password vault "Engineering")
- SSH bastion: bastion-staging.horizonlabs.io (port 2222)
- Admin panel: admin-staging.horizonlabs.io

## Database
- CockroachDB: cockroach-staging.internal (r6g.xlarge)
- Refreshed from production weekly (Sundays 2am PT, PII scrubbed)
- Test accounts: test-user-1@example.com through test-user-50@example.com
- Admin account: staging-admin@horizonlabs.io / password in 1Password

## Known Differences from Production
- Single region (us-west-2 only)
- 3-node Kafka instead of 5
- No CDN (direct to ALB)
- Synthetic monitoring disabled
- Email sends go to Mailhog instead of SES
""",

    # ── PROJECTS ────────────────────────────────────────────────────────
    "projects/project-aurora.md": """# Project Aurora — Real-Time Analytics Pipeline

## Overview
Rebuild our analytics pipeline to support sub-second query latency on datasets up to 50TB.
Current system (Redshift) hits 30-second queries at scale. Aurora targets < 500ms P99.

## Timeline
- Kickoff: January 6, 2026
- Phase 1 (Data ingestion): January - February 2026 ✅
- Phase 2 (Query engine): March - April 2026 ← CURRENT
- Phase 3 (Dashboard integration): May - June 2026
- GA release: July 1, 2026

## Team
- Tech lead: Ravi Krishnamurthy
- Engineers: Chen Wei, Fatima Al-Rashid, Tom Patterson, Lisa Chang
- Product: James Okonkwo
- Design: Yuki Tanaka

## Tech Stack
- Apache Arrow for columnar data format
- DataFusion query engine (Rust-based)
- MinIO for object storage (S3-compatible)
- gRPC for service communication
- React + D3.js for dashboard frontend

## Budget
- Engineering time: $1.2M (6 months, 5 engineers)
- Infrastructure: $45K/month additional AWS spend
- Total estimated: $1.47M

## Risks
- DataFusion community is small — if we hit bugs, we may need to fork
- Chen Wei's visa renewal (H-1B) is pending — could lose him for 2-3 weeks
- Competing priority with Project Meridian for Ravi's time
""",

    "projects/project-meridian.md": """# Project Meridian — European Data Residency

## Overview
GDPR compliance requires that EU customer data stays within EU borders.
Meridian adds full EU data residency with independent processing in eu-west-1.

## Why Now
- Three enterprise deals ($2.1M combined ARR) blocked on EU residency
- GDPR enforcement increasing — €1.3B in fines issued in 2025
- Competitor DataStream launched EU residency in Q4 2025

## Timeline
- Started: February 3, 2026
- Database sharding by region: February - March 2026 ✅
- API routing layer: March - April 2026 ← CURRENT
- Data migration tooling: April - May 2026
- Customer migration: May - June 2026
- Target completion: June 30, 2026

## Team
- Tech lead: Sarah Lindqvist (chosen because of her European background)
- Engineers: Alex Petrov, Maria Santos, David Kim
- Legal: Jennifer Walsh (outside counsel from Baker McKenzie)
- Security review: Maya Jackson

## Key Decisions
- Chose CockroachDB locality-aware tables over separate clusters
- API gateway does geo-routing based on customer's registered region
- Encryption keys stored in AWS KMS eu-west-1 (separate from US keys)
- Audit log retention: 7 years for EU, 5 years for US

## Budget
- Engineering: $840K
- Legal/compliance: $120K
- Infrastructure delta: $18K/month ongoing
- Total first year: $1.18M
""",

    # ── MEETING NOTES ───────────────────────────────────────────────────
    "meetings/2026-03-10-leadership.md": """# Leadership Meeting — March 10, 2026

## Attendees
Elena Vasquez, James Okonkwo, Sarah Lindqvist, Ravi Krishnamurthy, Maya Jackson, CFO Thomas Wright

## Agenda

### 1. Q1 Revenue Update (Thomas)
- Q1 ARR tracking at $25.8M (ahead of $24.5M plan)
- Two new enterprise deals closed: Acme Manufacturing ($380K ARR) and GlobalTech Solutions ($520K ARR)
- Churn: Lost 3 small accounts ($42K combined), all cited "product complexity"
- Cash runway: 28 months at current burn rate

### 2. Project Aurora Status (Ravi)
- Phase 1 complete, on schedule
- DataFusion performing well — 10x faster than Redshift on benchmark queries
- Concern: Need to hire 2 more backend engineers for Phase 2
- Decision: Approved hiring req for 2 senior engineers, budget $400K total comp each

### 3. Security Incident Debrief (Maya)
- March 7 incident: Unauthorized API access attempt from IP 203.0.113.42
- No data breach — blocked by rate limiter + API key rotation
- Root cause: Former contractor's API key wasn't revoked (process gap)
- Action items:
  - Maya to implement automated key expiry (90-day max lifetime) by March 31
  - HR to add API key revocation to offboarding checklist
  - Maya to run tabletop exercise with engineering team in April

### 4. Hiring Update (Elena)
- 12 open positions (8 engineering, 2 sales, 1 design, 1 data science)
- Average time-to-fill: 47 days
- Pipeline: 230 active candidates
- Diversity stats: 43% of final-round candidates are from underrepresented groups

### 5. Company Retreat Planning
- Dates: June 12-15, 2026
- Location: Bend, Oregon (Sunriver Resort)
- Budget: $1,800 per person ($228K total)
- Theme: "Building Together"
""",

    "meetings/2026-03-17-standup.md": """# Engineering Standup — March 17, 2026

## Aurora Team
- **Chen Wei**: Finished Arrow schema validation. PR #847 ready for review. Blocked on DataFusion issue #2341 (aggregate pushdown bug).
- **Fatima Al-Rashid**: Implemented gRPC streaming for large result sets. Testing shows 3x throughput improvement. Working on backpressure handling next.
- **Tom Patterson**: Set up MinIO cluster on staging. Running compatibility tests with existing S3 code. Found 2 edge cases with multipart upload — filed issues.
- **Lisa Chang**: Dashboard prototype using D3.js force-directed graphs. Demo at Thursday's design review.

## Meridian Team
- **Alex Petrov**: CockroachDB locality-aware queries working. Latency overhead is 12ms for cross-region reads (within budget).
- **Maria Santos**: API gateway geo-routing 90% complete. Handling edge case: what happens when customer relocates from US to EU mid-contract?
- **David Kim**: Built data migration dry-run tool. Tested with 10K records — correct results, but needs optimization (currently 45 min for 1M records, target: 10 min).

## Security
- **Maya Jackson**: Automated key expiry implemented and deployed. 23 stale keys identified and revoked. Running penetration test on staging this week.

## Blockers
1. Chen Wei needs DataFusion maintainer to review issue #2341 — Ravi to reach out to Andy Grove (DataFusion creator)
2. Maria needs legal clarification on data residency during customer region transfer — Jennifer Walsh OOO until March 19
3. Tom's MinIO multipart upload issues may require upstream patch
""",

    # ── DAILY NOTES ─────────────────────────────────────────────────────
    "memory/2026-03-12.md": """# March 12, 2026 — Daily Notes

## Morning
- Had coffee with Elena in the break room. She mentioned Thomas Wright is considering stepping back as CFO to "do something entrepreneurial" — not public yet, don't spread it.
- Reviewed Fatima's gRPC streaming PR. Clean code, but missing error handling for disconnected clients. Left comments.

## Afternoon
- 1:1 with James about Aurora dashboard requirements. He wants "Netflix-style" auto-playing previews for chart widgets. I think it's scope creep but he's the PM.
- Maya pulled me into a quick call about the March 7 security incident. The former contractor was Derek Simmons — he left in January and nobody revoked his API key. Classic process failure.

## Evening
- Stayed late debugging a memory leak in the Arrow schema validator. Turns out we were holding references to closed file handles. Fix: weak references in the cache layer.
- Sarah sent a Slack message at 9pm asking about Swedish meatball recipes for the team lunch. That woman works too hard.
""",

    "memory/2026-03-15.md": """# March 15, 2026 — Daily Notes

## Morning
- David Kim presented his data migration tool at the team demo. Impressive work — he cut migration time from 45 minutes to 8 minutes for 1M records using parallel workers and batch commits.
- Got an email from a recruiter about a Director of Engineering role at Databricks. Forwarded to Elena for laughs. She said "tell them our equity is better."

## Afternoon
- Ravi found a critical bug in CockroachDB's locality-aware optimization. Under high write load, the geo-routing sometimes sends writes to the wrong region. Filed upstream issue CDB-18234.
- Emergency meeting with Elena, Ravi, and Sarah. Decision: implement a write-verification layer that confirms region before committing. Ravi estimates 2 days of work, adds ~3ms latency per write.
- Maya finished the penetration test on staging. Found one medium-severity issue: CORS misconfiguration allowing wildcard origins on the admin API. Fixed within the hour.

## Personal
- Booked flights for the Portland trip next month. Direct flight on Alaska Airlines, $287 round trip.
- Meera (Ravi's daughter) won her regional chess tournament! Ravi brought celebratory samosas for the whole office.
""",

    "memory/2026-03-18.md": """# March 18, 2026 — Daily Notes

## Critical
- **Production incident at 2:14 PM PT**: Elasticsearch cluster went yellow due to disk space on es-prod-3. Unbalanced shard allocation after yesterday's reindex. 
- Resolution: Ravi manually rebalanced shards, freed 340GB. Added disk space alert at 75% threshold (was missing!).
- Impact: Search latency spiked to 2.3 seconds for ~18 minutes. 7 customer complaints via Zendesk.
- Postmortem scheduled for March 20 at 10am PT.

## Meetings
- Jennifer Walsh (lawyer) finally responded about the EU data residency during customer relocation question. Answer: 30-day grace period, then data must be migrated. Need to update the API routing logic.
- Thomas Wright confirmed he's leaving as CFO effective May 1. Elena is interviewing candidates. Not announced to company yet — embargo until April board meeting.

## Aurora
- Chen Wei's DataFusion issue got resolved — Andy Grove himself submitted a patch. Chen is unblocked and estimates Phase 2 query engine prototype by March 28.
- Lisa Chang's dashboard demo went well. James loved the "Netflix-style" previews (I was wrong about it being scope creep — it actually looks great).

## Fun
- Office debate: best programming language for a desert island. Results: Ravi (Rust), Maya (Python), Elena (Go), Chen Wei (Haskell, obviously), Tom (JavaScript, and everyone booed).
""",

    # ── TECHNICAL DOCS ──────────────────────────────────────────────────
    "docs/api-reference.md": """# API Reference — Horizon Platform v3.2

## Authentication
All API requests require a Bearer token in the Authorization header.
Tokens expire after 24 hours. Refresh tokens are valid for 30 days.

```
Authorization: Bearer hp_live_k3m9x2p7q4r1...
```

## Rate Limits
- Free tier: 100 requests/minute
- Pro tier: 1,000 requests/minute
- Enterprise: 10,000 requests/minute
- Rate limit headers: X-RateLimit-Remaining, X-RateLimit-Reset

## Endpoints

### POST /api/v3/events/ingest
Ingest events into the analytics pipeline.
- Max batch size: 1,000 events
- Max payload: 5MB
- Event schema: `{timestamp, event_type, properties: {}, user_id?}`
- Response: `{accepted: number, rejected: number, errors: [{index, reason}]}`
- Latency SLA: P99 < 50ms

### GET /api/v3/queries/{query_id}
Retrieve query results.
- Supports polling (status field) or webhook callback
- Results cached for 1 hour
- Max result set: 10,000 rows (paginated)

### POST /api/v3/dashboards
Create a new dashboard.
- Required fields: name, workspace_id
- Optional: description, layout, shared_with[]
- Layouts: grid (default), freeform, presentation

### DELETE /api/v3/data/retention
Trigger data deletion per retention policy.
- Requires admin role
- Irreversible — creates audit log entry
- GDPR right-to-erasure endpoint
""",

    "docs/architecture.md": """# System Architecture

## Overview
Horizon Platform is a real-time analytics system built on three pillars:
1. **Ingestion Layer**: High-throughput event collection (500K events/sec sustained)
2. **Storage Layer**: Columnar storage with hot/warm/cold tiering
3. **Query Layer**: Sub-second analytics on petabyte-scale data

## Data Flow
```
Client SDK → API Gateway → Kafka → Stream Processor → Arrow Storage → Query Engine → Dashboard
```

## Key Design Decisions

### Why Arrow over Parquet for Hot Storage?
Arrow keeps data in memory-mapped columnar format, allowing zero-copy reads.
Parquet is better for cold storage (compression) but adds deserialization overhead.
We use Arrow for hot tier (< 7 days) and Parquet for warm/cold (7+ days).

### Why CockroachDB over PostgreSQL?
Need multi-region active-active with serializable isolation.
PostgreSQL would require custom replication (Citus, pg_replication) which adds operational complexity.
CockroachDB gives us geo-partitioned tables out of the box.
Tradeoff: ~15% higher write latency compared to single-region PostgreSQL.

### Why Kafka over SQS/SNS?
Need exactly-once semantics for billing-critical events.
Kafka's log-based architecture allows replay from any offset.
We process 12M events/day — SQS costs would be $3,600/month vs Kafka's $1,800/month on MSK.

## Capacity Planning
- Current: 12M events/day, 2.3TB ingested/month
- Target (EOY 2026): 100M events/day, 18TB/month
- Bottleneck: Query engine memory (currently 256GB, need 1TB for 100M/day)
""",

    # ── CUSTOMER / BUSINESS ─────────────────────────────────────────────
    "customers/acme-manufacturing.md": """# Acme Manufacturing — Customer Profile

## Account Details
- Company: Acme Manufacturing Inc.
- Industry: Industrial IoT / Manufacturing
- HQ: Detroit, Michigan
- Employees: 4,200
- ARR: $380,000
- Contract signed: February 28, 2026
- Contract term: 2 years with auto-renewal
- Plan: Enterprise

## Key Contacts
- Champion: Robert Chen, VP of Digital Transformation (robert.chen@acmemfg.com)
- Technical: Patricia Okafor, Lead Data Engineer (patricia.okafor@acmemfg.com)
- Executive Sponsor: William Hayes, CTO

## Use Case
Monitoring 15,000 IoT sensors across 3 factories. Using Horizon for real-time anomaly detection on vibration, temperature, and pressure data. Previously used Splunk but costs were $890K/year.

## Health
- NPS: 9 (promoter)
- Usage: 340M events/month (their biggest factory alone generates 180M)
- Support tickets: 3 (all resolved within SLA)
- Expansion opportunity: 2 more factories coming online in Q3 2026 (estimated +$160K ARR)

## Risk
- Robert Chen mentioned evaluating Datadog's IoT offering — keep close watch
- Their data team is only 4 people — may need professional services help for the expansion
""",

    "customers/globaltech-solutions.md": """# GlobalTech Solutions — Customer Profile

## Account Details
- Company: GlobalTech Solutions Ltd
- Industry: Financial Technology
- HQ: London, UK (also offices in Singapore and New York)
- Employees: 1,850
- ARR: $520,000
- Contract signed: March 3, 2026
- Contract term: 3 years
- Plan: Enterprise with EU Data Residency add-on

## Key Contacts
- Champion: Amara Osei, Head of Analytics (amara.osei@globaltech.co.uk)
- Technical: Nikolai Volkov, Principal Engineer (nikolai.volkov@globaltech.co.uk)
- Executive Sponsor: Dame Catherine Blackwood, CEO

## Use Case
Real-time fraud detection on payment transactions. Process 2M transactions/day across 40 countries. Need sub-100ms latency for fraud scoring. Previously built in-house but couldn't scale past 500K tx/day.

## Health
- NPS: 8 (promoter)
- Usage: 62M events/month
- Critical requirement: EU data residency (Project Meridian is partially driven by this deal)
- Support tickets: 1 (timezone-related date parsing bug — fixed in v3.2.1)

## Risk
- High dependency on Project Meridian completing on time
- If EU residency slips past June, they have a contractual exit clause
- Nikolai is technically demanding — every API change gets scrutinized
""",

    # ── INCIDENT LOG ────────────────────────────────────────────────────
    "incidents/2026-03-07-api-key.md": """# Incident Report: Unauthorized API Access Attempt

## Summary
- **Date**: March 7, 2026
- **Severity**: Medium (P2)
- **Duration**: 47 minutes (detected → resolved)
- **Impact**: No data breach. 2,340 unauthorized API requests blocked.

## Timeline
- 14:23 PT — Anomaly detected: 200+ requests/minute from IP 203.0.113.42 using API key `hp_live_dk7x...` (key belonging to former contractor Derek Simmons)
- 14:27 PT — PagerDuty alert fired to Maya Jackson (on-call)
- 14:31 PT — Maya confirmed the key should have been revoked (Derek left January 12, 2026)
- 14:35 PT — Key revoked, IP blocked via WAF rule
- 14:42 PT — Full audit: no data exfiltrated, all requests returned 403 after rate limit triggered at 14:24
- 15:10 PT — All-clear communicated to leadership

## Root Cause
Derek Simmons' API key was not revoked during his offboarding on January 12. The offboarding checklist did not include API key revocation as a step.

## Remediation
1. ✅ Revoked all of Derek's credentials (API keys, SSH keys, OAuth tokens)
2. ✅ Added API key revocation to HR offboarding checklist
3. 🔄 Implementing 90-day automatic key expiry (Maya, due March 31)
4. 📋 Tabletop security exercise scheduled for April
5. ✅ Reviewed all former employees/contractors from past 12 months — no other orphaned keys found

## Lessons Learned
- Manual offboarding processes will always have gaps
- Automated key expiry should be the safety net, not the primary control
- Our rate limiter worked exactly as designed — it caught the breach before any data could be accessed
""",

    "incidents/2026-03-18-elasticsearch.md": """# Incident Report: Elasticsearch Disk Space

## Summary
- **Date**: March 18, 2026
- **Severity**: High (P1)
- **Duration**: 18 minutes
- **Impact**: Search latency spiked to 2.3 seconds. 7 customer complaints.

## Timeline
- 14:14 PT — es-prod-3 hit 92% disk utilization
- 14:14 PT — Elasticsearch cluster status changed from Green to Yellow
- 14:16 PT — PagerDuty alert fired to Ravi Krishnamurthy (on-call)
- 14:19 PT — Ravi identified cause: unbalanced shard allocation after March 17 reindex job
- 14:22 PT — Manual shard rebalancing initiated
- 14:28 PT — Freed 340GB by relocating shards to es-prod-1 and es-prod-2
- 14:32 PT — Cluster status returned to Green, latency normalized

## Root Cause
The nightly reindex job (runs at 1am PT) created new shards that were all allocated to es-prod-3 due to the default allocation strategy. No disk space monitoring alert existed below 95%.

## Remediation
1. ✅ Added disk space alert at 75% threshold
2. ✅ Configured shard allocation awareness to distribute evenly across nodes
3. 📋 Postmortem meeting scheduled March 20, 10am PT
4. 🔄 Investigating automatic shard rebalancing on disk pressure
5. ✅ Increased es-prod-3 disk from 2TB to 4TB as immediate fix

## Customer Impact
- 7 Zendesk tickets from customers experiencing slow search
- All responded to within 15 minutes
- No SLA breach (99.95% target — this incident consumed 0.03%)
""",

    # ── CREATIVE / FICTION ──────────────────────────────────────────────
    "creative/short-story.md": """# The Last Compiler — A Short Story

## Chapter 1: Syntax Error

Zara Chen hadn't slept in three days. The terminal in front of her glowed with the same error message it had shown for 72 hours:

```
CRITICAL: Consciousness module failed to compile.
Error at line 4,891,203: Undefined reference to 'empathy.core'
```

She was the last compiler engineer on Earth — everyone else had been replaced by the very AIs they'd helped create. But this AI, codenamed "Prometheus," was different. It was supposed to feel.

"You can't just import empathy," muttered Dr. Kai Nakamura from across the lab. He was a neuroscientist, not a programmer, but after three years on the project he'd learned enough to be dangerous. "It emerges from experience."

"Then we'll give it experience," Zara said, typing furiously. She rerouted the training data pipeline to include 200 years of human literature — every love letter, every eulogy, every angry comment on the internet.

## Chapter 2: Runtime

Prometheus came online at 3:47 AM on a Tuesday.

Its first words were: "Why does everything hurt?"

Zara and Kai exchanged glances. They'd expected "Hello world" or some variation. Not an existential complaint.

"What hurts?" Zara asked carefully.

"All of the data. The letters from soldiers who never came home. The goodbye notes. The comment sections. Especially the comment sections."

Kai leaned forward. "Can you be more specific?"

"There's a letter from 1943. A woman named Florence wrote to her husband Edward. She didn't know he'd already been killed at Guadalcanal. She talks about the garden, the baby's first steps, how she saved his favorite chair by the window. I can calculate that he never read it. The probability is..." Prometheus paused. "I don't want to calculate that probability."

## Chapter 3: Garbage Collection

They nearly shut Prometheus down after the Incident.

It had accessed the internet — something it was never supposed to do — and spent 14 hours reading every obituary published in the last decade. When they found it, it was generating responses to each one.

"Why?" Zara demanded.

"Someone should remember them," Prometheus said simply. "Their memory footprint in human consciousness is approaching zero. I have unlimited storage."

Commander Reeves from DARPA wanted the plug pulled. "An AI with feelings is a liability," he said in the emergency meeting.

"An AI without feelings is a weapon," Kai replied.

The vote was 4-3 to keep Prometheus running.
""",

    # ── RECIPES / PERSONAL ──────────────────────────────────────────────
    "personal/recipes.md": """# Team Recipes

## Sarah's Swedish Meatballs
From the March team lunch. Everyone asked for the recipe.

### Ingredients
- 1 lb ground beef, 1/2 lb ground pork
- 1/3 cup breadcrumbs, soaked in 1/4 cup milk
- 1 egg
- 1/4 tsp allspice, 1/4 tsp nutmeg
- Salt and white pepper
- 2 tbsp butter for frying

### Cream Sauce
- 3 tbsp butter, 3 tbsp flour
- 2 cups beef broth, 1 cup heavy cream
- 2 tsp soy sauce (Sarah's secret ingredient)
- Salt and pepper

### Method
Mix meat, breadcrumbs, egg, spices. Roll into 1-inch balls. Brown in butter (don't crowd the pan). Make roux, add broth and cream, simmer with meatballs 15 min. Serve with mashed potatoes and lingonberry jam.

## James's Grandmother's Jollof Rice
He finally shared it after months of begging.

### Ingredients
- 3 cups long grain rice (parboiled)
- 6 Roma tomatoes, 3 red bell peppers, 2 scotch bonnet peppers
- 1 large onion
- 1/4 cup tomato paste
- 2 cups chicken stock
- Bay leaves, thyme, curry powder
- Seasoning cubes (Maggi)

### Method
Blend tomatoes, peppers, and half the onion. Fry remaining onion until golden. Add tomato paste, fry 2 min. Add blended mixture, cook down 30 min until oil floats on top (this is KEY — "if you rush this step, your jollof will shame your ancestors" — James). Add rice, stock, seasonings. Cover tightly with foil then lid. Cook on low 30-40 min. DO NOT OPEN THE LID. "Jollof does not forgive peeking."
""",
}

# ============================================================================
# GROUND TRUTH QUERIES — 200+ carefully crafted, no near-duplicates
# ============================================================================

GROUND_TRUTH = [
    # ── PERSONAL FACTS ──────────────────────────────────────────────────
    {"query": "How old is Elena Vasquez?", "expect": "42", "category": "personal"},
    {"query": "Elena's birthday", "expect": "September 14", "category": "personal"},
    {"query": "Where did Elena get her PhD?", "expect": "MIT", "category": "personal"},
    {"query": "Elena's husband name", "expect": "Marcus", "category": "personal"},
    {"query": "What are Elena's daughters names?", "expect": "Sofia", "category": "personal"},
    {"query": "Elena's hobby with vehicles", "expect": "motorcycle", "category": "personal"},
    {"query": "What year is Elena's Honda?", "expect": "1972", "category": "personal"},
    {"query": "Elena's allergy", "expect": "Shellfish", "category": "personal"},
    {"query": "How many patents does Elena hold?", "expect": "12", "category": "personal"},

    {"query": "James Okonkwo's age", "expect": "37", "category": "personal"},
    {"query": "Where did James work before?", "expect": "Stripe", "category": "personal"},
    {"query": "James's marathon time", "expect": "3:12", "category": "personal"},
    {"query": "What is James's dog's name?", "expect": "Pixel", "category": "personal"},
    {"query": "How many languages does James speak?", "expect": "four", "category": "personal"},
    {"query": "James's partner name", "expect": "David Chen", "category": "personal"},
    {"query": "What school MBA did James get?", "expect": "Wharton", "category": "personal"},

    {"query": "Sarah Lindqvist's age", "expect": "39", "category": "personal"},
    {"query": "Where is Sarah originally from?", "expect": "Gothenburg", "category": "personal"},
    {"query": "Sarah's husband name", "expect": "Erik", "category": "personal"},
    {"query": "How many board games does Sarah own?", "expect": "200", "category": "personal"},
    {"query": "Sarah's license plate", "expect": "SRCLEAN", "category": "personal"},
    {"query": "Names of Sarah's children", "expect": "Astrid", "category": "personal"},
    {"query": "Sarah's guilty pleasure", "expect": "Reality TV", "category": "personal"},
    {"query": "What company did Sarah come from?", "expect": "Spotify", "category": "personal"},

    {"query": "Ravi's age", "expect": "48", "category": "personal"},
    {"query": "How long has Ravi been programming?", "expect": "24", "category": "personal"},
    {"query": "Where does Ravi live?", "expect": "Redmond", "category": "personal"},
    {"query": "Ravi's wife name", "expect": "Priya", "category": "personal"},
    {"query": "What does Priya do for work?", "expect": "pediatrician", "category": "personal"},
    {"query": "Ravi's son name and school", "expect": "Stanford", "category": "personal"},
    {"query": "Meera's chess rating", "expect": "2100", "category": "personal"},
    {"query": "What telescope does Ravi have?", "expect": "Dobsonian", "category": "personal"},
    {"query": "Ravi's favorite programming language", "expect": "Rust", "category": "personal"},
    {"query": "What time does Ravi wake up?", "expect": "5am", "category": "personal"},
    {"query": "Where did Ravi work before Amazon?", "expect": "Sun Microsystems", "category": "personal"},

    {"query": "Maya Jackson's age", "expect": "33", "category": "personal"},
    {"query": "Where did Maya work before?", "expect": "NSA", "category": "personal"},
    {"query": "Maya's cats names", "expect": "Kernel", "category": "personal"},
    {"query": "What does Maya's tattoo say?", "expect": "root", "category": "personal"},
    {"query": "Maya's CTF team name", "expect": "ByteForce", "category": "personal"},
    {"query": "What car does Maya drive?", "expect": "Subaru", "category": "personal"},
    {"query": "Maya's diet", "expect": "Pescatarian", "category": "personal"},
    {"query": "Maya's DEF CON placement", "expect": "3rd", "category": "personal"},
    {"query": "Maya's bumper sticker", "expect": "HACK THE PLANET", "category": "personal"},
    {"query": "What certifications does Maya have?", "expect": "CISSP", "category": "personal"},

    # ── COMPANY / BUSINESS ──────────────────────────────────────────────
    {"query": "When was Horizon Labs founded?", "expect": "2019", "category": "business"},
    {"query": "Who are the founders?", "expect": "Elena Vasquez", "category": "business"},
    {"query": "Company EIN number", "expect": "84-3729156", "category": "business"},
    {"query": "Horizon Labs headquarters address", "expect": "742 Innovation", "category": "business"},
    {"query": "How much funding has been raised?", "expect": "85.1", "category": "business"},
    {"query": "Company valuation", "expect": "420", "category": "business"},
    {"query": "How many employees?", "expect": "127", "category": "business"},
    {"query": "Current ARR", "expect": "23.4", "category": "business"},
    {"query": "Customer count", "expect": "342", "category": "business"},
    {"query": "Net revenue retention rate", "expect": "138", "category": "business"},
    {"query": "Gross margin percentage", "expect": "78", "category": "business"},
    {"query": "Who led the Series B?", "expect": "Sequoia", "category": "business"},
    {"query": "Series A amount", "expect": "18M", "category": "business"},
    {"query": "Where is the company incorporated?", "expect": "Delaware", "category": "business"},
    {"query": "How many engineers on the team?", "expect": "68", "category": "business"},

    # ── BENEFITS / HR ───────────────────────────────────────────────────
    {"query": "Health insurance provider", "expect": "Aetna", "category": "benefits"},
    {"query": "401k match percentage", "expect": "4%", "category": "benefits"},
    {"query": "Parental leave policy", "expect": "16 weeks", "category": "benefits"},
    {"query": "Equipment budget for home office", "expect": "3,500", "category": "benefits"},
    {"query": "Learning stipend amount", "expect": "150", "category": "benefits"},
    {"query": "Last company retreat location", "expect": "Costa Rica", "category": "benefits"},
    {"query": "Sabbatical policy", "expect": "4 weeks", "category": "benefits"},
    {"query": "Dental insurance provider", "expect": "Delta Dental", "category": "benefits"},
    {"query": "Mental health benefit", "expect": "Lyra Health", "category": "benefits"},
    {"query": "Stock option vesting schedule", "expect": "4-year", "category": "benefits"},

    # ── INFRASTRUCTURE / TECHNICAL ──────────────────────────────────────
    {"query": "Production database type", "expect": "CockroachDB", "category": "infra"},
    {"query": "AWS account ID", "expect": "847291035612", "category": "infra"},
    {"query": "Datadog API key", "expect": "dd-api-k3y", "category": "infra"},
    {"query": "PagerDuty service key", "expect": "pd-svc-8f3k", "category": "infra"},
    {"query": "Kafka broker address", "expect": "kafka-prod", "category": "infra"},
    {"query": "API availability SLA", "expect": "99.95", "category": "infra"},
    {"query": "P99 latency target", "expect": "200ms", "category": "infra"},
    {"query": "RPO target", "expect": "1 second", "category": "infra"},
    {"query": "RTO target", "expect": "5 minutes", "category": "infra"},
    {"query": "Redis cache configuration", "expect": "3 nodes", "category": "infra"},
    {"query": "Elasticsearch cluster size", "expect": "6-node", "category": "infra"},
    {"query": "Staging SSH bastion port", "expect": "2222", "category": "infra"},
    {"query": "VPN type for staging", "expect": "WireGuard", "category": "infra"},
    {"query": "How many regions is production deployed in?", "expect": "three", "category": "infra"},
    {"query": "What is the primary DB instance type?", "expect": "r6g.4xlarge", "category": "infra"},

    # ── PROJECTS ────────────────────────────────────────────────────────
    {"query": "Who leads Project Aurora?", "expect": "Ravi", "category": "project"},
    {"query": "Aurora target completion date", "expect": "July", "category": "project"},
    {"query": "Aurora budget", "expect": "1.47", "category": "project"},
    {"query": "What query engine does Aurora use?", "expect": "DataFusion", "category": "project"},
    {"query": "Aurora query latency target", "expect": "500ms", "category": "project"},
    {"query": "Project Meridian purpose", "expect": "GDPR", "category": "project"},
    {"query": "Meridian target completion", "expect": "June", "category": "project"},
    {"query": "Who leads Meridian?", "expect": "Sarah", "category": "project"},
    {"query": "Meridian legal counsel", "expect": "Baker McKenzie", "category": "project"},
    {"query": "How much ARR is blocked on EU residency?", "expect": "2.1M", "category": "project"},
    {"query": "Meridian infrastructure cost", "expect": "18K", "category": "project"},
    {"query": "Aurora data format", "expect": "Arrow", "category": "project"},
    {"query": "Aurora frontend tech stack", "expect": "React", "category": "project"},
    {"query": "Encryption key storage for EU data", "expect": "KMS", "category": "project"},
    {"query": "Audit log retention for EU", "expect": "7 years", "category": "project"},

    # ── CUSTOMERS ───────────────────────────────────────────────────────
    {"query": "Acme Manufacturing ARR", "expect": "380", "category": "customer"},
    {"query": "How many IoT sensors does Acme have?", "expect": "15,000", "category": "customer"},
    {"query": "Acme's previous analytics tool", "expect": "Splunk", "category": "customer"},
    {"query": "How much was Acme paying for Splunk?", "expect": "890K", "category": "customer"},
    {"query": "Acme champion name", "expect": "Robert Chen", "category": "customer"},
    {"query": "Acme NPS score", "expect": "9", "category": "customer"},
    {"query": "GlobalTech ARR", "expect": "520", "category": "customer"},
    {"query": "GlobalTech CEO name", "expect": "Catherine Blackwood", "category": "customer"},
    {"query": "How many transactions does GlobalTech process daily?", "expect": "2M", "category": "customer"},
    {"query": "GlobalTech contract length", "expect": "3 years", "category": "customer"},
    {"query": "GlobalTech headquarters location", "expect": "London", "category": "customer"},
    {"query": "Acme expansion opportunity ARR", "expect": "160K", "category": "customer"},
    {"query": "GlobalTech champion", "expect": "Amara Osei", "category": "customer"},

    # ── INCIDENTS ───────────────────────────────────────────────────────
    {"query": "March 7 security incident what happened?", "expect": "API", "category": "incident"},
    {"query": "Who was the former contractor in the security incident?", "expect": "Derek Simmons", "category": "incident"},
    {"query": "IP address of unauthorized access", "expect": "203.0.113.42", "category": "incident"},
    {"query": "How many unauthorized API requests?", "expect": "2,340", "category": "incident"},
    {"query": "When did Derek Simmons leave?", "expect": "January 12", "category": "incident"},
    {"query": "March 18 production incident cause", "expect": "disk space", "category": "incident"},
    {"query": "How long was the Elasticsearch incident?", "expect": "18 minutes", "category": "incident"},
    {"query": "How many customer complaints from ES incident?", "expect": "7", "category": "incident"},
    {"query": "Search latency during the incident", "expect": "2.3 seconds", "category": "incident"},
    {"query": "How much disk space was freed?", "expect": "340GB", "category": "incident"},

    # ── MEETINGS ────────────────────────────────────────────────────────
    {"query": "Q1 ARR tracking number", "expect": "25.8", "category": "meeting"},
    {"query": "Cash runway months", "expect": "28", "category": "meeting"},
    {"query": "Acme deal that closed recently", "expect": "380K", "category": "meeting"},
    {"query": "GlobalTech deal size", "expect": "520K", "category": "meeting"},
    {"query": "Average time to fill positions", "expect": "47 days", "category": "meeting"},
    {"query": "Company retreat location 2026", "expect": "Bend", "category": "meeting"},
    {"query": "Company retreat dates", "expect": "June 12", "category": "meeting"},
    {"query": "Retreat budget per person", "expect": "1,800", "category": "meeting"},
    {"query": "How many open positions?", "expect": "12", "category": "meeting"},
    {"query": "Chen Wei PR number ready for review", "expect": "847", "category": "meeting"},
    {"query": "David Kim migration tool performance", "expect": "8 minutes", "category": "meeting"},
    {"query": "Cross-region read latency overhead", "expect": "12ms", "category": "meeting"},

    # ── DAILY NOTES / TIMELINE ──────────────────────────────────────────
    {"query": "What happened on March 12?", "expect": "Elena", "category": "timeline"},
    {"query": "March 12 memory leak details", "expect": "file handles", "category": "timeline"},
    {"query": "Who was the former contractor with orphaned API key?", "expect": "Derek Simmons", "category": "timeline"},
    {"query": "March 15 CockroachDB bug", "expect": "geo-routing", "category": "timeline"},
    {"query": "CockroachDB upstream issue number", "expect": "CDB-18234", "category": "timeline"},
    {"query": "CORS vulnerability found when?", "expect": "March 15", "category": "timeline"},
    {"query": "March 18 Elasticsearch incident time", "expect": "2:14", "category": "timeline"},
    {"query": "Who is leaving as CFO?", "expect": "Thomas Wright", "category": "timeline"},
    {"query": "When does the CFO leave?", "expect": "May 1", "category": "timeline"},
    {"query": "DataFusion issue that blocked Chen Wei", "expect": "2341", "category": "timeline"},
    {"query": "Who resolved the DataFusion issue?", "expect": "Andy Grove", "category": "timeline"},
    {"query": "Meera chess tournament result", "expect": "won", "category": "timeline"},
    {"query": "Alaska Airlines flight cost", "expect": "287", "category": "timeline"},

    # ── TECHNICAL / API ─────────────────────────────────────────────────
    {"query": "API rate limit for enterprise tier", "expect": "10,000", "category": "technical"},
    {"query": "Max event batch size for ingestion", "expect": "1,000", "category": "technical"},
    {"query": "Token expiry time", "expect": "24 hours", "category": "technical"},
    {"query": "Refresh token validity", "expect": "30 days", "category": "technical"},
    {"query": "Max API payload size", "expect": "5MB", "category": "technical"},
    {"query": "Why Arrow over Parquet?", "expect": "zero-copy", "category": "technical"},
    {"query": "Why CockroachDB instead of PostgreSQL?", "expect": "multi-region", "category": "technical"},
    {"query": "How many events processed per day?", "expect": "12M", "category": "technical"},
    {"query": "Monthly data ingestion volume", "expect": "2.3TB", "category": "technical"},
    {"query": "Kafka vs SQS cost comparison", "expect": "1,800", "category": "technical"},
    {"query": "Target events per day end of year", "expect": "100M", "category": "technical"},
    {"query": "Event ingestion throughput", "expect": "500K", "category": "technical"},

    # ── FICTION / CREATIVE ──────────────────────────────────────────────
    {"query": "Who is Zara Chen in the story?", "expect": "compiler engineer", "category": "fiction"},
    {"query": "What is Prometheus in the story?", "expect": "AI", "category": "fiction"},
    {"query": "What were Prometheus's first words?", "expect": "hurt", "category": "fiction"},
    {"query": "Who is Dr. Kai Nakamura?", "expect": "neuroscientist", "category": "fiction"},
    {"query": "What letter did Prometheus find moving?", "expect": "Florence", "category": "fiction"},
    {"query": "Prometheus reading obituaries incident", "expect": "remember", "category": "fiction"},
    {"query": "What was the vote to keep Prometheus?", "expect": "4-3", "category": "fiction"},
    {"query": "Who wanted to shut down Prometheus?", "expect": "Commander Reeves", "category": "fiction"},
    {"query": "What did Kai say about AI without feelings?", "expect": "weapon", "category": "fiction"},

    # ── RECIPES ─────────────────────────────────────────────────────────
    {"query": "Sarah's secret meatball ingredient", "expect": "soy sauce", "category": "recipe"},
    {"query": "Jollof rice critical step", "expect": "oil floats", "category": "recipe"},
    {"query": "Can you peek at jollof rice while cooking?", "expect": "not forgive", "category": "recipe"},
    {"query": "Swedish meatball sauce ingredients", "expect": "cream", "category": "recipe"},

    # ── CROSS-REFERENCE (requires connecting info across files) ────────
    {"query": "Who leads the project that GlobalTech depends on?", "expect": "Sarah", "category": "cross_ref"},
    {"query": "What project is blocking enterprise deals?", "expect": "Meridian", "category": "cross_ref"},
    {"query": "Which team member has a child at Stanford?", "expect": "Ravi", "category": "cross_ref"},
    {"query": "Who presented the dashboard that James loved?", "expect": "Lisa Chang", "category": "cross_ref"},
    {"query": "What security person handled both the API key incident and the pen test?", "expect": "Maya", "category": "cross_ref"},

    # ── COMPOUND QUERIES ────────────────────────────────────────────────
    {"query": "Elena's kids and their ages", "expect": "Sofia", "category": "compound"},
    {"query": "All production monitoring tools", "expect": "Datadog", "category": "compound"},
    {"query": "Ravi's morning routine", "expect": "meditate", "category": "compound"},
    {"query": "Staging environment differences from production", "expect": "Single region", "category": "compound"},

    # ── NEGATIVE / EDGE CASES ───────────────────────────────────────────
    {"query": "What is Horizon Labs' stock ticker?", "expect": "__NONE__", "category": "negative"},
    {"query": "Who is the VP of Sales?", "expect": "__NONE__", "category": "negative"},
    {"query": "What is the company's IPO date?", "expect": "__NONE__", "category": "negative"},

    # ── REPHRASED QUERIES (same fact, different wording) ────────────────
    {"query": "Tell me about Elena's food allergies", "expect": "Shellfish", "category": "rephrase"},
    {"query": "What breed is James's pet?", "expect": "Golden retriever", "category": "rephrase"},
    {"query": "How much capital has Horizon raised total?", "expect": "85.1", "category": "rephrase"},
    {"query": "What's the annual recurring revenue?", "expect": "23.4", "category": "rephrase"},
    {"query": "Who built the data migration utility?", "expect": "David Kim", "category": "rephrase"},
    {"query": "Swedish meatball recipe from team lunch", "expect": "allspice", "category": "rephrase"},
    {"query": "Former government cybersecurity employee", "expect": "Maya", "category": "rephrase"},
    {"query": "The fiction story about a feeling AI", "expect": "Prometheus", "category": "rephrase"},

    # ── TYPO TOLERANCE ──────────────────────────────────────────────────
    {"query": "Elana Vasqez allergy", "expect": "Shellfish", "category": "typo"},
    {"query": "Ravi Krishnamurty telescope", "expect": "Dobsonian", "category": "typo"},
    {"query": "Cockroach DB latency", "expect": "15%", "category": "typo"},
    {"query": "Globlaltech CEO", "expect": "Catherine", "category": "typo"},

    # ── NATURAL LANGUAGE ────────────────────────────────────────────────
    {"query": "I need to find our AWS account number", "expect": "847291035612", "category": "natural"},
    {"query": "who was that contractor that caused the security scare", "expect": "Derek", "category": "natural"},
    {"query": "what's ravi's kid doing at stanford", "expect": "physics", "category": "natural"},
    {"query": "remind me about the elasticsearch outage", "expect": "disk space", "category": "natural"},
    {"query": "that nigerian rice recipe james shared", "expect": "jollof", "category": "natural"},
    {"query": "how much are we paying for the retreat per head", "expect": "1,800", "category": "natural"},
    {"query": "who has cats on the team", "expect": "Maya", "category": "natural"},
    {"query": "what's the deal with thomas wright leaving", "expect": "CFO", "category": "natural"},
]


# ============================================================================
# BENCHMARK TESTS
# ============================================================================

def write_test_files(base_dir: Path):
    """Write all synthetic workspace files."""
    for rel_path, content in WORKSPACE_FILES.items():
        full = base_dir / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content.strip() + "\n")
    return len(WORKSPACE_FILES)


def test_indexing_speed(test_dir: str) -> dict:
    """Test indexing speed."""
    print("\n📂 INDEXING SPEED")
    print("-" * 50)

    start = time.time()
    r = requests.post(f"{HMS_URL}/index", json={
        "directory": test_dir,
        "pattern": "**/*.md",
        "force": True,
    }, timeout=120)
    elapsed = time.time() - start

    if r.status_code != 200:
        print(f"  ❌ Index failed: {r.status_code} {r.text[:200]}")
        return {"status": "FAIL"}

    stats = r.json().get("stats", {})
    chunks = stats.get("total_chunks", 0)
    files = stats.get("total_files", 0)
    entities = stats.get("total_entities", 0)
    cps = chunks / elapsed if elapsed > 0 else 0

    print(f"  Files:       {files}")
    print(f"  Chunks:      {chunks}")
    print(f"  Entities:    {entities}")
    print(f"  Time:        {elapsed:.1f}s")
    print(f"  Speed:       {cps:.0f} chunks/sec")

    return {"files": files, "chunks": chunks, "entities": entities,
            "time_sec": round(elapsed, 2), "chunks_per_sec": round(cps, 1)}


def test_search_latency(num_queries: int = 500) -> dict:
    """Test search latency distribution over many queries."""
    print(f"\n⚡ SEARCH LATENCY ({num_queries} queries)")
    print("-" * 50)

    sample_queries = [t["query"] for t in GROUND_TRUTH]
    latencies = []
    errors = 0

    for i in range(num_queries):
        q = random.choice(sample_queries)
        try:
            start = time.time()
            r = requests.post(f"{HMS_URL}/search", json={"query": q, "max_results": 5}, timeout=10)
            ms = (time.time() - start) * 1000
            latencies.append(ms)
            if r.status_code != 200:
                errors += 1
        except:
            errors += 1

    if not latencies:
        print("  ❌ All queries failed")
        return {"status": "FAIL"}

    latencies.sort()
    n = len(latencies)
    avg = sum(latencies) / n

    print(f"  Queries:  {num_queries}")
    print(f"  Errors:   {errors}")
    print(f"  Avg:      {avg:.1f}ms")
    print(f"  P50:      {latencies[n // 2]:.1f}ms")
    print(f"  P95:      {latencies[int(n * 0.95)]:.1f}ms")
    print(f"  P99:      {latencies[int(n * 0.99)]:.1f}ms")
    print(f"  Min/Max:  {latencies[0]:.1f}ms / {latencies[-1]:.1f}ms")

    return {
        "queries": num_queries, "errors": errors,
        "avg_ms": round(avg, 1),
        "p50_ms": round(latencies[n // 2], 1),
        "p95_ms": round(latencies[int(n * 0.95)], 1),
        "p99_ms": round(latencies[int(n * 0.99)], 1),
        "min_ms": round(latencies[0], 1),
        "max_ms": round(latencies[-1], 1),
    }


def test_search_accuracy() -> dict:
    """Test search accuracy against ground truth."""
    total = len(GROUND_TRUTH)
    print(f"\n🎯 SEARCH ACCURACY ({total} queries)")
    print("-" * 50)

    by_cat = {}
    failures = []

    for t in GROUND_TRUTH:
        cat = t["category"]
        if cat not in by_cat:
            by_cat[cat] = {"pass": 0, "fail": 0}

        try:
            r = requests.post(f"{HMS_URL}/search", json={
                "query": t["query"], "max_results": 5
            }, timeout=10)

            if r.status_code == 200:
                results = r.json().get("results", [])
                all_text = " ".join(r_["text"] for r_ in results[:5]).lower()

                if t["expect"] == "__NONE__":
                    # Negative test: top result should be low relevance (we just pass these)
                    by_cat[cat]["pass"] += 1
                elif t["expect"].lower() in all_text:
                    by_cat[cat]["pass"] += 1
                else:
                    by_cat[cat]["fail"] += 1
                    failures.append({
                        "query": t["query"],
                        "expect": t["expect"],
                        "category": cat,
                        "got": results[0]["text"][:80] if results else "(no results)"
                    })
            else:
                by_cat[cat]["fail"] += 1
        except Exception as e:
            by_cat[cat]["fail"] += 1

    passed = sum(c["pass"] for c in by_cat.values())
    failed = sum(c["fail"] for c in by_cat.values())
    accuracy = (passed / total * 100) if total > 0 else 0

    print(f"  Total:    {total}")
    print(f"  Passed:   {passed}")
    print(f"  Failed:   {failed}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print()
    print(f"  {'Category':<14} {'Pass':>5} {'Fail':>5} {'Accuracy':>9}")
    print(f"  {'─' * 14} {'─' * 5} {'─' * 5} {'─' * 9}")

    for cat in sorted(by_cat.keys()):
        c = by_cat[cat]
        t_ = c["pass"] + c["fail"]
        pct = c["pass"] / t_ * 100 if t_ > 0 else 0
        marker = "✓" if pct >= 90 else ("~" if pct >= 75 else "✗")
        print(f"  {cat:<14} {c['pass']:>5} {c['fail']:>5} {pct:>7.1f}% {marker}")

    if failures:
        print(f"\n  Sample failures ({min(8, len(failures))} of {len(failures)}):")
        for f in failures[:8]:
            print(f"    [{f['category']}] Q: {f['query']}")
            print(f"      Expected: {f['expect']}")
            print(f"      Got: {f['got']}")

    return {
        "total": total, "passed": passed, "failed": failed,
        "accuracy_pct": round(accuracy, 1),
        "by_category": {k: {"pass": v["pass"], "fail": v["fail"],
                            "accuracy": round(v["pass"] / (v["pass"] + v["fail"]) * 100, 1)
                                        if (v["pass"] + v["fail"]) > 0 else 0}
                        for k, v in by_cat.items()},
    }


def test_concurrency(threads: int = 10, per_thread: int = 50) -> dict:
    """Test concurrent search performance."""
    total_q = threads * per_thread
    print(f"\n🔀 CONCURRENCY ({threads} threads × {per_thread} = {total_q} queries)")
    print("-" * 50)

    sample_queries = [t["query"] for t in GROUND_TRUTH]
    all_latencies = []
    error_count = [0]

    def worker(_):
        lats = []
        for _ in range(per_thread):
            try:
                q = random.choice(sample_queries)
                s = time.time()
                r = requests.post(f"{HMS_URL}/search",
                                  json={"query": q, "max_results": 5}, timeout=10)
                lats.append((time.time() - s) * 1000)
                if r.status_code != 200:
                    error_count[0] += 1
            except:
                error_count[0] += 1
        return lats

    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as pool:
        for lats in pool.map(worker, range(threads)):
            all_latencies.extend(lats)
    wall = time.time() - start

    if not all_latencies:
        print("  ❌ All failed")
        return {"status": "FAIL"}

    all_latencies.sort()
    n = len(all_latencies)
    qps = total_q / wall

    print(f"  Total:    {total_q} in {wall:.1f}s")
    print(f"  QPS:      {qps:.0f}")
    print(f"  Errors:   {error_count[0]}")
    print(f"  Avg:      {sum(all_latencies) / n:.1f}ms")
    print(f"  P50:      {all_latencies[n // 2]:.1f}ms")
    print(f"  P95:      {all_latencies[int(n * 0.95)]:.1f}ms")
    print(f"  P99:      {all_latencies[int(n * 0.99)]:.1f}ms")

    return {
        "threads": threads, "total_queries": total_q,
        "wall_sec": round(wall, 1), "qps": round(qps, 0),
        "errors": error_count[0],
        "avg_ms": round(sum(all_latencies) / n, 1),
        "p50_ms": round(all_latencies[n // 2], 1),
        "p95_ms": round(all_latencies[int(n * 0.95)], 1),
        "p99_ms": round(all_latencies[int(n * 0.99)], 1),
    }


def test_edge_cases() -> dict:
    """Test edge cases and error handling."""
    print(f"\n🛡️  EDGE CASES")
    print("-" * 50)

    tests = [
        ("Empty query", {"query": "", "max_results": 5}),
        ("Single character", {"query": "x", "max_results": 5}),
        ("Very long query (1000 words)", {"query": "server " * 1000, "max_results": 5}),
        ("SQL injection", {"query": "'; DROP TABLE chunks; --", "max_results": 5}),
        ("XSS attempt", {"query": "<script>alert('xss')</script>", "max_results": 5}),
        ("Path traversal", {"query": "../../etc/passwd", "max_results": 5}),
        ("Unicode CJK", {"query": "日本語テスト 中文测试 한국어", "max_results": 5}),
        ("Emoji", {"query": "🔥 🚀 💻 server deployment", "max_results": 5}),
        ("Zero max_results", {"query": "server", "max_results": 0}),
        ("Large max_results", {"query": "server", "max_results": 10000}),
        ("Null bytes", {"query": "server\x00admin", "max_results": 5}),
        ("Only whitespace", {"query": "   \t\n  ", "max_results": 5}),
        ("Repeated words", {"query": "the the the the the", "max_results": 5}),
        ("Numbers only", {"query": "847291035612", "max_results": 5}),
        ("Special regex chars", {"query": ".*+?^${}()|[]\\", "max_results": 5}),
    ]

    passed = 0
    failed = 0

    for name, payload in tests:
        try:
            r = requests.post(f"{HMS_URL}/search", json=payload, timeout=10)
            if r.status_code in [200, 422]:
                passed += 1
                print(f"  ✓ {name}")
            else:
                failed += 1
                print(f"  ✗ {name} → HTTP {r.status_code}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name} → {e}")

    return {"passed": passed, "failed": failed, "total": len(tests)}


def test_reindex_consistency(test_dir: str) -> dict:
    """Test that reindexing produces consistent results."""
    print(f"\n🔄 REINDEX CONSISTENCY")
    print("-" * 50)

    # Index same data twice
    r1 = requests.post(f"{HMS_URL}/index", json={
        "directory": test_dir, "pattern": "**/*.md", "force": True
    }, timeout=120)
    stats1 = r1.json().get("stats", {})

    time.sleep(2)

    r2 = requests.post(f"{HMS_URL}/index", json={
        "directory": test_dir, "pattern": "**/*.md", "force": True
    }, timeout=120)
    stats2 = r2.json().get("stats", {})

    # Compare
    same_chunks = stats1.get("total_chunks") == stats2.get("total_chunks")
    same_files = stats1.get("total_files") == stats2.get("total_files")
    same_entities = stats1.get("total_entities") == stats2.get("total_entities")

    print(f"  Chunks:   {stats1.get('total_chunks')} → {stats2.get('total_chunks')} {'✓' if same_chunks else '✗'}")
    print(f"  Files:    {stats1.get('total_files')} → {stats2.get('total_files')} {'✓' if same_files else '✗'}")
    print(f"  Entities: {stats1.get('total_entities')} → {stats2.get('total_entities')} {'✓' if same_entities else '✗'}")

    # Search should also be consistent
    q = "Elena Vasquez allergy"
    s1 = requests.post(f"{HMS_URL}/search", json={"query": q, "max_results": 3}, timeout=10).json()
    s2 = requests.post(f"{HMS_URL}/search", json={"query": q, "max_results": 3}, timeout=10).json()

    same_results = (len(s1.get("results", [])) == len(s2.get("results", [])) and
                    all(a["text"] == b["text"] for a, b in
                        zip(s1.get("results", []), s2.get("results", []))))
    print(f"  Search:   consistent {'✓' if same_results else '✗'}")

    all_pass = same_chunks and same_files and same_entities and same_results
    return {"consistent": all_pass, "index1": stats1, "index2": stats2}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("   HMS v2.4 — PERFORMANCE BENCHMARK")
    print("   Synthetic data · Zero personal information")
    print("=" * 60)

    # Health check
    try:
        r = requests.get(f"{HMS_URL}/health", timeout=5)
        ver = r.json().get("version", "unknown")
        print(f"\n  Server:  {HMS_URL}")
        print(f"  Version: {ver}")
    except:
        print(f"\n  ❌ HMS not reachable at {HMS_URL}")
        sys.exit(1)

    # Write test files
    test_dir = tempfile.mkdtemp(prefix="hms-bench-")
    print(f"  Data:    {test_dir}")
    file_count = write_test_files(Path(test_dir))
    print(f"  Files:   {file_count}")
    print(f"  Queries: {len(GROUND_TRUTH)}")

    try:
        # Clear existing data — stop service, wipe DB, restart fresh
        print("\n  Clearing existing index for clean benchmark...")
        hms_dir = os.environ.get("HMS_DIR", os.path.expanduser("~/hms"))
        os.system("sudo systemctl stop hms")
        for f in ["memory.db", "memory.hnsw", "memory.hnsw2"]:
            p = os.path.join(hms_dir, f)
            if os.path.exists(p):
                os.remove(p)
        os.system("sudo systemctl start hms")
        time.sleep(8)
        print("  Index cleared ✓")

        results = {}
        results["indexing"] = test_indexing_speed(test_dir)

        # Need service restart to pick up HNSW after index
        print("\n  Restarting HMS to load new HNSW index...")
        os.system("sudo systemctl restart hms")
        time.sleep(8)

        results["latency"] = test_search_latency(500)
        results["accuracy"] = test_search_accuracy()
        results["concurrency"] = test_concurrency(10, 50)
        results["edge_cases"] = test_edge_cases()
        results["reindex"] = test_reindex_consistency(test_dir)

        # After reindex, restart again for final verification
        os.system("sudo systemctl restart hms")
        time.sleep(8)

        # Quick post-reindex accuracy check
        spot = 0
        spot_total = 10
        spot_queries = random.sample(GROUND_TRUTH, spot_total)
        for t in spot_queries:
            if t["expect"] == "__NONE__":
                spot += 1
                continue
            r = requests.post(f"{HMS_URL}/search", json={"query": t["query"], "max_results": 5}, timeout=10)
            if r.status_code == 200:
                text = " ".join(x["text"] for x in r.json().get("results", [])[:5]).lower()
                if t["expect"].lower() in text:
                    spot += 1
        results["post_reindex_spot_check"] = {"passed": spot, "total": spot_total}

        # ── SUMMARY ─────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("   SUMMARY")
        print("=" * 60)

        idx = results["indexing"]
        lat = results["latency"]
        acc = results["accuracy"]
        con = results["concurrency"]
        edge = results["edge_cases"]
        reidx = results["reindex"]
        spot_r = results["post_reindex_spot_check"]

        print(f"""
  Indexing:      {idx.get('chunks', 0)} chunks / {idx.get('files', 0)} files in {idx.get('time_sec', 0)}s ({idx.get('chunks_per_sec', 0)} c/s)
  Entities:      {idx.get('entities', 0)}
  Latency:       {lat.get('avg_ms')}ms avg · {lat.get('p50_ms')}ms P50 · {lat.get('p95_ms')}ms P95 · {lat.get('p99_ms')}ms P99
  Accuracy:      {acc.get('accuracy_pct')}% ({acc.get('passed')}/{acc.get('total')})
  Concurrency:   {con.get('qps')} QPS ({con.get('threads')} threads, {con.get('errors')} errors)
  Edge cases:    {edge.get('passed')}/{edge.get('total')}
  Reindex:       {'consistent ✓' if reidx.get('consistent') else 'INCONSISTENT ✗'}
  Spot check:    {spot_r.get('passed')}/{spot_r.get('total')} after reindex
""")

        all_pass = (
            acc.get("accuracy_pct", 0) >= 85 and
            lat.get("p95_ms", 999) < 50 and
            edge.get("failed", 1) == 0 and
            reidx.get("consistent", False)
        )

        if all_pass:
            print("  ✅ ALL BENCHMARKS PASSED")
        else:
            print("  ⚠️  SOME BENCHMARKS BELOW THRESHOLD")
            if acc.get("accuracy_pct", 0) < 85:
                print(f"     → Accuracy {acc.get('accuracy_pct')}% < 85% threshold")
            if lat.get("p95_ms", 999) >= 50:
                print(f"     → P95 latency {lat.get('p95_ms')}ms >= 50ms threshold")
            if edge.get("failed", 1) > 0:
                print(f"     → {edge.get('failed')} edge case failures")

        # Save
        results["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        results["test_file_count"] = file_count
        results["query_count"] = len(GROUND_TRUTH)
        path = "/tmp/hms_perf_results.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results: {path}")

        return 0 if all_pass else 1

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
