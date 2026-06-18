"""
PASS 1 TIMING DIAGNOSTIC
Times each sub-step of Stages 0, 1, 2 independently.
No code changes — read-only profiling.
"""
import json
import os
import pickle
import time

import numpy as np

BASE = r'C:\Users\pranj\Desktop\Redrob Hackathon'
PRECOMPUTED = os.path.join(BASE, 'precomputed')
CANDIDATES_PATH = os.path.join(BASE, 'candidates.jsonl')
DATA_DIR = os.path.join(BASE, 'data')

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def hms(t): return f"{t*1000:.1f}ms" if t < 1 else f"{t:.3f}s"

print("=" * 70)
print("PASS 1 TIMING DIAGNOSTIC")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────
# STAGE 0 — Artifact loading, each file timed independently
# ─────────────────────────────────────────────────────────────────────
print("\n── STAGE 0: ARTIFACT LOADING ──")

for fname in ['bm25_index.pkl', 'candidate_ids.pkl', 'lgbm_model.pkl']:
    fpath = os.path.join(PRECOMPUTED, fname)
    size_mb = os.path.getsize(fpath) / 1e6
    t0 = time.perf_counter()
    with open(fpath, 'rb') as f:
        obj = pickle.load(f)
    elapsed = time.perf_counter() - t0
    extra = ""
    if fname == 'bm25_index.pkl':
        extra = f"  [type={type(obj).__name__}, idf_len={len(getattr(obj, 'idf', []))}]"
        bm25 = obj
    elif fname == 'candidate_ids.pkl':
        extra = f"  [n_ids={len(obj)}]"
        candidate_ids = obj
    elif fname == 'lgbm_model.pkl':
        extra = f"  [type={type(obj).__name__}]"
        model = obj
    print(f"  {fname:25s}  {size_mb:7.1f} MB   load={hms(elapsed)}{extra}")

# Check if BM25Okapi stores corpus internally
print("\n  BM25Okapi internal fields:")
for attr in ['corpus', 'corpus_size', 'avgdl', 'doc_freqs', 'idf', 'doc_len', 'nd']:
    val = getattr(bm25, attr, 'MISSING')
    if attr == 'corpus':
        if hasattr(val, '__len__'):
            print(f"    .corpus          present, len={len(val)}, type={type(val)}")
            if len(val) > 0:
                print(f"    .corpus[0]       type={type(val[0])}, len={len(val[0]) if hasattr(val[0], '__len__') else 'N/A'}")
        else:
            print(f"    .corpus          {val}")
    elif attr == 'doc_freqs':
        if hasattr(val, '__len__'):
            print(f"    .doc_freqs       present, len={len(val)}, sample={list(val[0].items())[:3] if val else '[]'}")
        else:
            print(f"    .doc_freqs       {val}")
    else:
        print(f"    .{attr:16s} {val if not hasattr(val, '__len__') else f'len={len(val)}'}")

# ─────────────────────────────────────────────────────────────────────
# STAGE 1 — Dual-pass retrieval, each sub-step timed independently
# ─────────────────────────────────────────────────────────────────────
print("\n── STAGE 1: DUAL-PASS RETRIEVAL (sub-step breakdown) ──")

from jd_parser import parse_jd
from retrieval import tokenize_query

jd_config = parse_jd(os.path.join(DATA_DIR, 'skill_aliases.json'))

query_a_terms = jd_config.get_all_query_terms()
query_a_tokens = tokenize_query(query_a_terms)
query_b_tokens = tokenize_query(jd_config.production_keywords)
print(f"  Pass A query tokens : {len(query_a_tokens)}")
print(f"  Pass B query tokens : {len(query_b_tokens)}")

# Time Pass A
t0 = time.perf_counter()
scores_a = bm25.get_scores(query_a_tokens)
t_pass_a = time.perf_counter() - t0
print(f"\n  Pass A get_scores() : {hms(t_pass_a)}")

# Time Pass B
t0 = time.perf_counter()
scores_b = bm25.get_scores(query_b_tokens)
t_pass_b = time.perf_counter() - t0
print(f"  Pass B get_scores() : {hms(t_pass_b)}")

# Time union/dedup
top_n = 5000
t0 = time.perf_counter()
combined_scores = np.maximum(scores_a, scores_b)
top_n_actual = min(top_n, len(candidate_ids))
top_indices = np.argpartition(combined_scores, -top_n_actual)[-top_n_actual:]
top_indices = top_indices[np.argsort(combined_scores[top_indices])[::-1]]
top_candidates = [candidate_ids[i] for i in top_indices]
top_scores = {candidate_ids[i]: float(combined_scores[i]) for i in top_indices}
t_union = time.perf_counter() - t0
print(f"  Union/sort/dedup    : {hms(t_union)}")

# Time rare-term safety net
t0 = time.perf_counter()
rare_pool_ids = set()
rare_pool_scores = {}
for rare_term in jd_config.rare_terms:
    rare_tokens = tokenize_query([rare_term])
    rare_scores = bm25.get_scores(rare_tokens)
    rare_nonzero = np.where(rare_scores > 0)[0]
    for idx in rare_nonzero:
        cid = candidate_ids[idx]
        if cid not in top_scores:
            rare_pool_ids.add(cid)
            rare_pool_scores[cid] = max(rare_pool_scores.get(cid, 0.0), float(rare_scores[idx]))
t_rare = time.perf_counter() - t0
print(f"  Rare-term safety net: {hms(t_rare)}  (added {len(rare_pool_ids)} candidates)")

all_scores = {**top_scores, **rare_pool_scores}
all_ordered = sorted(all_scores.keys(), key=lambda c: all_scores[c], reverse=True)
print(f"  Stage 1 total       : {hms(t_pass_a + t_pass_b + t_union + t_rare)}")
print(f"  Stage 1 candidates  : {len(all_ordered)}")

stage1_ids = all_ordered
stage1_set = set(stage1_ids)

# ─────────────────────────────────────────────────────────────────────
# STAGE 2 — Does it re-read the file? Time the streaming read.
# ─────────────────────────────────────────────────────────────────────
print("\n── STAGE 2: RECORD LOADING ──")
print(f"  candidates.jsonl size: {os.path.getsize(CANDIDATES_PATH)/1e6:.1f} MB")
print(f"  Stage 1 produced {len(stage1_ids)} candidate IDs to look up")
print()

# Does Stage 1 touch candidates.jsonl at all?
# Stage 1 only calls bm25.get_scores() on a pre-loaded index — NO file read.
# Stage 2 opens candidates.jsonl from scratch to find the full records.
print("  Stage 1 file reads   : 0  (uses precomputed bm25_index.pkl only)")
print("  Stage 2 file reads   : 1  (streams candidates.jsonl again from scratch)")
print("  => candidates.jsonl is read TWICE total: once in precompute.py (offline),")
print("     zero times in Stage 1, once in Stage 2 at rank time.")
print()

# Time Stage 2 streaming read
t0 = time.perf_counter()
found = {}
malformed = 0
lines_read = 0
with open(CANDIDATES_PATH, 'r', encoding='utf-8') as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        lines_read += 1
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        cid = c.get('candidate_id')
        if cid and cid in stage1_set:
            found[cid] = c
            if len(found) == len(stage1_set):
                break
t_stage2_total = time.perf_counter() - t0

# Also time just the JSON parsing vs disk I/O
t0 = time.perf_counter()
raw_bytes = 0
with open(CANDIDATES_PATH, 'rb') as f:
    for line in f:
        raw_bytes += len(line)
        if raw_bytes > 50_000_000:  # Read first 50MB to sample I/O speed
            break
t_50mb_io = time.perf_counter() - t0

print(f"  Stage 2 stream+parse : {hms(t_stage2_total)}")
print(f"  Lines read before all {len(stage1_set)} found: {lines_read:,}")
print(f"  % of file read       : {lines_read/100000*100:.1f}%")
print(f"  Malformed lines      : {malformed}")
print()
print(f"  Disk I/O speed check (first 50MB raw): {hms(t_50mb_io)}")
print(f"  Estimated file I/O rate: {50 / t_50mb_io:.0f} MB/s")

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Stage 0 (artifact load) : dominated by bm25_index.pkl")
print(f"  Stage 1 (retrieval)     : Pass A={hms(t_pass_a)}, Pass B={hms(t_pass_b)}, union={hms(t_union)}, rare={hms(t_rare)}")
print(f"  Stage 2 (record load)   : {hms(t_stage2_total)}")
print(f"  FILE READS: candidates.jsonl read 0x in Stage1, 1x in Stage2 ({lines_read:,} lines until done)")
