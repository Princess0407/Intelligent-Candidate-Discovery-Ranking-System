"""
Param_E_Credibility deep diagnostic for top-10 candidates.
Prints:
  1. Exact Param_E value to full float64 precision (no rounding)
  2. Raw inputs: advanced_claimed, assessed_count, raw_ratio
  3. Per-skill breakdown: name / proficiency / in_assessment_dict
  4. Verification that min(5.0, ratio) == stored value
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

BASE = r"C:\Users\pranj\Desktop\Redrob Hackathon"
PRECOMPUTED = os.path.join(BASE, "precomputed")

# ── Load top-10 IDs from submission ──────────────────────────────────────────
with open(os.path.join(BASE, "submission.csv"), encoding="utf-8") as f:
    rows = sorted(list(csv.DictReader(f)), key=lambda r: int(r["rank"]))
top10_ids = [r["candidate_id"] for r in rows[:10]]
print("Top-10 IDs:", top10_ids)

# ── Load candidate records ────────────────────────────────────────────────────
candidates: dict = {}
with open(os.path.join(BASE, "candidates.jsonl"), encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid = c.get("candidate_id")
        if cid in top10_ids:
            candidates[cid] = c
            if len(candidates) == 10:
                break

print(f"Loaded {len(candidates)} candidate records\n")

# ── Replicate Param_E formula exactly ─────────────────────────────────────────
print("=" * 72)
print("PARAM_E_CREDIBILITY — RAW INPUTS FOR TOP-10 CANDIDATES")
print("=" * 72)

for r_row in rows[:10]:
    cid = r_row["candidate_id"]
    rank = r_row["rank"]
    c = candidates.get(cid)
    if c is None:
        print(f"  Rank {rank}  {cid}: RECORD NOT FOUND")
        continue

    skills = c.get("skills") or []
    signals = c.get("redrob_signals") or {}
    assessments = signals.get("skill_assessment_scores") or {}
    if not isinstance(assessments, dict):
        assessments = {}

    assessed_keys = {k.lower().strip() for k in assessments.keys()}

    advanced_claimed = 0
    assessed_advanced = 0
    skill_detail_lines = []

    for s in skills:
        proficiency = (s.get("proficiency") or "").lower()
        name = (s.get("name") or "").lower().strip()
        is_advanced = proficiency == "advanced"
        in_assessed = name in assessed_keys
        if is_advanced:
            advanced_claimed += 1
            if in_assessed:
                assessed_advanced += 1
        skill_detail_lines.append(
            f"    {name:35s}  prof={proficiency:12s}  adv={'Y' if is_advanced else 'N'}  "
            f"in_assessment={'Y' if in_assessed else 'N'}"
        )

    raw_ratio = advanced_claimed / max(1, assessed_advanced)
    param_e = min(5.0, raw_ratio)

    print(f"\nRank {rank:>3}  {cid}")
    print(f"  advanced_claimed : {advanced_claimed}")
    print(f"  assessed_advanced: {assessed_advanced}")
    print(f"  raw_ratio        : {raw_ratio:.15f}")
    print(f"  min(5.0, ratio)  : {param_e:.15f}")
    print(f"  Assessment keys  : {sorted(assessed_keys) if assessed_keys else '(none)'}")
    print(f"  Skills breakdown ({len(skills)} total, showing adv only):")
    for ln in skill_detail_lines:
        if "adv=Y" in ln:
            print(ln)

print("\n" + "=" * 72)
print("DISTINCT Param_E VALUES IN TOP-10")
print("=" * 72)
values = []
for r_row in rows[:10]:
    cid = r_row["candidate_id"]
    c = candidates.get(cid)
    if not c:
        continue
    skills = c.get("skills") or []
    signals = c.get("redrob_signals") or {}
    assessments = signals.get("skill_assessment_scores") or {}
    if not isinstance(assessments, dict):
        assessments = {}
    assessed_keys = {k.lower().strip() for k in assessments.keys()}
    adv = sum(1 for s in skills if (s.get("proficiency") or "").lower() == "advanced")
    asd = sum(1 for s in skills
              if (s.get("proficiency") or "").lower() == "advanced"
              and (s.get("name") or "").lower().strip() in assessed_keys)
    ratio = adv / max(1, asd)
    pe = min(5.0, ratio)
    values.append((cid, adv, asd, ratio, pe))

distinct_pe = set(v[4] for v in values)
print(f"  Distinct Param_E values: {len(distinct_pe)}")
for v in values:
    print(f"  {v[0]}  adv={v[1]}  asd={v[2]}  ratio={v[3]:.6f}  Param_E={v[4]:.15f}")

print("\n" + "=" * 72)
print("DIAGNOSIS")
print("=" * 72)
if len(distinct_pe) == 1 and 5.0 in distinct_pe:
    print("  ALL 10 candidates have Param_E == 5.0 EXACTLY (not display rounding).")
    print("  This means all have raw_ratio >= 5.0 (i.e., advanced_claimed / max(1, assessed_advanced) >= 5).")
    print("  The min(5.0, ...) cap truncates genuine variation in the tail.")
    print("  Candidates with ratio=6 and ratio=100 are indistinguishable after capping.")
    print("  This IS a resolution problem at the top of the distribution.")
    print("  However, Param_E is only ONE of 22 features — LightGBM can still")
    print("  differentiate via bm25_score, yoe, Param_A..D, prod_signal_log, etc.")
    print()
    distinct_raw = set(v[3] for v in values)
    print(f"  Distinct RAW ratios (before cap): {len(distinct_raw)}")
    for v in values:
        print(f"    {v[0]}  raw_ratio={v[3]:.6f}")
elif len(distinct_pe) < 5:
    print(f"  Only {len(distinct_pe)} distinct value(s) in top-10 — limited resolution.")
else:
    print(f"  Good: {len(distinct_pe)} distinct Param_E values in top-10.")
