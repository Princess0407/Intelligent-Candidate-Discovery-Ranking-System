"""Diagnostic: analyze concern selection for top 10 candidates."""
import json, csv, os, pickle
import numpy as np

BASE = r'C:\Users\pranj\Desktop\Redrob Hackathon'

# Load submission
with open(os.path.join(BASE, 'submission.csv'), encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: int(r['rank']))
top10_ids = [r['candidate_id'] for r in rows[:10]]
print('Top 10 candidate IDs:', top10_ids)

# Load those candidates from JSONL
candidates = {}
with open(os.path.join(BASE, 'candidates.jsonl'), encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            if c.get('candidate_id') in top10_ids:
                candidates[c['candidate_id']] = c
                if len(candidates) == 10:
                    break
        except Exception:
            pass

print(f'Loaded {len(candidates)} candidates')

from features import build_feature_vector
from jd_parser import parse_jd
from retrieval import tokenize_query

jd_config = parse_jd(os.path.join(BASE, 'data', 'skill_aliases.json'))
with open(os.path.join(BASE, 'precomputed', 'bm25_index.pkl'), 'rb') as f:
    bm25 = pickle.load(f)
with open(os.path.join(BASE, 'precomputed', 'candidate_ids.pkl'), 'rb') as f:
    cids_all = pickle.load(f)

query_tokens = tokenize_query(jd_config.get_all_query_terms() + jd_config.production_keywords)
scores_raw = bm25.get_scores(query_tokens)
bm25_dict = {cids_all[i]: float(scores_raw[i]) for i in range(len(cids_all))}
median_bm25 = float(np.median(list(bm25_dict.values())))

print("\n" + "="*70)
print("CONCERN ANALYSIS — TOP 10 CANDIDATES")
print("="*70)

for cid in top10_ids:
    c = candidates.get(cid)
    if not c:
        print(f'{cid}: NOT FOUND')
        continue
    bs = bm25_dict.get(cid, 0.0)
    fv = build_feature_vector(c, jd_config, bs, median_bm25)

    # Replicate _get_severity_ranked_concern logic exactly
    concerns = []
    if fv.get('flag_consulting_only', 0) > 0.5:
        concerns.append(('consulting_only', 0.9))
    if fv.get('flag_title_desc_mismatch', 0) > 0.5:
        concerns.append(('title_mismatch', 0.8))
    if fv.get('consistency_score', 1.0) < 0.5:
        concerns.append(('consistency_violation', 0.85))
    if fv.get('flag_langchain_dabbler', 0) > 0.5:
        concerns.append(('langchain_only', 0.75))
    if fv.get('flag_cv_specialist', 0) > 0.5:
        concerns.append(('cv_specialist', 0.7))
    if fv.get('flag_template_desc', 0) > 0.5:
        concerns.append(('template_desc', 0.65))
    if fv.get('Param_E_Credibility', 1.0) > 2.5:
        concerns.append(('low_credibility', 0.6))

    if not concerns:
        notice = (c.get('redrob_signals') or {}).get('notice_period_days') or 0
        if notice > 90:
            concerns.append(('long_notice', 0.3))

    rank = next(r['rank'] for r in rows if r['candidate_id'] == cid)
    print(f"\nRank {rank}  {cid}")
    print(f"  Param_E_Credibility : {fv['Param_E_Credibility']:.4f}")
    print(f"  consistency_score   : {fv['consistency_score']:.4f}")
    print(f"  flag_consulting_only: {fv['flag_consulting_only']}")
    print(f"  flag_title_desc_mism: {fv['flag_title_desc_mismatch']}")
    print(f"  flag_langchain_dabl : {fv['flag_langchain_dabbler']}")
    print(f"  flag_cv_specialist  : {fv['flag_cv_specialist']}")
    print(f"  flag_template_desc  : {fv['flag_template_desc']}")
    concerns.sort(key=lambda x: x[1], reverse=True)
    print(f"  Eligible concerns (sorted by severity): {concerns}")
    print(f"  SELECTED: {concerns[0] if concerns else 'NONE'}")

print("\n" + "="*70)
print("SEVERITY SCORE COMPARISON — same concern types, hard-coded values")
print("="*70)
print("  consulting_only     -> severity 0.90")
print("  consistency_violatn -> severity 0.85")
print("  title_mismatch      -> severity 0.80")
print("  langchain_only      -> severity 0.75")
print("  cv_specialist       -> severity 0.70")
print("  template_desc       -> severity 0.65")
print("  low_credibility     -> severity 0.60  <-- ALWAYS LOWEST among non-notice")
print("  long_notice         -> severity 0.30")

print("\n" + "="*70)
print("PHRASING VARIANT POOL SIZE — _get_severity_ranked_concern in reasoning.py")
print("="*70)
print("Each concern type returns ONE hard-coded string (no rotation pool).")
print("There is only 1 variant per concern type — zero rotation.")
print("The n-gram collision check adds [Rank N] prefix but the CONCERN SENTENCE itself is identical.")
