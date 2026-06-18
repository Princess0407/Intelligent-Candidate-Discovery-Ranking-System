"""
test_malformed.py — Test pipeline against deliberately broken records.
Run: python test_malformed.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import (
    c1_timeline_impossibility, c2_signup_anomaly, c3_salary_inversion,
    c4_assessment_contradiction, c5_engagement_mismatch,
    consistency_score, build_feature_vector
)
from jd_parser import parse_jd

jd = parse_jd('./data/skill_aliases.json')

# ==========================================================================
# TEST 1: Maximally malformed record — every optional field null/missing/empty
# ==========================================================================
broken = {
    'candidate_id': 'CAND_BROKEN01',
    'profile': {
        'anonymized_name': 'Broken User',
        'headline': '',
        'summary': None,
        'location': None,
        'country': None,
        'years_of_experience': None,  # null YoE
        'current_title': 'AI Engineer',
        'current_company': 'Unknown',
        'current_company_size': '1-10',
        'current_industry': 'Technology'
    },
    'career_history': [],  # empty
    'education': [],
    'skills': [],  # empty
    'redrob_signals': {
        'signup_date': '2099-01-01',  # FUTURE -> signup anomaly
        'last_active_date': '2025-01-01',
        'open_to_work_flag': True,
        'profile_views_received_30d': 0,
        'applications_submitted_30d': 0,
        'recruiter_response_rate': None,  # null
        'avg_response_time_hours': 0,
        'skill_assessment_scores': None,  # null object
        'connection_count': 0,
        'endorsements_received': 0,
        'notice_period_days': None,  # null
        'expected_salary_range_inr_lpa': {'min': 50.0, 'max': 10.0},  # INVERTED
        'preferred_work_mode': 'remote',
        'willing_to_relocate': False,
        'github_activity_score': -1,
        'search_appearance_30d': 0,
        'saved_by_recruiters_30d': 0,
        'interview_completion_rate': 0,
        'offer_acceptance_rate': -1,
        'verified_email': False,
        'verified_phone': False,
        'linkedin_connected': False,
        'profile_completeness_score': 0
    }
}

print("=" * 60)
print("TEST 1: Maximally malformed record")
print("=" * 60)
c1 = c1_timeline_impossibility(broken)
c2 = c2_signup_anomaly(broken)
c3 = c3_salary_inversion(broken)
c4 = c4_assessment_contradiction(broken)
c5 = c5_engagement_mismatch(broken, 0.0, 5.0)
cs = consistency_score(broken, 0.0, 5.0)

print(f"c1 (timeline):      {c1:.1f}  (expect 1.0 — empty skills, no violation)")
print(f"c2 (signup):        {c2:.1f}  (expect 0.0 — 2099 > 2025)")
print(f"c3 (salary inv):    {c3:.1f}  (expect 0.0 — min 50 > max 10)")
print(f"c4 (assessment):    {c4:.1f}  (expect 1.0 — no skills)")
print(f"c5 (engagement):    {c5:.1f}  (expect 1.0 — bm25=0 not > median=5)")
print(f"consistency:        {cs:.1f}  (expect 0.0 — c2 and c3 fire)")

fv = build_feature_vector(broken, jd, bm25_score=0.0, stage1_bm25_median=5.0)
print(f"\nFeature vector: {len(fv)} features (expect 22)")
print(f"  yoe: {fv['yoe']}")
print(f"  hard_req_coverage: {fv['hard_req_coverage']:.2f}")
print(f"  consistency_score: {fv['consistency_score']:.2f}")

import math
all_finite = all(math.isfinite(v) for v in fv.values())

print(f"  All finite floats: {all_finite}")
print(f"  PASS" if all_finite and len(fv) == 22 else f"  FAIL")

# ==========================================================================
# TEST 2: Missing redrob_signals entirely
# ==========================================================================
print()
print("=" * 60)
print("TEST 2: Missing redrob_signals entirely")
print("=" * 60)
no_signals = {
    'candidate_id': 'CAND_BROKEN02',
    'profile': {
        'anonymized_name': 'No Signals',
        'headline': 'NLP Engineer',
        'summary': 'Background in NLP and retrieval systems.',
        'location': 'Pune',
        'country': 'India',
        'years_of_experience': 5.0,
        'current_title': 'NLP Engineer',
        'current_company': 'StartupX',
        'current_company_size': '51-200',
        'current_industry': 'Technology'
    },
    'career_history': [{
        'company': 'StartupX', 'title': 'NLP Engineer',
        'start_date': '2020-01-01', 'end_date': None,
        'duration_months': 48, 'is_current': True,
        'industry': 'Technology', 'company_size': '51-200',
        'description': 'Built BM25 retrieval pipeline for production search at scale.'
    }],
    'education': [],
    'skills': [
        {'name': 'BM25', 'proficiency': 'advanced', 'endorsements': 5, 'duration_months': 36},
        {'name': 'Python', 'proficiency': 'expert', 'endorsements': 20, 'duration_months': 60},
    ],
    # redrob_signals MISSING entirely — not even the key
}

try:
    fv2 = build_feature_vector(no_signals, jd, bm25_score=8.5, stage1_bm25_median=5.0)
    print(f"Feature extraction succeeded: {len(fv2)} features")
    print(f"  Param_B_Availability (no signals): {fv2['Param_B_Availability']:.3f}")
    print(f"  consistency_score: {fv2['consistency_score']:.2f}")
    all_finite2 = all(math.isfinite(v) for v in fv2.values())
    print(f"  All finite floats: {all_finite2}")
    print(f"  PASS" if all_finite2 and len(fv2) == 22 else f"  FAIL")
except Exception as e:
    print(f"FAIL: Exception raised: {e}")

# ==========================================================================
# TEST 3: Malformed JSON line in a JSONL stream
# ==========================================================================
print()
print("=" * 60)
print("TEST 3: Malformed JSON line in JSONL stream")
print("=" * 60)
import json, io

good_line = json.dumps({'candidate_id': 'CAND_GOOD01', 'profile': {'years_of_experience': 3}})
bad_line = '{not valid json at all'
empty_line = ''
another_good = json.dumps({'candidate_id': 'CAND_GOOD02', 'profile': {'years_of_experience': 5}})

stream = "\n".join([good_line, bad_line, empty_line, another_good])
candidates = []
malformed = 0
for line in stream.split("\n"):
    line = line.strip()
    if not line:
        continue
    try:
        candidates.append(json.loads(line))
    except json.JSONDecodeError:
        malformed += 1

print(f"Parsed {len(candidates)} valid records, {malformed} malformed (expect: 2 valid, 1 malformed)")
print(f"PASS" if len(candidates) == 2 and malformed == 1 else f"FAIL")

print()
print("=" * 60)
print("ALL DEFENSIVE TESTS COMPLETE")
print("=" * 60)
