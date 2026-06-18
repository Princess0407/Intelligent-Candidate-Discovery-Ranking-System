"""
Diagnostic: CAND_0089350 full skills breakdown + hard_req_coverage audit.
No code changes - read-only.
"""
import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'C:\Users\pranj\Desktop\Redrob Hackathon'

# Load the candidate
target_id = 'CAND_0089350'
candidate = None
with open(os.path.join(BASE, 'candidates.jsonl'), encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            c = json.loads(line)
            if c.get('candidate_id') == target_id:
                candidate = c
                break
        except json.JSONDecodeError:
            pass

if candidate is None:
    print(f'ERROR: {target_id} not found')
    sys.exit(1)

print('=' * 70)
print(f'CANDIDATE: {target_id}')
print('=' * 70)

# 1. Full profile
profile = candidate.get('profile', {}) or {}
print(f'\nProfile:')
print(f'  current_title  : {profile.get("current_title")}')
print(f'  current_company: {profile.get("current_company")}')
print(f'  yoe            : {profile.get("years_of_experience")}')
print(f'  location       : {profile.get("location")}')

# 2. Full skills array sorted by duration_months descending
skills = candidate.get('skills', []) or []
print(f'\nFull skills array ({len(skills)} skills), sorted by duration_months DESC:')
print(f'  {"#":>2}  {"name":40s}  {"proficiency":14s}  {"duration_months":>15}  {"endorsements":>12}')
print('  ' + '-' * 90)
skills_sorted = sorted(skills, key=lambda s: s.get('duration_months') or 0, reverse=True)
for i, s in enumerate(skills_sorted, 1):
    name = s.get('name', '') or ''
    prof = s.get('proficiency', '') or ''
    dur  = s.get('duration_months')
    end  = s.get('endorsements', 0) or 0
    print(f'  {i:>2}  {name:40s}  {prof:14s}  {str(dur) if dur is not None else "None":>15}  {end:>12}')

# 3. Hard requirement coverage — replicate features.py logic
print()
print('=' * 70)
print('HARD REQUIREMENT COVERAGE AUDIT')
print('=' * 70)

from jd_parser import parse_jd, hard_req_coverage_score
jd_config = parse_jd(os.path.join(BASE, 'data', 'skill_aliases.json'))

print(f'\nJD hard_requirements ({len(jd_config.hard_requirements)} total):')
for req_name, aliases in jd_config.hard_requirements.items():
    print(f'  {req_name}: {aliases[:5]}{"..." if len(aliases) > 5 else ""}')

# Compute coverage the same way features.py does
candidate_skill_names = {s.get('name', '').lower().strip() for s in skills}
career_text = ' '.join(
    (ch.get('description', '') or '').lower()
    for ch in candidate.get('career_history', []) or []
)

print(f'\nCandidate skill names (lowercase): {sorted(candidate_skill_names)}')
print(f'\nCareer text (first 400 chars): {career_text[:400]}...')

print(f'\nPer-requirement match trace:')
matched = []
for req_name, aliases in jd_config.hard_requirements.items():
    skill_hit  = [a for a in aliases if a in candidate_skill_names]
    career_hit = [a for a in aliases if a in career_text] if not skill_hit else []
    hit = bool(skill_hit or career_hit)
    if hit:
        matched.append(req_name)
    print(f'  {req_name:35s} MATCH={str(hit):5s}  skill_hits={skill_hit}  career_hits={career_hit[:3]}')

coverage_score = len(matched) / max(1, len(jd_config.hard_requirements))
print(f'\nMatched: {len(matched)} / {len(jd_config.hard_requirements)} = {coverage_score:.4f}')
print(f'Coverage label: ', end='')
if coverage_score >= 0.8:   print('HIGH (>= 0.8)')
elif coverage_score >= 0.6: print('MODERATE (>= 0.6)')
elif coverage_score >= 0.4: print('PARTIAL (>= 0.4)')
elif coverage_score > 0:    print('LOW (> 0)')
else:                       print('NONE (= 0)')

# 4. What _get_hard_req_matches() returns for the opening sentence
print()
print('=' * 70)
print('OPENING SENTENCE — hard_req_matches (used in reasoning opener)')
print('=' * 70)
from reasoning import _get_hard_req_matches
hard_matches = _get_hard_req_matches(candidate, jd_config)
print(f'\n_get_hard_req_matches() returned: {hard_matches}')
print(f'Opening sentence claims: "{", ".join(hard_matches[:2]).replace(chr(95), " ")}."')

# 5. Which of those matches are genuine (skill) vs career-text-only
print()
print('Match source breakdown:')
for req_name in hard_matches:
    aliases = jd_config.hard_requirements.get(req_name, [])
    skill_hit  = [a for a in aliases if a in candidate_skill_names]
    career_hit = [a for a in aliases if a in career_text]
    src = 'SKILL_NAME' if skill_hit else 'CAREER_TEXT'
    print(f'  {req_name:35s} source={src}  matching_alias={skill_hit or career_hit[:3]}')
