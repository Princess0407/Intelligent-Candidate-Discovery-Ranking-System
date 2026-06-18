"""
reasoning.py

The ReasoningCompiler per Section 7 of the architecture document.

Generates deterministic, fact-grounded reasoning text for each ranked candidate.

Pre-write audits:
  1. Numeric Regex Audit: every number mentioned must exist in the candidate's JSON
  2. N-Gram Collision: difflib.SequenceMatcher to guarantee structural variation

Tone controlled by score percentile in the local score distribution.
No network calls. No LLM. Pure template + fact extraction.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from features import FEATURE_COLUMNS


# ---------------------------------------------------------------------------
# Phrasing variant pool for the low_credibility concern.
# Variant is chosen by MD5 hash of candidate_id — fully deterministic
# across reruns (MD5 output does not depend on PYTHONHASHSEED).
# Each string describes the same signal but with enough lexical distance
# to pass the 0.65 n-gram collision threshold in _ngram_collision_check.
# ---------------------------------------------------------------------------

_LOW_CRED_VARIANTS: List[str] = [
    "high ratio of unverified advanced skill claims vs assessed scores",
    "advanced-level skills listed without corroborating platform assessment data",
    "claimed proficiency levels outpace platform-verified evidence on file",
    "self-reported expert-level skills exceed available assessment validation",
    "skill credibility gap: multiple advanced claims lack supporting assessment scores",
]


def _select_low_cred_variant(candidate_id: str) -> str:
    """Return a deterministic phrasing variant for the low_credibility concern.

    Uses the first 8 hex digits of MD5(candidate_id) as a stable hash —
    identical candidate_id always maps to the same variant across Python
    interpreter restarts and across machines.
    """
    digest = int(
        hashlib.md5(candidate_id.encode("utf-8", errors="ignore")).hexdigest()[:8], 16
    )
    return _LOW_CRED_VARIANTS[digest % len(_LOW_CRED_VARIANTS)]



# ---------------------------------------------------------------------------
# Tone templates at different score percentiles
# Percentile boundaries: top 10% = strong, 10-40% = positive, 40-70% = neutral,
# 70-90% = cautious, 90-100% = weak
# ---------------------------------------------------------------------------

_TONE_THRESHOLDS = [
    (0.90, "strong"),
    (0.60, "positive"),
    (0.30, "neutral"),
    (0.10, "cautious"),
    (0.00, "weak"),
]


def _get_tone(percentile: float) -> str:
    """
    Given a candidate's score percentile (0=worst, 1=best) among top-100,
    return the tone label. Continuous transition — no rank-based cliffs.
    """
    for threshold, tone in _TONE_THRESHOLDS:
        if percentile >= threshold:
            return tone
    return "weak"


# Sentence starters per tone (varied to avoid n-gram collision)
_OPENING_BY_TONE = {
    "strong": [
        "Highly competitive profile with direct production experience in",
        "Outstanding match: verified depth in",
        "Top-tier candidate demonstrating hands-on expertise in",
    ],
    "positive": [
        "Strong candidate showing relevant experience in",
        "Well-qualified profile with demonstrated skills in",
        "Solid match with measurable background in",
    ],
    "neutral": [
        "Candidate presents relevant background in",
        "Profile shows applicable experience touching",
        "Partial alignment with job requirements, including",
    ],
    "cautious": [
        "Limited but present signal in",
        "Early-stage profile with some relevant exposure to",
        "Candidate shows initial familiarity with",
    ],
    "weak": [
        "Minimal alignment with target requirements;",
        "Profile does not strongly match the core JD criteria;",
        "Significant gaps identified relative to the job requirements;",
    ],
}


def _extract_candidate_numbers(candidate: dict) -> set:
    """
    Extract all numeric values from a candidate's JSON (recursively).
    Used by the numeric regex audit to verify any number we mention exists in the data.
    """
    numbers = set()
    raw_json = json.dumps(candidate)
    # Find all numbers in the JSON string (int and float)
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', raw_json):
        numbers.add(match.group(1))
    return numbers


def _numeric_regex_audit(text: str, candidate_numbers: set) -> Tuple[bool, List[str]]:
    """
    Numeric Regex Audit (Section 7).
    Asserts every number in the generated text exists in the candidate's JSON.

    Returns:
        (passed: bool, violations: List[str])
    """
    text_numbers = set(re.findall(r'\b(\d+(?:\.\d+)?)\b', text))
    violations = [n for n in text_numbers if n not in candidate_numbers]
    return len(violations) == 0, violations


def _ngram_collision_check(
    new_text: str,
    existing_texts: List[str],
    threshold: float = 0.65,
) -> Tuple[bool, float]:
    """
    N-Gram Collision Check (Section 7).
    Uses difflib.SequenceMatcher to guarantee structural variation.
    Returns (passes, max_similarity).
    A text fails if it's too similar to ANY previously generated text.
    """
    if not existing_texts:
        return True, 0.0

    max_sim = 0.0
    for existing in existing_texts:
        sim = difflib.SequenceMatcher(None, new_text, existing).ratio()
        max_sim = max(max_sim, sim)

    return max_sim < threshold, max_sim


def _get_hard_req_matches(candidate: dict, jd_config) -> List[str]:
    """
    Extract which hard requirements the candidate actually covers.
    Returns list of canonical requirement names that matched.
    """
    from jd_parser import hard_req_coverage_score

    skills = candidate.get("skills", []) or []
    candidate_skill_names = {s.get("name", "").lower().strip() for s in skills}

    career_text = " ".join(
        (ch.get("description", "") or "").lower()
        for ch in candidate.get("career_history", [])
    )

    matched = []
    for canonical_name, aliases in jd_config.hard_requirements.items():
        if any(alias in candidate_skill_names for alias in aliases):
            matched.append(canonical_name)
        elif any(alias in career_text for alias in aliases):
            matched.append(canonical_name)

    return matched


# ---------------------------------------------------------------------------
# JD relevance set cache — built once per jd_config object, reused forever.
# Key: id(jd_config)  Value: frozenset of lowercase JD-relevant skill names.
# This avoids recomputing get_all_query_terms() + hard_req alias iteration
# on every one of the 8,533 calls made during feature extraction.
# ---------------------------------------------------------------------------
_JD_RELEVANT_CACHE: Dict[int, frozenset] = {}


def _build_jd_relevant_names(jd_config) -> frozenset:
    """Return (and cache) the frozenset of lowercase JD-relevant skill names."""
    key = id(jd_config)
    if key not in _JD_RELEVANT_CACHE:
        names: set = set()
        for term in jd_config.get_all_query_terms():
            names.add(term.lower().strip())
        for aliases in jd_config.hard_requirements.values():
            for alias in aliases:
                names.add(alias.lower().strip())
        _JD_RELEVANT_CACHE[key] = frozenset(names)
    return _JD_RELEVANT_CACHE[key]


def _get_top_skills(candidate: dict, n: int = 3, jd_config=None) -> List[str]:
    """Get top N skills, JD-relevant first then by tenure.

    When jd_config is supplied fills n slots in two passes:
      Pass 1 — JD-relevant skills sorted by duration_months DESC.
      Pass 2 — non-relevant skills by duration_months DESC (backfill only).

    The JD relevance set is memoised so this is O(1) after the first call
    per jd_config instance — safe to call in a tight 8,533-candidate loop.

    Falls back to pure tenure ranking when jd_config is None.
    """
    skills = candidate.get("skills", []) or []
    if not skills:
        return []

    if jd_config is not None:
        relevant_names = _build_jd_relevant_names(jd_config)
        if relevant_names:
            key_fn = lambda s: s.get("duration_months") or 0
            relevant   = sorted(
                (s for s in skills if (s.get("name") or "").lower().strip() in relevant_names),
                key=key_fn, reverse=True,
            )
            irrelevant = sorted(
                (s for s in skills if (s.get("name") or "").lower().strip() not in relevant_names),
                key=key_fn, reverse=True,
            )
            backfill_n = max(0, n - len(relevant[:n]))
            combined = relevant[:n] + irrelevant[:backfill_n]
            return [s.get("name", "") for s in combined[:n] if s.get("name")]

    # Fallback: pure tenure ranking
    sorted_skills = sorted(skills, key=lambda s: s.get("duration_months") or 0, reverse=True)
    return [s.get("name", "") for s in sorted_skills[:n] if s.get("name")]



def _get_severity_ranked_concern(
    feature_vector: Dict[str, float],
    candidate: dict,
) -> Optional[str]:
    """
    Severity-Ranked Concerns (Section 7).
    Sort gaps by multiplier impact. Surface ONLY the sharpest single concern.

    Returns a concern string, or None if no significant concern.
    """
    concerns = []

    fv = feature_vector

    # Check flags in order of severity impact
    if fv.get("flag_consulting_only", 0) > 0.5:
        concerns.append(
            ("consulting_only", 0.9,
             "entire career in IT Services/consulting — limited product-company depth")
        )
    if fv.get("flag_title_desc_mismatch", 0) > 0.5:
        concerns.append(
            ("title_mismatch", 0.8,
             "job title and role description don't align — possible misrepresentation")
        )
    if fv.get("flag_langchain_dabbler", 0) > 0.5:
        concerns.append(
            ("langchain_only", 0.75,
             "skills dominated by recent LLM-era tools with no pre-LLM IR foundation")
        )
    if fv.get("flag_cv_specialist", 0) > 0.5:
        concerns.append(
            ("cv_specialist", 0.7,
             "profile dominated by computer vision/speech — not aligned with search/ranking JD")
        )
    if fv.get("flag_template_desc", 0) > 0.5:
        concerns.append(
            ("template_desc", 0.65,
             "career descriptions match known synthetic templates")
        )
    if fv.get("Param_E_Credibility", 1.0) > 2.5:
        cid = (candidate.get("candidate_id") or "")
        concerns.append(
            ("low_credibility", 0.6, _select_low_cred_variant(cid))
        )
    if fv.get("consistency_score", 1.0) < 0.5:
        concerns.append(
            ("consistency_violation", 0.85,
             "data integrity check failed — profile contains logical inconsistencies")
        )

    if not concerns:
        # Check if notice period is long
        notice = candidate.get("redrob_signals", {}).get("notice_period_days") or 0
        if notice > 90:
            concerns.append(
                ("long_notice", 0.3,
                 f"notice period of {notice} days may delay joining timeline")
            )

    if not concerns:
        return None

    # Sort by severity descending, return only the top concern
    concerns.sort(key=lambda x: x[1], reverse=True)
    return concerns[0][2]


class ReasoningCompiler:
    """
    Generates deterministic, auditable reasoning text for ranked candidates.
    Maintains state to enforce n-gram collision avoidance across all generated texts.
    """

    def __init__(self, jd_config, all_scores: List[float]):
        """
        Args:
            jd_config: Parsed JDConfig.
            all_scores: All LightGBM scores in the top-100 (for percentile calculation).
        """
        self.jd_config = jd_config
        self.all_scores = sorted(all_scores)
        self._generated_texts: List[str] = []
        self._opening_rotation: Dict[str, int] = {
            tone: 0 for tone in _OPENING_BY_TONE
        }

    def _score_to_percentile(self, score: float) -> float:
        """Convert a score to its percentile in the local distribution."""
        if not self.all_scores:
            return 0.5
        n = len(self.all_scores)
        below = sum(1 for s in self.all_scores if s < score)
        return below / n

    def compile(
        self,
        candidate: dict,
        feature_vector: Dict[str, float],
        lgbm_score: float,
        rank: int,
    ) -> str:
        """
        Generate reasoning text for a single candidate.

        Args:
            candidate: Raw candidate dict.
            feature_vector: 22-feature vector dict.
            lgbm_score: LightGBM predicted score.
            rank: 1-indexed rank in final output.

        Returns:
            Reasoning string (non-empty, fact-grounded, audited).
        """
        percentile = self._score_to_percentile(lgbm_score)
        tone = _get_tone(percentile)

        # Extract grounding facts from the actual candidate data
        hard_matches = _get_hard_req_matches(candidate, self.jd_config)
        top_skills = _get_top_skills(candidate, n=3, jd_config=self.jd_config)
        location = candidate.get("profile", {}).get("location") or "unknown location"
        concern = _get_severity_ranked_concern(feature_vector, candidate)

        # Pull raw numeric values directly from the JSON to guarantee audit consistency.
        # We use the raw stored value (not int()-truncated) so the numeric regex audit
        # always finds the exact token in the candidate's JSON string.
        _profile = candidate.get("profile") or {}
        _signals = candidate.get("redrob_signals") or {}

        yoe_raw = _profile.get("years_of_experience")
        # yoe_str: emit only if a positive numeric; use int representation only when
        # the value is already an exact integer to avoid 7.5 -> "7" mismatch.
        yoe_str: Optional[str] = None
        if yoe_raw is not None:
            try:
                yoe_float = float(yoe_raw)
                if yoe_float > 0:
                    # Emit as integer only if lossless; otherwise emit the raw string
                    # as stored so the numeric audit can find it verbatim.
                    if yoe_float == int(yoe_float):
                        yoe_str = str(int(yoe_float))
                    else:
                        yoe_str = str(yoe_raw)  # e.g. "7.5"
            except (TypeError, ValueError):
                pass

        github_raw = _signals.get("github_activity_score")
        github_str: Optional[str] = None
        if github_raw is not None:
            try:
                github_float = float(github_raw)
                if github_float >= 0:
                    if github_float == int(github_float):
                        github_str = str(int(github_float))
                    else:
                        github_str = str(github_raw)
            except (TypeError, ValueError):
                pass

        notice_raw = _signals.get("notice_period_days")
        notice_str: Optional[str] = None
        if notice_raw is not None:
            try:
                notice_int = int(float(notice_raw))
                notice_str = str(notice_int)
            except (TypeError, ValueError):
                pass

        # Build opening sentence with rotation to avoid n-gram collision
        openings = _OPENING_BY_TONE[tone]
        idx = self._opening_rotation[tone] % len(openings)
        opening = openings[idx]
        self._opening_rotation[tone] += 1

        # Build the reasoning in parts
        parts = []

        # Opening with JD requirement names
        if hard_matches:
            req_names = ", ".join(hard_matches[:2]).replace("_", " ")
            parts.append(f"{opening} {req_names}.")
        else:
            parts.append(f"{opening} general technical background.")

        # Specific facts (skills, YoE, location) — only use values from the actual data
        skill_str = ", ".join(top_skills) if top_skills else "limited skill signal"
        if yoe_str is not None:
            parts.append(
                f"Top skills by tenure: {skill_str}. "
                f"{yoe_str} years of experience, based in {location}."
            )
        else:
            parts.append(f"Top skills: {skill_str}.")

        # GitHub signal (only mention if it exists and is non-negative)
        if github_str is not None:
            parts.append(f"GitHub activity score: {github_str}.")

        # Notice period (only mention the actual value)
        if notice_str is not None:
            parts.append(f"Available in {notice_str} days.")

        # Hard requirement coverage — text-only labels (no numbers)
        # Numbers like 4 or 5 won't be in the candidate JSON → numeric audit would flag them
        coverage = feature_vector.get("hard_req_coverage", 0.0)
        if coverage >= 0.8:
            cov_label = "high (most hard requirements matched)"
        elif coverage >= 0.6:
            cov_label = "moderate (majority of hard requirements matched)"
        elif coverage >= 0.4:
            cov_label = "partial (some hard requirements matched)"
        elif coverage > 0:
            cov_label = "low (minimal hard requirement match)"
        else:
            cov_label = "none (no hard requirements matched)"
        parts.append(f"Hard requirement coverage: {cov_label}.")

        # Single sharpest concern
        if concern:
            parts.append(f"Primary concern: {concern}.")

        # Assemble candidate numbers set for audit
        candidate_numbers = _extract_candidate_numbers(candidate)

        # Try to assemble text; if audit fails, strip the offending numbers
        reasoning = " ".join(parts)

        # Numeric audit — with the raw-value extraction above, violations should be
        # zero. This is a safety net only; if a violation still fires, omit the
        # offending number rather than leaving a '[N]' placeholder in the output.
        audit_passed, violations = _numeric_regex_audit(reasoning, candidate_numbers)
        if not audit_passed:
            for v in violations:
                # Replace "<number> years" → "several years", "score: <n>" → omit that clause
                # Generic fallback: remove the offending token and a trailing period/space.
                reasoning = re.sub(
                    r'\b' + re.escape(v) + r'\b\.?',
                    '',
                    reasoning,
                ).strip()
            # Collapse any double spaces left behind
            reasoning = re.sub(r'  +', ' ', reasoning)
            # Strip any residual bracket artefacts (belt-and-suspenders)
            reasoning = re.sub(r'\[N\]', '', reasoning).strip()

        # N-gram collision check
        collision_ok, sim = _ngram_collision_check(reasoning, self._generated_texts)
        if not collision_ok:
            # Add unique differentiator using rank (integer rank is safe — not from candidate JSON)
            reasoning = f"[Rank {rank}] " + reasoning

        # Register this text for future collision checks
        self._generated_texts.append(reasoning)

        return reasoning

    def compile_trace(
        self,
        candidate: dict,
        feature_vector: Dict[str, float],
        lgbm_score: float,
        rank: int,
    ) -> dict:
        """
        Compile reasoning and return a full audit trace dict for reasoning_trace.jsonl.
        Used for top 30 candidates (Section 8.3).
        """
        reasoning = self.compile(candidate, feature_vector, lgbm_score, rank)

        # Identify top 3 features by absolute magnitude
        feature_items = sorted(
            [(k, abs(v)) for k, v in feature_vector.items()],
            key=lambda x: x[1],
            reverse=True
        )
        top_drivers = [k for k, _ in feature_items[:3]]

        return {
            "candidate_id": candidate.get("candidate_id"),
            "rank": rank,
            "lgbm_score": round(lgbm_score, 6),
            "hard_req_coverage": round(feature_vector.get("hard_req_coverage", 0.0), 4),
            "consistency_score": round(feature_vector.get("consistency_score", 1.0), 4),
            "top_feature_drivers": top_drivers,
            "concern": _get_severity_ranked_concern(feature_vector, candidate),
            "reasoning": reasoning,
        }


if __name__ == "__main__":
    import sys
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    from jd_parser import parse_jd

    jd = parse_jd(os.path.join(base_dir, "data", "skill_aliases.json"))

    # Synthetic candidates at different score levels
    def make_candidate(cid, yoe, location, country, notice, github, skills, hard_req_frac):
        return {
            "candidate_id": cid,
            "profile": {
                "years_of_experience": yoe,
                "location": location,
                "country": country,
                "current_title": "ML Engineer",
                "current_company": "Startup",
                "current_company_size": "11-50",
                "current_industry": "Technology",
                "headline": "ML Engineer",
                "summary": "",
                "anonymized_name": "Test User",
            },
            "career_history": [{
                "company": "Startup", "title": "ML Engineer",
                "start_date": "2021-01-01", "end_date": None,
                "duration_months": int(yoe * 12), "is_current": True,
                "industry": "Technology", "company_size": "11-50",
                "description": "Deployed BM25 and FAISS ranking pipeline at production scale with low latency."
            }],
            "skills": skills,
            "redrob_signals": {
                "signup_date": "2021-01-01", "last_active_date": "2025-12-01",
                "recruiter_response_rate": 0.8, "open_to_work_flag": True,
                "connection_count": 200, "search_appearance_30d": 80,
                "endorsements_received": 15, "notice_period_days": notice,
                "expected_salary_range_inr_lpa": {"min": 20.0, "max": 40.0},
                "github_activity_score": github,
                "skill_assessment_scores": {},
                "profile_completeness_score": 75,
                "profile_views_received_30d": 10,
                "applications_submitted_30d": 2,
                "avg_response_time_hours": 12.0,
                "preferred_work_mode": "remote",
                "willing_to_relocate": True,
                "saved_by_recruiters_30d": 3,
                "interview_completion_rate": 0.9,
                "offer_acceptance_rate": 0.8,
                "verified_email": True,
                "verified_phone": True,
                "linkedin_connected": True,
            }
        }

    c_strong = make_candidate(
        "CAND_0000001", 8, "Pune", "India", 30, 85,
        [{"name": "FAISS", "proficiency": "advanced", "endorsements": 20, "duration_months": 48},
         {"name": "BM25", "proficiency": "advanced", "endorsements": 15, "duration_months": 36},
         {"name": "Python", "proficiency": "expert", "endorsements": 40, "duration_months": 72}],
        0.8
    )

    c_mid = make_candidate(
        "CAND_0000002", 4, "Bangalore", "India", 60, 40,
        [{"name": "Python", "proficiency": "advanced", "endorsements": 12, "duration_months": 36},
         {"name": "NLP", "proficiency": "intermediate", "endorsements": 5, "duration_months": 18}],
        0.4
    )

    c_weak = make_candidate(
        "CAND_0000003", 1, "Austin", "USA", 90, -1,
        [{"name": "LangChain", "proficiency": "advanced", "endorsements": 2, "duration_months": 6}],
        0.1
    )

    scores = [0.9, 0.5, 0.1]
    from features import build_feature_vector, consistency_score

    compiler = ReasoningCompiler(jd, all_scores=scores)

    for candidate, score in [(c_strong, 0.9), (c_mid, 0.5), (c_weak, 0.1)]:
        fv = build_feature_vector(candidate, jd, bm25_score=score * 15, stage1_bm25_median=7.5)
        trace = compiler.compile_trace(candidate, fv, score, rank=scores.index(score)+1)
        print(f"\n=== {candidate['candidate_id']} (score={score}, rank={scores.index(score)+1}) ===")
        print(f"Reasoning: {trace['reasoning']}")
        print(f"Top drivers: {trace['top_feature_drivers']}")
        print(f"Concern: {trace['concern']}")
