"""
Base Paper Evaluation Suite for HCAD-RAG.

Directly implements the evaluation protocols from the foundational base paper:
"Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation"
(Hyper-RAG, arXiv:2504.08758v1, Sections 4.1, 4.2, 1.5, and 5.2).

Protocols:
1. Scoring-Based Assessment (P_eval_score, Section 4.1 & 5.2):
   - 5 dimensions on a 0-100 scale using a 5-level hierarchical rubric:
     * Comprehensiveness (0-100): Covers all relevant aspects without omissions.
     * Diversity (0-100): Richness of domain context beyond direct query.
     * Empowerment (0-100): Factual credibility, hallucination suppression, and clinical confidence.
     * Logical (0-100): Coherence and absence of self-contradiction.
     * Readability (0-100): Structural clarity, accessibility, and medical formatting.
   - Overall Score = arithmetic mean of the 5 dimensions.

2. Selection-Based Assessment (P_eval_select, Section 4.2 & 5.2):
   - Blind pairwise evaluation (Model A vs. Model B) across 8 qualitative criteria:
     * Comprehensiveness, Empowerment, Accuracy, Relevance, Coherence,
       Clarity, Logical, Flexibility.
   - Head-to-head cumulative win rate (%).

3. Retrieval Latency Efficiency (Section 1.5):
   - Wall-clock retrieval time (seconds) isolating local DB search and tensor processing.
"""

import json
import logging
import math
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.llm_client import UnifiedLLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ScoringResult:
    """Quantitative evaluation across the base paper's 5 dimensions."""
    comprehensiveness: float
    diversity: float
    empowerment: float
    logical: float
    readability: float
    overall_score: float
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class SelectionResult:
    """Pairwise selection evaluation across the base paper's 8 criteria."""
    comprehensiveness_winner: str
    empowerment_winner: str
    accuracy_winner: str
    relevance_winner: str
    coherence_winner: str
    clarity_winner: str
    logical_winner: str
    flexibility_winner: str
    overall_winner: str
    criterion_rationales: Dict[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# ---------------------------------------------------------------------------
# Base Paper Prompts 
# ---------------------------------------------------------------------------

SCORING_PROMPT_SYSTEM = """You are an expert medical evaluator tasked with evaluating clinical answers by using the relevant documents based on five criteria: Comprehensiveness, Diversity, Empowerment, Logical, and Readability."""

SCORING_PROMPT_USER = """Evaluate the answer to the clinical question using the relevant reference document based on five criteria: Comprehensiveness, Diversity, Empowerment, Logical, and Readability.

- Comprehensiveness:
Measure whether the answer comprehensively covers all key aspects of the question and whether there are omissions.
Level 1 (0-20): The answer is extremely one-sided, leaving out key parts or important aspects of the question.
Level 2 (20-40): The answer has some content, but it misses many important aspects of the question and is not comprehensive enough.
Level 3 (40-60): The answer is more comprehensive, covering the main aspects of the question, but there are still some omissions.
Level 4 (60-80): The answer is comprehensive, covering most aspects of the question, with few omissions.
Level 5 (80-100): The answer is extremely comprehensive, covering all aspects of the question with no omissions, enabling the reader to gain a complete understanding.

- Diversity:
Evaluates the richness of the content, including additional related biomedical knowledge beyond the direct answer (Level 1: 0-20 to Level 5: 80-100).

- Empowerment:
Measures clinical credibility, factual accuracy, and appropriate uncertainty (Level 1: 0-20 to Level 5: 80-100). When retrieved evidence is supplied, verify cited claims against it. Give credit for citations only when the cited evidence directly supports the claim; citation tags alone earn no credit. Do not penalize uncited claims when they are supported by the reference document or established clinical knowledge. Penalize unsupported or contradicted high-impact claims regardless of citation style.

- Logical:
Determines the coherence and clarity of the response, ensuring that the arguments are logically structured, well-articulated, and free of contradictions (Level 1: 0-20 to Level 5: 80-100).

- Readability:
Examines the organization and formatting of the response, ensuring it is clear, professional, and easy to understand (Level 1: 0-20 to Level 5: 80-100).

--- Inputs ---
Question: {question}
Relevant Reference Document: {reference_document}
Retrieved Evidence Available to This System: {answer_evidence}
System Answer: {answer}

--- Output Instructions ---
Provide numeric scores strictly on a scale from 0 to 100 for each indicator based on the rubric.
Respond strictly in JSON format as follows:
{{
  "comprehensiveness": <score 0-100>,
  "diversity": <score 0-100>,
  "empowerment": <score 0-100>,
  "logical": <score 0-100>,
  "readability": <score 0-100>,
  "rationale": "<summary explanation>"
}}
"""

SELECTION_PROMPT_SYSTEM = """You are an expert clinical evaluator tasked with evaluating two answers to the same question based on eight criteria: Comprehensiveness, Empowerment, Accuracy, Relevance, Coherence, Clarity, Logical, and Flexibility."""

SELECTION_PROMPT_USER = """Evaluate two answers to the same clinical question by using the relevant documents based on eight criteria:
1. Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
2. Empowerment: How well does the answer help the reader make informed clinical judgments with accurate, appropriately supported claims? When evidence is supplied, verify citations against it. Do not reward citation tags by themselves or penalize uncited claims supported by the reference or established clinical knowledge.
3. Accuracy: How well does the answer align with factual truth and avoid hallucination based on the reference and any answer-specific retrieved evidence supplied below?
4. Relevance: How precisely does the answer address the core aspects of the question without including unnecessary information?
5. Coherence: How well does the system integrate and synthesize information from multiple sources into a logically flowing response?
6. Clarity: How well does the system provide complete information while avoiding unnecessary verbosity and redundancy?
7. Logical: How well does the system maintain consistent logical arguments without contradicting itself across the response?
8. Flexibility: How well does the system handle various question formats, tones, and levels of complexity?

For each criterion, choose the better answer (either "Answer 1" or "Answer 2") and explain why. Then, select an overall winner between Answer 1 and Answer 2.

--- Inputs ---
Question: {question}
Relevant Reference Document: {reference_document}
Retrieved Evidence Available to Answer 1: {evidence_1}
Answer 1: {answer_1}
Retrieved Evidence Available to Answer 2: {evidence_2}
Answer 2: {answer_2}

--- Output Instructions ---
Respond strictly in JSON format as follows:
{{
  "comprehensiveness": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "empowerment": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "accuracy": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "relevance": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "coherence": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "clarity": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "logical": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "flexibility": {{"winner": "Answer 1" | "Answer 2" | "Tie", "reason": "<brief reason>"}},
  "overall_winner": "Answer 1" | "Answer 2" | "Tie",
  "summary": "<overall comparative verdict>"
}}
"""

# ---------------------------------------------------------------------------
# Base Paper Evaluator Engine
# ---------------------------------------------------------------------------

class BasePaperEvaluator:
    """
    Evaluator strictly compliant with Hyper-RAG .
    Provides Scoring-Based Assessment (0-100), Selection-Based Assessment (Win-Rate),
    and Retrieval Latency profiling.
    """

    CRITERIA_LIST = [
        "comprehensiveness",
        "empowerment",
        "accuracy",
        "relevance",
        "coherence",
        "clarity",
        "logical",
        "flexibility",
    ]

    def __init__(self, judge_llm: Optional[UnifiedLLMClient] = None):
        """Initialize the configured deterministic judge (Phi-4 via Transformers on Kaggle)."""
        if judge_llm is None:
            from src.config import JUDGE_HF_MODEL
            self.judge = UnifiedLLMClient(model=JUDGE_HF_MODEL, temperature=0.0, max_tokens=768)
        else:
            self.judge = judge_llm

    def _extract_json(self, raw_text: str) -> Dict[str, Any]:
        """Robustly extract JSON payload from LLM markdown fences or raw strings."""
        clean_text = (raw_text or "").strip()
        # Strip chain-of-thought tags (e.g. DeepSeek-R1 <think>...</think>)
        clean_text = re.sub(r"<think>[\s\S]*?</think>", "", clean_text).strip()

        # Try multiple JSON extraction strategies

        # 1. Try to find JSON in markdown code fences first (most reliable)
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # 2. Try to find the first complete JSON object
        idx = clean_text.find("{")
        if idx != -1:
            try:
                decoder = json.JSONDecoder()
                obj, end_idx = decoder.raw_decode(clean_text[idx:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass

            # 3. Try to find balanced braces
            brace_count = 0
            for i, char in enumerate(clean_text[idx:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = clean_text[idx:idx+i+1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError:
                            break

        # 4. Try to find JSON without markdown fences - look for standalone JSON
        standalone_match = re.search(r'\{[^{}]*\}', clean_text, re.DOTALL)
        if standalone_match:
            try:
                return json.loads(standalone_match.group())
            except json.JSONDecodeError:
                pass

        logger.warning(f"Failed to parse LLM evaluation JSON directly. Falling back to heuristic recovery. Snippet: {clean_text[:200]}")
        return self._heuristic_json_recovery(clean_text)

    def _heuristic_json_recovery(self, text: str) -> Dict[str, Any]:
        """Fallback parser when LLM outputs markdown text or malformed JSON."""
        res: Dict[str, Any] = {}

        try:
            text_lower = (text or "").lower()

            # 1. Scoring dimensions - handle multiple pattern formats
            scoring_keys = ["comprehensiveness", "diversity", "empowerment", "logical", "readability", "overall_score"]
            for key in scoring_keys:
                patterns = [
                    # Nested object: "comprehensiveness": {"score": 85}
                    rf'(?:\"{key}\"|\'{key}\'|\b{key}\b)\s*:\s*\{{[\s\S]*?\"score\"\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                    # JSON-style with quotes: "comprehensiveness": 85
                    rf'(?:\"{key}\"|\'{key}\')\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                    # Unquoted key: comprehensiveness: 85
                    rf'\b{key}\b\s*:\s*([0-9]+(?:\.[0-9]+)?)',
                    # Markdown bold: **Comprehensiveness:** 85
                    rf'\*\*{key}\*\*\s*:?\s*([0-9]+(?:\.[0-9]+)?)',
                    # Score for key: Score for Comprehensiveness: 85
                    rf'(?:score|rating|level)\s*(?:for|of)?\s*{key}\s*:?\s*([0-9]+(?:\.[0-9]+)?)',
                ]

                for pattern in patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        try:
                            score = float(match.group(1))
                            res[key] = max(0.0, min(100.0, score))
                            break
                        except ValueError:
                            continue

            # 2. Pairwise Selection criteria - comprehensive pattern matching
            for crit in self.CRITERIA_LIST:
                # Pattern 1: JSON-style with nested object (supports multiline)
                crit_match = re.search(
                    rf'(?:\"{crit}\"|\'{crit}\'|\b{crit}\b)\s*:\s*\{{[\s\S]*?(?:\"winner\"|winner)\s*:\s*\"?(Answer 1|Answer 2|Tie)\"?',
                    text, re.IGNORECASE
                )
                if not crit_match:
                    # Pattern 2: Markdown-style with Winner mention
                    crit_match = re.search(
                        rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?(?:Winner|winner|choice|answer)\s*:?\s*(Answer 1|Answer 2|Tie)',
                        text, re.IGNORECASE
                    )
                if not crit_match:
                    # Pattern 3: Simple mention of Answer 1 or Answer 2 after criterion
                    answer_1_match = re.search(rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?\banswer\s*1\b', text, re.IGNORECASE)
                    answer_2_match = re.search(rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?\banswer\s*2\b', text, re.IGNORECASE)
                    tie_match = re.search(rf'(?:\*\*{crit}\*\*|\b{crit}\b)[\s\S]{0,200}?\btie\b', text, re.IGNORECASE)

                    if answer_1_match:
                        res[crit] = {"winner": "Answer 1"}
                    elif answer_2_match:
                        res[crit] = {"winner": "Answer 2"}
                    elif tie_match:
                        res[crit] = {"winner": "Tie"}

            # 3. Overall winner - comprehensive patterns
            overall_patterns = [
                r'(?:Overall Winner|overall_winner|Final Winner|Best Answer|Chooses?)\s*:?\s*(Answer 1|Answer 2|Tie)',
                r'(?:Winner|Victor|Preferred|Selects?)\s*:?\s*(Answer 1|Answer 2|Tie)',
                r'Final\s+(?:decision|result|outcome|answer)\s*:?\s*(Answer 1|Answer 2|Tie)',
            ]

            overall_match = None
            for pattern in overall_patterns:
                overall_match = re.search(pattern, text, re.IGNORECASE)
                if overall_match:
                    break

            if overall_match:
                res["overall_winner"] = overall_match.group(1)
            elif any(crit in res for crit in self.CRITERIA_LIST):
                # Compute majority winner across criteria if overall_winner was not explicitly stated
                winners = [res[c].get("winner", "Tie") for c in self.CRITERIA_LIST if c in res and isinstance(res[c], dict)]
                count_1 = sum(1 for w in winners if "Answer 1" in w or "1" in w)
                count_2 = sum(1 for w in winners if "Answer 2" in w or "2" in w)
                if count_1 > count_2:
                    res["overall_winner"] = "Answer 1"
                elif count_2 > count_1:
                    res["overall_winner"] = "Answer 2"
                else:
                    res["overall_winner"] = "Tie"

            if "rationale" not in res and "summary" not in res:
                res["rationale"] = "Extracted via regex fallback."

        except Exception as e:
            logger.warning(f"Heuristic JSON recovery failed: {e}")
            # Return minimal valid structure
            res = {"rationale": f"Recovery failed: {e}"}

        return res

    # -----------------------------------------------------------------------
    # 1. Scoring-Based Assessment (0-100 Scale, 5 Dimensions)
    # -----------------------------------------------------------------------

    @staticmethod
    def _normalize_dim_score(val: Any, default: float = 75.0) -> float:
        """Convert raw judge output to a standard 0-100 scale.

        Our scoring prompt explicitly asks for 'numeric scores strictly on a
        scale from 0 to 100', so we treat all numeric outputs as 0-100 values.
        """
        if isinstance(val, dict):
            val = val.get("score", val.get("level", default))
        try:
            score = float(val)
        except (ValueError, TypeError):
            return default

        return max(0.0, min(100.0, score))

    def evaluate_scoring(
        self,
        question: str,
        answer: str,
        reference_document: str,
        answer_evidence: str = "No retrieved evidence was supplied for this answer.",
        retries: int = 1,
        fallback_on_error: bool = False,
    ) -> ScoringResult:
        """
        Execute Scoring-Based Assessment (P_eval_scoring(q, R, T_0)).
        Returns scores across the 5 dimensions on a 0-100 scale.
        """
        messages = [
            {"role": "system", "content": SCORING_PROMPT_SYSTEM},
            {"role": "user", "content": SCORING_PROMPT_USER.format(
                question=question,
                reference_document=reference_document,
                answer=answer,
                answer_evidence=answer_evidence,
            )},
        ]

        score_fields = ("comprehensiveness", "diversity", "empowerment", "logical", "readability")
        last_exc: Optional[Exception] = None

        for attempt in range(max(1, retries + 1)):
            try:
                raw_response = self.judge.chat(messages, temperature=0.0)
                data = self._extract_json(raw_response)
                logger.debug("Judge raw evaluation response: %s", raw_response)
                logger.debug("Judge parsed data: %s", data)

                missing = [field for field in score_fields if field not in data]
                if missing:
                    raise ValueError(f"Judge response is missing score fields: {', '.join(missing)}")

                normalized = {}
                for field in score_fields:
                    value = data[field]
                    if isinstance(value, dict):
                        value = value.get("score", value.get("level"))
                    try:
                        numeric_value = float(value)
                    except (TypeError, ValueError) as exc:
                        raise ValueError(f"Judge returned a non-numeric {field} score") from exc
                    if not math.isfinite(numeric_value):
                        raise ValueError(f"Judge returned a non-finite {field} score")
                    normalized[field] = max(0.0, min(100.0, numeric_value))

                comp = normalized["comprehensiveness"]
                div = normalized["diversity"]
                emp = normalized["empowerment"]
                log = normalized["logical"]
                read = normalized["readability"]

                overall = (comp + div + emp + log + read) / 5.0
                rationale = str(data.get("rationale") or data.get("justification") or "")

                return ScoringResult(
                    comprehensiveness=comp,
                    diversity=div,
                    empowerment=emp,
                    logical=log,
                    readability=read,
                    overall_score=overall,
                    rationale=rationale,
                )
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Judge scoring attempt %d failed: %s", attempt + 1, exc
                )

        if fallback_on_error:
            logger.warning(
                "All scoring attempts failed (%s); returning neutral 50.0 fallback scores.",
                last_exc,
            )
            return ScoringResult(
                comprehensiveness=50.0,
                diversity=50.0,
                empowerment=50.0,
                logical=50.0,
                readability=50.0,
                overall_score=50.0,
                rationale=f"Fallback due to judge failure: {last_exc}",
            )
        assert last_exc is not None
        raise last_exc

    # -----------------------------------------------------------------------
    # 2. Selection-Based Assessment (Pairwise Win-Rate, 8 Criteria)
    # -----------------------------------------------------------------------

    def evaluate_selection(
        self,
        question: str,
        answer_a: str,
        answer_b: str,
        reference_document: str,
        label_a: str = "System A",
        label_b: str = "System B",
        evidence_a: str = "No retrieved evidence was supplied for this answer.",
        evidence_b: str = "No retrieved evidence was supplied for this answer.",
        retries: int = 1,
        fallback_on_error: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute Selection-Based Assessment (P_eval_select(q, R_a, R_b)).
        Compares Answer A vs. Answer B blindly across 8 criteria.
        """
        messages = [
            {"role": "system", "content": SELECTION_PROMPT_SYSTEM},
            {"role": "user", "content": SELECTION_PROMPT_USER.format(
                question=question,
                reference_document=reference_document,
                answer_1=answer_a,
                answer_2=answer_b,
                evidence_1=evidence_a,
                evidence_2=evidence_b,
            )},
        ]

        last_exc: Optional[Exception] = None

        for attempt in range(max(1, retries + 1)):
            try:
                raw_response = self.judge.chat(messages, temperature=0.0)
                data = self._extract_json(raw_response)

                missing_criteria = [crit for crit in self.CRITERIA_LIST if crit not in data]
                if missing_criteria:
                    raise ValueError(
                        "Judge response is missing pairwise criteria: "
                        + ", ".join(missing_criteria)
                    )

                criteria_winners = {}
                for crit in self.CRITERIA_LIST:
                    crit_obj = data.get(crit, {})
                    if isinstance(crit_obj, dict):
                        w = crit_obj.get("winner")
                    elif isinstance(crit_obj, str):
                        w = crit_obj
                    else:
                        raise ValueError(f"Judge returned malformed result for {crit}")

                    # Map Answer 1 / Answer 2 back to caller labels
                    w_str = str(w or "").strip()
                    normalized_winner = w_str.casefold().replace(" ", "")
                    if normalized_winner in {"answer1", "1"}:
                        mapped_w = label_a
                    elif normalized_winner in {"answer2", "2"}:
                        mapped_w = label_b
                    elif normalized_winner == "tie":
                        mapped_w = "Tie"
                    else:
                        raise ValueError(f"Judge returned invalid winner for {crit}: {w_str!r}")
                    criteria_winners[crit] = mapped_w

                overall_raw = str(data.get("overall_winner", "Tie")).strip().casefold().replace(" ", "")
                if overall_raw in {"answer1", "1"}:
                    final_winner = label_a
                elif overall_raw in {"answer2", "2"}:
                    final_winner = label_b
                elif overall_raw == "tie":
                    final_winner = "Tie"
                else:
                    # Derive overall winner from majority of 8 qualitative criteria
                    count_a = sum(1 for w in criteria_winners.values() if w == label_a)
                    count_b = sum(1 for w in criteria_winners.values() if w == label_b)
                    if count_a > count_b:
                        final_winner = label_a
                    elif count_b > count_a:
                        final_winner = label_b
                    else:
                        final_winner = "Tie"

                return {
                    "criteria_winners": criteria_winners,
                    "overall_winner": final_winner,
                    "summary": data.get("summary", ""),
                }
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Judge selection attempt %d failed: %s", attempt + 1, exc
                )

        if fallback_on_error:
            logger.warning(
                "All selection attempts failed (%s); returning neutral Tie fallback.",
                last_exc,
            )
            return {
                "criteria_winners": {crit: "Tie" for crit in self.CRITERIA_LIST},
                "overall_winner": "Tie",
                "summary": f"Fallback due to judge failure: {last_exc}",
            }
        assert last_exc is not None
        raise last_exc

    # -----------------------------------------------------------------------
    # 3. Efficiency & Latency Benchmark
    # -----------------------------------------------------------------------

    @staticmethod
    def measure_retrieval_latency(retriever_fn, query: str, runs: int = 3) -> Tuple[float, Any]:
        """
        Measure pure wall-clock retrieval time (seconds) isolating local DB search,
        graph traversal, and tensor projection from LLM inference latency.
        """
        if runs <= 0:
            raise ValueError("runs must be a positive integer")
        # one untimed WARMUP so cold caches do not poison
        # the measurement; report the MEDIAN plus p95 instead of the mean
        # (a single cold sample previously dominated the committed numbers:
        # 14.0 s "average" for vanilla RAG vs 0.98 s for HCAD).
        retriever_fn(query)
        latencies = []
        result = None
        for _ in range(runs):
            t0 = time.perf_counter()
            result = retriever_fn(query)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
        ordered = sorted(latencies)
        median = ordered[len(ordered) // 2] if ordered else 0.0
        p95 = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] if ordered else 0.0
        logger.info("Latency median=%.3fs p95=%.3fs runs=%d", median, p95, runs)
        return median, result
