
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
from openai import OpenAI
import json, os, re, random
from agents.base import BaseAgent, PipelineState
from utils import RunConfig, load_json_schema, validate_records_with_schema, to_tsv


@dataclass
class AnnotateConfig:
    # prompting
    mode: str = "zeroshot"
    fewshot_path: Optional[str] = "artifacts/examples_fewshot_v1.jsonl"
    temperature: float = 0.0

    # inference / votes
    self_consistency_votes: int = 1
    max_new_tokens: int = 100

    # backend
    provider: str = "openai"        # now default
    api_base: Optional[str] = None
    model_name: str = "gpt-4.1"#"gpt-4.1-mini"

    # I/O
    output_tsv: str = "results/checkpoints/node2_annotated.tsv"
    output_schema_path: Optional[str] = "artifacts/schema/output_label_schema.json"

    # meta
    prompt_version: str = "v1"
    taxonomy_version: str = "v1"

    # ORCAS-I label set
    allowed_types: Tuple[str, ...] = (
        "navigational", "factual", "transactional", "instrumental", "abstain"
    )


# =======================
# Prompt templates
# =======================
TAXONOMY_BLOCK = """You are performing a single-label taxonomy classification for the following query based on these categories:

- Navigational: go to or open a specific site/app/page (e.g., "facebook login", "bbc sport").
- Factual: seek facts, definitions, or knowledge (e.g., "what is backprop", "symptoms of flu").
- Transactional: intent to perform an action, purchase, subscribe, or download (e.g., "buy iphone 13", "download vscode").
- Instrumental: how-to, instructions, or tool usage (e.g., "install pandas", "how to reset iphone").
- Abstain: insufficient or ambiguous to decide confidently.

Classify the query into exactly ONE of these labels.
Return ONLY strict JSON in the following format:
{"type": "...", "confidence": 0.0}
"""


SYSTEM_ANNOTATOR_ZS = """
You are a senior query intent annotator.
Your task is to classify a single search query into one of the intent categories.

Taxonomy definitions:
- navigational → The query is to reach, open, or access a specific site, page, or app (e.g., "facebook login", "bbc sport").
- factual → The query seeks factual information, definitions, or knowledge (e.g., "what is backprop", "symptoms of flu").
- transactional → The query intends to perform an action such as buying, subscribing, downloading, or registering (e.g., "buy iphone 13", "download vscode").
- instrumental → The query is about how to do something or use a tool (e.g., "install pandas", "how to reset iphone").
- abstain → The query is ambiguous or lacks enough information to decide confidently.

Return ONLY a valid JSON object with exactly these keys:
{
  "type": "<one of: navigational, factual, transactional, instrumental, abstain>",
  "confidence": <float between 0 and 1>,
}

Do not include any text outside the JSON.
"""

# Taxonomy definitions:
# - navigational → The query is to reach, open, or access a specific site, page, or app (e.g., "facebook login", "bbc sport").
# - factual → The query seeks factual information, definitions, or knowledge (e.g., "what is backprop", "symptoms of flu").
# - transactional → The query intends to perform an action such as buying, subscribing, downloading, or registering (e.g., "buy iphone 13", "download vscode").
# - instrumental → The query is about how to do something or use a tool (e.g., "install pandas", "how to reset iphone").
# - abstain → The query is ambiguous or lacks enough information to decide confidently.


SYSTEM_ANNOTATOR_FS = """
You are a senior query intent annotator. Your task is to classify a single search query into one of the intent categories.

You will see several EXAMPLES (query → label). Use them as guidance. If examples conflict or do not cover the case, fall back to these rules:

Taxonomy definitions:
- navigational (brand/domain/URL/login/homepage) →
- transactional (buy/subscribe/register/download/apply/pay/book/reserve/renew/cancel) →
- instrumental (how to/steps/use/install/fix/reset/configure/setup/recipe/tutorial/guide) →
- factual (what/when/who/why/meaning/symptoms/definition/info) →
- abstain (insufficient/ambiguous).

Return ONLY a valid JSON object with exactly these keys:
{
  "type": "<one of: navigational, factual, transactional, instrumental, abstain>",
  "confidence": <float between 0 and 1>,
}

Do not include any text outside the JSON.
 """

def build_user_prompt_zeroshot(query_text: str) -> str:
    # Short + strict; the SYSTEM prompt carries the taxonomy/rules.
    return (
        'Task: classify the query into exactly one label and return JSON only.\n'
        'Allowed labels: navigational, factual, transactional, instrumental, abstain.\n'
        f'Query: "{query_text}"'
    )

def build_user_prompt_fewshot(query_text: str, shots: List[Dict[str, Any]]) -> str:
    # Up to 8 concise examples:  Example: "query" → label
    lines: List[str] = []
    for ex in shots[:8]:
        q = str(ex.get("query", "")).strip()
        t = str(ex.get("type", "")).strip().lower()
        if q and t:
            lines.append(f'Example: "{q}" → {t}')
    examples = "\n".join(lines)
    return (
        (examples + "\n\n") if examples else ""
    ) + 'Now classify the following query and return JSON only.\n' + f'Query: "{query_text}"'


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_json(text: str) -> str:
    m = JSON_BLOCK_RE.search(text or "")
    if not m:
        return json.dumps({"type": "abstain", "confidence": 0.0})
    return m.group(0)


def normalize_type(t: str, allowed: Tuple[str, ...]) -> str:
    t = (t or "").strip().lower()
    aliases = {
        "info": "factual", "informational": "factual", "informative": "factual",
        "nav": "navigational", "navigation": "navigational",
        "transact": "transactional", "purchase": "transactional",
        "howto": "instrumental", "how-to": "instrumental", "procedural": "instrumental",
    }
    t = aliases.get(t, t)
    return t if t in allowed else "abstain"


def clamp_conf(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        v = 0.0
    return max(0.0, min(1.0, v))

def build_openai_client(provider: str, api_base: Optional[str]) -> OpenAI:
    if provider == "openai":
        token = os.getenv("OPENAI_API_KEY")
        if not token:
            raise RuntimeError("OPENAI_API_KEY not set in .env")
        base = api_base or "https://api.openai.com/v1"
        return OpenAI(base_url=base, api_key=token)
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def openai_chat_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
    force_json: bool = True,
) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_new_tokens),
    }
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}

    tries = 0
    while True:
        tries += 1
        try:
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as e:
            if tries >= 2:
                return json.dumps({"type": "abstain", "confidence": 0.0})


# =======================
# Main Agent
# =======================
class LLMAnnotateAgent(BaseAgent):
    """Node 2: Annotate queries using OpenAI GPT-4.1-mini."""

    def __init__(self, run_cfg: RunConfig, ann_cfg: AnnotateConfig) -> None:
        self.run_cfg = run_cfg
        self.ann_cfg = ann_cfg
        self.schema = load_json_schema(ann_cfg.output_schema_path) if ann_cfg.output_schema_path else None

        if ann_cfg.provider == "openai":
            self.client = build_openai_client("openai", ann_cfg.api_base)
        else:
            raise ValueError("Only provider='openai' is supported in this config.")

        # NEW: few-shot examples (only when requested)
        # Few-shot examples (only when requested)
        self._fewshots: List[Dict[str, Any]] = []
        if (self.ann_cfg.mode or "zeroshot").lower() == "fewshot":
            self._fewshots = self._read_jsonl(self.ann_cfg.fewshot_path)
            random.Random(42).shuffle(self._fewshots)  # stable variety

    def _one_vote(self, query_text: str) -> Dict[str, Any]:
        mode = (self.ann_cfg.mode or "zeroshot").lower()
        if mode == "fewshot" and getattr(self, "_fewshots", None):
            system_prompt = SYSTEM_ANNOTATOR_FS
            user_prompt = build_user_prompt_fewshot(query_text, self._fewshots)
        else:
            system_prompt = SYSTEM_ANNOTATOR_ZS
            user_prompt = build_user_prompt_zeroshot(query_text)

        raw = openai_chat_json(
            client=self.client,
            model=self.ann_cfg.model_name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=self.ann_cfg.max_new_tokens,
            temperature=self.ann_cfg.temperature,
        )

        try:
            parsed = json.loads(extract_json(raw))
        except Exception:
            parsed = {"type": "abstain", "confidence": 0.0}

        t = normalize_type(parsed.get("type", ""), self.ann_cfg.allowed_types)
        conf = clamp_conf(parsed.get("confidence", 0.0))
        return {"type": t, "confidence": conf}

    def _annotate_one(self, query_text: str) -> Dict[str, Any]:
        """Runs one annotation; if self_consistency_votes>1, aggregates votes."""
        v = int(getattr(self.ann_cfg, "self_consistency_votes", 1) or 1)
        if v <= 1:
            return self._one_vote(query_text)

        votes = [self._one_vote(query_text) for _ in range(v)]

        from collections import defaultdict
        counts, confs = defaultdict(int), defaultdict(list)
        for r in votes:
            lbl = r.get("type", "abstain")
            c = float(r.get("confidence", 0.0) or 0.0)
            counts[lbl] += 1
            confs[lbl].append(c)


        # pick label by (count → mean confidence → max confidence)
        def score(lbl):
            mean_c = sum(confs[lbl]) / len(confs[lbl]) if confs[lbl] else 0.0
            max_c = max(confs[lbl]) if confs[lbl] else 0.0
            return (counts[lbl], mean_c, max_c)

        best = max(counts.keys(), key=score)
        best_conf = sum(confs[best]) / len(confs[best]) if confs[best] else 0.0
        return {"type": best, "confidence": best_conf}

    def _read_jsonl(self, path: Optional[str]) -> List[Dict[str, Any]]:
        if not path:
            return []
        rows: List[Dict[str, Any]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                        q = str(ex.get("query", "")).strip()
                        t = str(ex.get("type", "")).strip().lower()
                        if q and t:
                            rows.append({"query": q, "type": t})
                    except Exception:
                        # skip malformed line
                        pass
        except FileNotFoundError:
            pass
        return rows

    # in agents/annotate.py, inside class LLMAnnotateAgent(BaseAgent):

    def run(self, state: PipelineState, **_) -> PipelineState:
        records = state.get("records", [])
        if not records:
            raise ValueError("Node 2: no records found in state. Did Node 1 run?")

        out_rows: List[Dict[str, Any]] = []
        for i, rec in enumerate(records, 1):
            has_label = "ai_label" in rec and rec.get("ai_label") is not None
            needs_retry = bool(rec.get("needs_retry", False))

            # annotate if no label yet (first pass), or flagged for retry
            if (not has_label) or needs_retry:
                res = self._annotate_one(rec.get("query", ""))
                row = dict(rec)
                row.update({
                    "ai_label": res["type"],
                    "ai_confidence": res["confidence"],
                    "model_name": self.ann_cfg.model_name,
                    "provider": self.ann_cfg.provider,
                    "prompt_version": self.ann_cfg.prompt_version,
                    "taxonomy_version": self.ann_cfg.taxonomy_version,
                })
                # this was a retry if it had needs_retry
                if needs_retry:
                    row["retry_count"] = int(row.get("retry_count", 0)) + 1
                # clear retry flag after annotating
                row["needs_retry"] = False
            else:
                # keep existing
                row = dict(rec)

            out_rows.append(row)
            if i % 10 == 0:
                print(f"[annotate] {i}/{len(records)} done")


        to_tsv(pd.DataFrame(out_rows), self.ann_cfg.output_tsv)
        state["records"] = out_rows
        state["annotations"] = [
            {"qid": r.get("qid"), "query": r.get("query"),
             "ai_label": r.get("ai_label"), "ai_confidence": r.get("ai_confidence")}
            for r in out_rows
        ]
        return state
