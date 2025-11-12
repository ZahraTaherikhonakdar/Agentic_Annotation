from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
import json
import os
import random

import pandas as pd
from openai import OpenAI

from agents.base import BaseAgent, PipelineState
from utils import to_tsv

@dataclass
class EvalConfig:
    # files
    output_tsv: str = "results/eval/metrics_by_class.tsv"
    output_json: str = "results/eval/summary.json"

    # labels/fields
    human_col: str = "human_label_gold"      # from Node 1 normalization (optional)
    ai_col_candidates: Tuple[str, ...] = ("final_type", "ai_label")
    allowed_types: Tuple[str, ...] = (
        "navigational", "factual", "transactional", "instrumental", "abstain"
    )

    # Metric toggles
    use_prf: bool = True                 # P/R/F1 requires human labels
    use_kappa: bool = True               # Cohen's kappa vs human
    use_judge: Optional[bool] = None     # If None, fall back to judge_enabled for back-compat

    # LLM-as-Judge (either vs human or audit without human)
    judge_enabled: bool = False          # legacy flag (back-compat)
    judge_mode: str = "compare_to_human" # "compare_to_human" | "audit"
    judge_model: str = "gpt-4.1-mini"
    judge_sample_size: int = 500
    judge_temperature: float = 0.0
    judge_max_tokens: int = 120
    judge_output_tsv: str = "results/eval/judge_sample.tsv"
    judge_seed: int = 42

    # misc
    drop_nan: bool = True

def _confusion(labels: List[str], preds: List[str], classes: List[str]) -> pd.DataFrame:
    idx = {c: i for i, c in enumerate(classes)}
    mat = [[0]*len(classes) for _ in classes]
    for y, yhat in zip(labels, preds):
        if y in idx and yhat in idx:
            mat[idx[y]][idx[yhat]] += 1
    return pd.DataFrame(mat, index=classes, columns=classes)

def _precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r/(p+r) if (p+r) else 0.0
    return p, r, f1

def _cohen_kappa(labels: List[str], preds: List[str], classes: List[str]) -> float:
    cm = _confusion(labels, preds, classes).values
    n = cm.sum()
    if n == 0:
        return 0.0
    po = sum(cm[i][i] for i in range(len(classes))) / n
    row_marg = cm.sum(axis=1)
    col_marg = cm.sum(axis=0)
    pe = sum(row_marg[i] * col_marg[i] for i in range(len(classes))) / (n*n)
    denom = (1 - pe)
    return (po - pe) / denom if denom else 0.0

def _binary_kappa(labels: List[str], preds: List[str], positive: str) -> float:
    y = [1 if y == positive else 0 for y in labels]
    yhat = [1 if p == positive else 0 for p in preds]
    tp = sum(1 for a,b in zip(y,yhat) if a==1 and b==1)
    tn = sum(1 for a,b in zip(y,yhat) if a==0 and b==0)
    fp = sum(1 for a,b in zip(y,yhat) if a==0 and b==1)
    fn = sum(1 for a,b in zip(y,yhat) if a==1 and b==0)
    n = tp+tn+fp+fn
    if n == 0: return 0.0
    po = (tp+tn)/n
    p_yes_true = (tp+fn)/n
    p_yes_pred = (tp+fp)/n
    pe = p_yes_true*p_yes_pred + (1-p_yes_true)*(1-p_yes_pred)
    denom = (1-pe)
    return (po-pe)/denom if denom else 0.0

def _pick_ai_col(df: pd.DataFrame, candidates: Tuple[str, ...]) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No AI label column found among {candidates}")

# LLM-as-Judge prompts
JUDGE_TAXONOMY = """Taxonomy:
- navigational → reach/open a specific site/page/app (e.g., "facebook login", "bbc sport")
- factual → seek knowledge, facts, or definitions (e.g., "what is backprop", "symptoms of flu")
- transactional → intent to act (buy/subscribe/download/register) (e.g., "buy iphone 13", "download vscode")
- instrumental → how-to or tool usage (e.g., "install pandas", "how to reset iphone")
- abstain → ambiguous or insufficient information
"""

# Compare-to-human (your original behavior)
JUDGE_SYSTEM_HUMAN = (
    "You are an expert quality auditor for query intent labeling. "
    "Given the taxonomy and a (query, human_label, ai_label), "
    "decide if the AI label is CORRECT or INCORRECT with a short reason. "
    "Return JSON ONLY with keys: decision ('correct'|'incorrect'), confidence (0..1), rationale (<=120 chars)."
)

# Audit mode: no human label needed; judge provides its own label too
JUDGE_SYSTEM_AUDIT = (
    "You are an expert quality auditor for query intent labeling. "
    "Given the taxonomy and a (query, ai_label), first classify the query yourself, "
    "then decide if the AI label matches your classification. "
    "Return JSON ONLY with keys: "
    "judge_label (one of: navigational, factual, transactional, instrumental, abstain), "
    "decision ('correct'|'incorrect'), confidence (0..1), rationale (<=120 chars)."
)

def _build_openai() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=key)

def _judge_one_compare_to_human(client: OpenAI, model: str, query: str, human: str, ai: str,
                                temperature: float, max_tokens: int) -> Dict[str, Any]:
    user = (
        f"{JUDGE_TAXONOMY}\n"
        f'Query: "{query}"\n'
        f"Human label: {human}\n"
        f"AI label: {ai}\n"
        f'Return JSON ONLY: {{"decision":"correct|incorrect","confidence":0.0,"rationale":"..."}}'
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":JUDGE_SYSTEM_HUMAN},
                      {"role":"user","content":user}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_format={"type":"json_object"},
        )
        txt = resp.choices[0].message.content or "{}"
        data = json.loads(txt)
        dec = str(data.get("decision","")).strip().lower()
        if dec not in ("correct","incorrect"):
            dec = "incorrect"
        conf = float(data.get("confidence", 0.0))
        rat = str(data.get("rationale",""))[:120]
        return {"decision": dec, "confidence": max(0.0, min(1.0, conf)), "rationale": rat}
    except Exception as e:
        return {"decision": "incorrect", "confidence": 0.0, "rationale": f"API error: {e}"}

def _judge_one_audit(client: OpenAI, model: str, query: str, ai: str,
                     temperature: float, max_tokens: int) -> Dict[str, Any]:
    user = (
        f"{JUDGE_TAXONOMY}\n"
        f'Query: "{query}"\n'
        f"AI label: {ai}\n"
        f'Return JSON ONLY: {{"judge_label":"navigational|factual|transactional|instrumental|abstain",'
        f'"decision":"correct|incorrect","confidence":0.0,"rationale":"..."}}'
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"system","content":JUDGE_SYSTEM_AUDIT},
                      {"role":"user","content":user}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            response_format={"type":"json_object"},
        )
        txt = resp.choices[0].message.content or "{}"
        data = json.loads(txt)
        jl = str(data.get("judge_label","")).strip().lower()
        if jl not in ("navigational","factual","transactional","instrumental","abstain"):
            jl = "abstain"
        dec = str(data.get("decision","")).strip().lower()
        if dec not in ("correct","incorrect"):
            dec = "incorrect"
        conf = float(data.get("confidence", 0.0))
        rat = str(data.get("rationale",""))[:120]
        return {"judge_label": jl, "decision": dec, "confidence": max(0.0, min(1.0, conf)), "rationale": rat}
    except Exception as e:
        return {"judge_label":"abstain", "decision":"incorrect", "confidence":0.0, "rationale": f"API error: {e}"}


class EvaluateAgent(BaseAgent):
    """
    Evaluates AI-agent annotations vs. human labels (when present):
      - per-class Precision/Recall/F1, macro-F1, Accuracy
      - Cohen's kappa (AI vs human)
      - optional LLM-as-Judge:
          * compare_to_human (needs human) or
          * audit (no human needed; computes AI↔Judge agreement)
    Writes:
      - metrics_by_class.tsv
      - summary.json
      - judge_sample.tsv (optional)
    """

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg

    def run(self, state: PipelineState, **_) -> PipelineState:
        rows = state.get("records", [])
        if not rows:
            raise ValueError("Evaluate: no records to evaluate.")

        df = pd.DataFrame(rows)

        # Human labels may be absent; handle safely
        has_human = (self.cfg.human_col in df.columns) and bool(df[self.cfg.human_col].notna().any())
        if self.cfg.drop_nan and has_human:
            df = df.dropna(subset=[self.cfg.human_col])

        ai_col = _pick_ai_col(df, self.cfg.ai_col_candidates)
        classes = list(self.cfg.allowed_types)

        # sanitize AI predictions
        y_pred = [str(x).strip().lower() for x in df[ai_col].tolist()]

        # Prepare per-class frame with placeholders
        per_cls_rows: List[Dict[str, Any]] = []
        for c in classes:
            per_cls_rows.append({
                "class": c,
                "precision": None,
                "recall": None,
                "f1": None,
                "support": None,
                "judge_score": None,
                "cohen_kappa": None,
            })

        acc = None
        macro_p = None
        macro_r = None
        macro_f1 = None
        overall_kappa = None
        ai_judge_kappa = None

        if has_human and (self.cfg.use_prf or self.cfg.use_kappa):
            y_true = [str(x).strip().lower() for x in df[self.cfg.human_col].tolist()]
            cm = _confusion(y_true, y_pred, classes)

            if self.cfg.use_prf:
                per_cls = []
                for i, c in enumerate(classes):
                    tp = int(cm.iloc[i, i])
                    fp = int(cm.iloc[:, i].sum() - tp)
                    fn = int(cm.iloc[i, :].sum() - tp)
                    p, r, f1 = _precision_recall_f1(tp, fp, fn)
                    support = int(cm.iloc[i, :].sum())
                    per_cls.append((c, p, r, f1, support))
                # fill into rows
                rowmap = {r["class"]: r for r in per_cls_rows}
                for c, p, r, f1, support in per_cls:
                    rowmap[c]["precision"] = p
                    rowmap[c]["recall"] = r
                    rowmap[c]["f1"] = f1
                    rowmap[c]["support"] = support

                correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
                acc = correct / len(y_true) if y_true else 0.0
                macro_p = sum(r["precision"] for r in per_cls_rows if r["precision"] is not None) / len(classes)
                macro_r = sum(r["recall"] for r in per_cls_rows if r["recall"] is not None) / len(classes)
                macro_f1 = sum(r["f1"] for r in per_cls_rows if r["f1"] is not None) / len(classes)

            if self.cfg.use_kappa:
                y_true = [str(x).strip().lower() for x in df[self.cfg.human_col].tolist()]
                overall_kappa = _cohen_kappa(y_true, y_pred, classes)
                # one-vs-rest κ per class
                for r in per_cls_rows:
                    r["cohen_kappa"] = _binary_kappa(y_true, y_pred, r["class"])

        # If no human labels, at least provide AI class counts as "support"
        if not has_human:
            counts = pd.Series(y_pred).value_counts()
            rowmap = {r["class"]: r for r in per_cls_rows}
            for c in classes:
                rowmap[c]["support"] = int(counts.get(c, 0))

        use_judge = self.cfg.use_judge if self.cfg.use_judge is not None else self.cfg.judge_enabled
        judge_summary_overall = None
        if use_judge:
            client = _build_openai()
            sample_df = df.sample(
                n=min(self.cfg.judge_sample_size, len(df)),
                random_state=self.cfg.judge_seed
            )
            judge_rows = []

            if self.cfg.judge_mode == "compare_to_human":
                # Requires human labels; skip gracefully if absent
                if has_human:
                    correct_by_class = {c: 0 for c in classes}
                    total_by_class = {c: 0 for c in classes}
                    for _, r in sample_df.iterrows():
                        q = str(r.get("query", ""))
                        human = str(r.get(self.cfg.human_col, "")).strip().lower()
                        ai = str(r.get(ai_col, "")).strip().lower()
                        res = _judge_one_compare_to_human(
                            client, self.cfg.judge_model, q, human, ai,
                            self.cfg.judge_temperature, self.cfg.judge_max_tokens
                        )
                        judge_rows.append({
                            "qid": r.get("qid"),
                            "query": q,
                            "human_label_gold": human,
                            "ai_label": ai,
                            "judge_decision": res["decision"],
                            "judge_confidence": res["confidence"],
                            "judge_rationale": res["rationale"],
                        })
                        if ai in total_by_class:
                            total_by_class[ai] += 1
                            if res["decision"] == "correct":
                                correct_by_class[ai] += 1
                    # per-class judge_score (grouped by AI label)
                    judge_scores = {
                        c: (correct_by_class[c] / total_by_class[c]) if total_by_class[c] else None
                        for c in classes
                    }
                    for r in per_cls_rows:
                        r["judge_score"] = judge_scores[r["class"]]
                    # overall
                    total_correct = sum(v for v in correct_by_class.values())
                    total_count = sum(v for v in total_by_class.values())
                    judge_summary_overall = (total_correct / total_count) if total_count else None
                else:
                    # no-op if no human labels
                    judge_summary_overall = None

            else:  # audit mode (no human needed)
                correct_by_class = {c: 0 for c in classes}
                total_by_class = {c: 0 for c in classes}
                ai_labels_sample = []
                judge_labels_sample = []

                for _, r in sample_df.iterrows():
                    q = str(r.get("query", ""))
                    ai = str(r.get(ai_col, "")).strip().lower()
                    res = _judge_one_audit(
                        client, self.cfg.judge_model, q, ai,
                        self.cfg.judge_temperature, self.cfg.judge_max_tokens
                    )
                    jl = res["judge_label"]
                    judge_rows.append({
                        "qid": r.get("qid"),
                        "query": q,
                        "ai_label": ai,
                        "judge_label": jl,
                        "judge_decision": res["decision"],
                        "judge_confidence": res["confidence"],
                        "judge_rationale": res["rationale"],
                    })
                    if ai in total_by_class:
                        total_by_class[ai] += 1
                        if res["decision"] == "correct":
                            correct_by_class[ai] += 1
                    ai_labels_sample.append(ai)
                    judge_labels_sample.append(jl)

                # per-class judge_score (grouped by AI label)
                judge_scores = {
                    c: (correct_by_class[c] / total_by_class[c]) if total_by_class[c] else None
                    for c in classes
                }
                for r in per_cls_rows:
                    r["judge_score"] = judge_scores[r["class"]]

                # overall judge score
                total_correct = sum(correct_by_class.values())
                total_count = sum(total_by_class.values())
                judge_summary_overall = (total_correct / total_count) if total_count else None

                # AI↔Judge kappa on the sample (store in summary; also reuse TSV's cohen_kappa if no human)
                ai_judge_kappa = _cohen_kappa(judge_labels_sample, ai_labels_sample, classes)

            # write judge sample for auditing
            to_tsv(pd.DataFrame(judge_rows), self.cfg.judge_output_tsv)

        # Build dataframes
        per_cls_df = pd.DataFrame(per_cls_rows)
        # ALL row
        all_row = pd.DataFrame([{
            "class": "ALL",
            "precision": (sum(r for r in per_cls_df["precision"].dropna()) / len(per_cls_df["precision"].dropna()))
                         if per_cls_df["precision"].notna().any() else None,
            "recall": (sum(r for r in per_cls_df["recall"].dropna()) / len(per_cls_df["recall"].dropna()))
                      if per_cls_df["recall"].notna().any() else None,
            "f1": (sum(r for r in per_cls_df["f1"].dropna()) / len(per_cls_df["f1"].dropna()))
                  if per_cls_df["f1"].notna().any() else None,
            "support": int(len(df)),
            "cohen_kappa": overall_kappa if (overall_kappa is not None) else (ai_judge_kappa if ai_judge_kappa is not None else None),
            "judge_score": judge_summary_overall,
        }])

        full_df = pd.concat([per_cls_df, all_row], ignore_index=True)

        # write TSV (keep fixed columns for compatibility)
        to_tsv(full_df[["class", "precision", "recall", "f1", "support", "judge_score", "cohen_kappa"]],
               self.cfg.output_tsv)

        # summary JSON
        def _f(x):
            return float(x) if x is not None else None

        summary = {
            "overall": {
                "accuracy": _f(acc),
                "macro_precision": _f(macro_p),
                "macro_recall": _f(macro_r),
                "macro_f1": _f(macro_f1),
                "cohen_kappa": _f(overall_kappa),
                "judge_score": _f(judge_summary_overall),
                "ai_judge_kappa": _f(ai_judge_kappa),
                "has_human_labels": bool(has_human),
                "judge_mode": self.cfg.judge_mode if (
                    self.cfg.use_judge if self.cfg.use_judge is not None else self.cfg.judge_enabled) else None,
            }
        }

        os.makedirs(os.path.dirname(self.cfg.output_json), exist_ok=True)
        with open(self.cfg.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # stash in state
        meta = state.get("meta", {})
        meta.update({
            "eval_table_path": self.cfg.output_tsv,
            "eval_summary_path": self.cfg.output_json,
        })
        state["meta"] = meta
        state["eval"] = summary
        return state
