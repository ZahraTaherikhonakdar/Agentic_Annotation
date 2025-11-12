
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import os
from agents.base import PipelineState
from utils import RunConfig
from agents.load_queries import LoadQueriesAgent
from agents.annotate import LLMAnnotateAgent, AnnotateConfig
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv, find_dotenv
from agents.validate import ValidateAgent, ValidationConfig
from agents.evaluate import EvaluateAgent, EvalConfig
# Load .env (ensure OPENAI_API_KEY is defined there)
load_dotenv(find_dotenv(), override=True)

def node_load_queries(state: PipelineState, cfg: RunConfig) -> PipelineState:
    agent = LoadQueriesAgent(
        config=cfg,
        output_tsv=cfg.output_checkpoint,   # TSV checkpoint path
        schema_path=None,                   # optional strict schema validation
    )
    return agent.run(state)


def node_annotate(state: PipelineState, run_cfg: RunConfig, ann_cfg: AnnotateConfig) -> PipelineState:
    agent = LLMAnnotateAgent(run_cfg, ann_cfg)
    return agent.run(state)

def build_graph(cfg: RunConfig):
    graph = StateGraph(PipelineState)
    graph.add_node("load_queries", lambda s: node_load_queries(s, cfg))

    # Use OpenAI GPT-4.1-mini explicitly
    ann_cfg = AnnotateConfig(
        mode="zeroshot", #fewshot
        fewshot_path="artifacts/examples_fewshot_v1.jsonl",
        self_consistency_votes=1,
        temperature=0.0,
        max_new_tokens=120,

        # backend selection
        provider="openai",               # custom provider tag (you can handle this in annotate.py)
        api_base="https://api.openai.com/v1",
        model_name="gpt-4.1",

        # I/O + schema
        output_tsv="results/checkpoints/node2_annotated.tsv",
        output_schema_path="artifacts/schema/output_label_schema.json",

        # meta
        prompt_version=cfg.prompt_version,
        taxonomy_version=cfg.taxonomy_version,
    )

    graph.add_node("annotate", lambda s: node_annotate(s, cfg, ann_cfg))
    graph.add_node("validate", lambda s: ValidateAgent(ValidationConfig(
        conf_low=0.60, max_retries=1  # tune as you like
    )).run(s))

    graph.add_node("evaluate", lambda s: EvaluateAgent(EvalConfig(
        output_tsv="results/eval/metrics_by_class.tsv",
        output_json="results/eval/summary.json",
        use_prf=True,  # no human labels → skip P/R/F1
        use_kappa=True,  # no human labels → skip κ vs human
        use_judge=True,  # run judge
        judge_mode="compare_to_human",  # judge provides its own label; no gold needed -- "compare_to_human", # judge checks AI vs human gold
        judge_model="gpt-4.1-mini",  # use a different model than the annotator
        judge_sample_size=500,
    )).run(s))

    graph.add_edge(START, "load_queries")
    graph.add_edge("load_queries", "annotate")
    graph.add_edge("annotate", "validate")
    #graph.add_edge(START, "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"annotate": "annotate", "evaluate": "evaluate"}
    )
    graph.add_edge("evaluate", END)

    # inside class LLMAnnotateAgent(BaseAgent)


    return graph.compile()

def route_after_validate(state: PipelineState) -> str:
    for r in state.get("records", []):
        if r.get("needs_retry"):
            return "annotate"   # loop back to the SAME annotate node
    return "evaluate"                # or "judge_eval" if you have that node

if __name__ == "__main__":
    cfg = RunConfig(
        dataset_path=r"D:\RePair\data\preprocessed\orcas\original data\orcas.tsv",
        split="validation",  # "train" | "validation" | "test"
        lowercase=True,
        strip_punct=True,
        keep_cols=["qid", "query", "level_1", "level_2", "label", "data_split", "did", "url"],
        sample_n=5000, # 467smaller sample for faster test, None: run on the full 2 M dataset
        output_dir="results",
        output_checkpoint="results/checkpoints/node1_loaded.tsv",
        prompt_version="v1",
        taxonomy_version="v1",
        seed=42,
    )

    app = build_graph(cfg)
    final_state: PipelineState = app.invoke(PipelineState())

    print(f"[Node1] Loaded records: {final_state.get('meta', {}).get('n_records', 0)}")
    print(f"[Node1] Checkpoint: {final_state.get('meta', {}).get('checkpoint_path')}")

    if final_state.get("records"):
        print(f"[Node1] First record: {final_state['records'][0]}")

    # Show annotated results
    anns = final_state.get("annotations", [])
    if not anns:
        print("\nNo annotations found.")
    else:
        print(f"\nShowing first {min(5, len(anns))} annotated queries:\n")
        for i, ann in enumerate(anns[:5], 1):
            print(f"{i}. QID: {ann.get('qid')}")
            print(f"   Query: {ann.get('query')}")
            print(f"   → AI Label: {ann.get('ai_label')}  (Conf: {ann.get('ai_confidence'):.2f})")
