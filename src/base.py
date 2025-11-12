from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict


class PipelineState(TypedDict, total=False):
    """
    Global, shared state passed between all nodes.

    INPUT/DATA
      - records: normalized rows loaded by Node 1
      - gold_labels: optional qid -> human label map

    INTERMEDIATE ARTIFACTS
      - annotations: outputs from Node 2 (LLMAnnotate)
      - judged: outputs from Judge node (if/when added)
      - errors: pipeline warnings/errors (strings)

    METADATA
      - meta: counts, paths, timestamps, misc run info
      - config: snapshot of run config for reproducibility
    """
  
    records: List[Dict[str, Any]]
    gold_labels: Dict[str, str]

    annotations: List[Dict[str, Any]]
    judged: List[Dict[str, Any]]
    errors: List[str]

    # METADATA
    meta: Dict[str, Any]
    config: Dict[str, Any]


@dataclass
class AgentContext:
    """Room for shared clients, caches, etc."""
    pass


class BaseAgent:
    def run(self, state: PipelineState, **kwargs) -> PipelineState:
        raise NotImplementedError
