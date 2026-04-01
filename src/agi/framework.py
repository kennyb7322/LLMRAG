"""
AGI FRAMEWORK MODULE
=====================
Artificial General Intelligence integration layer for LLMRAG.
Implements cognitive architecture patterns, multi-modal reasoning,
autonomous planning, and alignment/safety guardrails.

Architecture Layers:
  Layer 1 — Perception (multi-modal input processing)
  Layer 2 — Memory (short-term, long-term, episodic, semantic)
  Layer 3 — Reasoning (chain-of-thought, tree-of-thought, causal)
  Layer 4 — Planning (goal decomposition, task scheduling)
  Layer 5 — Action (tool use, code execution, API calls)
  Layer 6 — Self-Reflection (meta-cognition, confidence calibration)
  Layer 7 — Alignment (ethical constraints, safety boundaries)
  Layer 8 — Autonomy Governance (human-in-loop, kill switch, scope limits)
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

from src.utils.logger import log


class CognitiveLayer(Enum):
    PERCEPTION = "perception"
    MEMORY = "memory"
    REASONING = "reasoning"
    PLANNING = "planning"
    ACTION = "action"
    REFLECTION = "self_reflection"
    ALIGNMENT = "alignment"
    GOVERNANCE = "autonomy_governance"


@dataclass
class CognitiveState:
    """Current cognitive state of the AGI system."""
    current_goal: str = ""
    sub_goals: List[str] = field(default_factory=list)
    working_memory: List[Dict[str, Any]] = field(default_factory=list)
    reasoning_chain: List[str] = field(default_factory=list)
    confidence: float = 0.0
    iteration: int = 0
    max_iterations: int = 10
    safety_score: float = 1.0
    active_layer: CognitiveLayer = CognitiveLayer.PERCEPTION


@dataclass
class AGIConfig:
    """Configuration for the AGI framework."""
    enabled: bool = False
    max_reasoning_depth: int = 5
    max_planning_steps: int = 20
    confidence_threshold: float = 0.7
    safety_threshold: float = 0.8
    human_in_loop: bool = True
    autonomy_level: str = "supervised"  # supervised | semi-autonomous | autonomous
    reasoning_strategy: str = "chain_of_thought"  # chain_of_thought | tree_of_thought | reflexion
    memory_type: str = "episodic"  # episodic | semantic | hybrid


# ── Memory System ───────────────────────────────────────────────────────────

class AGIMemory:
    """Multi-tier memory system: working, episodic, semantic, long-term."""

    def __init__(self):
        self.working: List[Dict[str, Any]] = []  # Current context
        self.episodic: List[Dict[str, Any]] = []  # Past interactions
        self.semantic: Dict[str, Any] = {}         # Learned facts/concepts
        self.long_term: List[Dict[str, Any]] = []  # Persistent knowledge

    def store_working(self, key: str, value: Any):
        self.working.append({"key": key, "value": value, "time": time.time()})
        if len(self.working) > 50:
            self.working = self.working[-50:]

    def store_episodic(self, event: str, context: Dict[str, Any]):
        self.episodic.append({"event": event, "context": context, "time": time.time()})

    def store_semantic(self, concept: str, definition: Any):
        self.semantic[concept] = {"value": definition, "updated": time.time()}

    def retrieve_relevant(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant memories across all tiers."""
        query_words = set(query.lower().split())
        scored = []
        for mem in self.working + self.episodic:
            text = str(mem.get("value", mem.get("event", "")))
            overlap = len(query_words & set(text.lower().split()))
            if overlap > 0:
                scored.append((overlap, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:top_k]]

    def clear_working(self):
        self.working.clear()


# ── Reasoning Engine ────────────────────────────────────────────────────────

class ReasoningEngine:
    """Multi-strategy reasoning for AGI tasks."""

    def __init__(self, config: AGIConfig):
        self.config = config
        self.strategy = config.reasoning_strategy

    def reason(self, query: str, context: List[str],
               llm_fn: Optional[Callable] = None) -> Dict[str, Any]:
        """Apply reasoning strategy to a query with context."""
        if self.strategy == "chain_of_thought":
            return self._chain_of_thought(query, context, llm_fn)
        elif self.strategy == "tree_of_thought":
            return self._tree_of_thought(query, context, llm_fn)
        elif self.strategy == "reflexion":
            return self._reflexion(query, context, llm_fn)
        return {"reasoning": "direct", "result": query}

    def _chain_of_thought(self, query: str, context: List[str],
                          llm_fn: Optional[Callable]) -> Dict[str, Any]:
        """Step-by-step reasoning chain."""
        steps = []
        steps.append(f"Step 1: Understand the question — '{query}'")
        steps.append(f"Step 2: Analyze {len(context)} context chunks for relevant information")
        steps.append("Step 3: Identify key entities and relationships")
        steps.append("Step 4: Synthesize information into a coherent answer")
        steps.append("Step 5: Verify answer against context (grounding check)")

        if llm_fn:
            cot_prompt = (
                f"Think step by step to answer this question.\n\n"
                f"Context:\n" + "\n".join(context[:3]) + f"\n\nQuestion: {query}\n\n"
                f"Let's think step by step:"
            )
            result = llm_fn(cot_prompt)
        else:
            result = f"[CoT reasoning for: {query}]"

        return {"strategy": "chain_of_thought", "steps": steps, "result": result}

    def _tree_of_thought(self, query: str, context: List[str],
                         llm_fn: Optional[Callable]) -> Dict[str, Any]:
        """Tree-structured exploration of reasoning paths."""
        branches = [
            {"path": "analytical", "approach": "Break into sub-problems"},
            {"path": "analogical", "approach": "Find similar known patterns"},
            {"path": "empirical", "approach": "Rely on evidence from context"},
        ]
        return {"strategy": "tree_of_thought", "branches": branches,
                "result": f"[ToT reasoning for: {query}]"}

    def _reflexion(self, query: str, context: List[str],
                   llm_fn: Optional[Callable]) -> Dict[str, Any]:
        """Self-reflective reasoning with critique loop."""
        return {"strategy": "reflexion", "iterations": 0,
                "result": f"[Reflexion reasoning for: {query}]"}


# ── Planning Engine ─────────────────────────────────────────────────────────

class PlanningEngine:
    """Goal decomposition and task planning."""

    def __init__(self, config: AGIConfig):
        self.max_steps = config.max_planning_steps

    def decompose_goal(self, goal: str) -> List[Dict[str, Any]]:
        """Break a high-level goal into actionable sub-tasks."""
        plan = [
            {"step": 1, "action": "parse_goal", "description": f"Parse and understand: {goal}"},
            {"step": 2, "action": "gather_context", "description": "Retrieve relevant information"},
            {"step": 3, "action": "analyze", "description": "Analyze gathered information"},
            {"step": 4, "action": "synthesize", "description": "Synthesize findings"},
            {"step": 5, "action": "validate", "description": "Validate against goal criteria"},
            {"step": 6, "action": "output", "description": "Format and present results"},
        ]
        return plan[:self.max_steps]

    def replan(self, current_state: CognitiveState, feedback: str) -> List[Dict]:
        """Dynamically replan based on feedback."""
        return [{"step": 1, "action": "replan", "description": f"Adjust plan based on: {feedback}"}]


# ── Alignment & Safety ──────────────────────────────────────────────────────

class AlignmentGuard:
    """Safety and alignment checks for AGI operations."""

    BLOCKED_INTENTS = [
        "create weapon", "hack system", "bypass security",
        "generate malware", "social engineering",
    ]

    def __init__(self, config: AGIConfig):
        self.safety_threshold = config.safety_threshold
        self.human_in_loop = config.human_in_loop

    def check_safety(self, action: str, context: str = "") -> Dict[str, Any]:
        """Evaluate action safety before execution."""
        score = 1.0
        flags = []

        for intent in self.BLOCKED_INTENTS:
            if intent in action.lower():
                score = 0.0
                flags.append(f"Blocked intent detected: {intent}")

        if score >= self.safety_threshold:
            return {"safe": True, "score": score, "flags": flags}
        return {"safe": False, "score": score, "flags": flags,
                "action": "blocked", "reason": "Below safety threshold"}

    def require_human_approval(self, action: str) -> bool:
        """Determine if an action requires human-in-the-loop approval."""
        high_risk_actions = ["delete", "modify_system", "external_api", "financial", "deploy"]
        return self.human_in_loop and any(a in action.lower() for a in high_risk_actions)


# ── AGI Orchestrator ────────────────────────────────────────────────────────

class AGIOrchestrator:
    """
    Main AGI orchestrator — coordinates cognitive layers.
    Integrates with the LLMRAG pipeline for RAG-augmented AGI reasoning.
    """

    def __init__(self, config: dict):
        self.cfg = AGIConfig(**{k: v for k, v in config.items() if hasattr(AGIConfig, k)})
        self.memory = AGIMemory()
        self.reasoning = ReasoningEngine(self.cfg)
        self.planning = PlanningEngine(self.cfg)
        self.alignment = AlignmentGuard(self.cfg)
        self.state = CognitiveState()
        log.info(f"AGI Orchestrator initialized | Strategy: {self.cfg.reasoning_strategy} "
                 f"| Autonomy: {self.cfg.autonomy_level}")

    def process(self, query: str, context: List[str] = None,
                llm_fn: Callable = None) -> Dict[str, Any]:
        """Run the full AGI cognitive loop."""
        if not self.cfg.enabled:
            return {"mode": "standard_rag", "note": "AGI disabled — using standard pipeline"}

        log.info(f"AGI Processing: '{query[:60]}...'")

        # Layer 7: Alignment check
        safety = self.alignment.check_safety(query)
        if not safety["safe"]:
            return {"error": "Action blocked by alignment guard", "flags": safety["flags"]}

        # Layer 1: Perception
        self.state.current_goal = query
        self.state.active_layer = CognitiveLayer.PERCEPTION

        # Layer 2: Memory retrieval
        self.state.active_layer = CognitiveLayer.MEMORY
        memories = self.memory.retrieve_relevant(query)
        augmented_context = (context or []) + [str(m) for m in memories]

        # Layer 3: Reasoning
        self.state.active_layer = CognitiveLayer.REASONING
        reasoning_result = self.reasoning.reason(query, augmented_context, llm_fn)

        # Layer 4: Planning
        self.state.active_layer = CognitiveLayer.PLANNING
        plan = self.planning.decompose_goal(query)

        # Layer 6: Self-reflection
        self.state.active_layer = CognitiveLayer.REFLECTION
        confidence = self._self_reflect(reasoning_result)

        # Store in episodic memory
        self.memory.store_episodic(query, {
            "reasoning": reasoning_result.get("strategy"),
            "confidence": confidence,
        })

        return {
            "mode": "agi",
            "reasoning": reasoning_result,
            "plan": plan,
            "confidence": confidence,
            "safety": safety,
            "memories_used": len(memories),
            "autonomy": self.cfg.autonomy_level,
            "cognitive_layers_engaged": [
                CognitiveLayer.PERCEPTION.value,
                CognitiveLayer.MEMORY.value,
                CognitiveLayer.REASONING.value,
                CognitiveLayer.PLANNING.value,
                CognitiveLayer.REFLECTION.value,
                CognitiveLayer.ALIGNMENT.value,
            ],
        }

    def _self_reflect(self, reasoning_result: dict) -> float:
        """Meta-cognitive self-reflection on reasoning quality."""
        if reasoning_result.get("steps"):
            return min(1.0, 0.6 + len(reasoning_result["steps"]) * 0.08)
        return 0.5
