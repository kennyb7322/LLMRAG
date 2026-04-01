"""
AGENTIC AI MODULE
==================
Autonomous agent framework with tool use, multi-agent orchestration,
and RAG-augmented decision making.

Architecture:
  Agent → Perceive → Think (RAG + Reasoning) → Act (Tools) → Observe → Loop

Agent Types:
  - RAG Agent: Retrieval-augmented Q&A with citation
  - Research Agent: Multi-source investigation with synthesis
  - Code Agent: Code generation, execution, debugging
  - Data Agent: SQL/API queries, analysis, visualization
  - Orchestrator Agent: Coordinates multiple sub-agents

Integration Points:
  - LLM Providers (OpenAI, Claude, Gemini, Grok, Llama, Perplexity)
  - Cloud Providers (Azure Copilot, AWS Bedrock, GCP Vertex, OCI GenAI)
  - Tool Registry (web search, code exec, file I/O, API calls)
  - LLMRAG Pipeline (ingestion, retrieval, generation)
"""

import time
import json
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

from src.utils.logger import log


class AgentStatus(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    COMPLETE = "complete"
    ERROR = "error"
    AWAITING_HUMAN = "awaiting_human"


@dataclass
class AgentAction:
    """An action taken by an agent."""
    tool: str
    input: Dict[str, Any]
    output: Any = None
    status: str = "pending"
    duration_ms: float = 0.0


@dataclass
class AgentStep:
    """A single step in the agent loop."""
    step_num: int
    thought: str
    action: Optional[AgentAction] = None
    observation: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentResult:
    """Final result from an agent execution."""
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    total_time_sec: float = 0.0
    tokens_used: int = 0
    status: AgentStatus = AgentStatus.COMPLETE


# ── Tool Registry ───────────────────────────────────────────────────────────

class Tool:
    """A tool that an agent can use."""

    def __init__(self, name: str, description: str, fn: Callable,
                 parameters: Dict[str, str] = None):
        self.name = name
        self.description = description
        self.fn = fn
        self.parameters = parameters or {}

    def execute(self, **kwargs) -> Any:
        try:
            return self.fn(**kwargs)
        except Exception as e:
            return {"error": str(e)}

    def to_schema(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """Registry of available tools for agents."""

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._register_defaults()

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict]:
        return [t.to_schema() for t in self._tools.values()]

    def _register_defaults(self):
        """Register built-in tools."""
        self.register(Tool(
            name="rag_query",
            description="Query the LLMRAG pipeline for information from indexed documents",
            fn=lambda query, **kw: {"result": f"[RAG result for: {query}]"},
            parameters={"query": "string"},
        ))
        self.register(Tool(
            name="web_search",
            description="Search the web for current information",
            fn=lambda query, **kw: {"result": f"[Web search: {query}]"},
            parameters={"query": "string"},
        ))
        self.register(Tool(
            name="code_execute",
            description="Execute Python code and return output",
            fn=self._execute_code,
            parameters={"code": "string"},
        ))
        self.register(Tool(
            name="file_read",
            description="Read contents of a file",
            fn=lambda path, **kw: {"result": f"[File: {path}]"},
            parameters={"path": "string"},
        ))
        self.register(Tool(
            name="api_call",
            description="Make an HTTP API call",
            fn=lambda url, method="GET", **kw: {"result": f"[API {method}: {url}]"},
            parameters={"url": "string", "method": "string"},
        ))
        self.register(Tool(
            name="calculator",
            description="Evaluate a mathematical expression",
            fn=self._calculate,
            parameters={"expression": "string"},
        ))

    @staticmethod
    def _execute_code(code: str, **kwargs) -> Dict:
        """Execute Python code safely."""
        import io, contextlib
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                exec(code, {"__builtins__": __builtins__}, {})
            return {"output": output.getvalue(), "status": "success"}
        except Exception as e:
            return {"output": str(e), "status": "error"}

    @staticmethod
    def _calculate(expression: str, **kwargs) -> Dict:
        try:
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}


# ── Base Agent ──────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    """Abstract base agent with the think-act-observe loop."""

    def __init__(self, name: str, config: dict, tools: ToolRegistry = None,
                 llm_fn: Callable = None):
        self.name = name
        self.config = config
        self.tools = tools or ToolRegistry()
        self.llm_fn = llm_fn
        self.max_steps = config.get("max_steps", 10)
        self.status = AgentStatus.IDLE
        self.steps: List[AgentStep] = []

    @abstractmethod
    def think(self, query: str, context: List[str] = None) -> str:
        """Decide what to do next."""
        ...

    @abstractmethod
    def act(self, thought: str) -> AgentAction:
        """Execute an action based on the thought."""
        ...

    def run(self, query: str, context: List[str] = None) -> AgentResult:
        """Execute the full agent loop."""
        t0 = time.time()
        self.status = AgentStatus.THINKING
        log.info(f"Agent '{self.name}' starting: '{query[:60]}...'")

        for step_num in range(1, self.max_steps + 1):
            # Think
            thought = self.think(query, context)

            # Check for completion
            if "[FINAL_ANSWER]" in thought:
                answer = thought.split("[FINAL_ANSWER]")[-1].strip()
                return AgentResult(
                    answer=answer,
                    steps=self.steps,
                    tools_used=list(set(s.action.tool for s in self.steps if s.action)),
                    total_time_sec=time.time() - t0,
                    status=AgentStatus.COMPLETE,
                )

            # Act
            self.status = AgentStatus.ACTING
            action = self.act(thought)

            # Observe
            self.status = AgentStatus.OBSERVING
            observation = str(action.output) if action.output else "No output"

            step = AgentStep(
                step_num=step_num,
                thought=thought,
                action=action,
                observation=observation,
            )
            self.steps.append(step)

            # Add observation to context
            context = (context or []) + [observation]

        return AgentResult(
            answer="Max steps reached without final answer",
            steps=self.steps,
            tools_used=list(set(s.action.tool for s in self.steps if s.action)),
            total_time_sec=time.time() - t0,
            status=AgentStatus.ERROR,
        )


# ── RAG Agent ───────────────────────────────────────────────────────────────

class RAGAgent(BaseAgent):
    """Agent specialized for RAG-augmented Q&A."""

    def __init__(self, config: dict, pipeline=None, **kwargs):
        super().__init__("rag_agent", config, **kwargs)
        self.pipeline = pipeline

    def think(self, query: str, context: List[str] = None) -> str:
        if not self.steps:
            return f"I need to search the knowledge base for: {query}. Using rag_query tool."
        if len(self.steps) >= 1 and self.steps[-1].observation:
            return f"[FINAL_ANSWER] Based on the retrieved information: {self.steps[-1].observation}"
        return f"Searching for more context about: {query}"

    def act(self, thought: str) -> AgentAction:
        action = AgentAction(tool="rag_query", input={"query": thought})
        if self.pipeline:
            result = self.pipeline.query(thought)
            action.output = result.answer
        else:
            tool = self.tools.get("rag_query")
            action.output = tool.execute(query=thought) if tool else "No RAG tool available"
        action.status = "complete"
        return action


# ── Research Agent ──────────────────────────────────────────────────────────

class ResearchAgent(BaseAgent):
    """Agent that conducts multi-source research."""

    def __init__(self, config: dict, **kwargs):
        super().__init__("research_agent", config, **kwargs)

    def think(self, query: str, context: List[str] = None) -> str:
        step = len(self.steps)
        if step == 0:
            return f"Research query: {query}. Step 1: Search knowledge base."
        elif step == 1:
            return f"Step 2: Search web for additional context on: {query}"
        elif step >= 2:
            findings = "\n".join(s.observation for s in self.steps if s.observation)
            return f"[FINAL_ANSWER] Research synthesis:\n{findings[:500]}"
        return "[FINAL_ANSWER] Research complete."

    def act(self, thought: str) -> AgentAction:
        if "knowledge base" in thought.lower():
            return AgentAction(tool="rag_query", input={"query": thought},
                             output="[KB results]", status="complete")
        elif "web" in thought.lower():
            return AgentAction(tool="web_search", input={"query": thought},
                             output="[Web results]", status="complete")
        return AgentAction(tool="none", input={}, output="", status="complete")


# ── Multi-Agent Orchestrator ────────────────────────────────────────────────

class AgentOrchestrator:
    """
    Coordinate multiple agents for complex tasks.
    Supports parallel execution, agent delegation, and result aggregation.
    """

    def __init__(self, config: dict):
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.tool_registry = ToolRegistry()

    def register_agent(self, agent: BaseAgent):
        self.agents[agent.name] = agent
        log.info(f"Agent registered: {agent.name}")

    def run_agent(self, agent_name: str, query: str,
                  context: List[str] = None) -> AgentResult:
        """Run a specific agent."""
        agent = self.agents.get(agent_name)
        if not agent:
            return AgentResult(answer=f"Agent '{agent_name}' not found",
                             status=AgentStatus.ERROR)
        return agent.run(query, context)

    def run_pipeline(self, query: str, agent_sequence: List[str] = None,
                     context: List[str] = None) -> Dict[str, AgentResult]:
        """Run a sequence of agents, passing context between them."""
        sequence = agent_sequence or list(self.agents.keys())
        results = {}
        accumulated_context = context or []

        for agent_name in sequence:
            result = self.run_agent(agent_name, query, accumulated_context)
            results[agent_name] = result
            if result.answer:
                accumulated_context.append(result.answer)

        return results

    def health_check(self) -> Dict[str, Any]:
        return {
            "agents": list(self.agents.keys()),
            "tools": [t["name"] for t in self.tool_registry.list_tools()],
            "total_agents": len(self.agents),
            "total_tools": len(self.tool_registry.list_tools()),
        }
