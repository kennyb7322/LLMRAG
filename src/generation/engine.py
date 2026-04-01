"""
GENERATION + OUTPUT MODULE
===========================
Context assembly → prompt construction → LLM call → structured output.
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from src.retrieval.engine import RetrievalResult
from src.utils.logger import log


@dataclass
class GenerationOutput:
    """Final output from the generation pipeline."""
    answer: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_prompt: str = ""
    tokens_used: int = 0


class ContextAssembler:
    """Assemble retrieved chunks into a context window for the LLM."""

    def __init__(self, config: dict):
        self.max_tokens = config.get("max_context_tokens", 4000)
        self.ordering = config.get("ordering", "relevance")
        self.include_citations = config.get("include_citations", True)

    def assemble(self, results: List[RetrievalResult]) -> tuple:
        """
        Assemble context string and citation mappings.
        Returns (context_string, citations_list).
        """
        if self.ordering == "relevance":
            results.sort(key=lambda x: x.score, reverse=True)
        elif self.ordering == "chronological":
            results.sort(key=lambda x: x.metadata.get("modified", ""))

        context_parts = []
        citations = []
        total_chars = 0
        char_limit = self.max_tokens * 4  # Rough token-to-char estimate

        for i, result in enumerate(results):
            if total_chars + len(result.content) > char_limit:
                break
            
            citation_ref = f"[{i + 1}]"
            context_parts.append(f"{citation_ref} {result.content}")
            citations.append({
                "ref": citation_ref,
                "source": result.metadata.get("filename", result.source or "unknown"),
                "chunk_id": result.chunk_id,
                "score": round(result.score, 4),
            })
            total_chars += len(result.content)

        context = "\n\n".join(context_parts)
        log.info(f"Context assembled: {len(context_parts)} chunks, ~{total_chars} chars")
        return context, citations


class PromptBuilder:
    """Construct the final prompt for the LLM."""

    DEFAULT_TEMPLATE = """You are a helpful AI assistant. Answer the question based ONLY on the provided context.
If the context does not contain sufficient information to answer, clearly state that.
Include citation references (e.g., [1], [2]) when referencing specific information.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, template: str = None, guardrails: dict = None):
        self.template = template or self.DEFAULT_TEMPLATE
        self.guardrails = guardrails or {}

    def build(self, question: str, context: str,
              system_prompt: str = None) -> Dict[str, Any]:
        """Build the prompt payload for the LLM."""
        user_prompt = self.template.format(context=context, question=question)

        system = system_prompt or (
            "You are a precise, helpful AI assistant. "
            "Ground your answers in the provided context. "
            "Cite sources using reference numbers. "
            "If uncertain, express your confidence level."
        )

        return {
            "system": system,
            "user": user_prompt,
        }


class GenerationEngine:
    """LLM interaction layer with multi-provider support."""

    def __init__(self, config: dict):
        self.config = config
        self.llm_cfg = config.get("llm", {})
        self.provider = self.llm_cfg.get("provider", "openai")
        self.model = self.llm_cfg.get("model", "gpt-4o-mini")
        self.temperature = self.llm_cfg.get("temperature", 0.1)
        self.max_tokens = self.llm_cfg.get("max_tokens", 4096)

        self.context_assembler = ContextAssembler(
            config.get("context_assembly", {})
        )
        self.prompt_builder = PromptBuilder(
            template=config.get("prompt_template"),
            guardrails=config.get("guardrails", {}),
        )

    def generate(self, query: str,
                 retrieval_results: List[RetrievalResult]) -> GenerationOutput:
        """Full generation pipeline: assemble → prompt → LLM → output."""
        log.info(f"═══ GENERATION: '{query[:60]}...' ═══")

        # 1. Assemble context
        context, citations = self.context_assembler.assemble(retrieval_results)

        # 2. Build prompt
        prompt = self.prompt_builder.build(query, context)

        # 3. Call LLM
        answer, tokens = self._call_llm(prompt)

        # 4. Build output
        output = GenerationOutput(
            answer=answer,
            citations=citations,
            sources=list(set(c["source"] for c in citations)),
            confidence=self._estimate_confidence(answer, retrieval_results),
            raw_prompt=prompt["user"][:500],
            tokens_used=tokens,
        )

        # 5. Apply guardrails
        output = self._apply_guardrails(output)

        log.info(f"Generation complete | Confidence: {output.confidence:.2f}")
        return output

    def _call_llm(self, prompt: Dict[str, str]) -> tuple:
        """Route to the correct LLM provider."""
        if self.provider == "openai":
            return self._call_openai(prompt)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt)
        elif self.provider == "ollama":
            return self._call_ollama(prompt)
        elif self.provider == "local":
            return self._call_local(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_openai(self, prompt: dict) -> tuple:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content
            tokens = response.usage.total_tokens if response.usage else 0
            return answer, tokens
        except ImportError:
            log.error("openai not installed — pip install openai")
            return "Error: OpenAI SDK not installed.", 0

    def _call_anthropic(self, prompt: dict) -> tuple:
        try:
            import anthropic
            client = anthropic.Anthropic(
                api_key=os.environ.get("ANTHROPIC_API_KEY", "")
            )
            anth_model = self.llm_cfg.get("anthropic", {}).get(
                "model", "claude-sonnet-4-20250514"
            )
            response = client.messages.create(
                model=anth_model,
                max_tokens=self.max_tokens,
                system=prompt["system"],
                messages=[{"role": "user", "content": prompt["user"]}],
            )
            answer = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return answer, tokens
        except ImportError:
            log.error("anthropic not installed — pip install anthropic")
            return "Error: Anthropic SDK not installed.", 0

    def _call_ollama(self, prompt: dict) -> tuple:
        try:
            import requests
            ollama_cfg = self.llm_cfg.get("ollama", {})
            base_url = ollama_cfg.get("base_url", "http://localhost:11434")
            model = ollama_cfg.get("model", "llama3")
            
            resp = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]},
                    ],
                    "stream": False,
                },
                timeout=120,
            )
            data = resp.json()
            answer = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)
            return answer, tokens
        except Exception as e:
            log.error(f"Ollama error: {e}")
            return f"Error calling Ollama: {e}", 0

    def _call_local(self, prompt: dict) -> tuple:
        """Call a locally hosted model via llama-cpp-python."""
        try:
            from llama_cpp import Llama
            local_cfg = self.llm_cfg.get("local", {})
            model_path = local_cfg.get("model_path", "./models/llm")
            ctx_len = local_cfg.get("context_length", 8192)
            gpu_layers = local_cfg.get("gpu_layers", -1)

            llm = Llama(
                model_path=model_path,
                n_ctx=ctx_len,
                n_gpu_layers=gpu_layers,
            )
            full_prompt = f"{prompt['system']}\n\nUser: {prompt['user']}\n\nAssistant:"
            output = llm(full_prompt, max_tokens=self.max_tokens, temperature=self.temperature)
            answer = output["choices"][0]["text"]
            tokens = output.get("usage", {}).get("total_tokens", 0)
            return answer, tokens
        except ImportError:
            log.error("llama-cpp-python not installed")
            return "Error: llama-cpp-python not installed.", 0

    @staticmethod
    def _estimate_confidence(answer: str,
                             results: List[RetrievalResult]) -> float:
        """Estimate answer confidence based on retrieval quality."""
        if not results:
            return 0.0
        avg_score = sum(r.score for r in results) / len(results)
        has_citations = any(f"[{i}]" in answer for i in range(1, len(results) + 1))
        low_confidence_phrases = [
            "i don't know", "i'm not sure", "cannot determine",
            "no information", "not enough context", "unclear",
        ]
        mentions_uncertainty = any(p in answer.lower() for p in low_confidence_phrases)
        
        confidence = avg_score
        if has_citations:
            confidence = min(1.0, confidence + 0.1)
        if mentions_uncertainty:
            confidence = max(0.0, confidence - 0.3)
        return round(confidence, 3)

    def _apply_guardrails(self, output: GenerationOutput) -> GenerationOutput:
        """Apply safety guardrails to the output."""
        guardrails = self.config.get("guardrails", {})
        
        threshold = guardrails.get("confidence_threshold", 0.3)
        if guardrails.get("refuse_low_confidence") and output.confidence < threshold:
            output.answer = (
                f"⚠ Low confidence ({output.confidence:.0%}). "
                f"The retrieved context may not sufficiently answer this question.\n\n"
                f"Original response: {output.answer}"
            )
        
        return output
