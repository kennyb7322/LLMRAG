"""
STREAMING MODULE
=================
Server-Sent Events (SSE) streaming for real-time token-by-token output.
Supports all LLM providers with unified streaming interface.

Features:
  - SSE endpoint for chat streaming
  - Token-by-token delivery
  - Streaming with citations
  - Multi-provider streaming abstraction
  - WebSocket support (optional)
  - Backpressure handling
"""

import os
import json
import asyncio
from typing import AsyncGenerator, Dict, Any, Optional
from dataclasses import dataclass, field

from src.utils.logger import log


@dataclass
class StreamChunk:
    """A single chunk in a streaming response."""
    token: str = ""
    done: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_type: str = "token"  # token | citation | source | status | error


class StreamingEngine:
    """Unified streaming interface across LLM providers."""

    def __init__(self, config: dict):
        self.config = config
        self.provider = config.get("llm", {}).get("provider", "openai")

    async def stream_generate(
        self,
        prompt: Dict[str, str],
        model: str = "",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Stream LLM response token by token."""
        if self.provider == "openai":
            async for chunk in self._stream_openai(prompt, model, temperature, max_tokens):
                yield chunk
        elif self.provider == "anthropic":
            async for chunk in self._stream_anthropic(prompt, model, temperature, max_tokens):
                yield chunk
        elif self.provider == "ollama":
            async for chunk in self._stream_ollama(prompt, model, temperature, max_tokens):
                yield chunk
        else:
            yield StreamChunk(token="Streaming not supported for this provider", done=True)

    async def _stream_openai(self, prompt, model, temperature, max_tokens):
        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
            stream = await client.chat.completions.create(
                model=model or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": prompt.get("system", "")},
                    {"role": "user", "content": prompt.get("user", "")},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield StreamChunk(token=delta.content)
                if chunk.choices[0].finish_reason:
                    yield StreamChunk(done=True, metadata={"finish": chunk.choices[0].finish_reason})
        except Exception as e:
            yield StreamChunk(token=f"Error: {e}", done=True, chunk_type="error")

    async def _stream_anthropic(self, prompt, model, temperature, max_tokens):
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
            async with client.messages.stream(
                model=model or "claude-sonnet-4-20250514",
                max_tokens=max_tokens,
                system=prompt.get("system", ""),
                messages=[{"role": "user", "content": prompt.get("user", "")}],
            ) as stream:
                async for text in stream.text_stream:
                    yield StreamChunk(token=text)
            yield StreamChunk(done=True)
        except Exception as e:
            yield StreamChunk(token=f"Error: {e}", done=True, chunk_type="error")

    async def _stream_ollama(self, prompt, model, temperature, max_tokens):
        try:
            import aiohttp
            ollama_cfg = self.config.get("llm", {}).get("ollama", {})
            base_url = ollama_cfg.get("base_url", "http://localhost:11434")
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{base_url}/api/chat",
                    json={
                        "model": model or ollama_cfg.get("model", "llama3"),
                        "messages": [
                            {"role": "system", "content": prompt.get("system", "")},
                            {"role": "user", "content": prompt.get("user", "")},
                        ],
                        "stream": True,
                    },
                ) as resp:
                    async for line in resp.content:
                        if line:
                            data = json.loads(line)
                            token = data.get("message", {}).get("content", "")
                            if token:
                                yield StreamChunk(token=token)
                            if data.get("done"):
                                yield StreamChunk(done=True)
        except Exception as e:
            yield StreamChunk(token=f"Error: {e}", done=True, chunk_type="error")


def create_sse_response(chunk: StreamChunk) -> str:
    """Format a StreamChunk as an SSE event string."""
    data = {
        "token": chunk.token,
        "done": chunk.done,
        "type": chunk.chunk_type,
    }
    if chunk.metadata:
        data["metadata"] = chunk.metadata
    return f"data: {json.dumps(data)}\n\n"
