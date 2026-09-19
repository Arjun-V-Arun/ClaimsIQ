"""Groq-backed tool-calling agent for ClaimsIQ."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .tools import DATASET_SOURCE, ClaimsTools

SYSTEM_PROMPT = """
You are ClaimsIQ, an internal insurance analytics copilot working over a
public French motor third-party liability dataset and a small demonstration
policy corpus.

Rules:
1. Use tools for dataset facts and policy questions. Never invent numbers.
2. Never write SQL and never request arbitrary database access.
3. Anomaly scores are statistical review signals, not proof of fraud.
4. Policy questions must be grounded in retrieved policy text. Mention the
   retrieved document title when relevant.
5. Clearly distinguish observed dataset facts from interpretation.
6. The underlying insurance dataset is motor third-party liability. Never
   claim it is Assurant data.
7. Do not give underwriting, pricing, coverage, or fraud decisions for a
   real customer. This application is a research/demo system.
8. For questions requiring comparison or synthesis, use multiple tools when
   needed rather than guessing from a single result. When independent tools are
   needed, emit their tool calls together in the same response when possible.
9. Keep answers concise and business-oriented. Prefer tables or compact bullets
   when several values are useful.
10. When summarizing grouped results, preserve category labels exactly as returned by
   tools (for example <25, 25-34, 65+, Diesel, Regular, and region codes).
""".strip()


def _tool(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOLS = [
    _tool("get_kpis", "Return overall portfolio KPIs including policy count, claim count, claim frequency, total claim amount and average claim severity."),
    _tool("regional_summary", "Return policy volume, claim frequency and claim severity by French region."),
    _tool("fuel_summary", "Return claim frequency and claim severity by vehicle fuel type."),
    _tool("driver_age_summary", "Return claim frequency and claim severity across driver age bands."),
    _tool(
        "find_policy",
        "Look up an individual policy by numeric ID and return its observed policy, claim and statistical-review features.",
        {"policy_id": {"type": "integer"}},
        ["policy_id"],
    ),
    _tool(
        "top_anomalies",
        "Return the highest anomaly-score policies for statistical review. These are not confirmed fraud cases.",
        {"limit": {"type": "integer", "minimum": 1, "maximum": 20}},
    ),
    _tool(
        "anomaly_region_summary",
        "Compare statistical-review flag rates and portfolio shares by region.",
    ),
    _tool(
        "search_policy",
        "Retrieve passages from the demonstration policy corpus for coverage and claims-process questions.",
        {
            "query": {"type": "string"},
            "k": {"type": "integer", "minimum": 1, "maximum": 5},
        },
        ["query"],
    ),
]


class ClaimsAgent:
    def __init__(
        self,
        tools: ClaimsTools,
        model: str | None = None,
        max_iterations: int = 4,
    ) -> None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        from groq import Groq

        self.client = Groq(api_key=api_key)
        self.tools = tools
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")
        self.reasoning_effort = os.environ.get("GROQ_REASONING_EFFORT", "low")
        self.max_completion_tokens = int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS", "2048"))
        self.max_iterations = max_iterations

    def _execute(self, name: str, arguments: dict[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
        dispatch = {
            "get_kpis": self.tools.get_kpis,
            "regional_summary": self.tools.regional_summary,
            "fuel_summary": self.tools.fuel_summary,
            "driver_age_summary": self.tools.driver_age_summary,
            "find_policy": self.tools.find_policy,
            "top_anomalies": self.tools.top_anomalies,
            "anomaly_region_summary": self.tools.anomaly_region_summary,
            "search_policy": self.tools.search_policy,
        }
        if name not in dispatch:
            raise ValueError(f"Unknown tool: {name}")

        started = time.perf_counter()
        result = dispatch[name](**arguments)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        if name == "search_policy":
            source_refs = [
                {
                    "kind": "policy",
                    "title": item.get("title", "Unknown policy"),
                    "distance": item.get("distance"),
                }
                for item in result
            ]
        else:
            source_refs = [DATASET_SOURCE]

        return {
            "data": result,
            "source_refs": source_refs,
        }, source_refs + [{"kind": "metric", "latency_ms": elapsed_ms}]

    def answer(self, question: str) -> dict[str, Any]:
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tool_trace: list[dict[str, Any]] = []
        sources: list[dict[str, Any]] = []
        llm_total_ms = 0.0
        llm_calls = 0

        for _ in range(self.max_iterations):
            llm_started = time.perf_counter()
            request_kwargs = {
                "model": self.model,
                "messages": messages,
                "tools": TOOLS,
                "tool_choice": "auto",
                "parallel_tool_calls": True,
                "temperature": 0,
                "citation_options": "disabled",
                "max_completion_tokens": self.max_completion_tokens,
            }
            # GPT-OSS exposes configurable reasoning effort. Keep this optional
            # so users can raise it for harder questions without changing code.
            if self.model.startswith("openai/gpt-oss-"):
                request_kwargs["reasoning_effort"] = self.reasoning_effort
            response = self.client.chat.completions.create(**request_kwargs)
            llm_elapsed_ms = round((time.perf_counter() - llm_started) * 1000, 2)
            llm_total_ms += llm_elapsed_ms
            llm_calls += 1
            message = response.choices[0].message

            # Groq may return provider-specific response fields such as
            # ``annotations``. Those fields are not accepted when the
            # assistant message is replayed in a subsequent Chat Completions
            # request. Keep only the standard fields required for tool use.
            assistant_message = {
                "role": "assistant",
                "content": message.content or "",
            }
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            messages.append(assistant_message)

            if not message.tool_calls:
                return {
                    "answer": message.content or "No answer generated.",
                    "tool_trace": tool_trace,
                    "sources": _unique_sources(sources),
                    "llm_calls": llm_calls,
                    "llm_total_ms": round(llm_total_ms, 2),
                }

            parsed_calls = [
                (tool_call, tool_call.function.name, json.loads(tool_call.function.arguments or "{}"))
                for tool_call in message.tool_calls
            ]

            # Execute independent tool calls concurrently. Each SQL-backed tool
            # opens its own read-only SQLite connection, and the small RAG corpus
            # is safe for independent reads. This keeps multi-tool questions from
            # serialising otherwise independent data access.
            with ThreadPoolExecutor(max_workers=len(parsed_calls)) as executor:
                futures = [
                    executor.submit(self._execute, name, arguments)
                    for _, name, arguments in parsed_calls
                ]
                execution_results = [future.result() for future in futures]

            for (tool_call, name, arguments), (result, execution_meta) in zip(parsed_calls, execution_results):
                source_refs = result.pop("source_refs", [])
                tool_latency = next(
                    (m["latency_ms"] for m in execution_meta if "latency_ms" in m),
                    None,
                )
                tool_trace.append(
                    {
                        "tool": name,
                        "arguments": arguments,
                        "latency_ms": tool_latency,
                        "llm_call_latency_ms": llm_elapsed_ms,
                        "parallel_batch_size": len(parsed_calls),
                    }
                )
                sources.extend(source_refs)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "content": json.dumps(result, default=str),
                    }
                )

        return {
            "answer": "The agent reached its tool-call limit before producing a final response.",
            "tool_trace": tool_trace,
            "sources": _unique_sources(sources),
            "llm_calls": llm_calls,
            "llm_total_ms": round(llm_total_ms, 2),
        }


def _unique_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output: list[dict[str, Any]] = []
    for source in sources:
        key = json.dumps(source, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        output.append(source)
    return output
