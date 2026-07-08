"""
agent_loop — deterministic, LLM-free tooling for the agent-driven quant R&D loop.

This package is intentionally kept OUTSIDE the vendored `rdagent/` tree so that
periodic upstream merges of microsoft/RD-Agent stay cheap (see ADR 0002). It wraps
RD-Agent's execution machinery (Dockerized qlib backtests, the trace ledger, and a
statistical guardrail) as tools that *this session's agent* drives directly — the
agent is the brain (propose / code / judge); these tools are the hands.

A read-only web UI (`python -m agent_loop.report_ui --port 19900`) browses every loop's report —
gate decision, selection-vs-holdout comparison, backtest figures, and the agent's discussion — over the
same file-based state (see ADR 0003).

Nothing in here calls an LLM. No ANTHROPIC_API_KEY / LiteLLM is required.
"""
