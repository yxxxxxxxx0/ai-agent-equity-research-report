"""Shared LLM client used by every stage that is allowed to call a model.

Only the Research Planner, the segment agents (when LLM-backed) and the
synthesis layer (when LLM-backed) ever import from this package. Analytics,
QA and rendering never do - they stay deterministic.
"""
