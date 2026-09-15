"""ArcticDB-backed providers for the three acquisition branches.

Unlike Megadata, ArcticDB is not a REST API: data lives in named libraries,
addressed by symbol, and is read as a pandas DataFrame. These providers do not
use the Research Planner's ``data_requests``/``search_requests`` (those are
Megadata-endpoint shaped) - they read the library/symbol layout directly,
keyed off ``plan.ticker``/``plan.peers``.
"""
