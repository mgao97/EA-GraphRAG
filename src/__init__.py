"""EA-GraphRAG: Evidence-Aware GraphRAG.

The package implements a sufficiency-driven GraphRAG framework with an
Evidence Acquisition Controller (EAC) that dynamically chooses among
RETRIEVE, EXPAND, BRIDGE, VERIFY, STOP based on four evidence signals
(Semantic Relevance, Structural Information Gain, Reasoning Completeness,
Evidence Consistency).  See ``experiment.md`` for the full experimental
protocol.
"""
__version__ = "0.1.0"
