"""
Python API for WaveGC.

Usage::

    from src import run_graph, run_node

    run_graph("src/graph/configs/pcqm.yaml", repeat=3, overrides={"optim.max_epoch": 100})
    run_node("src/node/configs/arxiv.yaml")
"""

from src.api import run_graph, run_node

__all__ = ["run_graph", "run_node"]
