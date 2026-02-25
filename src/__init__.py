"""
Python API for WaveGC.

Usage::

    from src import run_graph, run_node, embed_nodes

    run_graph("src/graph/configs/pcqm.yaml", repeat=3, overrides={"optim.max_epoch": 100})
    run_node("src/node/configs/arxiv.yaml")

    emb, names = embed_nodes("results/omnipath/2026-02-25 09:06:14", gpus="all")
"""

from src.api import run_graph, run_node, embed_nodes

__all__ = ["run_graph", "run_node", "embed_nodes"]
