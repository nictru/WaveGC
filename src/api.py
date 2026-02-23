"""
Core Python API for running WaveGC experiments.

Both :func:`run_graph` and :func:`run_node` are thin, typed wrappers around
the shared :func:`_run_experiment` engine.  Config overrides are passed as a
plain Python ``dict`` whose keys use GraphGym dot-notation
(e.g. ``{"optim.max_epoch": 100, "wandb.use": True}``); values are
automatically stringified before being forwarded to GraphGym.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _overrides_to_opts(overrides: dict[str, Any]) -> list[str]:
    """Convert a ``{key: value}`` dict into a flat ``["key", "value", ...]`` list."""
    opts: list[str] = []
    for k, v in overrides.items():
        opts.extend([str(k), str(v)])
    return opts


def _prepend_task_src(task: str) -> None:
    """Insert the task-specific source directory at the front of sys.path."""
    task_dir = str(Path(__file__).parent / task)
    if task_dir not in sys.path:
        sys.path.insert(0, task_dir)


def _new_optimizer_config(cfg):
    from torch_geometric.graphgym.optim import OptimizerConfig
    return OptimizerConfig(
        optimizer=cfg.optim.optimizer,
        base_lr=cfg.optim.base_lr,
        weight_decay=cfg.optim.weight_decay,
        momentum=cfg.optim.momentum,
    )


def _new_scheduler_config(cfg):
    from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
    return ExtendedSchedulerConfig(
        scheduler=cfg.optim.scheduler,
        steps=cfg.optim.steps,
        lr_decay=cfg.optim.lr_decay,
        max_epoch=cfg.optim.max_epoch,
        reduce_factor=cfg.optim.reduce_factor,
        schedule_patience=cfg.optim.schedule_patience,
        min_lr=cfg.optim.min_lr,
        num_warmup_epochs=cfg.optim.num_warmup_epochs,
        train_mode=cfg.train.mode,
        eval_period=cfg.train.eval_period,
    )


def _set_out_dir(cfg, cfg_fname: str, name_tag: str) -> None:
    run_name = os.path.splitext(os.path.basename(cfg_fname))[0]
    run_name += f"-{name_tag}" if name_tag else ""
    cfg.out_dir = os.path.join(
        cfg.out_dir,
        run_name,
        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    )


def _set_run_dir(cfg, run_id) -> None:
    from torch_geometric.graphgym.config import makedirs_rm_exist
    cfg.run_dir = os.path.join(cfg.out_dir, str(run_id))
    if cfg.train.auto_resume:
        os.makedirs(cfg.run_dir, exist_ok=True)
    else:
        makedirs_rm_exist(cfg.run_dir)


def _loop_settings(cfg, repeat: int):
    if len(cfg.run_multiple_splits) == 0:
        seeds = [cfg.seed + x for x in range(repeat)]
        split_indices = [cfg.dataset.split_index] * repeat
        run_ids = seeds
    else:
        if repeat != 1:
            raise NotImplementedError(
                "Running multiple repeats of multiple splits in one run is not supported."
            )
        seeds = [cfg.seed] * len(cfg.run_multiple_splits)
        split_indices = cfg.run_multiple_splits
        run_ids = split_indices
    return run_ids, seeds, split_indices


# ---------------------------------------------------------------------------
# Core experiment engine
# ---------------------------------------------------------------------------

def _run_experiment(
    task: str,
    cfg_file: str,
    repeat: int,
    mark_done: bool,
    opts: list[str],
) -> None:
    """Low-level engine shared by the public API and the CLI."""
    import torch
    from torch_geometric.graphgym.config import cfg, dump_cfg, set_cfg, load_cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.logger import set_printing
    from torch_geometric.graphgym.optim import create_optimizer, create_scheduler
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.train import GraphGymDataModule, train
    from torch_geometric.graphgym.utils.comp_budget import params_count
    from torch_geometric.graphgym.utils.device import auto_select_device
    from torch_geometric.graphgym.register import train_dict
    from torch_geometric import seed_everything
    from graphgps.finetuning import load_pretrained_model_cfg, init_model_from_pretrained
    from graphgps.logger import create_logger
    from graphgps.agg_runs import agg_runs

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = argparse.Namespace(cfg_file=cfg_file, repeat=repeat, mark_done=mark_done, opts=opts)
    set_cfg(cfg)
    load_cfg(cfg, args)
    _set_out_dir(cfg, cfg_file, cfg.name_tag)
    dump_cfg(cfg)
    torch.set_num_threads(cfg.num_threads)

    for run_id, seed, split_index in zip(*_loop_settings(cfg, repeat)):
        _set_run_dir(cfg, run_id)
        set_printing()
        cfg.dataset.split_index = split_index
        cfg.seed = seed
        cfg.run_id = run_id
        seed_everything(cfg.seed)
        auto_select_device()
        if cfg.pretrained.dir:
            cfg = load_pretrained_model_cfg(cfg)
        logging.info(f"[*] Run ID {run_id}: seed={cfg.seed}, split_index={cfg.dataset.split_index}")
        logging.info(f"    Starting now: {datetime.datetime.now()}")

        loaders = create_loader()
        loggers = create_logger()
        model = create_model()
        if cfg.pretrained.dir:
            model = init_model_from_pretrained(
                model,
                cfg.pretrained.dir,
                cfg.pretrained.freeze_main,
                cfg.pretrained.reset_prediction_head,
                seed=cfg.seed,
            )
        optimizer = create_optimizer(model.parameters(), _new_optimizer_config(cfg))
        scheduler = create_scheduler(optimizer, _new_scheduler_config(cfg))
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info("Num parameters: %s", cfg.params)

        if cfg.train.mode == "standard":
            if cfg.wandb.use:
                logging.warning(
                    "[W] WandB logging is not supported with train.mode='standard', set it to 'custom'"
                )
            datamodule = GraphGymDataModule()
            train(model, datamodule, logger=True)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer, scheduler)

    try:
        agg_runs(cfg.out_dir, cfg.metric_best)
    except Exception as exc:
        logging.info(f"Failed when trying to aggregate multiple runs: {exc}")

    if mark_done:
        os.rename(cfg_file, f"{cfg_file}_done")

    logging.info(f"[*] All done: {datetime.datetime.now()}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_graph(
    cfg_file: str | Path,
    *,
    repeat: int = 1,
    mark_done: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Run a graph-level WaveGC experiment.

    Args:
        cfg_file: Path to the YAML configuration file
            (e.g. ``"src/graph/configs/pcqm.yaml"``).
        repeat: Number of repeated runs with incrementing seeds.
        mark_done: If ``True``, rename the config file with a ``_done``
            suffix after the run completes.
        overrides: Optional dict of GraphGym config overrides using
            dot-notation keys, e.g.
            ``{"optim.max_epoch": 100, "wandb.use": True}``.

    Example::

        from src import run_graph

        run_graph(
            "src/graph/configs/pcqm.yaml",
            repeat=3,
            overrides={"optim.max_epoch": 50, "gt.layers": 2},
        )
    """
    _prepend_task_src("graph")
    import graphgps  # noqa — registers graph-level modules
    _run_experiment(
        task="graph",
        cfg_file=str(cfg_file),
        repeat=repeat,
        mark_done=mark_done,
        opts=_overrides_to_opts(overrides or {}),
    )


def run_node(
    cfg_file: str | Path,
    *,
    repeat: int = 1,
    mark_done: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Run a node-level WaveGC experiment.

    Args:
        cfg_file: Path to the YAML configuration file
            (e.g. ``"src/node/configs/arxiv.yaml"``).
        repeat: Number of repeated runs with incrementing seeds.
        mark_done: If ``True``, rename the config file with a ``_done``
            suffix after the run completes.
        overrides: Optional dict of GraphGym config overrides using
            dot-notation keys, e.g.
            ``{"optim.max_epoch": 100, "wandb.use": True}``.

    Example::

        from src import run_node

        run_node(
            "src/node/configs/arxiv.yaml",
            overrides={"optim.base_lr": 0.001},
        )
    """
    _prepend_task_src("node")
    import graphgps  # noqa — registers node-level modules
    _run_experiment(
        task="node",
        cfg_file=str(cfg_file),
        repeat=repeat,
        mark_done=mark_done,
        opts=_overrides_to_opts(overrides or {}),
    )
