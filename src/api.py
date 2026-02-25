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


def _apply_gpus(gpus: int | str | list[int] | None) -> None:
    """Set ``CUDA_VISIBLE_DEVICES`` from a flexible *gpus* value.

    Must be called **before** importing torch.

    Args:
        gpus: Which GPUs to expose.

            * ``None`` — leave ``CUDA_VISIBLE_DEVICES`` untouched.
            * ``"all"`` — unset ``CUDA_VISIBLE_DEVICES`` (all GPUs visible).
            * ``"cpu"`` or ``-1`` — hide all GPUs (CPU-only mode).
            * ``int`` — single GPU index, e.g. ``0``.
            * ``list[int]`` — multiple GPU indices, e.g. ``[0, 1]``.
            * ``str`` — raw comma-separated indices, e.g. ``"0,1"``.
    """
    if gpus is None:
        return
    if isinstance(gpus, str) and gpus.lower() == "all":
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        return
    if gpus == -1 or (isinstance(gpus, str) and gpus.lower() == "cpu"):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    if isinstance(gpus, list):
        value = ",".join(str(g) for g in gpus)
    else:
        value = str(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = value


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

def _best_checkpoint(run_dir: Path, metric: str, metric_agg: str) -> "Path | None":
    """Return the path to the best epoch's checkpoint, or None if unavailable.

    Reads ``val/stats.json`` to find the best epoch, then checks whether a
    matching ``.ckpt`` file exists.  Falls back to the numerically highest
    checkpoint file if the exact epoch is missing.
    """
    try:
        from torch_geometric.graphgym.utils.io import json_to_dict_list
        stats = json_to_dict_list(str(run_dir / "val" / "stats.json"))
        if not stats:
            return None
        m = metric if metric != "auto" else ("auc" if "auc" in stats[0] else "accuracy")
        values = [s.get(m, 0.0) for s in stats]
        best_idx = (values.index(max(values)) if metric_agg == "argmax"
                    else values.index(min(values)))
        best_epoch = stats[best_idx]["epoch"]
    except Exception:
        best_epoch = None

    ckpt_dir = run_dir / "ckpt"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: int(p.stem))
    if not ckpts:
        return None

    if best_epoch is not None:
        exact = ckpt_dir / f"{best_epoch}.ckpt"
        if exact.exists():
            return exact
        # Best epoch checkpoint not on disk (e.g. old run with ckpt_best=False);
        # use the checkpoint with the closest epoch number.
        closest = min(ckpts, key=lambda p: abs(int(p.stem) - best_epoch))
        logging.warning(
            f"[embed] Checkpoint for best epoch {best_epoch} not found; "
            f"using epoch {closest.stem} instead."
        )
        return closest

    # Could not determine best epoch — fall back to the last saved checkpoint.
    return ckpts[-1]


def _save_embeddings(model, loaders, cfg, torch) -> None:
    """Extract node embeddings from the trained model and save to run_dir.

    Loads the best-epoch checkpoint (determined from val/stats.json), hooks
    into the WaveLayer Sequential just before the classification head, and
    writes ``embeddings.npy`` and ``node_names.txt`` into ``cfg.run_dir``.
    Silently skips if the model does not expose a ``model.layers`` Sequential.
    """
    import numpy as np

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        return

    # Load the best-epoch weights before extracting embeddings
    best_ckpt = _best_checkpoint(Path(cfg.run_dir), cfg.metric_best, cfg.metric_agg)
    if best_ckpt is not None:
        ckpt_data = torch.load(str(best_ckpt), map_location="cpu", weights_only=False)
        state_dict = ckpt_data.get("model_state", ckpt_data)
        model.load_state_dict(state_dict)
        logging.info(f"[embed] Loaded checkpoint: epoch {best_ckpt.stem} (best val)")
    else:
        logging.warning("[embed] No checkpoint found — using current (last) model weights.")

    _captured: dict = {}

    def _hook(module, inp, out):
        _captured["x"] = out.x.detach().cpu()

    handle = layers.register_forward_hook(_hook)
    model.eval()
    try:
        with torch.no_grad():
            for batch in loaders[0]:
                batch.split = "test"
                batch.to(torch.device(cfg.accelerator))
                model(batch)
    finally:
        handle.remove()

    if "x" not in _captured:
        logging.warning("[embed] Hook did not fire — skipping embedding save.")
        return

    embeddings = _captured["x"].float().numpy()
    out_dir = Path(cfg.run_dir)

    emb_path = out_dir / "embeddings.npy"
    np.save(str(emb_path), embeddings)

    # Node names — protein symbols for OmniPath, integer strings otherwise
    dataset_id = cfg.dataset.format.split("-", 1)[-1]
    names_path = (
        Path(cfg.dataset.dir) / dataset_id / "processed" / "protein_index.txt"
    )
    names_out = out_dir / "node_names.txt"
    if names_path.exists():
        names_out.write_text(names_path.read_text())
    else:
        names_out.write_text("\n".join(str(i) for i in range(len(embeddings))))

    logging.info(
        f"[embed] Saved embeddings {embeddings.shape} → {emb_path}"
    )


def _run_experiment(
    task: str,
    cfg_file: str,
    repeat: int,
    mark_done: bool,
    opts: list[str],
    gpus: int | str | list[int] | None = None,
) -> None:
    """Low-level engine shared by the public API and the CLI."""
    _apply_gpus(gpus)
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
    # Always checkpoint the best validation epoch — required for embedding extraction.
    cfg.train.enable_ckpt = True
    cfg.train.ckpt_best = True
    cfg.train.ckpt_clean = True
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

        _save_embeddings(model, loaders, cfg, torch)

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
    gpus: int | str | list[int] | None = None,
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
        gpus: GPU(s) to use. Accepts an ``int`` (``0``), a list of ints
            (``[0, 1]``), a comma-separated string (``"0,1"``),
            ``"all"`` (all available GPUs), or ``"cpu"`` / ``-1``
            (CPU-only). ``None`` leaves ``CUDA_VISIBLE_DEVICES`` untouched.

    Example::

        from src import run_graph

        run_graph(
            "src/graph/configs/pcqm.yaml",
            repeat=3,
            gpus=0,
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
        gpus=gpus,
    )


def run_node(
    cfg_file: str | Path,
    *,
    repeat: int = 1,
    mark_done: bool = False,
    overrides: dict[str, Any] | None = None,
    gpus: int | str | list[int] | None = None,
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
        gpus: GPU(s) to use. Accepts an ``int`` (``0``), a list of ints
            (``[0, 1]``), a comma-separated string (``"0,1"``),
            ``"all"`` (all available GPUs), or ``"cpu"`` / ``-1``
            (CPU-only). ``None`` leaves ``CUDA_VISIBLE_DEVICES`` untouched.

    Example::

        from src import run_node

        run_node(
            "src/node/configs/arxiv.yaml",
            gpus=[0, 1],
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
        gpus=gpus,
    )


def embed_nodes(
    run_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    gpus: int | str | list[int] | None = None,
) -> "tuple[np.ndarray, list[str]]":
    """Extract node embeddings from a trained WaveGC node model.

    Loads the saved config and checkpoint from *run_dir*, runs a single
    forward pass through encoder → pre_mp → wave_generator → WaveLayers
    (stopping before the classification head), and returns the resulting
    node representations.

    Args:
        run_dir: Top-level output directory of a finished run,
            e.g. ``"results/omnipath/2026-02-25 09:06:14"``.
            The function locates ``config.yaml`` there and the latest
            checkpoint under ``<run_dir>/0/ckpt/``.
        checkpoint: Explicit path to a ``.ckpt`` file.  Overrides the
            automatic discovery when supplied.
        gpus: GPU selection — same semantics as :func:`run_node`.

    Returns:
        embeddings: float32 numpy array of shape ``[N, dim_hidden]``.
        node_names: list of *N* node-name strings.  For OmniPath datasets
            these are gene symbols; otherwise ``"0"``, ``"1"``, …

    Example::

        from src.api import embed_nodes
        import pandas as pd

        emb, names = embed_nodes(
            "results/omnipath/2026-02-25 09:06:14",
            gpus="all",
        )
        df = pd.DataFrame(emb, index=names)
        df.to_parquet("omnipath_embeddings.parquet")
    """
    _apply_gpus(gpus)

    import numpy as np
    import torch
    from torch_geometric.graphgym.config import cfg, set_cfg, load_cfg
    from torch_geometric.graphgym.loader import create_loader
    from torch_geometric.graphgym.model_builder import create_model
    from torch_geometric.graphgym.utils.comp_budget import params_count
    from torch_geometric.graphgym.utils.device import auto_select_device

    _prepend_task_src("node")
    import graphgps  # noqa — registers node-level modules

    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {run_dir}")

    # Locate checkpoint --------------------------------------------------
    if checkpoint is None:
        ckpt_dir = run_dir / "0" / "ckpt"
        ckpts = sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: int(p.stem))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")
        checkpoint = ckpts[-1]
        logging.info(f"[embed_nodes] Using checkpoint: {checkpoint}")
    checkpoint = Path(checkpoint)

    # Load config --------------------------------------------------------
    set_cfg(cfg)
    args = argparse.Namespace(
        cfg_file=str(cfg_path), repeat=1, mark_done=False, opts=[]
    )
    load_cfg(cfg, args)
    auto_select_device()

    # Build dataset and model --------------------------------------------
    loaders = create_loader()
    model = create_model()
    cfg.params = params_count(model)

    # Load weights — GraphGym saves under 'model_state'; handle plain dicts too
    ckpt_data = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt_data.get("model_state", ckpt_data)
    model.load_state_dict(state_dict)
    model.eval()

    device = torch.device(cfg.accelerator)
    model.to(device)

    # Hook: capture batch.x after WaveLayers, before the classification head
    _captured: dict[str, torch.Tensor] = {}

    def _hook(module, inp, out):
        _captured["embeddings"] = out.x.detach().cpu()

    handle = model.model.layers.register_forward_hook(_hook)

    try:
        with torch.no_grad():
            for batch in loaders[0]:   # full graph — all nodes present
                batch.split = "test"
                batch.to(device)
                model(batch)
    finally:
        handle.remove()

    if "embeddings" not in _captured:
        raise RuntimeError(
            "Forward hook did not fire. "
            "Verify that model.model.layers is the WaveLayer Sequential."
        )

    embeddings: np.ndarray = _captured["embeddings"].float().numpy()

    # Node names — protein symbols for OmniPath, integer strings otherwise
    dataset_id = cfg.dataset.format.split("-", 1)[-1]   # e.g. "OmniPath"
    names_path = (
        Path(cfg.dataset.dir) / dataset_id / "processed" / "protein_index.txt"
    )
    if names_path.exists():
        node_names = names_path.read_text().strip().splitlines()
    else:
        node_names = [str(i) for i in range(len(embeddings))]

    logging.info(
        f"[embed_nodes] Extracted embeddings: {embeddings.shape}  "
        f"({len(node_names)} named nodes)"
    )
    return embeddings, node_names
