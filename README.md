# WaveGC

Official implementation of the ICML 2025 paper  
**"A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition"**

Built on top of [GraphGPS](https://github.com/rampasek/GraphGPS).

![WaveGC model](model.png)

> **Citation**
> ```bibtex
> @inproceedings{liu2025a,
>   title={A General Graph Spectral Wavelet Convolution via Chebyshev Order Decomposition},
>   author={Nian Liu and Xiaoxin He and Thomas Laurent and Francesco Di Giovanni
>           and Michael M. Bronstein and Xavier Bresson},
>   booktitle={Forty-second International Conference on Machine Learning},
>   year={2025},
>   url={https://openreview.net/forum?id=UTvdB2WPSp}
> }
> ```

---

## Installation

WaveGC uses [pixi](https://pixi.sh) to manage the environment, including CUDA-aware PyTorch and PyG wheels. No manual conda/pip steps are needed.

**1. Install pixi** (once per machine):
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**2. Clone and install the environment:**
```bash
git clone https://github.com/nictru/WaveGC.git
cd WaveGC
pixi install
```

This resolves all dependencies declared in `pyproject.toml` — including the correct CUDA 12.4 wheels for PyTorch 2.6, PyG extensions (`torch-scatter`, `torch-sparse`, etc.), and `rich-click` for the CLI.

**3. (Optional) Download the PCQM4Mv2-Contact dataset:**
```bash
cd src/graph/datasets/
gdown "https://drive.google.com/u/0/uc?id=1AsDG-9WZ5b11lzHl8Ns4Qb0HxlcUOlQq" && unzip pcqm4m-v2-contact.zip
```

---

## Repository structure

```
WaveGC/
├── src/
│   ├── graph/              # Graph-level task (long-range benchmarks)
│   │   ├── configs/        # YAML configs: coco, pcqm, pf, ps, voc
│   │   └── graphgps/       # Model, layers, loaders, encoders …
│   ├── node/               # Node-level task (short-range benchmarks)
│   │   ├── configs/        # YAML configs: arxiv, computer, corafull, cs, photo
│   │   └── graphgps/       # Model, layers, loaders, encoders …
│   ├── api.py              # Public Python API
│   ├── cli.py              # rich-click CLI entry point
│   └── __init__.py
└── pyproject.toml
```

---

## Running experiments

Experiments can be launched via the **CLI** or directly from **Python**.

### CLI

Activate the pixi environment first:
```bash
pixi shell
```

**Graph-level** (long-range datasets: `voc`, `pcqm`, `coco`, `pf`, `ps`):
```bash
wavegc graph --cfg src/graph/configs/pcqm.yaml --repeat 4 --gpus 0
```

**Node-level** (short-range datasets: `computer`, `corafull`, `cs`, `photo`, `arxiv`):
```bash
wavegc node --cfg src/node/configs/arxiv.yaml --repeat 10 --gpus 0
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `--cfg FILE` | *(required)* | Path to the YAML configuration file |
| `--repeat N` | `1` | Number of repeated runs with incrementing seeds |
| `--gpus IDS` | *(unset)* | GPU(s) to use — see table below |
| `--mark-done` | off | Rename the config file with a `_done` suffix on completion |
| `[OPTS]...` | | Extra `key value` pairs forwarded as GraphGym config overrides |

**`--gpus` values:**

| Value | Effect |
|---|---|
| `0` | Use GPU 0 |
| `0,1` | Use GPUs 0 and 1 |
| `all` | Expose all GPUs (unsets `CUDA_VISIBLE_DEVICES`) |
| `cpu` | Force CPU-only mode |
| *(omitted)* | Leave `CUDA_VISIBLE_DEVICES` untouched |

**Config overrides** are appended as positional `key value` pairs after all flags:
```bash
wavegc graph --cfg src/graph/configs/voc.yaml --repeat 4 --gpus 0,1 \
    optim.max_epoch 200 gt.layers 6 wandb.use True
```

Run `wavegc --help` or `wavegc graph --help` / `wavegc node --help` for full usage.

---

### Python API

```python
from src import run_graph, run_node
```

**Graph-level:**
```python
run_graph(
    "src/graph/configs/pcqm.yaml",
    repeat=4,
    gpus=0,
    overrides={"optim.max_epoch": 200, "gt.layers": 6},
)
```

**Node-level:**
```python
run_node(
    "src/node/configs/arxiv.yaml",
    repeat=10,
    gpus=[0, 1],
    overrides={"optim.base_lr": 0.001},
)
```

Both functions share the same signature:

```python
run_graph(
    cfg_file,           # path to the YAML config
    *,
    repeat=1,           # number of runs with incrementing seeds
    mark_done=False,    # rename config to <name>_done after completion
    gpus=None,          # int, list[int], "0,1", "all", "cpu", or None
    overrides=None,     # dict of GraphGym dot-notation overrides
)
```

**`gpus` values:**

| Value | Effect |
|---|---|
| `0` | Use GPU 0 |
| `[0, 1]` or `"0,1"` | Use GPUs 0 and 1 |
| `"all"` | Expose all GPUs |
| `"cpu"` or `-1` | Force CPU-only mode |
| `None` *(default)* | Leave `CUDA_VISIBLE_DEVICES` untouched |

Config `overrides` use GraphGym dot-notation keys and accept any Python value:
```python
overrides = {
    "optim.max_epoch": 100,
    "optim.base_lr": 3e-4,
    "wandb.use": True,
    "gt.dropout": 0.1,
}
```
