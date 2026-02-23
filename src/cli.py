"""
rich-click CLI for WaveGC.  All experiment logic lives in :mod:`src.api`.
"""

import rich_click as click

from src.api import run_graph, run_node

click.rich_click.USE_RICH_MARKUP = True
click.rich_click.SHOW_ARGUMENTS = True
click.rich_click.GROUP_ARGUMENTS_OPTIONS = True
click.rich_click.STYLE_ERRORS_SUGGESTION = "magenta italic"
click.rich_click.ERRORS_SUGGESTION = (
    "Try running [bold]wavegc --help[/bold] for usage information."
)

# ---------------------------------------------------------------------------
# Shared option decorators
# ---------------------------------------------------------------------------

_cfg_option = click.option(
    "--cfg",
    "cfg_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    help="Path to the YAML configuration file.",
    metavar="FILE",
)

_repeat_option = click.option(
    "--repeat",
    default=1,
    show_default=True,
    type=int,
    help="Number of repeated experiment runs.",
)

_mark_done_option = click.option(
    "--mark-done",
    is_flag=True,
    default=False,
    help="Rename the config file with a [cyan]_done[/cyan] suffix after the run completes.",
)

_opts_argument = click.argument("opts", nargs=-1, metavar="[OPTS]...")


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="WaveGC")
def cli():
    """[bold]WaveGC[/bold] — Wavelet-based Graph Convolution framework.

    Run graph-level or node-level experiments using YAML config files.
    """


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

@cli.command("graph")
@_cfg_option
@_repeat_option
@_mark_done_option
@_opts_argument
def graph_cmd(cfg_file, repeat, mark_done, opts):
    """Train a [bold]graph-level[/bold] WaveGC model.

    \b
    OPTS are additional key=value config overrides forwarded directly to
    GraphGym (e.g. [cyan]optim.max_epoch 100 wandb.use True[/cyan]).

    \b
    Example:
        wavegc graph --cfg src/graph/configs/pcqm.yaml --repeat 3
    """
    overrides = dict(zip(opts[::2], opts[1::2]))
    run_graph(cfg_file, repeat=repeat, mark_done=mark_done, overrides=overrides)


@cli.command("node")
@_cfg_option
@_repeat_option
@_mark_done_option
@_opts_argument
def node_cmd(cfg_file, repeat, mark_done, opts):
    """Train a [bold]node-level[/bold] WaveGC model.

    \b
    OPTS are additional key=value config overrides forwarded directly to
    GraphGym (e.g. [cyan]optim.max_epoch 100 wandb.use True[/cyan]).

    \b
    Example:
        wavegc node --cfg src/node/configs/arxiv.yaml
    """
    overrides = dict(zip(opts[::2], opts[1::2]))
    run_node(cfg_file, repeat=repeat, mark_done=mark_done, overrides=overrides)


if __name__ == "__main__":
    cli()
