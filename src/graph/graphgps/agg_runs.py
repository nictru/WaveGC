import logging
import os

import numpy as np
from rich.console import Console
from rich.table import Table
from rich import box

from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.utils.io import (
    dict_list_to_json,
    dict_list_to_tb,
    dict_to_json,
    json_to_dict_list,
    makedirs_rm_exist,
    string_to_python,
)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def is_seed(s):
    try:
        int(s)
        return True
    except Exception:
        return False


def is_split(s):
    if s in ['train', 'val', 'test']:
        return True
    else:
        return False


def join_list(l1, l2):
    assert len(l1) == len(l2), \
        'Results with different seeds must have the save format'
    for i in range(len(l1)):
        l1[i] += l2[i]
    return l1


def agg_dict_list(dict_list):
    """
    Aggregate a list of dictionaries: mean + std
    Args:
        dict_list: list of dictionaries

    """
    dict_agg = {'epoch': dict_list[0]['epoch']}
    for key in dict_list[0]:
        if key != 'epoch':
            value = np.array([dict[key] for dict in dict_list])
            dict_agg[key] = np.mean(value).round(cfg.round)
            dict_agg['{}_std'.format(key)] = np.std(value).round(cfg.round)
    return dict_agg


def name_to_dict(run):
    run = run.split('-', 1)[-1]
    cols = run.split('=')
    keys, vals = [], []
    keys.append(cols[0])
    for col in cols[1:-1]:
        try:
            val, key = col.rsplit('-', 1)
        except Exception:
            print(col)
        keys.append(key)
        vals.append(string_to_python(val))
    vals.append(cols[-1])
    return dict(zip(keys, vals))


def rm_keys(dict, keys):
    for key in keys:
        dict.pop(key, None)


def _print_results_table(results_best: dict) -> None:
    """Render best-epoch metrics as a rich table to stdout."""
    # Ordered set of metric keys we want to show (if present in the data)
    METRIC_KEYS = ['loss', 'accuracy', 'f1', 'auc', 'ap', 'mae', 'rmse', 'r2']
    SPLIT_STYLES = {'train': 'green', 'val': 'yellow', 'test': 'cyan'}

    # Collect the columns that actually exist across all splits
    present_metrics = []
    for key in METRIC_KEYS:
        if any(key in d for d in results_best.values()):
            present_metrics.append(key)

    best_epoch = next(iter(results_best.values())).get('epoch', '?')

    table = Table(
        title=f"Best results  ·  epoch {best_epoch}",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold",
        title_style="bold magenta",
    )
    table.add_column("split", style="bold", justify="left")
    for key in present_metrics:
        table.add_column(key, justify="right")
    table.add_column("time/epoch", justify="right", style="dim")

    for split in ['train', 'val', 'test']:
        if split not in results_best:
            continue
        d = results_best[split]
        style = SPLIT_STYLES.get(split, "")
        row = [f"[{style}]{split}[/{style}]"]
        for key in present_metrics:
            val = d.get(key)
            if val is None:
                row.append("—")
            elif key == 'loss':
                row.append(f"{val:.4f}")
            else:
                row.append(f"{val * 100:.2f}%")
        t = d.get('time_epoch') or d.get('time_iter')
        row.append(f"{t:.3f}s" if t is not None else "—")
        table.add_row(*row)

    Console().print(table)


def agg_runs(dir, metric_best='auto'):
    r'''
    Aggregate over different random seeds of a single experiment

    Args:
        dir (str): Directory of the results, containing 1 experiment
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
    results = {'train': None, 'val': None, 'test': None}
    results_best = {'train': None, 'val': None, 'test': None}
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)

            split = 'val'
            if split in os.listdir(dir_seed):
                dir_split = os.path.join(dir_seed, split)
                fname_stats = os.path.join(dir_split, 'stats.json')
                stats_list = json_to_dict_list(fname_stats)
                if metric_best == 'auto':
                    metric = 'auc' if 'auc' in stats_list[0] else 'accuracy'
                else:
                    metric = metric_best
                performance_np = np.array(  # noqa
                    [stats[metric] for stats in stats_list])
                best_epoch = \
                    stats_list[
                        eval("performance_np.{}()".format(cfg.metric_agg))][
                        'epoch']

            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    stats_list = json_to_dict_list(fname_stats)
                    stats_best = [
                        stats for stats in stats_list
                        if stats['epoch'] == best_epoch
                    ][0]
                    stats_list = [[stats] for stats in stats_list]
                    if results[split] is None:
                        results[split] = stats_list
                    else:
                        results[split] = join_list(results[split], stats_list)
                    if results_best[split] is None:
                        results_best[split] = [stats_best]
                    else:
                        results_best[split] += [stats_best]
    results = {k: v for k, v in results.items() if v is not None}  # rm None
    results_best = {k: v
                    for k, v in results_best.items()
                    if v is not None}  # rm None
    for key in results:
        for i in range(len(results[key])):
            results[key][i] = agg_dict_list(results[key][i])
    for key in results_best:
        results_best[key] = agg_dict_list(results_best[key])
    # save aggregated results
    for key, value in results.items():
        dir_out = os.path.join(dir, 'agg', key)
        makedirs_rm_exist(dir_out)
        fname = os.path.join(dir_out, 'stats.json')
        dict_list_to_json(value, fname)

        if cfg.tensorboard_agg:
            if SummaryWriter is None:
                raise ImportError(
                    'Tensorboard support requires `tensorboardX`.')
            writer = SummaryWriter(dir_out)
            dict_list_to_tb(value, writer)
            writer.close()
    for key, value in results_best.items():
        dir_out = os.path.join(dir, 'agg', key)
        fname = os.path.join(dir_out, 'best.json')
        dict_to_json(value, fname)

    # Pretty-print the best-epoch results as a rich table
    _print_results_table(results_best)

    logging.info('Results aggregated across runs saved in {}'.format(
        os.path.join(dir, 'agg')))
