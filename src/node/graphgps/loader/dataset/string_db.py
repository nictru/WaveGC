"""
STRING v12.0 protein interaction dataset for node-level classification.

Nodes   : unique human proteins (gene symbols) present in the STRING network
          after applying the required_score threshold.
Edges   : interactions with combined_score >= required_score, made undirected.
          No protein-complex nodes — STRING uses individual Ensembl protein IDs.
Features: multi-hot vector over GO_CC_CATEGORIES (GO Cellular Component).
Labels  : integer class of the primary GO CC category matched per protein;
          -1 for unannotated proteins (excluded from train/val/test masks).
"""

from __future__ import annotations

import gzip
import logging
import os
import os.path as osp

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

logger = logging.getLogger(__name__)

# Broad GO Cellular Component categories used as node label classes.
# Matched by substring against the `description` field in the enrichment
# terms file (case-insensitive).  Order matters: first match wins.
GO_CC_CATEGORIES = [
    "nucleus",
    "cytoplasm",
    "plasma membrane",
    "extracellular region",
    "mitochondrion",
    "endoplasmic reticulum",
    "Golgi apparatus",
    "cytoskeleton",
    "vesicle",
    "chromatin",
]

_DOWNLOAD_BASE = "https://stringdb-downloads.org/download"
_SPECIES = "9606"
_VERSION = "v12.0"


class STRINGDataset(InMemoryDataset):
    """Single-graph node-classification dataset built from STRING v12.0.

    Downloads three bulk files from stringdb-downloads.org on first use and
    caches the processed PyG ``Data`` object under ``<root>/processed/data.pt``.

    Args:
        root:           Directory where raw and processed files are stored.
        required_score: Minimum combined_score (0–1000) to include an edge.
                        700 = high confidence (default), 400 = medium,
                        900 = highest confidence.
        transform:      Optional transform applied at access time.
        pre_transform:  Optional transform applied before saving.
    """

    _RAW_FILES = [
        f"{_SPECIES}.protein.links.detailed.{_VERSION}.txt.gz",
        f"{_SPECIES}.protein.info.{_VERSION}.txt.gz",
        f"{_SPECIES}.protein.enrichment.terms.{_VERSION}.txt.gz",
    ]

    _URLS = [
        f"{_DOWNLOAD_BASE}/protein.links.detailed.{_VERSION}/{_SPECIES}.protein.links.detailed.{_VERSION}.txt.gz",
        f"{_DOWNLOAD_BASE}/protein.info.{_VERSION}/{_SPECIES}.protein.info.{_VERSION}.txt.gz",
        f"{_DOWNLOAD_BASE}/protein.enrichment.terms.{_VERSION}/{_SPECIES}.protein.enrichment.terms.{_VERSION}.txt.gz",
    ]

    def __init__(
        self,
        root: str,
        required_score: int = 700,
        transform=None,
        pre_transform=None,
    ):
        self.required_score = required_score
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def raw_file_names(self) -> list[str]:
        return self._RAW_FILES

    @property
    def processed_file_names(self) -> list[str]:
        return [f"data_score{self.required_score}.pt"]

    def download(self):
        import requests
        from rich.progress import (
            BarColumn,
            DownloadColumn,
            Progress,
            TextColumn,
            TimeRemainingColumn,
            TransferSpeedColumn,
        )

        os.makedirs(self.raw_dir, exist_ok=True)
        for url, fname in zip(self._URLS, self._RAW_FILES):
            dest = osp.join(self.raw_dir, fname)
            if osp.exists(dest):
                logger.info(f"[STRING] Already downloaded: {fname}")
                continue
            logger.info(f"[STRING] Downloading {fname} …")
            with requests.get(url, stream=True, timeout=300) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                with Progress(
                    TextColumn(f"[bold cyan]{fname}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                ) as progress:
                    task = progress.add_task("download", total=total or None)
                    with open(dest, "wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                            progress.advance(task, len(chunk))
            logger.info(f"[STRING] Saved to {dest}")

    def process(self):  # noqa: C901
        import pandas as pd

        links_path = osp.join(self.raw_dir, self._RAW_FILES[0])
        info_path = osp.join(self.raw_dir, self._RAW_FILES[1])
        terms_path = osp.join(self.raw_dir, self._RAW_FILES[2])

        # ------------------------------------------------------------------
        # 1.  Protein info: STRING ID → preferred_name (gene symbol)
        # ------------------------------------------------------------------
        logger.info("[STRING] Loading protein info …")
        with gzip.open(info_path, "rt") as fh:
            info = pd.read_csv(fh, sep="\t")
        # Column name may have a leading '#'
        id_col = [c for c in info.columns if "string_protein_id" in c][0]
        info = info.rename(columns={id_col: "string_protein_id"})
        id_to_gene: dict[str, str] = dict(
            zip(info["string_protein_id"], info["preferred_name"])
        )
        logger.info(f"[STRING] {len(id_to_gene):,} proteins in info table")

        # ------------------------------------------------------------------
        # 2.  Interactions filtered by required_score
        # ------------------------------------------------------------------
        logger.info(
            f"[STRING] Loading interactions (required_score ≥ {self.required_score}) …"
        )
        with gzip.open(links_path, "rt") as fh:
            links = pd.read_csv(fh, sep=" ")

        links = links[links["combined_score"] >= self.required_score].copy()
        logger.info(f"[STRING] {len(links):,} interactions after score filter")

        # Map STRING IDs to gene symbols; drop rows without a mapping
        links["gene_a"] = links["protein1"].map(id_to_gene)
        links["gene_b"] = links["protein2"].map(id_to_gene)
        links = links.dropna(subset=["gene_a", "gene_b"])
        links = links[
            links["gene_a"].str.strip().astype(bool)
            & links["gene_b"].str.strip().astype(bool)
        ]
        logger.info(f"[STRING] {len(links):,} interactions after symbol mapping")

        # ------------------------------------------------------------------
        # 3.  Build protein index
        # ------------------------------------------------------------------
        all_proteins = sorted(set(links["gene_a"]) | set(links["gene_b"]))
        protein_to_idx = {p: i for i, p in enumerate(all_proteins)}
        num_nodes = len(all_proteins)
        logger.info(f"[STRING] {num_nodes:,} unique proteins")

        # ------------------------------------------------------------------
        # 4.  Edge index + edge attributes (7 channel scores, normalised)
        # ------------------------------------------------------------------
        src_idx = torch.tensor(
            [protein_to_idx[g] for g in links["gene_a"]], dtype=torch.long
        )
        tgt_idx = torch.tensor(
            [protein_to_idx[g] for g in links["gene_b"]], dtype=torch.long
        )
        edge_index = torch.stack([src_idx, tgt_idx], dim=0)

        score_cols = ["nscore", "fscore", "pscore", "ascore", "escore", "dscore", "tscore"]

        def _score_col(col: str) -> torch.Tensor:
            if col in links.columns:
                return torch.tensor(
                    links[col].fillna(0).astype(float).values / 1000.0,
                    dtype=torch.float,
                )
            return torch.zeros(len(links), dtype=torch.float)

        edge_attr = torch.stack([_score_col(c) for c in score_cols], dim=1)  # [E, 7]

        # Make undirected
        edge_index, edge_attr = to_undirected(
            edge_index, edge_attr, num_nodes=num_nodes, reduce="max"
        )
        logger.info(
            f"[STRING] {edge_index.shape[1]:,} edges after making undirected"
        )

        # ------------------------------------------------------------------
        # 5.  GO CC annotations: node features (multi-hot) and labels
        # ------------------------------------------------------------------
        logger.info("[STRING] Loading GO CC enrichment terms …")
        with gzip.open(terms_path, "rt") as fh:
            terms = pd.read_csv(fh, sep="\t")
        id_col_t = [c for c in terms.columns if "string_protein_id" in c][0]
        terms = terms.rename(columns={id_col_t: "string_protein_id"})

        # Keep only GO Cellular Component entries
        # STRING uses the full name "Cellular Component (Gene Ontology)" as the category
        cc_terms = terms[terms["category"] == "Cellular Component (Gene Ontology)"].copy()
        cc_terms["gene"] = cc_terms["string_protein_id"].map(id_to_gene)
        cc_terms = cc_terms.dropna(subset=["gene"])

        cat_to_idx = {c: i for i, c in enumerate(GO_CC_CATEGORIES)}
        n_cats = len(GO_CC_CATEGORIES)

        # Pre-build a lookup: gene → list of matched category indices
        node_features = np.zeros((num_nodes, n_cats), dtype=np.float32)
        y = torch.full((num_nodes,), -1, dtype=torch.long)

        gene_first_label: dict[str, int] = {}

        for _, row in cc_terms.iterrows():
            gene = row["gene"]
            if gene not in protein_to_idx:
                continue
            desc = str(row.get("description", "")).lower()
            for cat_idx, cat in enumerate(GO_CC_CATEGORIES):
                if cat.lower() in desc:
                    node_idx = protein_to_idx[gene]
                    node_features[node_idx, cat_idx] = 1.0
                    if gene not in gene_first_label:
                        gene_first_label[gene] = cat_idx
                    break  # multi-hot: continue scanning other rows for same gene

        # Set label to first-encountered category (deterministic: iterate sorted)
        for gene, cat_idx in gene_first_label.items():
            y[protein_to_idx[gene]] = cat_idx

        x = torch.from_numpy(node_features)

        labelled_count = (y >= 0).sum().item()

        # Remap labels to contiguous 0..K-1 (not all 10 categories may appear)
        if labelled_count > 0:
            unique_labels = y[y >= 0].unique().sort().values
            y_remapped = y.clone()
            for new_idx, old_idx in enumerate(unique_labels.tolist()):
                y_remapped[y == int(old_idx)] = new_idx
            y = y_remapped
            actual_n_cats = len(unique_labels)
        else:
            actual_n_cats = 0

        logger.info(
            f"[STRING] {labelled_count:,}/{num_nodes:,} proteins labelled "
            f"across {actual_n_cats} GO CC categories"
        )

        # ------------------------------------------------------------------
        # 6.  Save
        # ------------------------------------------------------------------
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    num_nodes=num_nodes)
        data, slices = self.collate([data])
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"[STRING] Saved to {self.processed_paths[0]}")

        idx_path = osp.join(self.processed_dir, "protein_index.txt")
        with open(idx_path, "w") as fh:
            fh.write("\n".join(all_proteins))
        logger.info(f"[STRING] Protein index saved to {idx_path}")
