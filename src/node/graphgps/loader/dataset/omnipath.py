"""
OmniPath protein-interaction dataset for node-level classification.

Nodes   : unique proteins (gene symbols) present in the OmniPath core
          interaction network (manually curated, high-confidence subset).
Edges   : interactions from omnipath.interactions.OmniPath, made undirected.
Features: multi-hot vector over intercellular 'parent' categories
          (ligand, receptor, ecm, …) from Intercell; zero-vector for
          proteins with no annotation.
Labels  : integer class of the *first* intercellular parent category;
          -1 for unannotated proteins (excluded from train/val/test masks).
"""

from __future__ import annotations

import logging
import os
import os.path as osp

# Must be set before omnipath is imported. The autoload feature fires 6
# metadata requests on import with a hardcoded 3s timeout, producing noisy
# warnings on restricted networks. The local cache is sufficient.
os.environ.setdefault("OMNIPATH_AUTOLOAD", "0")

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected

logger = logging.getLogger(__name__)

# Canonical ordering of the Intercell generic parent categories.
# Derived from omnipath.requests.Intercell generic_categories(); kept fixed
# here so the feature dimension is stable across runs / versions.
INTERCELL_CATEGORIES = [
    "ligand",
    "receptor",
    "ecm",
    "cell_surface_ligand",
    "secreted",
    "transmembrane",
    "intracellular",
    "transporter",
    "adhesion",
    "gap_junction",
    "ion_channel",
    "nuclear_receptor",
    "matrix_adhesion",
    "tight_junction",
    "cell_surface",
]


class OmniPathDataset(InMemoryDataset):
    """Single-graph node-classification dataset built from OmniPath.

    Uses the core OmniPath interaction network (omnipath.interactions.OmniPath),
    a manually curated high-confidence subset (~8 800 proteins, ~85 000 edges).
    Downloads interaction and annotation data from the OmniPath REST API on
    first use and caches the processed PyG ``Data`` object under
    ``<root>/processed/data.pt``.

    Args:
        root: Directory where the processed file will be stored.
        transform: Optional transform applied at access time.
        pre_transform: Optional transform applied before saving.
    """

    def __init__(self, root: str, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return []  # data is fetched directly into process(); no raw files

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    def download(self):
        pass  # fetched inside process()

    def process(self):
        import omnipath.interactions as oi

        logger.info("[OmniPath] Fetching OmniPath core interactions …")
        interactions = oi.OmniPath.get(genesymbols=True)

        logger.info("[OmniPath] Fetching Intercell annotations …")
        import omnipath.requests as or_
        intercell = or_.Intercell.get(scope="generic")

        # ------------------------------------------------------------------
        # Build protein index
        # ------------------------------------------------------------------
        src_col = "source_genesymbol"
        tgt_col = "target_genesymbol"

        # Drop rows with missing gene symbols
        interactions = interactions.dropna(subset=[src_col, tgt_col])
        interactions = interactions[
            interactions[src_col].str.strip().astype(bool)
            & interactions[tgt_col].str.strip().astype(bool)
        ]

        # ------------------------------------------------------------------
        # Build protein / complex index
        # ------------------------------------------------------------------
        # Step 1 — collect every endpoint that appears in interactions
        #          (includes complex nodes like "ACVR1B_ACVR2A")
        endpoint_proteins = sorted(
            set(interactions[src_col]) | set(interactions[tgt_col])
        )
        protein_to_idx = {p: i for i, p in enumerate(endpoint_proteins)}

        # Step 2 — for every complex node, ensure each of its constituent
        #          genes also has an individual node in the graph.
        #          Complexes stay because they are important for graph
        #          structure; individual members are needed so that every
        #          protein gets its own embedding slot.
        complex_names = [p for p in endpoint_proteins if "_" in p]
        new_members = sorted({
            gene
            for c in complex_names
            for gene in c.split("_")
            if gene and gene not in protein_to_idx
        })
        for p in new_members:
            protein_to_idx[p] = len(protein_to_idx)

        all_proteins = endpoint_proteins + new_members
        num_nodes = len(all_proteins)
        logger.info(
            f"[OmniPath] {num_nodes} nodes: "
            f"{len(endpoint_proteins)} interaction endpoints "
            f"({len(complex_names)} complexes) + "
            f"{len(new_members)} complex-member proteins added"
        )

        # ------------------------------------------------------------------
        # Edge index + edge attributes
        # ------------------------------------------------------------------
        src_idx = torch.tensor(
            [protein_to_idx[p] for p in interactions[src_col]], dtype=torch.long
        )
        tgt_idx = torch.tensor(
            [protein_to_idx[p] for p in interactions[tgt_col]], dtype=torch.long
        )
        edge_index = torch.stack([src_idx, tgt_idx], dim=0)

        def _to_float_col(df, col):
            if col in df.columns:
                return torch.tensor(df[col].fillna(0).astype(float).values,
                                    dtype=torch.float)
            return torch.zeros(len(df), dtype=torch.float)

        edge_attr = torch.stack([
            _to_float_col(interactions, "is_stimulation"),
            _to_float_col(interactions, "is_inhibition"),
            _to_float_col(interactions, "is_directed"),
        ], dim=1)  # [E, 3]

        # Make undirected (mirrors what the arxiv pre-processing does)
        edge_index, edge_attr = to_undirected(edge_index, edge_attr,
                                              num_nodes=num_nodes,
                                              reduce="max")

        # ------------------------------------------------------------------
        # Complex-membership edges: complex node ↔ each constituent gene.
        # These are undirected structural edges with zero attributes
        # (no stimulation / inhibition / direction signal).
        # ------------------------------------------------------------------
        m_src, m_tgt = [], []
        for c in complex_names:
            c_idx = protein_to_idx[c]
            for gene in c.split("_"):
                if gene and gene in protein_to_idx:
                    m_src.append(c_idx)
                    m_tgt.append(protein_to_idx[gene])

        if m_src:
            # Add both directions so the graph stays undirected
            m_ei = torch.tensor(
                [m_src + m_tgt, m_tgt + m_src], dtype=torch.long
            )
            m_ea = torch.zeros(len(m_src) * 2, edge_attr.shape[1],
                               dtype=torch.float)
            edge_index = torch.cat([edge_index, m_ei], dim=1)
            edge_attr = torch.cat([edge_attr, m_ea], dim=0)
            logger.info(f"[OmniPath] Added {len(m_src)} complex-membership edges")

        # ------------------------------------------------------------------
        # Node features: multi-hot over INTERCELL_CATEGORIES
        # ------------------------------------------------------------------
        cat_to_idx = {c: i for i, c in enumerate(INTERCELL_CATEGORIES)}
        n_cats = len(INTERCELL_CATEGORIES)

        node_features = np.zeros((num_nodes, n_cats), dtype=np.float32)

        intercell = intercell.dropna(subset=["genesymbol", "parent"])
        for _, row in intercell.iterrows():
            gene = row["genesymbol"]
            parent = row["parent"]
            if gene in protein_to_idx and parent in cat_to_idx:
                node_features[protein_to_idx[gene], cat_to_idx[parent]] = 1.0

        x = torch.from_numpy(node_features)  # [N, n_cats]

        # ------------------------------------------------------------------
        # Node labels: primary intercell parent category (integer)
        # -1 for unannotated proteins
        # ------------------------------------------------------------------
        y = torch.full((num_nodes,), -1, dtype=torch.long)

        # Group by gene symbol and take the first category encountered
        first_cat = (
            intercell[intercell["genesymbol"].isin(protein_to_idx)]
            .groupby("genesymbol")["parent"]
            .first()
        )
        for gene, parent in first_cat.items():
            if parent in cat_to_idx:
                y[protein_to_idx[gene]] = cat_to_idx[parent]

        labelled_count = (y >= 0).sum().item()

        # Remap labels to contiguous integers 0..K-1.
        # INTERCELL_CATEGORIES defines 15 possible classes but not all may appear
        # as a "first" label for any protein in a given interaction subset. Gaps
        # in the label range confuse PyG's num_classes inference and cause the
        # model output dimension to disagree with actual label values.
        if labelled_count > 0:
            unique_labels = y[y >= 0].unique().sort().values
            y_remapped = y.clone()
            for new_idx, old_idx in enumerate(unique_labels.tolist()):
                y_remapped[y == int(old_idx)] = new_idx
            y = y_remapped
            actual_n_cats = len(unique_labels)
        else:
            actual_n_cats = 0

        logger.info(f"[OmniPath] {labelled_count}/{num_nodes} proteins labelled "
                    f"across {actual_n_cats} categories")

        # ------------------------------------------------------------------
        # Save
        # ------------------------------------------------------------------
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                    num_nodes=num_nodes)
        data, slices = self.collate([data])
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(f"[OmniPath] Saved to {self.processed_paths[0]}")

        # Store the protein index alongside for downstream reference
        idx_path = osp.join(self.processed_dir, "protein_index.txt")
        with open(idx_path, "w") as f:
            f.write("\n".join(all_proteins))
