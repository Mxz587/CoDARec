"""Dataset loading for CoDARec."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

SPLIT_FILES = {
    "train": "train_list.npy",
    "valid": "valid_list.npy",
    "test": "test_list.npy",
}
ITEM_EMBEDDING_FILE = "item_emb.npy"


@dataclass
class DataBundle:
    """User-item interaction matrices plus dataset statistics."""

    train: sp.csr_matrix
    valid: sp.csr_matrix
    test: sp.csr_matrix
    n_users: int
    n_items: int
    item_embeddings: Optional[np.ndarray] = None

    def describe(self) -> str:
        density = self.train.nnz / float(self.n_users * self.n_items) * 100
        lines = [
            f"users={self.n_users}  items={self.n_items}",
            f"interactions  train={self.train.nnz}  valid={self.valid.nnz}  "
            f"test={self.test.nnz}",
            f"train density={density:.4f}%",
        ]
        if self.item_embeddings is not None:
            lines.append(
                f"item embeddings={self.item_embeddings.shape} "
                f"({self.item_embeddings.dtype})"
            )
        return "\n".join(lines)


def load_pairs(path: str, keep_first_half: bool = True) -> np.ndarray:
    """Load an ``(n, 2)`` array of ``(user_id, item_id)`` pairs.

    The released sample splits store two consecutive user blocks in a single
    file.  ``keep_first_half`` keeps only the first block, which reproduces the
    subset used in the reported experiments.
    """
    pairs = np.load(path, allow_pickle=True)
    pairs = np.asarray(pairs)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError(f"{path}: expected an (n, 2) array, got {pairs.shape}")
    pairs = pairs.astype(np.int64, copy=False)
    if keep_first_half:
        pairs = pairs[: pairs.shape[0] // 2]
    if pairs.size == 0:
        raise ValueError(f"{path}: no interactions left after truncation")
    return pairs


def _to_matrix(pairs: np.ndarray, n_users: int, n_items: int) -> sp.csr_matrix:
    """Build a binary user-item CSR matrix."""
    matrix = sp.csr_matrix(
        (
            np.ones(pairs.shape[0], dtype=np.float64),
            (pairs[:, 0], pairs[:, 1]),
        ),
        shape=(n_users, n_items),
    )
    # Repeated pairs accumulate to values > 1; keep the matrix binary.
    matrix.data[:] = 1.0
    return matrix


def load_data(
    data_dir: str,
    keep_first_half: bool = True,
    with_item_embeddings: bool = True,
) -> DataBundle:
    """Load the three splits (and optional item embeddings) from ``data_dir``."""
    missing = [
        name for name, filename in SPLIT_FILES.items()
        if not os.path.isfile(os.path.join(data_dir, filename))
    ]
    if missing:
        raise FileNotFoundError(
            f"{data_dir} is missing the split file(s): {', '.join(missing)}"
        )

    splits = {
        name: load_pairs(os.path.join(data_dir, filename), keep_first_half)
        for name, filename in SPLIT_FILES.items()
    }

    n_users = int(max(split[:, 0].max() for split in splits.values())) + 1
    n_items = int(max(split[:, 1].max() for split in splits.values())) + 1

    matrices = {
        name: _to_matrix(split, n_users, n_items) for name, split in splits.items()
    }

    item_embeddings = None
    embedding_path = os.path.join(data_dir, ITEM_EMBEDDING_FILE)
    if with_item_embeddings and os.path.isfile(embedding_path):
        item_embeddings = np.load(embedding_path, allow_pickle=True).astype(np.float32)
        if item_embeddings.shape[0] != n_items:
            raise ValueError(
                f"{embedding_path}: {item_embeddings.shape[0]} rows but the splits "
                f"contain {n_items} items"
            )

    return DataBundle(
        train=matrices["train"],
        valid=matrices["valid"],
        test=matrices["test"],
        n_users=n_users,
        n_items=n_items,
        item_embeddings=item_embeddings,
    )


class InteractionDataset(Dataset):
    """One user history per item, densified on the fly.

    A scipy sparse matrix is accepted so that a large implicit-feedback matrix
    (``n_users x n_items``) is never materialised in memory; only the rows of
    the current batch are converted to dense ``float32`` tensors.
    """

    def __init__(self, data) -> None:
        if sp.issparse(data):
            self._matrix = data.tocsr().astype(np.float32)
            self._tensor = None
            self._n_users, self._n_items = self._matrix.shape
        else:
            tensor = torch.as_tensor(data, dtype=torch.float32)
            if tensor.dim() != 2:
                raise ValueError(
                    f"expected a 2-D interaction matrix, got {tuple(tensor.shape)}"
                )
            self._matrix = None
            self._tensor = tensor
            self._n_users, self._n_items = tensor.shape

    def __getitem__(self, index: int) -> torch.Tensor:
        if self._tensor is not None:
            return self._tensor[index]
        row = self._matrix.getrow(index)
        return torch.from_numpy(np.asarray(row.toarray(), dtype=np.float32).ravel())

    def __len__(self) -> int:
        return self._n_users

    @property
    def n_items(self) -> int:
        return self._n_items


__all__ = [
    "DataBundle",
    "InteractionDataset",
    "ITEM_EMBEDDING_FILE",
    "SPLIT_FILES",
    "load_data",
    "load_pairs",
]
