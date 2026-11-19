# Sample dataset shipped with CoDARec

A small, self-contained implicit-feedback dataset so that the repository runs
out of the box. The full production data behind the experiments cannot be
published, so this sample split is bundled for reproducibility.

## Files

| File | Shape | Dtype | Description |
| --- | --- | --- | --- |
| `train_list.npy` | `(1014486, 2)` | `int64` | `(user_id, item_id)` pairs |
| `valid_list.npy` | `(246676, 2)` | `int64` | validation ground truth |
| `test_list.npy` | `(141574, 2)` | `int64` | test ground truth |
| `item_emb.npy` | `(34395, 64)` | `float32` | item embeddings |

## Two user blocks per file

Each split file concatenates **two consecutive user blocks**:

* first half — users `0 … 20262` (20,263 users)
* second half — users `20262 … 54573`

The reference experiments only use the first block. `load_pairs(...,
keep_first_half=True)` — the default — keeps it, which yields the statistics
reported below: 20,263 users, 34,395 items, 507,243 / 123,338 / 70,787 training,
validation and test interactions (0.0728 % density). Pass
`--no-keep-first-half` to train on the full 54,574 users.

## Checksums (SHA-256)

```
20feb95133f7ae929c57bb770ca06fb149d8816d166aeb8690f54fe32168c0ba  item_emb.npy
5686d82a84a191944abb30460a143909b4fde472eed52a1060ccdadfa916b1a6  test_list.npy
957ca71718cb4523945206143d085ecdfe1f07e5616b7ff5e1cc871c17d9e8dc  train_list.npy
bc71e3302baa364d5ce3a692a3baae3f28c03eebccef8df818c4e7c0e72ed8c7  valid_list.npy
```

Verify with:

```bash
cd data/sample_dataset && sha256sum *.npy
```

## Using your own data

Create `data/<your_dataset>/` with the same four file names. Requirements:

* `(user_id, item_id)` pairs are zero-based and contiguous;
* `item_emb.npy` is optional unless you run with `--use-rl`;
* `n_users` / `n_items` are derived from the split files, so no config edit is needed.
