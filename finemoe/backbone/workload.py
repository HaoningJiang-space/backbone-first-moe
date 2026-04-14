import pickle
from copy import deepcopy
from math import ceil
from pathlib import Path


def load_state_dict(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def subset_state_dict(full_state_dict, keep_keys):
    return {key: deepcopy(full_state_dict[key]) for key in keep_keys}


def save_subset_state(path, full_state_dict, keep_keys):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(subset_state_dict(full_state_dict, keep_keys), f)
    return path


def split_sequence_keys(seq_keys, train_fraction):
    split_idx = max(1, min(len(seq_keys) - 1, int(round(len(seq_keys) * train_fraction))))
    return seq_keys[:split_idx], seq_keys[split_idx:]


def build_kfold_splits(seq_keys, cv_folds):
    fold_size = ceil(len(seq_keys) / cv_folds)
    folds = []
    for fold_idx in range(cv_folds):
        start = fold_idx * fold_size
        end = min(len(seq_keys), start + fold_size)
        test_keys = seq_keys[start:end]
        if not test_keys:
            continue
        train_keys = seq_keys[:start] + seq_keys[end:]
        folds.append(
            {
                "fold_index": fold_idx,
                "train_sequences": train_keys,
                "test_sequences": test_keys,
            }
        )
    return folds
