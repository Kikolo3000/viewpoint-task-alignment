"""Load pre-extracted features and labels from the Zenodo artifacts directory.

Replaces the simpg-dependent loaders used during mining/feature-extraction
with a thin loader over parquet + CSV artifacts. Piece alignment mirrors
the original pipeline.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

CORPORA = ["essen", "the_session", "symbtr", "dance9"]
VP_CONFIGS = ["M", "R", "B", "MR", "MB", "RB", "MRB"]
FEATURE_GROUPS = ["structural", "musical", "bop", "global"]

FEATURE_CONFIGS = {
    "global": ["D"],
    "structural": ["A"],
    "musical": ["B"],
    "bop": ["C"],
    "pattern_all": ["A", "B", "C"],
    "combined": ["A", "B", "C", "D"],
    "combined_no_bop": ["A", "B", "D"],
}

_LETTER_TO_GROUP = {"A": "structural", "B": "musical", "C": "bop", "D": "global"}


def artifact_features_path(
    corpus: str,
    vp_config: str,
    group: str,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    return Path(artifacts_dir) / "features" / f"{corpus}_{vp_config}_{group}.parquet"


def artifact_labels_path(
    corpus: str,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    return Path(artifacts_dir) / "labels" / f"{corpus}_labels.csv"


def load_feature_groups(
    corpus: str,
    vp_config: str,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> dict[str, pd.DataFrame]:
    """Load the four feature groups for a corpus × VP config."""
    groups: dict[str, pd.DataFrame] = {}
    for name in FEATURE_GROUPS:
        path = artifact_features_path(corpus, vp_config, name, artifacts_dir)
        if path.exists():
            df = pd.read_parquet(path)
            groups[name] = df
            logger.info("Loaded %s: %d × %d from %s", name, *df.shape, path)
        else:
            logger.warning("Feature group '%s' not found at %s", name, path)
    return groups


def load_labels(
    corpus: str,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
) -> pd.Series:
    """Load labels CSV as a Series indexed by piece title."""
    path = artifact_labels_path(corpus, artifacts_dir)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    df = pd.read_csv(path)
    if "piece" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Labels CSV {path} must have columns 'piece' and 'label'; got {list(df.columns)}"
        )
    return pd.Series(df["label"].values, index=df["piece"].values, name="label")


def align_pieces(
    feature_groups: dict[str, pd.DataFrame],
    labels: pd.Series,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Intersect feature-group indices and labels; return aligned groups + list."""
    if not feature_groups:
        raise ValueError("feature_groups is empty; nothing to align")

    indices = [df.index for df in feature_groups.values()]
    common_idx = indices[0]
    for idx in indices[1:]:
        common_idx = common_idx.intersection(idx)
    common_idx = common_idx.intersection(labels.index)

    aligned = {k: df.loc[common_idx] for k, df in feature_groups.items()}
    aligned_labels = labels.loc[common_idx].tolist()
    logger.info("Aligned to %d common pieces", len(common_idx))
    return aligned, aligned_labels


def filter_classes(
    feature_groups: dict[str, pd.DataFrame],
    labels: list[str],
    min_count: int = 10,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Drop classes with fewer than `min_count` samples."""
    counts = Counter(labels)
    keep = {cls for cls, cnt in counts.items() if cnt >= min_count}
    dropped = {cls: cnt for cls, cnt in counts.items() if cnt < min_count}
    if dropped:
        logger.info("Dropping classes with < %d samples: %s", min_count, dropped)

    valid_mask = [lbl in keep for lbl in labels]
    valid_labels = [lbl for lbl, m in zip(labels, valid_mask) if m]

    first_df = next(iter(feature_groups.values()))
    valid_idx = first_df.index[valid_mask]
    filtered = {k: df.loc[valid_idx] for k, df in feature_groups.items()}

    logger.info(
        "After filtering: %d pieces, %d classes",
        len(valid_labels),
        len(set(valid_labels)),
    )
    return filtered, valid_labels


def assemble_features(
    groups: dict[str, pd.DataFrame],
    config_name: str,
) -> pd.DataFrame:
    """Concatenate the feature groups requested by `config_name` along columns."""
    if config_name not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature config: {config_name}")

    dfs = []
    for letter in FEATURE_CONFIGS[config_name]:
        key = _LETTER_TO_GROUP[letter]
        if key not in groups:
            raise ValueError(
                f"Feature group '{key}' not available for config '{config_name}'"
            )
        dfs.append(groups[key])

    combined = pd.concat(dfs, axis=1)
    logger.info("Config '%s': %d pieces × %d features", config_name, *combined.shape)
    return combined


def load_corpus(
    corpus: str,
    vp_config: str,
    artifacts_dir: Path | str = DEFAULT_ARTIFACTS_DIR,
    min_class_count: int = 10,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Convenience: load + align + filter in one call."""
    groups = load_feature_groups(corpus, vp_config, artifacts_dir)
    if not groups:
        raise FileNotFoundError(
            f"No features found for {corpus}/{vp_config} under {artifacts_dir}"
        )
    labels = load_labels(corpus, artifacts_dir)
    groups, label_list = align_pieces(groups, labels)
    groups, label_list = filter_classes(groups, label_list, min_count=min_class_count)
    return groups, label_list
