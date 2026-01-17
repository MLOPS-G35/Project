from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    """Configuration object for the clustering pipeline."""

    seed: int = 42
    metrics_path: str = "reports/metrics.json"
    profile: bool = False
    profile_path: str = "reports/profile.pstats"

    # CSV data
    csv_path: str = "data/processed/combined.csv"

    # Clustering config
    id_col: str = "ScanDir ID"
    feature_cols: tuple[str, ...] = (
        "Age",
        "Gender",
        "Handedness",
        "Verbal IQ",
        "Performance IQ",
        "Full4 IQ",
        "ADHD Index",
        "Inattentive",
        "Hyper/Impulsive",
    )
    n_clusters: int = 5
