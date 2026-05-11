"""Hierarchical clustering on the residual correlation matrix."""

from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tqdm import tqdm
from loguru import logger

from src.models.covariance import load_covariance, cov2corr, COV_DIR
from src.config import cfg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CLUSTERS_DIR = PROJECT_ROOT / cfg.clustering.output_dir

def mantegna_dist(C):
    """Mantegna (1999) distance matrix: d_ij = sqrt(0.5*(1 - C_ij))."""
    return np.sqrt(np.clip(0.5 * (1 - C), 0, None))


def cluster_snapshot(Sigma, symbols, threshold=cfg.clustering.k_target):
    """
    Hierarchical cluster assignments for one rebalancing date.

    Computes the Mantegna distance matrix, runs single-linkage hierarchical
    clustering, and cuts the dendrogram at a distance threshold. If threshold
    is None, uses the median linkage distance which adapts to each date.

    Parameters
    ----------
    Sigma     : np.ndarray (N, N) cleaned covariance matrix
    symbols   : list[str]
    threshold : float or None. Dendrogram cut height.
 
    Returns
    -------
    pl.DataFrame: symbol | cluster
    """
    C, _ = cov2corr(Sigma)
    D = mantegna_dist(C)
    np.fill_diagonal(D, 0)
    condensed = squareform(D, checks=False)
    Z = linkage(condensed, method=cfg.clustering.linkage_method)
    K_target = threshold if threshold is not None else cfg.clustering.k_target
    labels = fcluster(Z, t=K_target, criterion='maxclust')
    labels = labels - 1
    df = pl.DataFrame({"symbol": symbols, "cluster": labels.tolist()})

    return df


def compute_all_clusters(threshold=cfg.clustering.k_target):
    """
    Compute and save hierarchical cluster assignments for all rebalancing dates.
    Skips dates already computed. Safe to re-run.
    """
    CLUSTERS_DIR.mkdir(parents=True, exist_ok=True)
    dates = sorted([p.stem for p in COV_DIR.glob("*.parquet")])
    logger.info(f"Computing hierarchical clusters for {len(dates)} dates.")

    for d in tqdm(dates, desc="Hierarchical clustering"):
        path = CLUSTERS_DIR / f"{d}.parquet"
        if path.exists():
            continue
        Sigma, symbols = load_covariance(d)
        cluster_snapshot(Sigma, symbols, threshold).write_parquet(path)

    logger.info("Done.")


def load_clusters(rebal_date):
    """Load cluster assignments for a given rebalancing date."""
    path = CLUSTERS_DIR / f"{rebal_date}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No clusters for {rebal_date}.")
    return pl.read_parquet(path)


if __name__ == "__main__":
    compute_all_clusters()