import torch
import numpy as np
from sklearn.neighbors import kneighbors_graph

SEED = 0

def set_seed(seed: int = SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)

def update_gamma(k: int, K: int, beta_base: float) -> float:
    k = torch.tensor(k, dtype=torch.float32)
    K = torch.tensor(K, dtype=torch.float32)
    beta = 1 - (1 - beta_base) * (torch.cos(torch.pi * k / K) + 1) / 2
    return beta.item()

def build_graph(data: np.ndarray, n_neighbors: int = 40) -> np.ndarray:
    connectivity = kneighbors_graph(data, n_neighbors=n_neighbors,
                                    mode='connectivity', include_self=True)
    return connectivity.toarray()

def get_labels_from_graph(n: int, connectivity: np.ndarray) -> torch.Tensor:
    labels = torch.ones((n, n)) * -1
    labels.fill_diagonal_(0)
    labels[connectivity == 1] = 1
    return labels
