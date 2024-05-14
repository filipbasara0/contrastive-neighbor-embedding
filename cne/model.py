import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR

from cne.utils import set_seed, update_gamma, build_graph, get_labels_from_graph


class Model(nn.Module):
    def __init__(self, init_data: torch.Tensor):
        super().__init__()
        self.embedding = nn.Parameter(init_data)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding[indices]


class EMAWrapper(nn.Module):
    def __init__(self, init_data: torch.Tensor, tau: float, b: float, scale: float):
        super().__init__()
        self.online_model = Model(init_data)
        self.ema_model = copy.deepcopy(self.online_model)
        self.ema_model.requires_grad_(False)
        self.tau = nn.Parameter(torch.ones([]) * tau)
        self.b = nn.Parameter(torch.ones([]) * b)
        self.scale = nn.Parameter(torch.tensor([scale] * len(init_data)))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.online_model(indices)
    
    @torch.inference_mode()
    def get_target_preds(self, indices: torch.Tensor) -> torch.Tensor:
        return self.ema_model(indices)
        
    def get_embedding(self) -> torch.Tensor:
        return self.ema_model.embedding
    
    def update_params(self, gamma: float) -> None:
        with torch.no_grad():
            for o_param, t_param in zip(self.online_model.parameters(),
                                        self.ema_model.parameters()):
                t_param.data.lerp_(o_param.data, 1. - gamma)


def cauchy_kernel(x1: torch.Tensor, x2: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    distances = torch.sum((x1[:, None] - x2[None, :]) ** 2, dim=2)
    distances *= scale.exp().clamp(0.6, 1.2)
    cauchy_similarity = 1 / (1 + distances)
    return cauchy_similarity


def contrastive_loss(x: torch.Tensor, labels: torch.Tensor, tau: nn.Parameter, b: nn.Parameter,
                     scale: torch.Tensor, max_tau: float, epoch: int, lambda_reg: float = 0.0001,
                     warmup_epochs: int = 40) -> torch.Tensor:
    n = x.size(0)
    l2_reg = lambda_reg * torch.sum(x ** 2)
    min_tau = 10.0
    logits = cauchy_kernel(x, x, scale) * tau.exp().clamp(min_tau, max_tau) + b
    if epoch < warmup_epochs:
        loss = -torch.sum(nn.functional.logsigmoid(labels * logits)) / n
        return loss + l2_reg

    probs = logits * labels
    probs = torch.sigmoid(probs)

    conf_penalty = torch.where(labels > 0, 1.0, probs)
    loss = -torch.sum(conf_penalty * nn.functional.logsigmoid(labels * logits)) / n
    return loss + l2_reg


class ContrastiveNeighboorEmbedding(nn.Module):
    def __init__(self, n_components: int = 2, n_neighbors: int = 40, temp: float = 20,
                 learning_rate: float = 0.2, scale: float = 1.0, max_epochs: int = 100,
                 batch_size: int = 300, gamma: float = 0.9, standardize_data: bool = False,
                 lambda_reg: float = 0.0001, warmup_epochs: int = 40, random_state: int = 42,
                 verbose: bool = True):
        super().__init__()
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.temp, self.bias, self.max_temp = np.log(temp), -temp, temp
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.scale = np.log(scale)
        self.embedding = None
        self.init_ema_gamma = gamma
        self.standardize_data = standardize_data
        self.verbose = verbose
        self.lambda_reg = lambda_reg
        self.warmup_epochs = warmup_epochs
        set_seed(random_state)

    def fit(self, X: np.ndarray) -> None:
        if self.standardize_data:
            from sklearn.preprocessing import StandardScaler
            X = StandardScaler().fit_transform(X)

        data = torch.tensor(X, dtype=torch.float32)
        init_data = torch.randn((len(data), self.n_components), dtype=torch.float32)

        self.batch_size = min(len(data), self.batch_size)

        self.model = EMAWrapper(init_data, self.temp, self.bias, self.scale)

        params = list(self.model.online_model.parameters()) + [self.model.tau, self.model.b]

        optimizer = optim.Adam(params, lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=0.001)

        connectivity = build_graph(data.numpy(), n_neighbors=self.n_neighbors)
        self.labels = get_labels_from_graph(len(data), connectivity)
        
        total_num_steps = (len(data) + self.batch_size - 1) // self.batch_size * self.max_epochs
        gamma = self.init_ema_gamma
        for epoch in range(self.max_epochs):
            gamma = self._train_epoch(data, optimizer, epoch, scheduler, total_num_steps, gamma)
            scheduler.step()
        self.embedding = self.model.get_embedding().detach().numpy()

    def transform(self) -> torch.Tensor:
        if self.embedding is None:
            raise RuntimeError("The model has not been trained yet!")
        return self.embedding

    def _train_epoch(self, data: torch.Tensor, optimizer: optim.Optimizer, epoch: int,
                     scheduler: CosineAnnealingLR, total_num_steps: int, gamma: float) -> float:
        n = len(data)
        permutation = torch.randperm(n)
        epoch_loss = 0.0
        for i in range(0, n, self.batch_size):
            indices = permutation[i:i+self.batch_size]
            batch_labels = self.labels[indices][:, indices]
            scales = self.model.scale[indices]
            optimizer.zero_grad()
            embedding = self.model(indices)
            loss = contrastive_loss(embedding, batch_labels, self.model.tau, self.model.b,
                                    scales, self.max_temp, epoch, self.lambda_reg, self.warmup_epochs)
            loss += contrastive_loss(self.model.get_target_preds(indices), batch_labels, self.model.tau,
                                     self.model.b, scales, self.max_temp, epoch, self.lambda_reg, self.warmup_epochs)
            loss /= 2
            loss.backward()
            optimizer.step()
            self.model.update_params(gamma)
            gamma = update_gamma(i // self.batch_size + 1 + epoch * (n // self.batch_size),
                                 total_num_steps, self.init_ema_gamma)

            epoch_loss += loss.item()

        if self.verbose and epoch % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / n}, Tau: {self.model.tau.exp().item()},'
                  f'Scale: {self.model.scale.exp().mean().item()}, Bias: {self.model.b.item()},'
                  f'LR: {scheduler.get_last_lr()[0]}, Gamma: {gamma}')

        return gamma