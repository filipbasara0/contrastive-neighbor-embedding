# Contrastive Neighbor Embedding (CNE)

Contrastive Neighbor Embedding (CNE) is a custom algorithm for non-linear dimensionality reduction. It leverages k-nearest neighbors as positive labels to create a contrastive learning framework that preserves local structures in high-dimensional data.

The project is in infant stages and is not optimized for large amounts of data.

## Features

- **Dimensionality Reduction**: Reduces high-dimensional data to lower dimensions while preserving local structures.
- **Contrastive Learning**: Uses a contrastive loss function to learn embeddings.
- **EMA Updates**: Incorporates Exponential Moving Average (EMA) updates for stable learning.
- **Customizable**: Allows for various hyperparameter settings including number of neighbors, learning rate, and regularization strength.

## Installation

To install the necessary dependencies, run:

```bash
pip install torch numpy scikit-learn
```

## Usage

### Fitting the Model

```python
import numpy as np
from cne import ContrastiveNeighborEmbedding

# Sample data
X = np.random.rand(200, 100)  # 200 samples with 100 features each

# Initialize and fit the model
cne = ContrastiveNeighborEmbedding(n_components=2, n_neighbors=15, max_epochs=50)
cne.fit(X)

# Transform the data
embedding = cne.transform()

# Visualize the embedding
import matplotlib.pyplot as plt
plt.scatter(embedding[:, 0], embedding[:, 1])
plt.title("2D Embedding of High-Dimensional Data")
plt.show()
```

### Parameters

- `n_components` (int): Number of dimensions for the output embedding.
- `n_neighbors` (int): Number of nearest neighbors to consider for constructing the graph.
- `temp` (float): Temperature parameter for the contrastive loss.
- `learning_rate` (float): Learning rate for the optimizer.
- `scale` (float): Initial scale for the Cauchy kernel.
- `max_epochs` (int): Maximum number of training epochs.
- `batch_size` (int): Batch size for training.
- `gamma` (float): Initial gamma value for EMA updates.
- `standardize_data` (bool): Whether to standardize data before training.
- `lambda_reg` (float): Regularization strength.
- `warmup_epochs` (int): Number of warm-up epochs for training.
- `verbose` (bool): Whether to print training progress.

## API Reference

### `ContrastiveNeighborEmbedding`

#### `__init__(self, n_components=2, n_neighbors=40, temp=20, learning_rate=0.2, scale=1.0, max_epochs=100, batch_size=300, gamma=0.9, standardize_data=False, lambda_reg=0.0001, warmup_epochs=40, verbose=True)`

Constructor for the CNE model.

#### `fit(self, X: np.ndarray) -> None`

Fits the model to the data `X`.

- `X` (np.ndarray): Input data of shape `(n_samples, n_features)`.

#### `transform(self) -> torch.Tensor`

Transforms the input data into the learned low-dimensional space.

## Contributing

Contributions are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
