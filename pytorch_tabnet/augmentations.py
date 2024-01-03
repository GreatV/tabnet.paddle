import paddle
from pytorch_tabnet.utils import define_device
import numpy as np


class RegressionSMOTE:
    """
    Apply SMOTE

    This will average a percentage p of the elements in the batch with other elements.
    The target will be averaged as well (this might work with binary classification
    and certain loss), following a beta distribution.
    """

    def __init__(self, device_name="auto", p=0.8, alpha=0.5, beta=0.5, seed=0):
        """"""
        self.seed = seed
        self._set_seed()
        self.device = define_device(device_name)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        if p < 0.0 or p > 1.0:
            raise ValueError("Value of p should be between 0. and 1.")

    def _set_seed(self):
        paddle.seed(seed=self.seed)
        np.random.seed(self.seed)
        return

    def __call__(self, X, y):
        batch_size = X.shape[0]
        random_values = paddle.rand(shape=batch_size)
        idx_to_change = random_values < self.p
        np_betas = np.random.beta(self.alpha, self.beta, batch_size) / 2 + 0.5
        random_betas = (
            paddle.to_tensor(data=np_betas).astype(dtype="float32")
        )
        index_permute = paddle.randperm(n=batch_size)
        X[idx_to_change] = random_betas[idx_to_change, None] * X[idx_to_change]
        X[idx_to_change] += (1 - random_betas[idx_to_change, None]) * X[index_permute][
            idx_to_change
        ].reshape(X[idx_to_change].shape)
        y[idx_to_change] = random_betas[idx_to_change, None] * y[idx_to_change]
        y[idx_to_change] += (1 - random_betas[idx_to_change, None]) * y[index_permute][
            idx_to_change
        ].reshape(y[idx_to_change].shape)
        return X, y


class ClassificationSMOTE:
    """
    Apply SMOTE for classification tasks.

    This will average a percentage p of the elements in the batch with other elements.
    The target will stay unchanged and keep the value of the most important row in the mix.
    """

    def __init__(self, device_name="auto", p=0.8, alpha=0.5, beta=0.5, seed=0):
        """"""
        self.seed = seed
        self._set_seed()
        self.device = define_device(device_name)
        self.alpha = alpha
        self.beta = beta
        self.p = p
        if p < 0.0 or p > 1.0:
            raise ValueError("Value of p should be between 0. and 1.")

    def _set_seed(self):
        paddle.seed(seed=self.seed)
        np.random.seed(self.seed)
        return

    def __call__(self, X, y):
        batch_size = X.shape[0]
        random_values = paddle.rand(shape=batch_size)
        idx_to_change = random_values < self.p
        np_betas = np.random.beta(self.alpha, self.beta, batch_size) / 2 + 0.5
        random_betas = (
            paddle.to_tensor(data=np_betas).astype(dtype="float32")
        )
        index_permute = paddle.randperm(n=batch_size)
        X[idx_to_change] = random_betas[idx_to_change, None] * X[idx_to_change]
        X[idx_to_change] += (1 - random_betas[idx_to_change, None]) * X[index_permute][
            idx_to_change
        ].reshape(X[idx_to_change].shape)
        return X, y
