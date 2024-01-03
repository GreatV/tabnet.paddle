import paddle
import numpy as np
import pytest
from pytorch_tabnet.metrics import UnsupervisedLoss, UnsupervisedLossNumpy


@pytest.mark.parametrize(
    "y_pred,embedded_x,obf_vars",
    [
        (
            np.random.uniform(low=-2, high=2, size=(20, 100)),
            np.random.uniform(low=-2, high=2, size=(20, 100)),
            np.random.choice([0, 1], size=(20, 100), replace=True),
        ),
        (
            np.random.uniform(low=-2, high=2, size=(30, 50)),
            np.ones((30, 50)),
            np.random.choice([0, 1], size=(30, 50), replace=True),
        ),
    ],
)
def test_equal_losses(y_pred, embedded_x, obf_vars):
    numpy_loss = UnsupervisedLossNumpy(
        y_pred=y_pred, embedded_x=embedded_x, obf_vars=obf_vars
    )
    torch_loss = UnsupervisedLoss(
        y_pred=paddle.to_tensor(data=y_pred, dtype="float64"),
        embedded_x=paddle.to_tensor(data=embedded_x, dtype="float64"),
        obf_vars=paddle.to_tensor(data=obf_vars, dtype="float64"),
    )
    assert np.isclose(numpy_loss, torch_loss.detach().numpy())
