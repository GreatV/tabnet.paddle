import paddle

"""
Other possible implementations:
https://github.com/KrisKorrel/sparsemax-pytorch/blob/master/sparsemax.py
https://github.com/msobroza/SparsemaxPytorch/blob/master/mnist/sparsemax.py
https://github.com/vene/sparse-structured-attention/blob/master/pytorch/torchsparseattn/sparsemax.py
"""


def _make_ix_like(input, dim=0):
    d = input.shape[dim]
    rho = paddle.arange(start=1, end=d + 1, dtype=input.dtype)
    view = [1] * input.dim()
    view[0] = -1
    x = rho.reshape(view)
    perm_0 = list(range(x.ndim))
    perm_0[0] = dim
    perm_0[dim] = 0
    return x.transpose(perm=perm_0)


class SparsemaxFunction(paddle.autograd.PyLayer):
    """
    An implementation of sparsemax (Martins & Astudillo, 2016). See
    :cite:`DBLP:journals/corr/MartinsA16` for detailed description.
    By Ben Peters and Vlad Niculae
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        """sparsemax: normalizing sparse transform (a la softmax)

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
        input : torch.Tensor
            any shape
        dim : int
            dimension along which to apply sparsemax

        Returns
        -------
        output : torch.Tensor
            same shape as input

        """
        ctx.dim = dim
        max_val = input.max(axis=dim, keepdim=True)
        input -= max_val
        tau, supp_size = SparsemaxFunction._threshold_and_support(input, dim=dim)
        output = paddle.clip(x=input - tau, min=0)
        ctx.save_for_backward(supp_size, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        supp_size, output = ctx.saved_tensors
        dim = ctx.dim
        grad_input = grad_output.clone()
        grad_input[output == 0] = 0
        v_hat = grad_input.sum(axis=dim) / supp_size.squeeze()
        v_hat = v_hat.unsqueeze(axis=dim)
        grad_input = paddle.where(
            condition=output != 0, x=grad_input - v_hat, y=grad_input
        )
        return grad_input, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        """Sparsemax building block: compute the threshold

        Parameters
        ----------
        input: torch.Tensor
            any dimension
        dim : int
            dimension along which to apply the sparsemax

        Returns
        -------
        tau : torch.Tensor
            the threshold value
        support_size : torch.Tensor

        """
        input_srt, _ = paddle.sort(descending=True, x=input, axis=dim), paddle.argsort(
            descending=True, x=input, axis=dim
        )
        input_cumsum = input_srt.cumsum(axis=dim) - 1
        rhos = _make_ix_like(input, dim)
        support = rhos * input_srt > input_cumsum
        support_size = support.sum(axis=dim).unsqueeze(axis=dim)
        tau = input_cumsum.take_along_axis(axis=dim, indices=support_size - 1)
        tau /= support_size
        return tau, support_size


sparsemax = SparsemaxFunction.apply


class Sparsemax(paddle.nn.Layer):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Sparsemax, self).__init__()

    def forward(self, input):
        return sparsemax(input, self.dim)


class Entmax15Function(paddle.autograd.PyLayer):
    """
    An implementation of exact Entmax with alpha=1.5 (B. Peters, V. Niculae, A. Martins). See
    :cite:`https://arxiv.org/abs/1905.05702 for detailed description.
    Source: https://github.com/deep-spin/entmax
    """

    @staticmethod
    def forward(ctx, input, dim=-1):
        ctx.dim = dim
        max_val, _ = input.max(dim=dim, keepdim=True)
        input = input - max_val
        input = input / 2
        tau_star, _ = Entmax15Function._threshold_and_support(input, dim)
        output = paddle.clip(x=input - tau_star, min=0) ** 2
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (Y,) = ctx.saved_tensors
        gppr = Y.sqrt()
        dX = grad_output * gppr
        q = dX.sum(axis=ctx.dim) / gppr.sum(axis=ctx.dim)
        q = q.unsqueeze(axis=ctx.dim)
        dX -= q * gppr
        return dX, None

    @staticmethod
    def _threshold_and_support(input, dim=-1):
        Xsrt, _ = paddle.sort(descending=True, x=input, axis=dim), paddle.argsort(
            descending=True, x=input, axis=dim
        )
        rho = _make_ix_like(input, dim)
        mean = Xsrt.cumsum(axis=dim) / rho
        mean_sq = (Xsrt**2).cumsum(axis=dim) / rho
        ss = rho * (mean_sq - mean**2)
        delta = (1 - ss) / rho
        delta_nz = paddle.clip(x=delta, min=0)
        tau = mean - paddle.sqrt(x=delta_nz)
        support_size = (tau <= Xsrt).sum(axis=dim).unsqueeze(axis=dim)
        tau_star = tau.take_along_axis(axis=dim, indices=support_size - 1)
        return tau_star, support_size


class Entmoid15(paddle.autograd.PyLayer):
    """A highly optimized equivalent of lambda x: Entmax15([x, 0])"""

    @staticmethod
    def forward(ctx, input):
        output = Entmoid15._forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def _forward(input):
        input, is_pos = abs(input), input >= 0
        tau = (input + paddle.sqrt(x=paddle.nn.functional.relu(x=8 - input**2))) / 2
        tau.masked_fill_(tau <= input, 2.0)
        y_neg = 0.25 * paddle.nn.functional.relu(x=tau - input) ** 2
        return paddle.where(condition=is_pos, x=1 - y_neg, y=y_neg)

    @staticmethod
    def backward(ctx, grad_output):
        return Entmoid15._backward(ctx.saved_tensors[0], grad_output)

    @staticmethod
    def _backward(output, grad_output):
        gppr0, gppr1 = output.sqrt(), (1 - output).sqrt()
        grad_input = grad_output * gppr0
        q = grad_input / (gppr0 + gppr1)
        grad_input -= q * gppr0
        return grad_input


entmax15 = Entmax15Function.apply
entmoid15 = Entmoid15.apply


class Entmax15(paddle.nn.Layer):
    def __init__(self, dim=-1):
        self.dim = dim
        super(Entmax15, self).__init__()

    def forward(self, input):
        return entmax15(input, self.dim)
