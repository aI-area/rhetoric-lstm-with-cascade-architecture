from chainer import function
from chainer.utils import type_check


def _as_mat(x):
    if x.ndim == 2:
        return x
    return x.reshape(len(x), -1)


class MultiplicateUnitFun(function.Function):
    def forward(self, inputs):
        x = _as_mat(inputs[0])
        y = _as_mat(inputs[1])

        if not type_check.same_types(*inputs):
            raise ValueError('numpy and cupy must not be used together\n'
                             'type(x): {0}, type(y): {1}'
                             .format(type(x), type(y)))

        # dot是矩阵乘法，并不是Hadarma乘积（hadarma既点乘）
        z = (x * y).astype(x.dtype, copy=False)
        # Output of tuple is expected
        return z,

    def backward(self, inputs, grad_outputs):
        x = _as_mat(inputs[0])
        y = _as_mat(inputs[1])
        goutput = grad_outputs[0]

        gx = (goutput * y).astype(x.dtype, copy=False).reshape(inputs[0].shape)
        gy = (goutput * x).astype(y.dtype, copy=False).reshape(inputs[1].shape)
        return gx, gy


def multiplicate(*inputs):
    return MultiplicateUnitFun()(*inputs)
