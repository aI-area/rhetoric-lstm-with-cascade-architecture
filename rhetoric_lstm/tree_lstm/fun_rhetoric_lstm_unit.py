import six

from chainer import cuda
from chainer import function


def _extract_gates(x, n_split=5):
    """Extract gates by split.

    This is different from ``_extract_gates`` in lstm.py,
    which is as follows::

            r = x.reshape((x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])
            return (r[:, :, i] for i in six.moves.range(4))

    In other words, it thinly slices ``x`` and merge them,
    while this thickly slices ``x``.

    """
    r = x.reshape(
        (x.shape[0], n_split, x.shape[1] // n_split) + x.shape[2:])
    return (r[:, i, :] for i in six.moves.range(n_split))


def _sigmoid(x):
    xp = cuda.get_array_module(x)
    half = x.dtype.type(0.5)
    return xp.tanh(x * half) * half + half


def _grad_sigmoid(x):
    return x * (1 - x)


def _grad_tanh(x):
    return 1 - x * x


_preamble = '''
template <typename T> __device__ T sigmoid(T x) {
    const T half = 0.5;
    return tanh(x * half) * half + half;
}
template <typename T> __device__ T grad_sigmoid(T y) { return y * (1 - y); }
template <typename T> __device__ T grad_tanh(T y) { return 1 - y * y; }

#define COMMON_ROUTINE \
    T aa = tanh(a); \
    T ai = sigmoid(i_); \
    T ao = sigmoid(o); \
'''


class TreeLSTM(function.Function):

    """TreeLSTM unit with N forget gates.

    This have variable inputs ``(c1, c2, ..., cN, x)``
    where ``x`` is (3 + N) times larger than each cell.
    Forget gates ``(f1, f2, ..., fN)`` can depend in
    different partitions of ``x[:, 3 * cell_units:]``.
    There are two outputs ``(c, h)``.

    """

    def forward(self, inputs):
        xp = cuda.get_array_module(*inputs)
        cs, x = inputs[:-1], inputs[-1]
        n_ary = len(cs)
        gates = list(_extract_gates(x, 3 + n_ary))
        a, i, o = gates[:3]
        fs = gates[3:]

        self.a = a
        self.i = _sigmoid(i)
        self.o = _sigmoid(o)
        self.fs = [_sigmoid(f) for f in fs]

        self.c = self.a * self.i + sum(f * c for f, c in zip(self.fs, cs))
        h = xp.tanh(self.o * self.c)
        return self.c, h

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        cs, x = inputs[:-1], inputs[-1]
        n_ary = len(cs)
        gc, gh = grad_outputs

        gx = xp.empty_like(x)
        gates = list(_extract_gates(gx, 3 + n_ary))
        ga, gi, go = gates[:3]
        gfs = gates[3:]

        # Consider the case that either gradient is not given
        if gc is None:
            gc = 0
        if gh is None:
            gh = 0

        co = xp.tanh(self.c * self.o)
        tmp = gh * self.o * _grad_tanh(co) + gc

        ga[:] = tmp * self.i
        gi[:] = tmp * self.a * _grad_sigmoid(self.i)
        go[:] = gh * _grad_tanh(co) * self.c * _grad_sigmoid(self.o)

        gcs = []
        for j in six.moves.range(0, n_ary):
            gfs[j][:] = tmp * cs[j] * _grad_sigmoid(self.fs[j])
            gcs.append(tmp * self.fs[j])
        return list(gcs) + [gx]


def tree_lstm(*inputs):
    """TreeLSTM unit as an activation function.

    This function implements TreeLSTM units both for
    N-ary TreeLSTM and Child-Sum TreeLSTM.
    Let the children cell states
    :math:`c_{\\text{1}}, c_{\\text{2}}, \dots, c_{\\text{N}}`,
    and the incoming signal :math:`x`.

    First, the incoming signal :math:`x` is split into (3 + N) arrays
    :math:`a, i, o, f_{\\text{1}}, f_{\\text{2}}, ..., f_{\\text{N}}`
    of the same shapes along the second axis.
    It means that :math:`x` 's second axis must have (3 + N) times
    of the length of each :math:`c_{n}`.

    The splitted input signals are corresponding to:

        - :math:`a` : sources of cell input
        - :math:`i` : sources of input gate
        - :math:`o` : sources of output gate
        - :math:`f_{n}` : sources of forget gate for n-th ary

    Second, it computes outputs as:

    .. math::

        c &= \\tanh(a) \\text{sigmoid}(i) \\\\
          & + c_{\\text{1}} \\text{sigmoid}(f_{\\text{1}}), \\\\
          & + c_{\\text{2}} \\text{sigmoid}(f_{\\text{2}}), \\\\
          & + ..., \\\\
          & + c_{\\text{N}} \\text{sigmoid}(f_{\\text{N}}), \\\\
        h &= \\tanh(c) \\text{sigmoid}(o).

    These are returned as a tuple of (N + 1) variables.

    Args:
        inputs (list of :class:`~chainer.Variable`): Variable arguments which
            include all cell vectors from child-nodes, and an input vector.
            Each of the cell vectors and the input vector is
            :class:`~chainer.Variable`.
            The input vector must have the second dimension whose size
            is (N + 3) times of that of each cell,
            where N denotes the total number of cells.

    Returns:
        tuple: Two :class:`~chainer.Variable` objects ``c`` and ``h``. ``c`` is
            the updated cell state. ``h`` indicates the outgoing signal.

    See the papers for details: `Improved Semantic Representations From \
    Tree-Structured Long Short-Term Memory Networks \
    <http://www.aclweb.org/anthology/P15-1150>`_ and
    `A Fast Unified Model for Parsing and Sentence Understanding \
    <https://arxiv.org/pdf/1603.06021.pdf>`_.

    Tai et al.'s N-Ary TreeLSTM is little extended in
    Bowman et al., and this link is based on
    the variant by Bowman et al.
    Specifically, eq. 10 in Tai et al. only has one :math:`W` matrix
    to be applied to :math:`x`, consistently for all children.
    On the other hand, Bowman et al.'s model has multiple matrices,
    each of which affects the forget gate for each child's cell individually.

    .. admonition:: Example

        Assuming ``y`` is the current input signal, ``c`` is the previous cell
        state, and ``h`` is the previous output signal from an
        ``n_ary_tree_lstm`` function.
        Each of ``y``, ``c`` and ``h`` has ``n_units`` channels.
        Using 2-ary (binary) TreeLSTM,
        most typical preparation of ``x`` is:

        >>> model = chainer.Chain(w=F.Linear(10, 5 * 10),
        ...                       v1=F.Linear(10, 5 * 10),
        ...                       v2=F.Linear(10, 5 * 10),)
        >>> y = np.random.uniform(-1, 1, (4, 10)).astype('f')
        >>> h1 = np.random.uniform(-1, 1, (4, 10)).astype('f')
        >>> h2 = np.random.uniform(-1, 1, (4, 10)).astype('f')
        >>> c1 = np.random.uniform(-1, 1, (4, 10)).astype('f')
        >>> c2 = np.random.uniform(-1, 1, (4, 10)).astype('f')
        >>> x = model.w(y) + model.v1(h1) + model.v2(h2)
        >>> c, h = F.n_ary_tree_lstm(c1, c2, x)

        It corresponds to calculate the input sources
        :math:`a, i, o, f_{\\text{1}}, f_{\\text{2}}`
        from the current input ``y`` and the children's outputs
        ``h1`` and ``h2``. Different parameters are used for different kind of
        input sources.

    """
    return TreeLSTM()(*inputs)
