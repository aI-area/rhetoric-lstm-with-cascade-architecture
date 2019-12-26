from tree_lstm import fun_rhetoric_lstm_unit
from chainer.links.connection import linear
from chainer import link

from tree_lstm import con_multiplicate_unit
import numpy
import chainer.functions as F


class RhetoricLSTMUnit(link.Chain):
    """N-ary TreeLSTM unit.

    This is a N-ary TreeLSTM unit as a chain.
    This link is a fixed-length arguments function, which compounds
    the states of all children nodes into the new states of
    a current (parent) node. *states* denotes the cell state, :math:`c`,
    and the output, :math:`h`, which are produced by this link.
    This link doesn't keep cell and hidden states internally.

    For example, this link is called such as
    ``func(c1, c2, h1, h2, x)`` if the number of children nodes
    was set 2 (``n_ary = 2``), while
    ``func(c1, c2, c3, h1, h2, h3, x)`` if that was 3
    (``n_ary = 3``).
    This function is *dependent* from an order of children nodes
    unlike Child-Sum TreeLSTM.
    Thus, the returns of ``func(c1, c2, h1, h2, x)`` are
    different from those of ``func(c2, c1, h2, h1, x)``.

    Args:
        in_size (int): Dimension of input vectors.
        out_size (int): Dimensionality of cell and output vectors.
        n_ary (int): The number of children nodes in a tree structure.

    """

    def __init__(self, in_size, out_size, normalizer_teq,
                 param_initialization, n_ary=2, boo_test_mode=False):

        assert(n_ary >= 2)

        if boo_test_mode:
            initialw_W_x_top = numpy.random.uniform(
                1, 1, (out_size, out_size)).astype(numpy.float32)
            initialw_W_x_bottom = numpy.random.uniform(
                2, 2, (3 * out_size, out_size)).astype(numpy.float32)
            initialw_W_x_reduction = numpy.random.uniform(
                3, 3, (out_size, in_size)).astype(numpy.float32)
            initialw_W_hl = numpy.random.uniform(
                4, 4, ((3 + n_ary) * out_size, out_size)).astype(numpy.float32)
            initialw_W_hr = numpy.random.uniform(
                5, 5, ((3 + n_ary) * out_size, out_size)).astype(numpy.float32)

            initial_bias_W_x_top = numpy.random.uniform(
                6, 6, out_size).astype(numpy.float32)
        else:
            if param_initialization is not None:
                initialw_W_x_top = param_initialization()
                initialw_W_x_bottom = param_initialization()
                initialw_W_x_reduction = param_initialization()
                initialw_W_hl = param_initialization()
                initialw_W_hr = param_initialization()
                initial_bias_W_x_top = numpy.random.uniform(
                    1, 2, out_size).astype(numpy.float32)
            else:
                initialw_W_x_top = param_initialization
                initialw_W_x_bottom = param_initialization
                initialw_W_x_reduction = param_initialization
                initialw_W_hl = param_initialization
                initialw_W_hr = param_initialization
                initial_bias_W_x_top = param_initialization

        if normalizer_teq is not None:
            super(RhetoricLSTMUnit, self).__init__(
                # bias in forget gates have special initialization seed
                # W_x_top is parameter $W^{(f)}$ used to share in $f_{tn}$
                # and $f_{ts}$ in equations.
                # W_x_bottom is other 3 parameter $W^{(*)}$ in equations.
                W_x_top=linear.Linear(out_size, out_size, nobias=False,
                                      initial_bias=initial_bias_W_x_top,
                                      initialW=initialw_W_x_top),
                W_x_bottom=linear.Linear(out_size, 3 * out_size, nobias=False,
                                         initialW=initialw_W_x_bottom),
                # Construct the bridge that DU vector and hidden state in
                # RST Tree
                # are connected with same dimension
                W_x_reduction=linear.Linear(in_size, out_size, nobias=True,
                                            initialW=initialw_W_x_reduction),

                # Parameter $U_n$ in equations
                W_hl=linear.Linear(out_size, (3 + n_ary) * out_size,
                                   nobias=True, initialW=initialw_W_hl),
                # Parameter $U_s$ in equations
                W_hr=linear.Linear(out_size, (3 + n_ary) * out_size,
                                   nobias=True, initialW=initialw_W_hr),

                # Parameter $K_n$
                W_multi_n=con_multiplicate_unit.MultiplicateUnit(out_size,
                                                                 out_size),
                # Parameter $K_s$
                W_multi_s=con_multiplicate_unit.MultiplicateUnit(out_size,
                                                                 out_size),
                normalizer_teq=normalizer_teq()
            )
        else:
            super(RhetoricLSTMUnit, self).__init__(
                # bias in forget gates have special initialization seed
                # W_x_top is parameter $W^{(f)}$ used to share in $f_{tn}$
                # and $f_{ts}$ in equations.
                # W_x_bottom is other 3 parameter $W^{(*)}$ in equations.
                W_x_top=linear.Linear(out_size, out_size, nobias=False,
                                      initial_bias=initial_bias_W_x_top,
                                      initialW=initialw_W_x_top),
                W_x_bottom=linear.Linear(out_size, 3 * out_size, nobias=False,
                                         initialW=initialw_W_x_bottom),
                # Construct the bridge that DU vector and hidden state in
                # RST Tree
                # are connected with same dimension
                W_x_reduction=linear.Linear(in_size, out_size, nobias=True,
                                            initialW=initialw_W_x_reduction),

                # Parameter $U_n$ in equations
                W_hl=linear.Linear(out_size, (3 + n_ary) * out_size,
                                   nobias=True, initialW=initialw_W_hl),
                # Parameter $U_s$ in equations
                W_hr=linear.Linear(out_size, (3 + n_ary) * out_size,
                                   nobias=True, initialW=initialw_W_hr),

                # Parameter $K_n$
                W_multi_n=con_multiplicate_unit.MultiplicateUnit(out_size,
                                                                 out_size),
                # Parameter $K_s$
                W_multi_s=con_multiplicate_unit.MultiplicateUnit(out_size,
                                                                 out_size)
            )
            self.normalizer_teq=normalizer_teq
        self.n_ary = n_ary

    def __call__(self, *cshsx):
        """Returns new cell state and output of N-ary TreeLSTM.

        Args:
            cshsx (list of :class:`~chainer.Variable`): Arguments which include
                all cell vectors and all output vectors of fixed-length
                children, and an input vector. The number of arguments must be
                same as ``n_ary * 2 + 1``.

        Returns:
            tuple of ~chainer.Variable: Returns :math:`(c_{new}, h_{new})`,
                where :math:`c_{new}` represents new cell state vector,
                and :math:`h_{new}` is new output vector.

        """

        # memory cells in nucleus and satellite child unit
        cs = cshsx[:self.n_ary]
        # hidden states in nucleus and satellite child unit
        hs = cshsx[self.n_ary:-2]
        rel = cshsx[-2]

        x_parent = cshsx[-1]
        # Connect DU vector in DU classifier with hidden state in RST Tree
        # if x_parent.shape[1] != cs[0].shape[1]:
        x_parent = self.W_x_reduction(x_parent)

        # Combine equation in all gates
        tree_lstm_in = \
            F.concat((self.W_x_bottom(x_parent),
                      F.concat((self.W_x_top(x_parent),
                                self.W_x_top(x_parent)), axis=1)

                      ), axis=1)

        tree_lstm_in += self.W_hl(self.W_multi_n(*(rel, hs[0])))
        tree_lstm_in += self.W_hr(self.W_multi_s(*(rel, hs[1])))
        if self.normalizer_teq is not None:
            tree_lstm_in = self.normalizer_teq(tree_lstm_in)

        return fun_rhetoric_lstm_unit\
            .tree_lstm(*(cs + (tree_lstm_in, )))

    def _pad_zero_nodes(self, vs, shape, dtype='f'):
        if any(v is None for v in vs):
            zero = self.xp.zeros(shape, dtype=dtype)
            return tuple(zero if v is None else v for v in vs)
        else:
            return vs
