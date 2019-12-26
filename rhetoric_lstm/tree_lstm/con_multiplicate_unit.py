from tree_lstm import fun_multiplicate_unit
from chainer import initializers
from chainer import link
from chainer import variable


class MultiplicateUnit(link.Link):
    def __init__(self, in_size, out_size=None,
                 initial_wx=None):
        super(MultiplicateUnit, self).__init__()

        if out_size is None:
            in_size, out_size = None, in_size
        self.out_size = out_size

        with self.init_scope():
            self.Wx \
                = variable.Parameter(initializers._get_initializer(initial_wx))
            if in_size is not None:
                self._initialize_params_wx(in_size)

    def _initialize_params_wx(self, in_size):
        self.Wx.initialize((self.out_size, in_size))

    def __call__(self, *cshsx):
        """Applies the linear layer.

        Args:
            x (~chainer.Variable): Batch of input vectors.

        Returns:
            ~chainer.Variable: Output of the linear layer.

        """
        if self.Wx.data is None:
            self._initialize_params_wx(cshsx[0].size // cshsx[0].shape[0])

        return fun_multiplicate_unit.multiplicate(*(cshsx[0], cshsx[1]))