import chainer.links as link
import chainer
import numpy as np


class RecursiveTreeLSTMNet(chainer.Chain):
    def __init__(self, n_vocab, n_units, hidden_unit, n_label, data_carrier,
                 initial_vector=None):
        super(RecursiveTreeLSTMNet, self).__init__(
            embed=link.EmbedID(n_vocab, n_units, initialW=initial_vector),
            treelstm=link.NaryTreeLSTM(n_units, hidden_unit, n_ary=2),
            w=link.Linear(hidden_unit, n_label))
        self.hidden_unit = hidden_unit
        self.data_carrier = data_carrier

    def leaf(self, x):
        word = self.data_carrier.array([x], np.int32)
        emb = self.embed(word)
        zero = self.xp.zeros((1, self.hidden_unit), 'f')
        c, h = self.treelstm(zero, zero, zero, zero, emb)
        return c, h

    def node(self, left, right):
        c_left, h_left = left
        c_right, h_right = right
        zero = self.xp.zeros((1, self.initialW.shape[0]), 'f')
        c, h = self.treelstm(c_left, c_right, h_left, h_right, zero)
        return c, h

    def label(self, v):
        h = v[1]
        return self.w(h)
