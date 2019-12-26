import numpy
import cupy as cp
import chainer
import chainer.functions as F
from chainer import reporter
from chainer.links.connection import linear

from model import rst_lstm
from model import thin_stack as ts_teq

class ThinStackRecursiveNet(chainer.Chain):

    def __init__(self,
                 n_label, xp,
                 flo_dropout,
                 flo_input_dp,
                 normalizer_teq, segment_model,
                 SEG_LEARN_MODE,
                 index2word, du_text2vector, du_unit, rst_unit, w2index,
                 rst_none_leaf_node_cst_tree, rst_none_leaf_node_cst_tree_text,

                 model_size, rst_rel2index,

                 du_text2index, x_normalizer_teq, param_initialization,
                 boo_test_mode=False
                 ):

        if boo_test_mode:
            initialw = numpy.random.uniform(
                1, 1, (rst_unit, du_unit)).astype(numpy.float32)
        else:
            if param_initialization is not None:
                initialw = param_initialization()
            else:
                initialw = param_initialization

        if x_normalizer_teq is not None:
            super(ThinStackRecursiveNet, self).__init__(
                classifier = rst_lstm.RstLSTMStack(
                    n_label, xp,
                    flo_dropout,
                    flo_input_dp,
                    normalizer_teq, segment_model,
                    SEG_LEARN_MODE,
                    index2word, du_text2vector, du_unit, rst_unit, w2index,
                    rst_none_leaf_node_cst_tree,
                    rst_none_leaf_node_cst_tree_text,
                    model_size, rst_rel2index, param_initialization,
                    boo_test_mode
                                                  ),
                W_x_reduction=linear.Linear(du_unit, rst_unit, nobias=True,
                                            initialW=initialw),
                x_normalizer_teq=x_normalizer_teq(du_unit),
                )
        else:
            super(ThinStackRecursiveNet, self).__init__(
                classifier = rst_lstm.RstLSTMStack(
                    n_label, xp,
                    flo_dropout,
                    flo_input_dp,
                    normalizer_teq, segment_model,
                    SEG_LEARN_MODE,
                    index2word, du_text2vector, du_unit, rst_unit, w2index,
                    rst_none_leaf_node_cst_tree,
                    rst_none_leaf_node_cst_tree_text,
                    model_size, rst_rel2index, param_initialization,
                    boo_test_mode
                ),
                W_x_reduction=linear.Linear(du_unit, rst_unit, nobias=True,
                                            initialW=initialw),
            )
            self.x_normalizer_teq=None

        self.du_index2text = {v: k for k, v in du_text2index.items()}
        self.du_text2vector = du_text2vector
        self.rst_unit = rst_unit

    def leaf(self, x):
        c, h = self.leaf_treelstm(self.embed(x))
        return c, h

    def node(self, c_left, c_right, h_left, h_right):
        c, h = self.node_treelstm(c_left, c_right, h_left, h_right)
        return c, h

    def label(self, v):
        h = v[1]
        return self.w(h)
        # return self.w(v)

    def concate_array(self, ary_index, dct_leaf_id2ary_id, dct_ary_id2vector,
                      dct_index2vector):
        lst_emb_index = []
        for index in ary_index:
            lst_emb_index.append(dct_leaf_id2ary_id[int(index.data)])
        ary_emb = dct_ary_id2vector[lst_emb_index]
        return ary_emb

    def __call__(self, *inputs, test_mode=False, inference_mode=False):
        if test_mode or inference_mode:
            inputs = inputs[0]

        batch = int(len(inputs) / 9)
        lefts = inputs[0: batch * 1]
        rights = inputs[batch * 1: batch * 2]
        dests = inputs[batch * 2: batch * 3]
        labels = inputs[batch * 3: batch * 4]
        sequences = inputs[batch * 4: batch * 5]
        leaf_labels = inputs[batch * 5: batch * 6]
        relations = inputs[batch * 6: batch * 7]
        non_leafs = inputs[batch * 7: batch * 8]
        levels = inputs[batch * 8: batch * 9]

        inds = numpy.argsort([-len(l) for l in lefts])
        # Sort all arrays in descending order and transpose them
        lefts = F.transpose_sequence([lefts[i] for i in inds])
        rights = F.transpose_sequence([rights[i] for i in inds])
        dests = F.transpose_sequence([dests[i] for i in inds])
        labels = F.transpose_sequence([labels[i] for i in inds])
        sequences = F.transpose_sequence([sequences[i] for i in inds])
        leaf_labels = F.transpose_sequence(
            [leaf_labels[i] for i in inds])
        relations = F.transpose_sequence(
            [relations[i] for i in inds])
        non_leafs = F.transpose_sequence(
            [non_leafs[i] for i in inds])
        levels = F.transpose_sequence(
            [levels[i] for i in inds])

        batch = len(inds)
        maxlen = len(sequences)

        loss = 0
        count = 0
        correct = 0

        stack_c = self.xp.zeros((batch, maxlen * 2, self.rst_unit), 'f')
        stack_h = self.xp.zeros((batch, maxlen * 2, self.rst_unit), 'f')

        dct_du_index2norm_vec = {}
        ary_norm_vec = None
        index = 0
        # Implement batch normalization on du vector
        # for i, (leafs, label) in enumerate(zip(sequences, leaf_labels)):
        for leafs in sequences:
            for leaf in leafs:
                if int(leaf.data) not in dct_du_index2norm_vec:
                    dct_du_index2norm_vec[int(leaf.data)] = index
                    index += 1
                    if ary_norm_vec is None:
                        ary_norm_vec \
                            = self.du_text2vector[self.du_index2text[
                            int(leaf.data)]]
                    else:
                        cur_emb = self.du_text2vector[self.du_index2text[
                            int(leaf.data)]]
                        ary_norm_vec = self.xp.concatenate((ary_norm_vec,
                                                            cur_emb))
        ary_norm_vec1 = ary_norm_vec

        for non_leaf in non_leafs:
            for single_non_leaf in non_leaf:
                if int(single_non_leaf.data) not in dct_du_index2norm_vec:
                    dct_du_index2norm_vec[int(single_non_leaf.data)] = index
                    index += 1
                    if ary_norm_vec is None:
                        ary_norm_vec = self.du_text2vector[self.du_index2text[
                            int(single_non_leaf.data)]]
                    else:
                        cur_emb = self.du_text2vector[self.du_index2text[
                            int(single_non_leaf.data)]]
                        ary_norm_vec = self.xp.concatenate((ary_norm_vec,
                                                            cur_emb))

        if self.x_normalizer_teq is not None:
            ary_norm_vec = self.x_normalizer_teq(ary_norm_vec)

        for i, (leafs, label) in enumerate(zip(sequences, leaf_labels)):
            batch = leafs.shape[0]
            ary_reduction_vector = self.concate_array(leafs,
                                                      dct_du_index2norm_vec,
                                                      ary_norm_vec, None)
            ary_reduction_vector = self.W_x_reduction(ary_reduction_vector)
            es = (self.xp.zeros((batch, self.rst_unit), 'f'), \
                  ary_reduction_vector)

            ds = self.xp.full((batch,), i, 'i')
            count += batch

            stack_c = ts_teq.thin_stack_set(stack_c, ds, es[0])
            stack_h = ts_teq.thin_stack_set(stack_h, ds, es[1])

        int_node_total_count = 0
        int_node_correct_count = 0
        int_index = 0
        lst_o = []
        lst_acc_index = []
        for left, right, dest, label, relation, non_leaf, level \
                in zip(lefts, rights, dests, labels, relations,
                       non_leafs, levels):
            l_c, stack_c = ts_teq.thin_stack_get(stack_c, left)
            l_h, stack_h = ts_teq.thin_stack_get(stack_h, left)

            r_c, stack_c = ts_teq.thin_stack_get(stack_c, right)
            r_h, stack_h = ts_teq.thin_stack_get(stack_h, right)

            ary_non_leaf_emb = self.concate_array(non_leaf,
                                                  dct_du_index2norm_vec,
                                                  ary_norm_vec, None)

            o = self.classifier.rst_node((l_c, l_h), (r_c, r_h),
                                         ary_non_leaf_emb, relation)
            lst_o.append(o)

            y = self.classifier.label(o)

            batch = l_h.shape[0]

            loss += F.softmax_cross_entropy(y, label, normalize=False) * batch

            count += batch

            predict = self.xp.argmax(y.data, axis=1)

            # Pick up the root node
            root_index = numpy.where(cp.asnumpy(level.data) == 1)[0].tolist()

            # Pick up the labeled nodes, discard unlabeled nodes
            # no matter root or non-leaf nodes
            label_cpu = cp.asnumpy(label.data)
            avail_index = numpy.where(label_cpu != -1)[0].tolist()

            avail_index = list(set(avail_index).intersection(set(root_index)))
            lst_acc_index.append(avail_index)

            avail_label_cpu = label_cpu[avail_index]
            avail_pred_cpu = cp.asnumpy(predict)[avail_index]

            correct += (avail_label_cpu == avail_pred_cpu).sum()

            int_node_total_count += len(avail_label_cpu)
            int_node_correct_count += (avail_label_cpu == avail_pred_cpu).sum()

            stack_c = ts_teq.thin_stack_set(stack_c, dest, o[0])
            stack_h = ts_teq.thin_stack_set(stack_h, dest, o[1])

            int_index += 1

        reporter.report({'loss': loss}, self)
        reporter.report({'accuracy': float(int_node_correct_count /
                                           int_node_total_count)}, self)

        if test_mode:
            return lst_o, lst_acc_index, (int_node_correct_count,
                                          int_node_total_count)
        else:
            return (int_node_correct_count, int_node_total_count)