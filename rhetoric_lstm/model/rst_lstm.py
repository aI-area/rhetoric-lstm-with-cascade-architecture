import copy
import time
import random
import collections

import numpy as np

import chainer
import chainer.links as links
import tree_lstm.con_rhetoric_lstm_unit as\
    PyConnection_multiplicative_LSTM_over_RST
import chainer.functions as function_set
import chainer.functions as chainer_functions
from chainer import reporter
from chainer import initializers

from utilities import process
import process.artificial_creater as artificial_creater

# import monitor.parameter_visualization as parameter_visualization


class RstLSTMStack(chainer.Chain):
    def __init__(self, n_label, xp,
                 flo_dropout, input_dp,
                 normalizer_teq, segment_model, seg_learn_mode,

                 index2word, du_text2vector, du_unit, rst_unit, w2index,
                 rst_none_leaf_node_cst_tree, rst_none_leaf_node_cst_tree_text,

                 model_size, rst_rel2index, param_initialization,
                 boo_test_mode=False):
        if boo_test_mode:
            nb_rel = 6
            initialW = xp.array([[0.1, 0.2, 0.3], [0.2, 0.1, 0.1],
                                 [0.3, 0.1, 0.1], [0.1, 0.1, 0.2],
                                 [0.2, 0.2, 0.1], [0.1, 0.1, 0.1]],
                                dtype=xp.float32)
            initialw_full_connected_layer = np.random.uniform(
                1, 1, (n_label, rst_unit)).astype(np.float32)
        else:
            nb_rel = 16
            if param_initialization is not None:
                initialW = param_initialization()
                initialw_full_connected_layer = param_initialization()
            else:
                initialW = param_initialization
                initialw_full_connected_layer = param_initialization

        super(RstLSTMStack, self).__init__(
            embed=links.EmbedID(nb_rel, rst_unit, initialW=initialW),
            rst_nonleaf_treelstm=PyConnection_multiplicative_LSTM_over_RST.
            RhetoricLSTMUnit(du_unit, rst_unit, normalizer_teq,
                             param_initialization, n_ary=2,
                             boo_test_mode=boo_test_mode),
                             full_connected_layer
                             =links.Linear(rst_unit, n_label,
                                           initialW=
                                           initialw_full_connected_layer),
        )
        self.name = 'predictor'
        self.array_type = xp
        self.flo_dropout = flo_dropout
        self.input_dp = input_dp
        self.segment_model = segment_model
        self.seg_learn_mode = seg_learn_mode
        self.index2word = index2word
        self.du_text2vector = du_text2vector
        self.du_unit = du_unit
        self.rst_unit = rst_unit
        self.w2index = w2index
        self.rst_none_leaf_node_cst_tree_text \
            = rst_none_leaf_node_cst_tree_text
        self.rst_none_leaf_node_cst_tree = rst_none_leaf_node_cst_tree
        self.n_label = n_label
        self.model_size = model_size
        self.rst_rel2index = rst_rel2index

    # x为当前结点的constituent的segment vector
    def rst_node(self, left, right, x, int_relation):
        x = function_set.dropout(x, self.input_dp)

        c_left, h_left = left
        c_right, h_right = right

        var_relation = self.embed(int_relation)

        if isinstance(h_left, chainer.Variable):
            c, h = self.rst_nonleaf_treelstm(c_left, c_right, h_left, h_right,
                                             var_relation,
                                             x)
        else:
            zero = self.xp.zeros(c_left.shape, 'f')
            c, h = self.rst_nonleaf_treelstm(zero, zero, zero, zero,
                                             var_relation,
                                             x)
        return c, h

    def label(self, v):
        h = v[1]
        h = function_set.dropout(h, self.flo_dropout)
        return self.full_connected_layer(h)

    def handle_leaf_node(self, node=None):
        """
        Obtain memory cell and hidden state of leaf node which is a
        single word,
        not phrase or sentence. E.g. leaf node in constituency parse tree.
        """
        memory_cell, hidden_state = self.segment_model.leaf(node)
        return memory_cell, hidden_state


    def handle_constituent_non_leaf_authentic_node(self, node, current_model,
                                                   entire_tree, evaluate_info,
                                                   root, lis_sequence,
                                                   extracted_segment_sentence,
                                                   hidden_size, rel2index):
        left_node, right_node = node

        left_loss, left, left_node_artificial_cutted, _ \
            = self.traverse(current_model, left_node, right_node, entire_tree,
                       hidden_size, rel2index, 0,
                       evaluate_info=evaluate_info, root=root,
                       lis_sequence=lis_sequence,
                       extracted_segment_sentence=extracted_segment_sentence)

        right_loss, right, right_node_artificial_cutted, _ \
            = self.traverse(current_model, right_node, left_node, entire_tree,
                       hidden_size, rel2index, 0,
                       evaluate_info=evaluate_info, root=root,
                       lis_sequence=lis_sequence,
                       extracted_segment_sentence=extracted_segment_sentence)
        mem_state = self.segment_model.node(left, right)
        return mem_state


    def handle_constituent_non_leaf_fake_node(self, node, left, right,
                                              rel2index):
        # mode = "joint"
        str_artificial_relation = node['artificial_relation']
        node.pop("artificial_relation")
        node.pop("artificial_left_leaf")
        node.pop("artificial_right_leaf")

        if self.seg_learn_mode != "joint":
            lis_segment_string = []
            process.transform2string(node, lis_segment_string)

            str_tmp = ""
            for seg in lis_segment_string:
                str_tmp += self.index2word[seg] + " "

            current_x \
                = self.du_text2vector[str_tmp[:-1] + "\n"]

        else:
            current_x = self.array_type.zeros((1, self.du_unit), 'f')
        node["artificial_relation"] = "Mark"

        if self.seg_learn_mode != "joint":
            lis_relation_direction = str_artificial_relation.split("_____")
            # print(lis_relation_direction)
            if lis_relation_direction[1] == "LeftToRight":
                # 第1、2个参数，是分别从左右结点传过来的h的值
                output = self.rst_node(left, right, current_x,
                                        rel2index[lis_relation_direction[0]])
            else:
                output = self.rst_node(right, left, current_x,
                                        rel2index[lis_relation_direction[0]])
        else:
            output = self.segment_model.node(left, right)
            # 最后一个参数填什么修辞关系都没影像，因为左右孩子节点的h和c必为零
            zero_hc = (self.array_type.zeros((1, self.rst_unit), 'f'),
                       self.array_type.zeros((1, self.rst_unit), 'f'))
            output = self.rst_node(zero_hc, zero_hc, output[1],
                                    rel2index["joint"])

        return node, output

    def delete_artificial_factor(self, node):
        if "artificial_relation" in node.keys():
            node.pop("artificial_relation")
        if "artificial_left_leaf" in node.keys():
            node.pop("artificial_left_leaf")
        if "artificial_right_leaf" in node.keys():
            node.pop("artificial_right_leaf")
        return node


    def get_constituent_segment_vector(self, special_node, enable_pass=False):
        lis_segment_string = []
        process.transform2string(special_node, lis_segment_string)
        str_tmp = ""
        for seg in lis_segment_string:
            str_tmp += self.index2word[seg] + " "

        constituent_segment_vector \
            = self.du_text2vector[str_tmp[:-1] + "\n"]

        if enable_pass:
            # lis_segment_string = []
            # transform_original2string(special_node, lis_segment_string)
            self.output_segment(lis_segment_string)
        return constituent_segment_vector


    def handle_rst_non_leaf_node(self, left_node_artificial_cutted,
                                 right_node_artificial_cutted,
                                 node, left, right,
                                 enable_pass, rel2index,
                                 hidden_size):
        left_node = left_node_artificial_cutted
        right_node = right_node_artificial_cutted
        left_node = self.delete_artificial_factor(left_node)
        right_node = self.delete_artificial_factor(right_node)
        node["node"] = left_node, right_node

        if self.seg_learn_mode != "joint":
            current_x = self.get_constituent_segment_vector(node, enable_pass)
        else:
            lis_segment_string = []
            process.transform2string(node, lis_segment_string)

            if lis_segment_string not in self.rst_none_leaf_node_cst_tree_text:
                lst_segment_text = []
                for sig_word in lis_segment_string:
                    for pair in self.w2index:
                        if self.w2index[pair] == sig_word:
                            lst_segment_text.append(pair)
                current_x = self.array_type.zeros((1, self.du_unit), 'f')
            else:
                _, seg_vector, _, _ \
                    = self.traverse(self, self.rst_none_leaf_node_cst_tree[
                    self.rst_none_leaf_node_cst_tree_text.index(
                        lis_segment_string)], None,
                               self.rst_none_leaf_node_cst_tree[
                                   self.rst_none_leaf_node_cst_tree_text.index(
                                       lis_segment_string)], hidden_size,
                               rel2index, 0)
                current_x = seg_vector[1]
        lis_relation_direction = node['relation'].split("_____")
        if lis_relation_direction[1] == "LeftToRight":
            output = self.rst_node(left, right, current_x,
                                    rel2index[lis_relation_direction[0]])
        else:
            output = self.rst_node(right, left, current_x,
                                    rel2index[lis_relation_direction[0]])
        return node, output


    def transform_original2string(self, single_tree, lis_string):
        if not isinstance(single_tree["node"], int):
            self.transform_original2string(single_tree["node"][0], lis_string)
            self.transform_original2string(single_tree["node"][1], lis_string)
        else:
            lis_string.append((single_tree["original_node"]))

    def output_segment(self, lis_segment_string):
        str_tmp = ""
        for seg in lis_segment_string:
            if isinstance(seg, int):
                str_tmp += self.index2word[seg] + " "
            else:
                str_tmp += seg + " "
        with open('../data/segment_text.txt', 'a') as f_output:
            f_output.write(str_tmp[:-1] + "\n")


    def softmax_output(self, enable_train, node, current_loss, y):
        """

        :param enable_train:
        :param node:
        :param current_loss:
        :param y:
        :return: nb_trained: number of sample that participate in
        training process
        """
        nb_trained = 0
        if enable_train:
            if node['label'] != 20 and node['label'] != 25:
                if node['label'] > 9:
                    label = self.array_type.array([node['label'] - 10],
                                                  np.int32)
                    t = chainer.Variable(label)
                    current_loss += chainer_functions.softmax_cross_entropy(y,
                                                                            t)
                    nb_trained = 1
        else:
            if node['label'] != 20 and node['label'] != 25:
                if node['label'] > 9:
                    label = self.array_type.array([node['label'] - 10],
                                                  np.int32)
                    current_loss \
                        += (chainer_functions
                            .softmax_cross_entropy(y, label)).data
                    nb_trained = 1
        return current_loss, nb_trained

    def predict(self, statistics, node, root, y):
        if statistics is not None:
            if node['label'] > 9 and node['label'] != 25:
                if node['label'] != 20 and node['label'] != 25:
                    boo_rst = False
                    if node['label'] > 9:
                        int_label_value = node['label'] - 10
                        boo_rst = True
                    else:
                        int_label_value = node['label']
                    predicted_value = self.array_type.argmax(y.data, axis=1)
                    if predicted_value[0] == int_label_value:
                        statistics['correct_node'] += 1
                    statistics['total_node'] += 1
                    if boo_rst:
                        if predicted_value[0] == int_label_value:
                            statistics['correct_rst_all'] += 1
                        statistics['total_rst_all'] += 1
                        if root:
                            if predicted_value[0] == int_label_value:
                                statistics['correct_rst_root'] += 1
                            statistics['total_rst_root'] += 1
                        else:
                            if predicted_value[0] == int_label_value:
                                statistics['correct_rst_node'] += 1
                            statistics['total_rst_node'] += 1
                    if root:
                        if predicted_value[0] == int_label_value:
                            statistics['correct_root'] += 1
                        statistics['total_root'] += 1

                        if not boo_rst:
                            if predicted_value[0] == int_label_value:
                                statistics['correct_constituent_root'] += 1
                            statistics['total_constituent_root'] += 1
                    elif isinstance(node['node'], int):
                        if predicted_value[0] == int_label_value:
                            statistics['correct_leaf'] += 1
                        statistics['total_leaf'] += 1
                    else:
                        if not boo_rst:
                            if predicted_value[0] == int_label_value:
                                statistics['correct_constituent_node'] += 1
                            statistics['total_constituent_node'] += 1
        return statistics

    def traverse(self, current_model, node, sibling, entire_tree, hidden_size,
                 rel2index,
                 nb_total_train_sample,
                 evaluate_info=None,
                 root=True, lis_sequence=[], extracted_segment_sentence=None):

        nb_lf_train_sample = nb_rt_train_sample = 0
        loss_value = 0
        output = (self.array_type.zeros((1, hidden_size), 'f'),
                  self.array_type.zeros((1, hidden_size), 'f'))
        y = None

        # Forward propagation along tree-structure
        # Current node is terminal leaf node that corresponds to a single
        # word, not EDU nodes in RST parse tree .
        if isinstance(node['node'], int):
            if self.seg_learn_mode == "joint":
                output = self.handle_leaf_node(node['node'])

        # Current node is intermediate node that corresponds to a phrase,
        # not DU
        # nodes in RST parse tree .
        elif(node['label'] < 9 or node['label'] == 25 or
             node['label'] == 26) and \
                "artificial_relation" \
                not in node.keys():
            if self.seg_learn_mode == "joint":
                output = self.handle_constituent_non_leaf_authentic_node(
                    node["node"], current_model, entire_tree, evaluate_info,
                    False, lis_sequence, extracted_segment_sentence,
                    hidden_size,
                    rel2index)

        # Current node is DU(EDU) nodes in RST parse tree .
        else:
            left_node, right_node = node['node']
            left_loss, left, left_node_artificial_cutted, nb_lf_train_sample \
                = self.traverse(current_model, left_node, right_node,
                                entire_tree,
                           hidden_size, rel2index, nb_total_train_sample,
                           evaluate_info=evaluate_info, root=False,
                           lis_sequence=lis_sequence,
                           extracted_segment_sentence
                                =extracted_segment_sentence)
            # nb_total_train_sample += nb_train_sample
            right_loss, right, right_node_artificial_cutted, \
                nb_rt_train_sample \
                = self.traverse(current_model, right_node, left_node,
                                entire_tree,
                           hidden_size, rel2index, nb_total_train_sample,
                           evaluate_info=evaluate_info, root=False,
                           lis_sequence=lis_sequence,
                           extracted_segment_sentence
                                =extracted_segment_sentence)

            # 这个点是Constituent Parser Tree和RST Parser Tree的交界处
            # 实际上就是RST树的叶子节点
            if "artificial_relation" in node.keys():
                node, output \
                    = self.handle_constituent_non_leaf_fake_node(node, left,
                                                                 right,
                                                            rel2index)
            else:
                if isinstance(extracted_segment_sentence, list):
                    enable_pass = True
                else:
                    enable_pass = False
                if not isinstance(extracted_segment_sentence, list):
                    node, output \
                        = self.handle_rst_non_leaf_node(
                        left_node_artificial_cutted,
                        right_node_artificial_cutted,
                        node, left,
                        right, enable_pass,
                        rel2index, hidden_size
                    )
                else:
                    lis_tmp = []
                    self.transform_original2string(node, lis_tmp)
                    self.output_segment(lis_tmp)

                    lis_tmp = []
                    self.transform_original2string(left_node_artificial_cutted,
                                                   lis_tmp)
                    self.output_segment(lis_tmp)

                    lis_tmp = []
                    self.transform_original2string(
                        right_node_artificial_cutted, lis_tmp)
                    self.output_segment(lis_tmp)

                if isinstance(extracted_segment_sentence, list):
                    extracted_segment_sentence.append(copy.deepcopy(node))
            if not isinstance(extracted_segment_sentence, list):
                loss_value = left_loss + right_loss
                y = current_model.label(output)

        # Evaluate model performance and prediction
        if not isinstance(extracted_segment_sentence, list):
            label = self.array_type.array([node['label']], np.int32)
            nb_trained = 0
            if self.n_label > 2 or (self.n_label <= 2 and label
                                    in [0, 1, 10, 11]):
                loss_value, nb_trained = self.softmax_output(
                    chainer.config.train, node, loss_value, y)
                self.predict(evaluate_info, node, root, y)

            nb_total_train_sample += nb_lf_train_sample \
                                     + nb_rt_train_sample + nb_trained
            return loss_value, output, copy.deepcopy(node), \
                   nb_total_train_sample
        else:
            return 0, output, copy.deepcopy(node), nb_total_train_sample


    def evaluate(self, test_trees):

        if self.model_size == "1":
            test_trees \
                = (artificial_creater.Generator(
                self.du_text2vector,
                copy.deepcopy(
                    test_trees),
                self.index2word)) \
                .trees_leaf_inserted

        result = collections.defaultdict(lambda: 0)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for tree in test_trees:
                lis_words = []
                current_loss, _, _, nb_trained_sample \
                    = self.traverse(self, tree, None,
                               tree, self.n_unit, self.rel2index, 0,
                               evaluate_info=result,
                               lis_sequence=lis_words,
                               extracted_segment_sentence=None)

        acc_node = 100.0 * result['correct_node'] / result['total_node']
        acc_root = 100.0 * result['correct_root'] / result['total_root']
        print(' Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
            acc_node, result['correct_node'], result['total_node']))
        print(' Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
            acc_root, result['correct_root'], result['total_root']))


    def forward(self, train_trees):
        # Store statistics of prediction
        result = collections.defaultdict(lambda: 0)

        if self.model_size == "1":
            train_trees \
                = (artificial_creater.Generator(self.du_text2vector,
                                                copy.deepcopy(
                                                    train_trees),
                                                self.index2word)) \
                .trees_leaf_inserted

        accum_loss = 0.0
        for tree in train_trees:
            loss, v, _, nb_train_sample \
                = self.traverse(self, tree, None,
                           tree, self.rst_unit,
                           self.rst_rel2index, 0,
                           evaluate_info=result)

            accum_loss += loss

        reporter.report({'loss': accum_loss}, self)
        reporter.report({'total': result['total_root']}, self)
        reporter.report({'correct': result['correct_root']}, self)
        return accum_loss