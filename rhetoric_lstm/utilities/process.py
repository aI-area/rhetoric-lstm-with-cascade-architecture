import numpy as np
import pickle

from chainer import initializers
import tensorflow as tf

import process.data_transformer as data_transformer


def obtain_all_experiment():
    lis_process = [
                    "dependency_5_150_1_1_0",
                    "bert_2_768_1_1_1"
                    "lstm_5_150_1_0_0", "lstm_5_150_1_0_1",
                    "lstm_5_150_1_1_0", "lstm_5_150_1_1_1"
                    "bilstm_5_300_1_0_0", "bilstm_5_300_1_0_1",
                    "bilstm_5_300_1_1_0", "bilstm_5_300_1_1_1"
                    "constituency_2_150_1_0_0", "constituency_2_150_1_0_1",
                    "constituency_2_150_1_1_0", "constituency_2_150_1_1_1"
                   ]

    return lis_process


def transform2string(single_tree, lis_string):
    if not isinstance(single_tree[1], int):
        transform2string(single_tree[1][0], lis_string)
        transform2string(single_tree[1][1], lis_string)
    else:
        lis_string.append((single_tree[1]))


def load_data(w2vector, n_label, w2index, max_size, flag_swap,
              mode, folder_prefix):
    # Load train set
    train_trees \
        = data_transformer.Parser(w2vector, n_label,
                                  folder_prefix +
                                  'data/sentiment_analysis/train.txt',
                                  w2index, max_size).trees
    int_index = 0
    lis_last_tree = []
    while int_index < len(train_trees):
        lis_tmp = []
        transform2string(train_trees[int_index], lis_tmp)
        if train_trees[int_index][0] < 10 or \
                train_trees[int_index][0] \
                == 25 or train_trees[int_index][0] == 26:  # 8667
            train_trees.pop(int_index)
        elif lis_tmp == lis_last_tree and flag_swap == "0":
            train_trees.pop(int_index)
        else:
            lis_last_tree = lis_tmp
            int_index += 1

    # Load develop set
    dev_trees \
        = data_transformer.Parser(w2vector, n_label,
                                  folder_prefix +
                                  'data/sentiment_analysis/dev.txt',
                                  w2index, max_size).trees

    # Load test set
    test_trees \
        = data_transformer.Parser(w2vector, n_label,
                                  folder_prefix +
                                  'data/sentiment_analysis/test.txt',
                                  w2index, max_size).trees

    int_index = 0
    while int_index < len(test_trees):
        if test_trees[int_index][0] < 10 or \
                test_trees[int_index][0] in (25, 26):
            test_trees.pop(int_index)
        else:
            int_index += 1

    if mode == "joint":
        # DU classifier, 2-array Tree-LSTM, need constituency tree
        # correspondingly
        # Constituency tree structure
        rst_none_leaf_node_cst_tree \
            = data_transformer\
            .Parser(w2vector, n_label,
                    folder_prefix + "sentiment_analysis/"
                    "constituent_normalized_segment_word",
                    w2index, max_size).trees
        # Text of below Constituency tree structure
        rst_none_leaf_node_cst_tree_text = []

        for tree in rst_none_leaf_node_cst_tree:
            tmp = []
            transform2string(tree, tmp)
            rst_none_leaf_node_cst_tree_text.append(tmp)
    else:
        rst_none_leaf_node_cst_tree = rst_none_leaf_node_cst_tree_text = None

    return train_trees, dev_trees, test_trees, \
        rst_none_leaf_node_cst_tree, rst_none_leaf_node_cst_tree_text


def init_file(model_dump_path):
    """  Delete existent folder and create empty folder under this path.
    Args:
        model_dump_path: Type 'string'. Path of log information.
    """
    if tf.gfile.Exists(model_dump_path):
        tf.gfile.DeleteRecursively(model_dump_path)
    tf.gfile.MakeDirs(model_dump_path)


def load_vob(du_generator_name, folder_prefix, xp):
    # Load dumped word representation, i.e. Glove.
    # Speed of reading is quite faster than original representation file.
    fil_pkl = open(folder_prefix +
                   "data/sentiment_analysis/glove.840B.300d.pic", 'rb')
    w2vector = pickle.load(fil_pkl)

    # Load dumped vocabulary that map word content to index.
    fil_pkl = open(folder_prefix +
                   'data/sentiment_analysis/vocab_300d no synonmy.pkl',
                   'rb')
    w2index = pickle.load(fil_pkl)
    index2word = dict((v, k) for k, v in w2index.items())

    fil_pkl = open(folder_prefix + 'data/sentiment_analysis/s2v_'
                   + du_generator_name
                   + ".pkl", 'rb')
    du_text2vector = pickle.load(fil_pkl)

    return w2vector, w2index, du_text2vector, index2word


def voi_generate_vocabulary(vocab, word2vec, typ_data):
    index2vector = np.empty((len(vocab), word2vec.vectors[0].shape[0]),
                            dtype=typ_data)
    for item in vocab:
        if item in word2vec.vocab:
            index2vector[vocab[item]] = word2vec.word_vec(item)
        else:
            index2vector[vocab[item]] \
                = initializers.generate_array(initializers.normal.Normal(0.82),
                                              (1,
                                               word2vec.vectors[0].shape[0]),
                                              np)
    return index2vector
