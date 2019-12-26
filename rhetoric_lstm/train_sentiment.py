import argparse
import collections
import numpy as np
import chainer
from chainer import cuda
from chainer import optimizers, initializers
from tensorboardX import SummaryWriter
from chainer.training import extensions

from model import thin_stack_rst_lstm as ts_rst_lstm
from utilities import process
from model import tree_lstm
from process import stacker

def analysis_sample_distribution(board):
    if board['total_node'] == 0:
        acc_node = 0
    else:
        acc_node = 100.0 * board['correct_node'] / board['total_node']
    if board['total_root'] == 0:
        acc_root = 0
    else:
        acc_root = 100.0 * board['correct_root'] / board['total_root']
    if board['total_rst_all'] == 0:
        acc_rst = 0
    else:
        acc_rst = 100.0 * board['correct_rst_all'] / board['total_rst_all']

    if (board['total_node'] - board['total_root'] - board['total_leaf']) \
            == 0:
        acc_non_leaf_root = 0
    else:
        acc_non_leaf_root \
            = 100.0 * (board['correct_node'] - board['correct_root'] -
                       board['correct_leaf']) / (board['total_node'] -
                                                 board['total_root'] -
                                                 board['total_leaf'])
    if board['total_rst_root'] == 0:
        acc_rst_root = 0
    else:
        acc_rst_root \
            = 100.0 * board['correct_rst_root'] / board['total_rst_root']

    if board['total_rst_node'] == 0:
        acc_rst_node = 0
    else:
        acc_rst_node \
            = 100.0 * board['correct_rst_node'] / board['total_rst_node']
    # =====
    str_log = ('Tree accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_node, board['correct_node'], board['total_node']))
    print(str_log)
    # =====
    str_log = ('Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_root, board['correct_root'], board['total_root']))
    print(str_log)
    str_log \
        = ('Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'
           .format(acc_non_leaf_root, board['correct_node']
                   - board['correct_root'] - board['correct_leaf'],
                   board['total_node'] - board['total_root']
                   - board['total_leaf']))
    print(str_log)
    # =====
    str_log = ('RST Tree accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_rst, board['correct_rst_all'], board['total_rst_all']))
    print(str_log)
    str_log = ('RST Root accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_rst_root, board['correct_rst_root'], board['total_rst_root']))
    print(str_log)
    str_log = ('RST Node accuracy: {0:.2f} %% ({1:,d}/{2:,d})'.format(
        acc_rst_node, board['correct_rst_node'], board['total_rst_node']))
    print(str_log)
    return acc_rst_root


def evaluate(model_load, evaluated_trees, n_unit, rel2index,
             extracted_seg_sentence=None):
    board = collections.defaultdict(lambda: 0)
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        sum_loss = 0
        sum_train_sample = 0
        for single_tree in evaluated_trees:
            lis_words = []
            current_loss, _, _, nb_trained_sample \
                = traverse(model_load, single_tree, None,
                           single_tree, n_unit, rel2index, 0,
                           evaluate_info=board,
                           lis_sequence=lis_words,
                           extracted_segment_sentence=extracted_seg_sentence)
            sum_loss += current_loss
            sum_train_sample += nb_trained_sample
    acc_root = analysis_sample_distribution(board)
    return acc_root, sum_loss/sum_train_sample


def generate_relation_vob(str_rel_cluster):
    if str_rel_cluster == "0":
        rel2index = {"joint": 0, "elaboration": 1, "background": 2,
                     "attribution": 3, "condition": 4, "cause": 5,
                     "enablement": 6, "same-unit": 7, "contrast": 8,
                     "temporal": 9, "comparison": 10, "explanation": 11,
                     "manner-means": 12, "evaluation": 13,
                     "summary": 14, "textual-organization": 15}
    else:
        rel2index = {"joint": 0, "elaboration": 1, "background": 2,
                     "attribution": 3, "condition": 2, "cause": 2,
                     "enablement": 2, "same-unit": 4, "contrast": 5,
                     "temporal": 2, "comparison": 2, "explanation": 2,
                     "manner-means": 2, "evaluation": 2, "summary": 2,
                     "textual-organization": 2}
    return rel2index


def setup_tree_lstm(w2index_vob, w2vector_vob, data_type, nb_unit, nb_label,
                    data_carrier):
    if SEG_LEARN_MODE == "joint":
        index2vector \
            = process.voi_generate_vocabulary(w2index_vob, w2vector_vob,
                                              data_type)
        tree_model \
            = tree_lstm.RecursiveTreeLSTMNet(len(w2index_vob),
                                             w2vector_vob.vectors[0].shape[0],
                                             nb_unit, nb_label,
                                             data_carrier, index2vector)

        tree_model.embed.disable_update()
        tree_model.to_gpu()
    else:
        tree_model = None
    return tree_model

def delete_original_node(node):
    if isinstance(node["node"], int):
        if "original_node" in node.keys():
            node.pop("original_node")
    else:
        delete_original_node(node["node"][0])
        delete_original_node(node["node"][1])

def output_segment(lis_segment_string):
    str_tmp = ""
    for seg in lis_segment_string:
        if isinstance(seg, int):
            str_tmp += index2word[seg] + " "
        else:
            str_tmp += seg + " "
    with open('../data/segment_text.txt', 'a') as f_output:
        f_output.write(str_tmp[:-1] + "\n")

def optimizer_setup(model_armed, model_optimizer):
    model_optimizer.setup(model_armed)
    str_optimizer_type = str(type(model_optimizer))
    if flo_l2 > 0:
        model_optimizer.add_hook(chainer.optimizer.WeightDecay(flo_l2))

    if flo_relation_learning_rate > 0:
        for trainable_variable in model_armed.classifier.embed.params():
            trainable_variable.update_rule.hyperparam.alpha \
                = flo_relation_learning_rate
            pass
    else:
        model_armed.classifier.embed.disable_update()
    return model_armed, model_optimizer, str_optimizer_type

def convert(batch, device):
    if device is None:
        def to_device(x):
            return x
    elif device < 0:
        to_device = cuda.to_cpu
    else:
        def to_device(x):
            return cuda.to_gpu(x, device, cuda.Stream.null)

    return tuple(
        [to_device(d['lefts']) for d in batch] +
        [to_device(d['rights']) for d in batch] +
        [to_device(d['dests']) for d in batch] +
        [to_device(d['labels']) for d in batch] +
        [to_device(d['leafs']) for d in batch] +
        [to_device(d['leaf_labels']) for d in batch] +
        [to_device(d['relations']) for d in batch] +
        [to_device(d['non_leafs']) for d in batch] +
        [to_device(d['levels']) for d in batch]
    )


if __name__ == "__main__":
    lis_process = process.obtain_all_experiment()
    for item in lis_process:
        for _ in range(0, 4):
            lis_parameter = item.split("_")

            parser = argparse.ArgumentParser()
            parser.add_argument('--gpu', '-g', default=0, type=int,
                                help='GPU ID (negative value indicates CPU)')
            parser.add_argument('--epoch', '-e', default=200, type=int,
                                help='number of epochs to learn')
            parser.add_argument('--du_unit', '-du_unit',
                                default=int(lis_parameter[2]),
                                type=int,
                                help='number of units in DU generator')
            parser.add_argument('--rst_unit', '-rst_unit', default=150,
                                type=int,
                                help='number of units in RST classifier')
            parser.add_argument('--batchsize', '-b', type=int, default=25,
                                help='learning minibatch size')
            parser.add_argument('--label', '-l', type=int,
                                default=int(lis_parameter[1]),
                                help='number of labels')
            parser.add_argument('--epocheval', '-p', type=int, default=1,
                                help='number of epochs per evaluation')
            parser.add_argument('--test', dest='test', action='store_true')
            parser.add_argument('--mode', '-m', default="pipeline",
                                type=str,
                                help='Mode of Stacking on DU generator')
            parser.add_argument('--rel_cluster', '-rel_cluster', type=str,
                                default=lis_parameter[5],
                                help='If relation cluster is implemented')
            parser.add_argument('--model_size', '-model_size', type=str,
                                default=lis_parameter[3],
                                help='If consider leaf nodes '
                                     'in rst parse tree')

            parser.add_argument('--rst_lr', '-r_lr', type=float, default=0.005,
                                help='learning rate of rst classifier')
            parser.add_argument('--relation_lr', '-rel_lr', type=float,
                                default=0.005,
                                help='learning rate of relation embedding in'
                                     ' rst classifier')
            parser.add_argument('--du_lr', '-du_lr', type=float, default=0,
                                help='learning rate of du classifier')
            parser.add_argument('--word_lr', '-word_lr', type=float, default=0,
                                help='learning rate of word embedding in '
                                     'du classifier')
            parser.add_argument('--rst_l2', '-rst_l2', type=float,
                                default=0.005,
                                help='l2 penalty of rst classifier')
            parser.add_argument('--du_l2', '-du_l2', type=float,
                                default=0.005,
                                help='l2 penalty of du classifier')
            parser.add_argument('--dropout', '-dropout', type=float,
                                default=0,
                                help='dropout value')
            parser.set_defaults(test=False)

            # Parse arguments
            args = parser.parse_args()
            if args.gpu >= 0:
                cuda.check_cuda_available()
            xp = cuda.cupy if args.gpu >= 0 else np
            typ_data = xp.float32
            if args.test:
                max_size = 99
            else:
                max_size = None
            n_epoch = args.epoch
            SEG_LEARN_MODE = args.mode
            du_unit = args.du_unit
            rst_unit = args.rst_unit
            n_label = args.label
            epoch_per_eval = args.epocheval
            nb_early_stop = -1

            if lis_parameter[0] == "bcnelmo":
                rst_unit = 20
                batchsize = 25  # args.batchsize
                flo_learning_rate = 5e-05  # args.rst_lr
                optimizer = optimizers.Adam(flo_learning_rate)  # AdaGrad
                flo_l2 = 0.1  # args.rst_l2
                flo_relation_learning_rate = 5e-04  # args.relation_lr 1e-04
                flo_dropout = 0.65  # args.dropout 0.2
                flo_input_dp = 0.3
                # options: chainer.links.LayerNormalization or None
                normalizer_teq = chainer.links.LayerNormalization
                # options: chainer.links.BatchNormalization or None
                x_normalizer_teq = chainer.links.BatchNormalization
                # options: initializers.LeCunUniform or None
                # When initializers.LeCunUniform is chosen, bias is not this,
                # but uniform random between 1 and 2
                param_initialization = initializers.LeCunUniform
            elif lis_parameter[0] == "bcnglove":
                rst_unit = 20
                batchsize = 25  # args.batchsize
                flo_learning_rate = 5e-05  # args.rst_lr
                optimizer = optimizers.Adam(flo_learning_rate)  # AdaGrad
                flo_l2 = 0.1  # args.rst_l2
                flo_relation_learning_rate = 5e-04  # args.relation_lr 1e-04
                flo_dropout = 0.65  # args.dropout 0.2
                flo_input_dp = 0.3
                # options: chainer.links.LayerNormalization or None
                normalizer_teq = chainer.links.LayerNormalization
                # options: chainer.links.BatchNormalization or None
                x_normalizer_teq = chainer.links.BatchNormalization
                # options: initializers.LeCunUniform or None
                # When initializers.LeCunUniform is chosen, bias is not this,
                # but uniform random between 1 and 2
                param_initialization = initializers.LeCunUniform
            elif lis_parameter[0] == "bert":
                rst_unit = 4
                batchsize = 25  # args.batchsize
                flo_learning_rate = 5e-05  # args.rst_lr
                optimizer = optimizers.Adam(flo_learning_rate)  # AdaGrad
                flo_l2 = 0.1  # args.rst_l2
                flo_relation_learning_rate = 5e-04  # args.relation_lr 1e-04
                flo_dropout = 0.5  # args.dropout 0.2
                flo_input_dp = 0.4
                normalizer_teq = chainer.links.LayerNormalization
                # options: chainer.links.BatchNormalization or None
                x_normalizer_teq = None
                # options: initializers.LeCunUniform or None
                param_initialization = None
            else:
                if n_label == 2:
                    rst_unit = 4
                    batchsize = 25  # args.batchsize
                    flo_learning_rate = 5e-05  # args.rst_lr
                    optimizer = optimizers.Adam(flo_learning_rate)  # AdaGrad
                    flo_l2 = 0.1  # args.rst_l2
                    flo_relation_learning_rate = 5e-04
                    flo_dropout = 0.5  # args.dropout 0.2
                    flo_input_dp = 0.4
                    normalizer_teq = chainer.links.LayerNormalization
                    # options: chainer.links.BatchNormalization or None
                    x_normalizer_teq = None
                    # options: initializers.LeCunUniform or None
                    param_initialization = None
                else:
                    rst_unit = 20
                    batchsize = 25  # args.batchsize
                    flo_learning_rate = 5e-05  # args.rst_lr
                    optimizer = optimizers.Adam(flo_learning_rate)  # AdaGrad
                    flo_l2 = 0.1  # args.rst_l2
                    flo_relation_learning_rate = 5e-04
                    flo_dropout = 0.65  # args.dropout 0.2
                    flo_input_dp = 0.3
                    # options: chainer.links.LayerNormalization or None
                    normalizer_teq = chainer.links.LayerNormalization
                    # options: chainer.links.BatchNormalization or None
                    x_normalizer_teq = chainer.links.BatchNormalization
                    # options: initializers.LeCunUniform or None
                    # When initializers.LeCunUniform is chosen, bias is not
                    # this,
                    # but uniform random between 1 and 2
                    param_initialization = initializers.LeCunUniform

            station_id = 1

            # unavaliable in pipeline mode
            flo_du_l2 = args.du_l2
            flo_constituent_learning_rate = args.du_lr
            optimizer_constituent \
                = optimizers.AdaGrad(lr=flo_constituent_learning_rate)
            flo_word_learning_rate = args.word_lr

            flag_rel_cluster = args.rel_cluster
            model_size = args.model_size
            rst_rel2index = generate_relation_vob(flag_rel_cluster)
            if station_id == 0:
                folder_prefix = "/home/bit0427/Documents/"
            else:
                folder_prefix = "/opt/"

            print('Dataset: ' + str(lis_parameter))
            print('----- Opt. hyperparams -----')
            print('rst_lr: ' + str(flo_learning_rate))
            print('rel_lr: ' + str(flo_relation_learning_rate))
            print('l2_coef: ' + str(flo_l2))
            print('batch size: ' + str(batchsize))
            print('input layer dropout rate: ' + str(flo_input_dp))
            print('hidden layer dropout rate: ' + str(flo_dropout))
            print('Max number of early stop: ' + str(nb_early_stop))
            print('----- Archi. hyperparams -----')
            print('nb. units per du hidden layer: ' + str(du_unit))
            print('nb. units per rst hidden layer: ' + str(rst_unit))
            print('normalization technique: ' + str(normalizer_teq))
            print('x_normalization technique: ' + str(x_normalizer_teq))
            print('parameter initializer: ' + str(param_initialization))
            print('Station ID : ' + str(station_id))

            # =====
            str_tb_folder = folder_prefix + "models/sentiment_analysis/log" \
                            + "__lr_" + str(flo_learning_rate) \
                            + "__rel_lr_" \
                            + str(flo_relation_learning_rate) \
                            + "__l2_" + str(flo_l2) \
                            + "__hidden_dp_" + str(flo_dropout) \
                            + "__input_dp_" + str(flo_input_dp) \
                            + "__normalization_technique_" \
                            + str(normalizer_teq)
            process.init_file(str_tb_folder)
            writer = SummaryWriter(str_tb_folder)
            print("Log folder path is : \"" + str_tb_folder + "\"")

            # Load necessary vocabularies which consists
            # of text(index) and vectors
            w2vector, w2index, du_text2vector, index2word \
                = process.load_vob(lis_parameter[0] + '_' + lis_parameter[1],
                                   folder_prefix, xp)

            # Initialize du classifier
            segment_model = setup_tree_lstm(w2index, w2vector,
                                            typ_data, du_unit, n_label, xp)
            if SEG_LEARN_MODE == "joint":
                optimizer_constituent.setup(segment_model)
                if flo_du_l2 > 0:
                    optimizer_constituent.add_hook(
                        chainer.optimizer.WeightDecay(flo_du_l2))

                if flo_word_learning_rate > 0:
                    for param in segment_model.embed.params():
                        param.update_rule.hyperparam.lr \
                            = flo_word_learning_rate
                else:
                    segment_model.embed.disable_update()

            # Load necessary sample files
            train_trees, _, test_trees, rst_none_leaf_node_cst_tree, \
            rst_none_leaf_node_cst_tree_text \
                = process.load_data(w2vector, n_label, w2index, max_size,
                                    lis_parameter[4], SEG_LEARN_MODE,
                                    folder_prefix)

            # linearize data
            du_text2index = {}
            train_data = [stacker.linearize_tree(index2word, t,
                                                 du_text2vector,
                                                 du_text2index,
                                                 rst_rel2index, xp)
                          for t in train_trees]
            train_iter = chainer.iterators.SerialIterator(train_data,
                                                          args.batchsize,
                                                          shuffle=True)
            test_data = [stacker.linearize_tree(index2word, t, du_text2vector,
                                                du_text2index, rst_rel2index,
                                                xp) for t in test_trees]

            test_iter = chainer.iterators.SerialIterator(test_data, batchsize,
                                                         repeat=False,
                                                         shuffle=False)

            model = ts_rst_lstm.ThinStackRecursiveNet(
                  n_label, xp,
                  flo_dropout,
                  flo_input_dp,
                  normalizer_teq, segment_model,
                  SEG_LEARN_MODE,
                  index2word, du_text2vector, du_unit, rst_unit, w2index,
                  rst_none_leaf_node_cst_tree,
                rst_none_leaf_node_cst_tree_text,
                  model_size, rst_rel2index,
                  du_text2index, x_normalizer_teq, param_initialization
            )

            chainer.cuda.get_device_from_id(args.gpu).use()
            model.to_gpu()

            model, optimizer, _ = optimizer_setup(model, optimizer)

            # Setup updater
            updater = chainer.training.StandardUpdater(
                train_iter, optimizer, converter=convert, device=args.gpu)

            # Setup trainer and run
            trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'),
                                               out='saved_models')

            trainer.extend(
                extensions.Evaluator(
                    test_iter, model, converter=convert, device=args.gpu),
                trigger=(epoch_per_eval, 'epoch'))

            def get_rel_tensor_value(cur_trainer):
                return model.classifier.embed.W.data

            trainer.extend(extensions.observe_value(
                "rel_tensor", get_rel_tensor_value), trigger=(1, 'epoch'))

            trainer.extend(extensions.LogReport())

            trainer.extend(extensions.PrintReport(
                ['epoch', 'main/loss', 'validation/main/loss',
                 'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

            trainer.extend(extensions.snapshot(
                filename='snapshot_epoch-{.updater.epoch}'))
            trainer.extend(extensions.snapshot_object(
                model, filename='model_epoch-{.updater.epoch}'))

            trainer.run()
            continue
