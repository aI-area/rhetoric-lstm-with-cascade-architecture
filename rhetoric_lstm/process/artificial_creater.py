import random


class Generator(object):
    def __init__(self, vocab_segment_vector, trees, vocab):
        self.vocab_segment_vector = vocab_segment_vector
        self.trees_leaf_inserted = []
        self.trees = trees
        self.vocab = vocab
        self.voi_leaf_insertion()
        self.changed = None

    def transform2string(self, single_tree, lis_string):
        if not isinstance(single_tree["node"], int):
            self.transform2string(single_tree["node"][0], lis_string)
            self.transform2string(single_tree["node"][1], lis_string)
        else:
            lis_string.append((single_tree["node"]))

    def voi_artificial_leaf_insertion(self, ord_tree, entire_tree):
        if ord_tree["relation"] != "None":
            self.voi_artificial_leaf_insertion(ord_tree["node"][0],
                                               entire_tree)
            self.voi_artificial_leaf_insertion(ord_tree["node"][1],
                                               entire_tree)
        elif isinstance(ord_tree['node'], int):
            pass
        else:
            if random.uniform(0, 1) >= -1:
                lis_segment_string = []
                self.transform2string(ord_tree, lis_segment_string)

                str_tmp = ""
                for item in lis_segment_string:
                    str_tmp += self.vocab[item] + " "

                complete_distribution \
                    = self.vocab_segment_vector[str_tmp[:-1] + "\n"]

                flo_epsilon = random.uniform(0, 1)
                ord_tree["artificial_left_leaf"]\
                    = flo_epsilon * complete_distribution
                ord_tree["artificial_right_leaf"]\
                    = (1 - flo_epsilon) * complete_distribution

                ord_tree["artificial_relation"] \
                    = str(random.sample(["joint", "elaboration", "background",
                                         "attribution", "same-unit",
                                         "contrast"], 1)[0]) + "_____" + \
                    str(random.sample(["LeftToRight", "RightToLeft"], 1)[0])
                self.changed = True

        return ord_tree

    def voi_leaf_insertion(self):
        for item in self.trees:
            self.changed = False
            if item['label'] >= 9 and item['label'] != 25 and \
                    item['label'] != 26:
                self.voi_artificial_leaf_insertion(item, item)
                # if self.changed == True:
                self.trees_leaf_inserted.append(item)
