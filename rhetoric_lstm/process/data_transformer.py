import re
import codecs

import nltk

class Parser(object):
    def __init__(self, word2vec, n_label, path, vocabulary, max_limit,
                 enable_original_word=False):
        self.tokens = None
        self.pos = None
        self.word2vec = word2vec
        self.n_label = n_label
        self.trees = self.__read_corpus(path, vocabulary, max_limit,
                                        enable_original_word)

    def __parse(self):
        assert self.pos < len(self.tokens)
        token = self.tokens[self.pos]
        assert token != ')'
        self.pos += 1
        if token == '(':
            children = []
            while True:
                assert self.pos < len(self.tokens)
                if self.tokens[self.pos] == ')':
                    self.pos += 1
                    break
                else:
                    children.append(self.__parse())
            return children
        else:
            return token

    def __convert_tree(self, vocab, exp,
                       int_label_range, enable_original_word=False):
        if len(exp) == 3:
            label, str_relation, leaf = exp
            original_leaf = leaf
            stemmer = nltk.stem.SnowballStemmer('english')
            leaf = leaf.lower()
            leaf = leaf.replace('\\', '')
            if leaf == '-lrb-':
                leaf = '('
            elif leaf == '-rrb-':
                leaf = ')'
            if leaf in vocab:
                leaf_transformed = vocab[leaf]
            elif stemmer.stem(leaf) in vocab:
                leaf_transformed = vocab[stemmer.stem(leaf)]
            else:
                if leaf in self.word2vec.vocab:
                    leaf_transformed = vocab[leaf]
                elif stemmer.stem(leaf) in self.word2vec.vocab:
                    leaf_transformed = vocab[stemmer.stem(leaf)]
                else:
                    leaf_transformed = vocab[leaf]

            if int_label_range == 5:
                pass
            else:
                if int(label) >= 0 and int(label) < 2:
                    label = 0
                elif int(label) > 2 and int(label) < 5:
                    label = 1
                elif int(label) == 2:
                    label = 26
                elif int(label) >= 10 and int(label) < 12:
                    label = 10
                elif int(label) > 12 and int(label) < 15:
                    label = 11
                elif int(label) == 12:
                    label = 21
            lst_node = []
            lst_node.append(int(label))
            lst_node.append(leaf_transformed)
            lst_node.append(str_relation)

            if enable_original_word:
                lst_node.append(original_leaf)


            return lst_node

        elif len(exp) == 4:
            label, str_relation, left, right = exp

            node = (self.__convert_tree(vocab, left, self.n_label,
                                        enable_original_word),
                    self.__convert_tree(vocab, right, self.n_label,
                                        enable_original_word))
            if int_label_range == 5:
                pass
            else:
                if int(label) >= 0 and int(label) < 2:
                    label = 0
                elif int(label) > 2 and int(label) < 5:
                    label = 1
                elif int(label) == 2:
                    label = 26
                elif int(label) >= 10 and int(label) < 12:
                    label = 10
                elif int(label) > 12 and int(label) < 15:
                    label = 11
                elif int(label) == 12:
                    label = 21

            lst_node = []
            lst_node.append(int(label))
            lst_node.append(node)
            lst_node.append(str_relation)
            return lst_node

    def __read_corpus(self, path, vocabulary, max_limit,
                      enable_original_word=False):
        with codecs.open(path, encoding='utf-8') as f:
            trees = []
            for line in f:
                line = line.strip()

                self.tokens = re.findall(r'\(|\)|[^\(\) ]+', line)
                self.pos = 0

                single_tree = self.__parse()

                trees.append(self.__convert_tree(vocabulary, single_tree,
                                                 self.n_label,
                                                 enable_original_word))
                if max_limit and len(trees) >= max_limit:
                    break
            return trees
