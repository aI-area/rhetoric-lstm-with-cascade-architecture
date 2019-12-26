
def traverse_child_tree(tree, lst_words, index2word):
    if type(tree[1]) is int:
        lst_words.append(index2word[tree[1]])
    else:
        traverse_child_tree(tree[1][0], lst_words, index2word)
        traverse_child_tree(tree[1][1], lst_words, index2word)


def linearize_tree(index2word, root, du_text2vector,
                   du_text2index, rst_rel2index, xp):
    # Left node indexes for all parent nodes
    lefts = []
    # Right node indexes for all parent nodes
    rights = []
    # Parent node indexes
    dests = []
    # All labels to predict for all parent nodes
    labels = []
    # indexes of non-leaf node
    non_leafs = []
    levels = []

    # All words of leaf nodes
    leafs = []
    # Leaf labels
    leaf_labels = []
    # rst relations
    relations = []

    # Current leaf node index
    leaf_index = [0]
    def traverse_leaf(exp, last_level_rel):
        if last_level_rel != "None" and exp[2] == "None":
            label, children, relation  = exp

            # DU vectors are archived
            if type(children) is tuple:
                lst_words = []
                traverse_child_tree(children[0], lst_words, index2word)
                traverse_child_tree(children[1], lst_words, index2word)
                context = " ".join(lst_words)
            else:
                context = index2word[children]

            if context + "\n" not in du_text2index:
                du_text2index[context + "\n"] = len(du_text2index)

            assert context + "\n" in du_text2vector
            leafs.append(du_text2index[context + "\n"])
            leaf_labels.append(int(label))
            leaf_index[0] += 1
        else:
            label, sibling, relation = exp
            rel_name, direction = relation.split("_____")

            if direction == "RightToLeft":
                right, left = sibling
            else:
                left, right = sibling

            traverse_leaf(left, relation)
            traverse_leaf(right, relation)

    traverse_leaf(root, root[2])

    # Current internal node index
    node_index = leaf_index
    leaf_index = [0]

    def traverse_node(exp, last_level_rel, boo_root):
        if last_level_rel != "None" and exp[2] == "None":
            leaf_index[0] += 1
            return leaf_index[0] - 1, 0
        else:
            label, children, relation = exp
            left, right = children

            # Judge direction of two adjcent spans
            rel_name, direction = relation.split("_____")
            if direction == "RightToLeft":
                left, right = right, left

            l, l_depth = traverse_node(left, relation, 0.0)
            l_depth += 1
            r, r_depth = traverse_node(right, relation, 0.0)
            r_depth += 1

            lefts.append(l)
            rights.append(r)
            dests.append(node_index[0])

            # Transform label value into standard value of
            # function softmax_cross_entropy and ignore nonsense label value
            # i.e. 20 and 25
            label = int(label) - 10
            if label >= 10:
                label = -1

            labels.append(int(label))
            relations.append(rst_rel2index[rel_name])

            # representation of current non-leaft node
            lst_words = []
            traverse_child_tree(exp, lst_words, index2word)
            context = " ".join(lst_words)

            if context + "\n" not in du_text2index:
                du_text2index[context + "\n"] = len(du_text2index)
            non_leafs.append(du_text2index[context + "\n"])
            levels.append(boo_root)

            node_index[0] += 1
            return node_index[0] - 1, max(l_depth, r_depth)

    _, depth = traverse_node(root, root[2], 1.0)
    assert len(lefts) == len(leafs) - 1

    return {
        'lefts': xp.array(lefts, 'i'),
        'rights': xp.array(rights, 'i'),
        'dests': xp.array(dests, 'i'),
        'leafs': xp.array(leafs, 'i'),
        'labels': xp.array(labels, 'i'),
        'leaf_labels': xp.array(leaf_labels, 'i'),
        'relations': xp.array(relations, 'i'),
        'non_leafs': xp.array(non_leafs, 'i'),
        'levels': xp.array(levels, 'i'),
    }