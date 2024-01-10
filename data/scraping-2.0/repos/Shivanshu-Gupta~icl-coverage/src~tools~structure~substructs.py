import ast
import nltk
import pandas as pd
import numpy as np

from nltk import Tree
from collections import defaultdict
from itertools import product
from functools import cache
from langchain.utilities.track import track
from rich import print

@cache
def all_partitions(size):
    import more_itertools
    return [[len(part) for part in p] for p in more_itertools.partitions(range(size))]

@cache
def partitions(size, n_parts):
    return [p for p in all_partitions(size) if len(p) == n_parts]

get_tree_size = lambda t: 1 + sum([get_tree_size(c) for c in t]) if isinstance(t, nltk.Tree) else 1

get_tuple_size = lambda t: 1 + sum([get_tuple_size(c) for c in t[1]]) if isinstance(t, tuple) else 1

get_tree_root = lambda t: t.label() if isinstance(t, nltk.Tree) else t

def tree_to_tuple(t):
    if isinstance(t, nltk.Tree):
        return (t.label(), tuple([tree_to_tuple(c) for c in t]))
    else:
        return t

def tuple_to_tree(t):
    if isinstance(t, tuple):
        return nltk.Tree(t[0], [tuple_to_tree(c) for c in t[1]])
    else:
        return t

def get_subtrees(t: nltk.Tree, max_size: int, context_type: str = None,
                 path: nltk.Tree = None, node: nltk.Tree = None,
                 result: list[nltk.Tree] = None) -> tuple[list[nltk.Tree], list[nltk.Tree]]:
    assert isinstance(t, nltk.Tree)

    if context_type in ['rule', 'nt'] and (path == None or node == None):
        assert path == None and node == None
        path = node = nltk.Tree(t.label(), [])

    if result == None:
        # result = set()
        result = defaultdict(lambda: 0)

    subtrees = defaultdict(lambda: 0, {(t.label(), tuple()): 1})
    # subtrees = set()
    all_child_trees = []
    for i in range(len(t)):
        if isinstance(t[i], nltk.Tree):
            if context_type == 'rule':
                node[:] = [nltk.Tree(c.label(), []) if isinstance(c, nltk.Tree) else c
                           for c in t]
                c_node = node[i]
            elif context_type == 'nt':
                node[:] = [nltk.Tree(t[i].label(), [])]
                c_node = node[0]
            else:
                c_node = None
            c_trees = get_subtrees(t[i], max_size, context_type,
                                   path, c_node, result)[0]
        else:
            c_trees = [t[i]]
            # c_trees = []
        all_child_trees.append(c_trees)
    for c_sizes in partitions(max_size + len(t) - 1, len(t)):
        c_trees = [[t for t in c_trees if get_tuple_size(t) < c_sz]
                    for c_sz, c_trees in zip(c_sizes, all_child_trees) if c_sz > 1]
        for c_tree_comb in product(*c_trees):
            # subtrees.add(tree_to_tuple(nltk.Tree(t.label(), c_tree_comb)))
            subtrees[(t.label(), c_tree_comb)] += 1
    if context_type in ['rule', 'nt']:
        # subtrees = [tuple_to_tree(st) for st in subtrees]
        for st in subtrees:
            node[:] = tuple_to_tree(st)[:]
            result[tree_to_tuple(path)] += 1
    else:
        result |= subtrees
        # result.update(subtrees)
        # subtrees = [tuple_to_tree(st) for st in subtrees]
    # print(subtrees)

    if node is not None: node[:] = []
    return list(subtrees.keys()), list(result.keys())

def get_templates(trn_pool):
    all_templates = set()
    template2ex = defaultdict(dict)
    ex2template = {}
    targets = trn_pool.anonymized_target
    for qid, target in track(zip(trn_pool['qid'], targets),
                            total=trn_pool.shape[0],
                            description="Extracting templates"):
        ex2template[qid] = {target: 1}
        all_templates.add(target)
        template2ex[target][qid] = 1
    print(f'{len(all_templates)} templates')
    return dict(template2ex), ex2template

def get_dep_tree_spacy(sent: str, nlp=None):
    def nltk_tree(node):
        from nltk import Tree
        if node.n_lefts + node.n_rights > 0:
            return Tree(node.orth_, [nltk_tree(child) for child in node.children])
        else:
            return Tree(node.orth_, [])
            # return node.orth_

    import spacy
    # spacy.require_gpu(7
    nlp = nlp or spacy.load('en_core_web_sm')
    doc = nlp(sent)
    # dep_trees = [Tree('<s>', [nltk_tree(sent.root)]) for sent in doc.sents]
    dep_trees = [nltk_tree(sent.root) for sent in doc.sents]
    if False: dep_trees = [t for t in dep_trees if isinstance(t, nltk.Tree)]
    # for t in dep_trees:
    #     assert isinstance(t, nltk.Tree)
    return dep_trees

def get_dep_tree_displacy(sent: str, nlp=None):
    def nltk_tree(words, arcs):
        from nltk import Tree
        trees = [Tree(word['text'], []) for word in words]
        root_cands = set(range(len(words)))
        for arc in arcs:
            trees[arc['start']].append(trees[arc['end']])
            if arc['end'] in root_cands:
                root_cands.remove(arc['end'])
        for t in trees:
            for i, c in enumerate(t):
                if len(c) == 0:
                    t[i] = c.label()
        # assert len(root_cands) == 1
        return [trees[i] for i in root_cands]


    import spacy
    from spacy import displacy
    # spacy.require_gpu(7
    nlp = nlp or spacy.load('en_core_web_sm')
    doc = nlp(sent)
    parse = displacy.parse_deps(doc, options={'collapse_punct': True, 'collapse_phrases': True})
    # dep_trees = [Tree('<s>', [t]) for t in nltk_tree(parse['words'], parse['arcs'])]
    dep_trees = [t for t in nltk_tree(parse['words'], parse['arcs'])]
    if False: dep_trees = [t for t in dep_trees if isinstance(t, nltk.Tree)]
    # for t in dep_trees:
    #     assert isinstance(t, nltk.Tree)
    return dep_trees

def get_dep_tree_stanza(sent: str, nlp=None):
    # https://stanfordnlp.github.io/stanza/depparse.html
    # http://stanza.run/
    def nltk_tree(sent):
        from nltk import Tree
        trees = {}
        tree = None
        for node in sent.words:
            if node.id not in trees:
                trees[node.id] = Tree(node.text, [])
            if node.head != 0:
                if node.head not in trees:
                    trees[node.head] = Tree(sent.words[node.head - 1].text, [])
                trees[node.head].append(trees[node.id])
            else:
                tree = trees[node.id]
        return tree

    import stanza
    if nlp is None:
        try:
            nlp = stanza.Pipeline('en')
        except:
            nlp = stanza.Pipeline('en')
    doc = nlp(sent)

    # dep_trees = [Tree('<s>', [nltk_tree(sent)]) for sent in doc.sentences]
    dep_trees = [nltk_tree(sent) for sent in doc.sentences]
    if False: dep_trees = [t for t in dep_trees if isinstance(t, nltk.Tree)]
    return dep_trees

def get_parser(parser):
    if parser in ['spacy', 'displacy']:
        import spacy
        # spacy.require_gpu(7)
        nlp = spacy.load('en_core_web_sm')
    elif parser == 'stanza':
        import stanza
        try:
            nlp = stanza.Pipeline('en')
        except:
            nlp = stanza.Pipeline('en')
    else:
        raise ValueError(f'Unknown parser: {parser}')
    return nlp


def get_depsubtrees(sents, max_size, context_type, verbose=True, parser='spacy', nlp=None):
    nlp = nlp or get_parser(parser)

    all_depsts = set()
    depst2ex = defaultdict(list)
    ex2depst = []

    for qid, sent in enumerate(track(sents, description="Extracting Dependency subtrees", disable=not verbose)):

        dep_trees = dict(
            stanza=get_dep_tree_stanza,
            spacy=get_dep_tree_spacy,
            displacy=get_dep_tree_displacy,
        )[parser](sent, nlp)

        ex_depsts = []
        for dep_tree in dep_trees:
            _, ex_depsts_sent = get_subtrees(dep_tree, max_size, context_type)
            ex_depsts.extend(ex_depsts_sent)

        ex2depst.append(ex_depsts)
        for depst in ex_depsts:
            all_depsts.add(depst)
            depst2ex[depst].append(qid)
    return dict(depst2ex), ex2depst

def get_sent_ngrams(sents, max_n, min_n=1, verbose=True):
    from nltk import ngrams
    from collections import defaultdict
    all_ngrams = set()
    ngram2ex = defaultdict(list)
    ex2ngram = []
    for qid, sent in enumerate(track(sents, description="Extracting ngrams",
                                     disable=not verbose)):
        ex_ngrams = [ng for n in range(min_n, max_n+1) for ng in ngrams(sent.split(), n) if ng]
        # ex2ngram[qid] = Counter(ex_ngrams)
        # ex2ngram[qid] = {ng: 1 for ng in ex_ngrams}
        ex2ngram.append(ex_ngrams)
        for ng in ex_ngrams:
            all_ngrams.add(ng)
            ngram2ex[ng].append(qid)
    return dict(ngram2ex), ex2ngram

def get_lfsubtrees(targets, max_size, context_type, verbose=True):
    from tools.structure.ast_parser import target_to_ast
    # trees = [target_to_ast(target) for target in trn_pool.target]
    all_subtrees = set()
    subtree2ex = defaultdict(list)
    ex2subtree = []
    for qid, target in enumerate(track(targets, description="Extracting LF subtrees",
                                       disable=not verbose)):
        tree = target_to_ast(target)
        _, ex_subtrees = get_subtrees(tree, max_size, context_type)
        ex2subtree.append(ex_subtrees)
        for subtree in ex_subtrees:
            all_subtrees.add(subtree)
            subtree2ex[subtree].append(qid)
    # print(f'{len(all_subtrees)} subtrees with max size {max_size} and context type {context_type}')
    # all_subtrees = [tuple_to_tree(st) for st in all_subtrees]
    # return {k: dict(v) for k, v in subtree2ex.items()}, ex2subtree
    return dict(subtree2ex), ex2subtree

def get_substructs(strings, substruct, subst_size, depparser='spacy', parser=None, verbose=False):
    from tools.structure.substructs import get_parser, get_depsubtrees, get_sent_ngrams, get_lfsubtrees
    if substruct == 'depst':
        parser = parser or get_parser(depparser)
        _, str2struct = get_depsubtrees(strings, subst_size, None, nlp=parser, verbose=verbose)
    elif substruct == 'ngram':
        _, str2struct = get_sent_ngrams(strings, subst_size, verbose=verbose)
    elif substruct == 'lfst':
        _, str2struct = get_lfsubtrees(strings, subst_size, None, verbose=verbose)
    else:
        raise ValueError(f"Unknown substruct {substruct}")
    str2struct = [set(d) for d in str2struct]
    return str2struct