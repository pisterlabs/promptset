from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import collections
from collections import defaultdict
import string

import copy
import numpy as np
from gym.spaces import Discrete, Box
import gym.spaces
import rllab.spaces

from pixelworld.spaces_gym import NamedDiscrete, NamedBox
import pixelworld.spaces_gym as spaces_gym
import pixelworld.spaces_rllab as spaces_rllab



def uniq(lst):
    seen = set()
    out = []
    for x in lst:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def uniq_pw_envs(pw_env_lst):
    unique_pw_envs = []
    pw_env_to_idx = {}
    for pw_env in pw_env_lst:
        try:
            seen_pw_env_idx = unique_pw_envs.index(pw_env)
            this_pw_env_idx = seen_pw_env_idx
        except ValueError:
            unique_pw_envs.append(pw_env)
            this_pw_env_idx = len(unique_pw_envs)-1

        pw_env_to_idx[pw_env] = this_pw_env_idx

    return unique_pw_envs, pw_env_to_idx


def print_nested_dict(x, prefix="", max_keys=0, next_max_keys=0):
    if isinstance(x, collections.Mapping):
        keys = sorted(x.keys())
        if max_keys > 0 and len(keys) > max_keys:
            keys = keys[:max_keys-1] + [None] + keys[-1:]
        for key in keys:
            if key is None:
                print("%s...(%s skipped)..." % (prefix, len(keys)-max_keys) )
            else:
                if isinstance(x[key], collections.Mapping):
                    print("%s%s:" % (prefix, key))
                    print_nested_dict(x[key], prefix+"  ", 
                        max_keys=next_max_keys, next_max_keys=next_max_keys)
                else:
                    print("%s%s: %s" % (prefix, key, x[key]))
    else:
        print("%s%s" % (prefix, x))


def set_nested(dict, keys, val, raise_on_reset=True):
    assert len(keys) > 0
    key = keys[0]
    if len(keys) == 1:      
        if key in dict and raise_on_reset:
            raise Exception("Setting key %s twice! old value %s new value %s" 
                                % (key, dict[key], val))
        dict[key] = val
    else:
        if key not in dict:
            dict[key] = {}
        set_nested(dict[key], keys[1:], val, raise_on_reset=raise_on_reset)


def recursive_merge_into(m1, m2, debug=False, list_new_keys=False):
    """Recursively merges m2 into m1, where m1 and m2 are nested dicts of dicts.
    Mutates m1, but not m2."""
    if debug:
        print("merge", "\n  m1:", m1, "\n  m2:", m2)
    if list_new_keys:
        new_keys = []
    for k in m2:
        if k in m1:
            if isinstance(m1[k], collections.Mapping) \
                and isinstance(m2[k], collections.Mapping):
                if debug:
                    print("  recursing into", k)
                recursive_merge_into(m1[k], m2[k])
            else:
                if debug:
                    print("  overwriting", k)
                m1[k] = copy.deepcopy(m2[k])
        else:
            if debug:
                print("  setting", k)
            if list_new_keys:
                new_keys.append(k)
            m1[k] = copy.deepcopy(m2[k])
    if list_new_keys:
        return new_keys


def flatten(lst):
    """Flatten a nested lists of lists into a single list. Strings are not 
    considered lists."""
    out = []
    for x in lst:
        if isinstance(x, basestring):
            out.append(x)
        elif isinstance(x, collections.Sequence):
            out.extend(flatten(x))
        else:
            out.append(x)
    return out


# borrowed from http://stackoverflow.com/questions/1446549/how-to-identify-binary-and-text-files-using-python
def is_text(filename):
    s=open(filename).read(512)
    text_characters = "".join(map(chr, range(32, 127)) + list("\n\r\t\b"))
    _null_trans = string.maketrans("", "")
    if not s:
        # Empty files are considered text
        return True
    if "\0" in s:
        # Files with null bytes are likely binary
        return False
    # Get the non-text characters (maps a character to itself then
    # use the 'remove' option to get rid of the text characters.)
    t = s.translate(_null_trans, text_characters)
    # If more than 30% non-text characters, then
    # this is considered a binary file
    if float(len(t))/float(len(s)) > 0.30:
        return False
    return True


def safe_annotate(obj, **annotation):
    """Annotate an object, squashing exceptions if not possible."""
    try:
        for k, v in annotation.items():
            setattr(obj, k, v)
    except:
        pass



# Borrowed and modified from OpenAI Gym
def np_random_seed(seed, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    rng.seed(_int_list_from_bigint(hash_seed(seed)))
    return rng

# Borrowed and modified from OpenAI Gym
def hash_seed(seed=None, max_bytes=8):
    """Any given evaluation is likely to have many PRNG's active at
    once. (Most commonly, because the environment is running in
    multiple processes.) There's literature indicating that having
    linear correlations between seeds of multiple PRNG's can correlate
    the outputs:

    http://blogs.unity3d.com/2015/01/07/a-primer-on-repeatable-random-numbers/
    http://stackoverflow.com/questions/1554958/how-different-do-random-seeds-need-to-be
    http://dl.acm.org/citation.cfm?id=1276928

    Thus, for sanity we hash the seeds before using them. (This scheme
    is likely not crypto-strength, but it should be good enough to get
    rid of simple correlations.)

    Args:
        seed (Optional[int]): None seeds from an operating system specific randomness source.
        max_bytes: Maximum number of bytes to use in the hashed seed.
    """
    if seed is None:
        seed = _seed(max_bytes=max_bytes)
    hash = hashlib.sha512(str(seed).encode('utf8')).digest()
    return _bigint_from_bytes(hash[:max_bytes])

# TODO: don't hardcode sizeof_int here
def _bigint_from_bytes(bytes):
    sizeof_int = 4
    padding = sizeof_int - len(bytes) % sizeof_int
    bytes += b'\0' * padding
    int_count = int(len(bytes) / sizeof_int)
    unpacked = struct.unpack("{}I".format(int_count), bytes)
    accum = 0
    for i, val in enumerate(unpacked):
        accum += 2 ** (sizeof_int * 8 * i) * val
    return accum

def _int_list_from_bigint(bigint):
    # Special case 0
    if bigint < 0:
        raise error.Error('Seed must be non-negative, not {}'.format(bigint))
    elif bigint == 0:
        return [0]

    ints = []
    while bigint > 0:
        bigint, mod = divmod(bigint, 2 ** 32)
        ints.append(mod)
    return ints


from scipy.misc import imresize

def better_imresize(arr, size, interp='bilinear', mode=None):
    # NOTE: this is to counter imresize using the first axis with dim. 3
    #       as the channel axis!
    if arr.shape[0] == 3 or arr.shape[1] == 3:
        if len(arr.shape) == 3:
            arr = np.transpose(arr, (2,0,1))
        elif len(arr.shape) ==2:
            arr = np.transpose(arr, (1,0))

    return imresize(arr, size, interp=interp, mode=mode)


def make_hashable(o):
    if isinstance(o, basestring):
        return o
    elif isinstance(o, collections.Mapping):
        return tuple([(k, make_hashable(v))  for k,v in o.items()])
    elif isinstance(o, collections.Sequence):
        return tuple([make_hashable(v) for v in o])
    else:
        return o

def compute_ancestors(graph, node):
    """Return the set of all proper ancestor nodes, i.e., those that can reach
    the given node by following a path of one or more outward edges in the 
    graph.
    """
    if node not in graph:
        return set() # node might not appear in graph if it doesn't have any ancestors
        #raise Exception("Node must be in graph.")
    fringe = set(graph[node])
    out = set()
    while len(fringe) > 0:
        node = fringe.pop()
        out.add(node)
        for next_node in graph[node]:
            if next_node not in out:
                fringe.add(next_node)
    return out

def reverse_graph(graph):
    rev_graph = defaultdict(set)
    for node, parents in graph.items():
        for parent in parents:
            rev_graph[parent].add(node)
    return rev_graph

def compute_descendants(graph, node):
    """Return the set of all proper descendant nodes, i.e., those that can be 
    reached from the given node by following a path of one or more outward edges
    in the graph.
    """
    return compute_ancestors(reverse_graph(graph), node)

def compute_depths(graph):
    rev_graph = reverse_graph(graph)
    num_parents = {node: len(parents) for node, parents in graph.items()}
    fringe = set([node for node, parents in graph.items() if len(parents) == 0])
    depths = {}
    while len(fringe) > 0:
        node = fringe.pop()
        depth = 0
        for parent_node in graph[node]:
            depth = max(depth, depths[parent_node] + 1)
        depths[node] = depth  
        for child_node in rev_graph[node]:
            assert num_parents[child_node] >= 1
            num_parents[child_node] -= 1
            if num_parents[child_node] == 0:
                fringe.add(child_node)
    
    for node in graph:
        assert depths[node] == max([0] + [depths[parent]+1 for parent in graph[node]])

    return depths


import threading

class IntervalTimer(object):
    def __init__(self, interval, function, args=[], kwargs={}):
        assert interval == int(interval)
        assert interval >= 1

        self.interval = int(interval)
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.timer = None
        self.running = False
        self.repeats_left = None

        self.main_thread = threading.enumerate()[0]
        if self.main_thread.name != 'MainThread':
            print("WARNING: thread[0] has name %s not MainThread" % (self.main_thread.name,))

    def _call(self):
        self.repeats_left -= 1
        if self.repeats_left <= 0:
            self.function(*self.args, **self.kwargs)
            self.repeats_left = self.interval
        if self.running:
            # Don't resume if main thread is not alive (e.g. due to an exception):
            if self.main_thread.is_alive():
                self.reset_timer()
            else:
                print("Main thread is no longer alive, shutting down IntervalTimer.")
    
    def reset_timer(self):
        if self.running:
            # Uses a 1 second timer so that _call can shut down the timer rapidly
            # if the main thread exits.
            self.timer = threading.Timer(1, self._call)
            self.timer.start()

    def start(self, delay=0):
        self.running = True
        self.repeats_left = self.interval + delay
        self.reset_timer()

    def stop(self, block=False):
        self.running = False
        if self.timer:
            self.timer.cancel()
            if block:
                self.timer.join()
            self.timer = None
