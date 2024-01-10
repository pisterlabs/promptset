# Taken from OpenAI baselines:
# https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
# https://github.com/openai/baselines/commit/69f25c6028d71e23a7b267b0e1071ad216290f91

import numpy as np


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.
        https://en.wikipedia.org/wiki/Segment_tree
        Can be used as regular array, but with two
        important differences:
            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient ( O(log segment size) )
               `reduce` operation which reduces `operation` over
               a contiguous subsequence of items in the array.
        Parameters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must form a mathematical group together with the set of
            possible values for array elements (i.e. be associative)
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [np.float64(neutral_element) for _ in range(2 * capacity)]
        self._operation = operation

        #print(self._value)

    def _reduce_helper(self, start, end, node, node_start, node_end):
        #print(start, end, node, node_start, node_end)
        if start == node_start and end == node_end:
            #print(self._value[node])
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            #print(1)
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                #print(2)
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                #print(3)
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.
            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))
        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences
        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        #print('reduce 0 - ', start, end, self._capacity)
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        #print('reduce 1 - ', start, end)
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        #print(idx, self._capacity)
        # idx += self._capacity
        idx = int(idx + self._capacity)
        #print(idx)
        self._value[idx] = np.float64(val)
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=lambda x, y: x+y,
            neutral_element=np.float64(0.0)
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        ret = super(SumSegmentTree, self).reduce(start, end)
        #print("sum", start, end, ret)
        return ret
        #return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum
        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.
        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix
        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        #print('minseg')
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=np.float64(np.inf) #float('inf')
        )
        #print(np.inf)

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)