
# Source adapted from openai-baselines :
# https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py
#
# This implementation follows the SumTree variant discussed in appendix B.2.1
# from the Priritized Experience Replay paper (https://arxiv.org/pdf/1511.05952.pdf)
#
# The proportianl prioritization variant uses a SumTree (the code below) to ...
# do efficient inserts, updates and samples. However, it seems that for min ...
# computation (to get max importance sampling weight wi) we need to access ...
# the min priority efficiently. One way could be to have an extra variable ...
# to hold the min and update it as new experiences, or updates in priority ...
# occur. Another approach is to use a separate structure (mintree) to allow ...
# O(log n  +  k) access to the min, which is the approach taken here.
#

import math
import numpy as np

class SegmenTree( object ) :

    def __init__( self, bufferSize, operator, neutralElement ) :
        """Creates a segment tree based on the 'operator' given
        
        Args:
            bufferSize (int)        : capacity of the tree (number of leaves it can hold)
            operator (lambda)       : operator to be applied to construct nodes in the tree
            neutralElement (float)  : neutral element respect to the operator given

        """

        super( SegmenTree, self ).__init__()

        # sanity-check: size of buffer should be power of 2
        assert ( (bufferSize > 0) and ( bufferSize & (bufferSize - 1) == 0 ) ), \
               'ERROR> buffer size for segmentree should be power of 2'               

        # capacity of the tree
        self._bufferSize = bufferSize

        # operator used for construction parents of the leaves
        self._operator = operator

        # neutral element respect to the operator
        self._neutralElement = neutralElement

        # data used for the tree representation in ...
        # array-like form, kind of similar to a heap
        self._tree = np.array( [ neutralElement for _ in range( 2 * bufferSize - 1 ) ], dtype = np.float32 )

        # actual buffer of the data (why not list?, perhaps it is just an np array of pointers)
        self._data = np.zeros( bufferSize, dtype = object )

        # position pointer in the buffer
        self._pos = 0

        # a flag to indicate the buffer has been filled
        self._filled = False

    def add( self, data, nodeval ) :
        """Adds a node with internal data 'data' and node value 'nodeval'
    
        Args:
            data (object)   : the actual data stored in the node
            nodeval         : the value of this node (e.g. priority)
        """

        # index in the tree buffer
        _indx = self._pos + self._bufferSize - 1

        # store the data in the appropriate position in the data buffer
        self._data[self._pos] = data

        # update the node value in the tree buffer
        self.update( _indx, nodeval )

        # move the pointer to the next appropriate position
        self._pos += 1
        if self._pos >= self._bufferSize :
            self._pos = 0
            self._filled = True # indicates that the buffer has been filled


    def update( self, index, nodeval ) :
        """Updates the tree, starting at the given index and then upwards

        Args:
            index (int)     : index of the node in tree to start the update
            nodeval (float) : value of the node in tree to start the update

        """

        # update the value of this node in the tree
        self._tree[index] = nodeval

        # recursively update the nodes above
        self._propagate( index )

    def _propagate( self, index ) :
        """Recursively update nodes above from the changes of the given node

        Args:
            index (int): index of the node in the tree that changed its value

        """
        # grab the subling node for this node
        _siblingIndx = ( index - 1 ) if index % 2 == 0  else ( index + 1 )

        # grab the parent node for this node
        _parentIndx = ( index - 1 ) // 2

        # do the operator in the parent node
        self._tree[_parentIndx] = self._operator( self._tree[index], 
                                                  self._tree[_siblingIndx] )

        if _parentIndx != 0 :
            self._propagate( _parentIndx )

class SumTree( SegmenTree ) :

    def __init__( self, bufferSize ) :
        """Creates a sumtree, inhering from a segmentree with sum operator
           and 0 as neutral element. All node-values are assumed to be positive.

           This structure is usually used in the context of efficient sampling
           based on the node-values as measures of likelihood

        Args:
            bufferSize (int) : capacity of the tree (number of leaves it can hold)

        """
        super( SumTree, self ).__init__( bufferSize, lambda x, y : x + y, 0 )

    def getNode( self, value ) :
        """Gets a node given a certain value that it should have (close to it)
        
        Args:
            value (float) : a value used to locate a node close to it

        """
        
        # search recursively in the tree
        _indx = self._retrieve( 0, value )

        # check if not filled and got to last valid element
        if not self._filled :
            _lastIndx = ( self._pos - 1 ) + self._bufferSize - 1
            if _indx > _lastIndx :
                _indx = _lastIndx

        # for this index, grab the corresponding index in the data buffer
        _dataIndx = _indx - self._bufferSize + 1

        # return the index in the tree, the node-value and the actual data
        return ( _indx, self._tree[_indx], self._data[_dataIndx] )

    def sum( self ) :
        """Returns the total cumsum of node-values (stored at the root)

        """
        return self._tree[0]

    def _retrieve( self, index, value ) :
        """Searchs recursively (starting from 'index') for a node with value close to 'value'

        Args:
            index (int)     : index in the tree where to start the search
            value (float)   : value (or close to it) of the resulting node we want

        """

        _leftIndx = 2 * index + 1
        _rightIndx = _leftIndx + 1

        if _leftIndx >= len( self._tree ) :
            return index

        if value <= self._tree[_leftIndx] :
            return self._retrieve( _leftIndx, value )
        else :
            return self._retrieve( _rightIndx, value - self._tree[_leftIndx] )


class MinTree( SegmenTree ) :

    def __init__( self, bufferSize ) :
        """Creates a mintree, inheriting from a segmentree with min operator
           and 'inf' as neutral element.

           This structure is usually used in the context of getting the min
           node value of all leaves

        """
        super( MinTree, self ).__init__( bufferSize, lambda x, y : min(x, y), float( 'inf' ) )

    def min( self ) :
        """Returns the minimum nodevalue in the tree

        """
        return self._tree[0]