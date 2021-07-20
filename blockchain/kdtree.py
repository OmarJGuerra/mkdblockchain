# -*- coding: utf-8 -*-


"""
A Python implementation of a kd-tree.

This package provides a simple implementation of a kd-tree in Python.
https://en.wikipedia.org/wiki/K-d_tree
"""

from __future__ import print_function

import heapq
import itertools
import operator
import math
from collections import deque
from functools import wraps
import block  # removed from blockchain
from blockchain_utils import BlockchainUtils as BU  # removed from blockchain

__author__ = u'Stefan Kögl <stefan@skoegl.net>'
__version__ = '0.16'
__website__ = 'https://github.com/stefankoegl/kdtree'
__license__ = 'ISC license'


class Node(dict, object):
    """
    A Node in a kd-tree.

    A tree is represented by its root node, and every node represents
    its subtree.
    """
    def __init__(self, data=None, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    def __repr__(self):
        return f'Node({self.data}, {self.left}, {self.right})'

    @property
    def is_leaf(self):
        """
        Returns True if a Node has no subnodes.
        eg:
                >>> Node().is_leaf
                True

                >>> Node( 1, left=Node(2) ).is_leaf
                False
        """
        return (not self.data) or \
               (all(not bool(c) for c, p in self.children))

    def preorder(self):
        """iterator for nodes: root, left, right"""
        if not self:
            return

        yield self

        if self.left:
            for x in self.left.preorder():
                yield x

        if self.right:
            for x in self.right.preorder():
                yield x

    def inorder(self):
        """iterator for nodes: left, root, right"""
        if not self:
            return

        if self.left:
            for x in self.left.inorder():
                yield x

        yield self

        if self.right:
            for x in self.right.inorder():
                yield x

    def postorder(self):
        """iterator for nodes: left, right, root"""
        if not self:
            return

        if self.left:
            for x in self.left.postorder():
                yield x

        if self.right:
            for x in self.right.postorder():
                yield x

        yield self

    @property
    def children(self):
        """
        Returns an iterator for the non-empty children of the Node.

        The children are returned as (Node, pos) tuples where pos is 0 for the
        left subnode and 1 for the right.
        eg:
                >>> len(list(create(dimensions=2).children))
                0

                >>> len(list(create([ (1, 2) ]).children))
                0

                >>> len(list(create([ (2, 2), (2, 1), (2, 3) ]).children))
                2
        """
        if self.left and self.left.data is not None:
            yield self.left, 0
        if self.right and self.right.data is not None:
            yield self.right, 1

    def set_child(self, index, child):
        """
        Sets one of the node's children.

        index 0 refers to the left child, 1 to the right child.
        """
        if index == 0:
            self.left = child
        else:
            self.right = child

    def height(self):
        """
        Returns height of the (sub)tree, without considering empty leaf-nodes.
        eg:
                >>> create(dimensions=2).height()
                0

                >>> create([ (1, 2) ]).height()
                1

                >>> create([ (1, 2), (2, 3) ]).height()
                2
        """
        min_height = int(bool(self))
        return max([min_height] + [c.height() + 1 for c, p in self.children])

    def get_child_pos(self, child):
        """
        Returns the position of the given child.

        If the given node is the left child, 0 is returned.
        If it's the right child, 1 is returned.
        Otherwise None.
        """
        for c, pos in self.children:
            if child == c:
                return pos

    def __repr__(self):
        return '<%(cls)s - %(data)s>' % \
               dict(cls=self.__class__.__name__, data=repr(self.data))

    def __nonzero__(self):
        return self.data is not None

    __bool__ = __nonzero__

    def __eq__(self, other):
        if isinstance(other, tuple):
            return self.data == other
        else:
            return self.data == other.data

    def __hash__(self):
        return BU.hash(self.data)


def require_axis(f):
    """Check if the object of the function has axis and sel_axis members."""
    @wraps(f)
    def _wrapper(self, *args, **kwargs):
        if None in (self.axis, self.sel_axis):
            raise ValueError('%(func_name) rrequires the node %(node)s '
                             'to have an axis and a sel_axis function' %
                             dict(func_name=f.__name__, node=repr(self)))

        return f(self, *args, **kwargs)

    return _wrapper


class KDNode(Node):
    """A Node that contains kd-tree specific data and methods."""

    def __init__(self, data=None, left=None, right=None, axis=None,
                 sel_axis=None, dimensions=None, st_hash=None):
        """
        Creates a new node for a kd-tree.

        If the node will be used within a tree, the axis and the sel_axis
        function should be supplied.
        sel_axis(axis) is used when creating subnodes of the current node. It
        receives the axis of the parent node and returns the axis of the child
        node.

        """
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions
        self.size = 0
        self.st_hash = st_hash
        if left is None and right is None:
            self.subtree_hash = BU.hash(self.data).hexdigest()
        else:
            self.subtree_hash = BU.hash(left).hexdigest() if right is None \
                else BU.hash(right).hexdigest() if left is None \
                else BU.hash(BU.hash(left).hexdigest() + BU.hash(right)).hexdigest()

    def to_json(self):
        j_data = {'axis': self.axis,
                'dimensions': self.dimensions, 'size': self.size,
                'subtree_hash': self.subtree_hash}

        json_transactions = []
        for t in self.data.transactions:
            json_transactions.append(t.toJSon())
        j_data['transactions'] = json_transactions
        return j_data

    def __len__(self):
        i = 0
        for b in self.inorder():
            i += 1
        return i

    def __repr__(self):
        return f'KDNode({self.data}, {self.axis}, {self.dimensions}, {self.st_hash}'

    @require_axis
    def add(self, point):  # point refers to a block for our purposes
        """
        Adds a point to the current node or iteratively descends to one of its children.

        Users should call add() only to the topmost tree.
        """


        current = self
        # print(f'current.left: {current.left} current.right: {current.right}')

        traversed_kdnodes = [current]

        while True:
            check_dimensionality([point], dimensions=current.dimensions)
            # Adding has hit an empty leaf-node, add here
            if current.data is None:
                current.data = point
                current.size += 1
                return current

            # split on self.axis, recurse either left or right
            if int(point[current.axis]) < int(current.data[current.axis]):
                if current.left is None:
                    parent = current.data
                    current.left = current.create_subnode(point)
                    current.size += 1
                    traversed_kdnodes.append(current.left)
                    # print(f'Traversed kd Nodes: {traversed_kdnodes}')
                    return current.left, parent, traversed_kdnodes
                else:
                    current.size += 1
                    current = current.left
                    traversed_kdnodes.append(current)
            else:
                if current.right is None:
                    parent = current.data
                    current.right = current.create_subnode(point)
                    current.size += 1
                    traversed_kdnodes.append(current.right)
                    # print(f'Traversed kd Nodes: {traversed_kdnodes}')
                    return current.right, parent, traversed_kdnodes
                else:
                    current.size += 1
                    current = current.right
                    traversed_kdnodes.append(current)


    def get_left_right_size(self):
        left_size = self.left.size if self.left is not None else 0
        right_size = self.right.size if self.right is not None else 0
        return left_size, right_size

    @require_axis
    def create_subnode(self, data):
        """ Creates a subnode for the current node """
        return self.__class__(data,
                              axis=self.sel_axis(self.axis),
                              sel_axis=self.sel_axis,
                              dimensions=self.dimensions)

    #def aggregate(self, other_tree):
    @require_axis
    def merge(self, other_tree):
        if self.st_hash != other_tree.st_hash:
            merging_tree = other_tree if other_tree.size < self.size else self
            merged_into_tree = other_tree if other_tree.size >= self.size else self

            for kdn in merging_tree.level_order():
                merged_into_tree.add_node(kdn)

            return merged_into_tree
        else:
            return self

    @require_axis
    def add_node(self, node):
        """ Adds a node to the tree and re-hashes it. """
        if node != self:
            if node.data[self.axis] < self.data[self.axis]:
                if not self.left:
                    self.left = node
                    self.left.st_hash = BU.hash(self.left.data).hexdigest()
                    right_hash = self.right.st_hash if self.right is not None else ''
                    self.st_hash = concat_hashes(self.left.st_hash, right_hash)
                    self.left.data.parent_hash = BU.hash(self.data.to_json).hexdigest()
                    return self.left
                else:
                    return self.left.add_node(node)
            else:
                if not self.right:
                    self.right = node
                    self.right.st_hash = BU.hash(self.right.data.to_json).hexdigest()
                    left_hash = self.left.st_hash if self.left is not None else ''
                    self.st_hash = concat_hashes(left_hash, self.right.st_hash)
                    self.right.data.parent_hash = BU.hash(self.data.to_json).hexdigest()
                    return self.right
                else:
                    return self.right.add_node(node)
        else:
            print("Node already in tree")

    @require_axis
    def node_in_tree(self, node):
        return True if self.search_node(node) is not None else False

    @require_axis
    def search_node(self, node):
        if node.data.coords == self.data.coords:
            return self
        else:
            if node.data.coords[self.axis] < self.data.coords[self.axis]:
                #print(f'{node.data.coords[self.axis]} < {self.data.coords[self.axis]}')
                if self.left is not None:
                    return self.left.search_node(node)
                else:
                    #print('failed search')
                    return
            else:
                #print(f'{node.data.coords[self.axis]} >= {self.data.coords[self.axis]}')
                if self.right is not None:
                    return self.right.search_node(node)
                else:
                    #print('failed search')
                    return

    @require_axis
    def search_by_coords(self, coordinates):
        #print(f'coords: {coordinates}, self: {self.data}')
        if set(coordinates) == set(self.data):
            return self
        else:
            if coordinates[self.axis] < self.data[self.axis]:
                #print(f'{coordinates[self.axis]} < {self.data[self.axis]}')
                if self.left is not None:
                    return self.left.search_by_coords(coordinates)
                else:
                    #print('failed search')
                    return
            else:
                #print(f'{coordinates[self.axis]} >= {self.data[self.axis]}')
                if self.right is not None:
                    return self.right.search_by_coords(coordinates)
                else:
                    #print('failed search')
                    return


    @require_axis
    def find_replacement(self):
        """
        Finds a replacement for the current node.

        The replacement is returned as a
        (replacement-node, replacements-parent-node) tuple
        """
        if self.right:
            child, parent = self.right.extreme_child(min, self.axis)
        else:
            child, parent = self.left.extreme_child(max, self.axis)

        return child, parent if parent is not None else self

    def should_remove(self, point, node):
        """ checks if self's point (and maybe identity) matches """
        if not self.data == point:
            return False

        return (node is None) or (node is self)

    @require_axis
    def remove(self, point, node=None):
        """
        Removes the node with the given point from the tree.

        Returns the new root node of the (sub)tree.

        If there are multiple points matching "point", only one is removed. The
        optional "node" parameter is used for checking the identity, once the
        removal candidate is decided.
        """
        # Recursion has reached an empty leaf node, nothing here to delete
        if not self:
            return

        # Recursion has reached the node to be deleted
        if self.should_remove(point, node):
            return self._remove(point)

        # Remove direct subnode
        if self.left and self.left.should_remove(point, node):
            self.left = self.left._remove(point)

        elif self.right and self.right.should_remove(point, node):
            self.right = self.right._remove(point)

        # Recurse to subtrees
        if point[self.axis] <= self.data[self.axis]:
            if self.left:
                self.left = self.left.remove(point, node)

        if point[self.axis] >= self.data[self.axis]:
            if self.right:
                self.right = self.right.remove(point, node)

        return self

    @require_axis
    def _remove(self, point):
        # we have reached the node to be deleted here

        # deleting a leaf node is trivial
        if self.is_leaf:
            self.data = None
            return self

        # we have to delete a non-leaf node here

        # find a replacement for the node (will be the new subtree-root)
        root, max_p = self.find_replacement()

        # self and root swap positions
        tmp_l, tmp_r = self.left, self.right
        self.left, self.right = root.left, root.right
        root.left, root.right = tmp_l if tmp_l is not root else self, tmp_r if tmp_r is not root else self
        self.axis, root.axis = root.axis, self.axis

        # Special-case if we have not chosen a direct child as the replacement
        if max_p is not self:
            pos = max_p.get_child_pos(root)
            max_p.set_child(pos, self)
            max_p.remove(point, self)

        else:
            root.remove(point, self)

        return root

    @property
    def is_balanced(self):
        """
        Returns True if the (sub)tree is balanced.

        The tree is balanced if the heights of both subtrees differ at most
        by 1
        """
        left_height = self.left.height() if self.left else 0
        right_height = self.right.height() if self.right else 0

        if abs(left_height - right_height) > 1:
            return False

        return all(c.is_balanced for c, _ in self.children)

    def rebalance(self):
        """Returns the (possibly new) root of the rebalanced tree."""

        return create([x.data for x in self.inorder()])

    def axis_dist(self, point, axis):
        """
        Squared distance at the given axis between
        the current Node and the given point.
        """
        return math.pow(self.data[axis] - point[axis], 2)

    def dist(self, point):
        """
        Squared distance between the current Node
        and the given point.
        """
        r = range(self.dimensions)
        return sum([self.axis_dist(point, i) for i in r])

    def search_knn(self, point, k, dist=None):
        """
        Return the k nearest neighbors of point and their distances.

        point must be an actual point, not a node.

        k is the number of results to return. The actual results can be less
        (if there aren't more nodes to return) or more in case of equal
        distances.

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any comparable tr_type.

        The result is an ordered list of (node, distance) tuples.
        """
        if k < 1:
            raise ValueError("k must be greater than 0.")

        if dist is None:
            def get_dist(n): return n.dist(point)
        else:
            def get_dist(n): return dist(n.data, point)

        """Changed from lambda to function definition as defined by PEP 8"""
        # if dist is None:
        #     get_dist = lambda n: n.dist(point)
        # else:
        #     get_dist = lambda n: dist(n.data, point)

        results = []

        self._search_node(point, k, results, get_dist, itertools.count())

        # We sort the final result by the distance in the tuple
        # (<KdNode>, distance).
        return [(node, -d) for d, _, node in sorted(results, reverse=True)]

    def _search_node(self, point, k, results, get_dist, counter):
        if not self:
            return

        node_dist = get_dist(self)

        # Add current node to the priority queue if it closer than
        # at least one point in the queue.
        #
        # If the heap is at its capacity, we need to check if the
        # current node is closer than the current farthest node, and if
        # so, replace it.
        item = (-node_dist, next(counter), self)
        if len(results) >= k:
            if -node_dist > results[0][0]:
                heapq.heapreplace(results, item)
        else:
            heapq.heappush(results, item)
        # get the splitting plane
        split_plane = self.data[self.axis]
        # get the squared distance between the point and the splitting plane
        # (squared since all distances are squared).
        plane_dist = point[self.axis] - split_plane
        plane_dist2 = plane_dist * plane_dist

        # Search the side of the splitting plane that the point is in
        if point[self.axis] < split_plane:
            if self.left is not None:
                self.left._search_node(point, k, results, get_dist, counter)
        else:
            if self.right is not None:
                self.right._search_node(point, k, results, get_dist, counter)

        # Search the other side of the splitting plane if it may contain
        # points closer than the farthest point in the current results.
        if -plane_dist2 > results[0][0] or len(results) < k:
            if point[self.axis] < self.data[self.axis]:
                if self.right is not None:
                    self.right._search_node(point, k, results, get_dist,
                                            counter)
            else:
                if self.left is not None:
                    self.left._search_node(point, k, results, get_dist,
                                           counter)

    @require_axis
    def search_nn(self, point, dist=None):
        """
        Search the nearest node of the given point.

        point must be an actual point, not a node. The nearest node to the
        point is returned. If a location of an actual node is used, the Node
        with this location will be returned (not its neighbor).

        dist is a distance function, expecting two points and returning a
        distance value. Distance values can be any comparable tr_type.

        The result is a (node, distance) tuple.
        """
        return next(iter(self.search_knn(point, 1, dist)), None)

    def _search_nn_dist(self, point, dist, results, get_dist):
        if not self:
            return

        node_dist = get_dist(self)

        if node_dist < dist:
            results.append(self.data)

        # get the splitting plane
        split_plane = self.data[self.axis]

        # Search the side of the splitting plane that the point is in
        if point[self.axis] <= split_plane + dist:
            if self.left is not None:
                self.left._search_nn_dist(point, dist, results, get_dist)
        if point[self.axis] >= split_plane - dist:
            if self.right is not None:
                self.right._search_nn_dist(point, dist, results, get_dist)

    @require_axis
    def search_nn_dist(self, point, distance, best=None):
        """
        Search the n nearest nodes of the given point which are within given
        distance.

        point must be a location, not a node. A list containing the n nearest
        nodes to the point within the distance will be returned.
        """
        results = []
        def get_dist(n): return n.dist(point)

        self._search_nn_dist(point, distance, results, get_dist)
        return results

    @require_axis
    def is_valid(self):
        """
        Checks recursively if the tree is valid.

        It is valid if each node splits correctly.
        """

        if not self:
            return True

        if self.left and self.data[self.axis] < self.left.data[self.axis]:
            return False

        if self.right and self.data[self.axis] > self.right.data[self.axis]:
            return False

        return all(c.is_valid() for c, _ in self.children) or self.is_leaf

    def extreme_child(self, sel_func, axis):
        """
        Returns a child of the subtree and its parent.

        The child is selected by sel_func which is either min or max
        (or a different function with similar semantics).
        """

        def max_key(child_parent): return child_parent[0].data[axis]

        # we don't know our parent, so we include None
        me = [(self, None)] if self else []

        child_max = [c.extreme_child(sel_func, axis) for c, _ in self.children]
        # insert self for unknown parents
        child_max = [(c, p if p is not None else self) for c, p in child_max]

        candidates = me + child_max

        if not candidates:
            return None, None

        return sel_func(candidates, key=max_key)


def verify_mkd_blockchain(root):
    st = root.subtree_hash
    if root.left is None and root.right is None:
        if root.subtree_hash == '0':
            return True
        new_hash = BU.hash(root.data).hexdigest()
        return new_hash == st
    elif root.left is None:
        if verify_mkd_blockchain(root.right):
            return st == root.right.subtree_hash
        else:
            return False
    elif root.right is None:
        if verify_mkd_blockchain(root.left):
            return st == root.left.subtree_hash
        else:
            return False
    else:
        if verify_mkd_blockchain(root.left) and verify_mkd_blockchain(root.right):
            new_hash = concat_hashes(root.left.subtree_hash, root.right.subtree_hash)
            return new_hash == root.subtree_hash


# Want to change to create_subtree_hash or create_st_hash
def create_subtree_hash(traversed_kdnodes):
    for kdnode in reversed(traversed_kdnodes):
        if kdnode.left is not None and kdnode.right is not None:
            #print(f'Left: {kdnode.left.subtree_hash} Right: {kdnode.right.subtree_hash}')
            kdnode.subtree_hash = concat_hashes(kdnode.left.subtree_hash, kdnode.right.subtree_hash)
        elif kdnode.left is not None and kdnode.right is None:
            #print(f'Left: {kdnode.left.subtree_hash}')
            kdnode.subtree_hash = kdnode.left.subtree_hash
        elif kdnode.right is not None and kdnode.left is None:
            #print(f'Right: {kdnode.right.subtree_hash}')
            kdnode.subtree_hash = kdnode.right.subtree_hash
        else:
            kdnode.subtree_hash = BU.hash(kdnode.data.to_json()).hexdigest()


def verify_subtree_hash(root, kdnode):
    if kdnode != root:
        if concat_hashes(root.left.subtree_hash, root.right.subtree_hash) == root.subtree_hash:
            if kdnode.data[root.axis] < root.data[root.axis]:
                if kdnode == root.left:
                    if concat_hashes(root.left.left.subtree_hash, root.left.right.subtree_hash) == kdnode.subtree_hash:
                        return print('Valid')
                    else:
                        return print('Invalid')
                else:
                    return verify_subtree_hash(root.left, kdnode)
            else:
                if kdnode == root.right:
                    if concat_hashes(root.right.left.subtree_hash, root.right.right.subtree_hash) == kdnode.subtree_hash:
                        return print('Valid')
                    else:
                        return print('Invalid')
                else:
                    return verify_subtree_hash(root.right, kdnode)
        else:
            print('Invalid')
    elif kdnode.subtree_hash == root.subtree_hash:
        print('Valid')
    else:
        print('Invalid')


def concat_hashes(hash1, hash2):
    return BU.hash(hash1 + hash2).hexdigest()


def concat_hash_list(hashes):
    concat = ''
    for h in hashes:
        concat += h
    return BU.hash(concat).hexdigest()


def create(point_list=None, dimensions=None, axis=0, sel_axis=None):
    """
    Creates a kd-tree from a list of points.

    All points in the list must be of the same dimensionality.

    If no point_list is given, a genesis tree is created. The number of
    dimensions has to be given instead.

    If no point_list is given, an empty tree is created. The number of
    dimensions has to be given instead.

    If both a point_list and dimensions are given, the numbers must agree.

    Axis is the axis on which the root-node should split.

    sel_axis(axis) is used when creating subnodes of a node. It receives the
    axis of the parent node and returns the axis of the child node.
    """

    if not point_list and not dimensions:
        raise ValueError('either point_list or dimensions must be provided')

    elif point_list:
        dimensions = check_dimensionality(point_list, dimensions)

    # by default cycle through the axis
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis + 1) % dimensions)

    if not point_list:
        return KDNode(sel_axis=sel_axis, axis=axis, dimensions=dimensions)

    # Sort point list and choose median as pivot element
    point_list = list(point_list)
    point_list.sort(key=lambda point: point[axis])
    median = len(point_list) // 2

    loc = point_list[median]
    left = create(point_list[:median], dimensions, sel_axis(axis))
    right = create(point_list[median + 1:], dimensions, sel_axis(axis))

    """may not be necessary"""
    # loc.block.point_list = point_list

    if left.data is None and right.data is None:
        hashed = BU.hash(loc)
    else:
        hashed = BU.hash(left.data) if right is None \
            else BU.hash(right.data) if left is None \
            else BU.hash(BU.hash(left.data) + BU.hash(right.data))

    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions, st_hash=hashed)


def create_root(dimensions, genesis_node_id, forger, sel_axis=None):
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis + 1) % dimensions)
    g_block = block.Block.genesis(genesis_node_id, forger)
    return KDNode(g_block, left=None, right=None, axis=0, sel_axis=sel_axis, st_hash='0')


def check_dimensionality(point_list, dimensions=None):
    dimensions = dimensions or len(point_list[0])
    for p in point_list:
        if len(p) != dimensions:
            raise ValueError('All Points in the point_list must have the same dimensionality')

    return dimensions


def level_order(tree, include_all=False):
    """
    Returns an iterator over the tree in level-order.

    If include_all is set to True, empty parts of the tree are filled
    with dummy entries and the iterator becomes infinite.
    """

    q = deque()
    q.append(tree)
    while q:
        node = q.popleft()
        yield node

        if include_all or node.left:
            q.append(node.left or node.__class__())

        if include_all or node.right:
            q.append(node.right or node.__class__())


def visualize(tree, max_level=100, node_width=10, left_padding=5):
    """Prints the tree to stdout."""

    height = min(max_level, tree.height() - 1)
    max_width = pow(2, height)

    per_level = 1
    in_level = 0
    level = 0

    for node in level_order(tree, include_all=True):

        if in_level == 0:
            print()
            print()
            print(' ' * left_padding, end=' ')

        width = int(max_width * node_width / per_level)

        node_str = ((str(node.data) + ', ' + node.subtree_hash) if node else '').center(width)
        print(node_str, end=' ')

        in_level += 1

        if in_level == per_level:
            in_level = 0
            per_level *= 2
            level += 1

        if level > height:
            break

    print()
    print()
