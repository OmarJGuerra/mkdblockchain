# -*- coding: utf-8 -*-


"""
A Python implementation of a kd-tree.

This package provides a simple implementation of a kd-tree in Python.
https://en.wikipedia.org/wiki/K-d_tree
"""

from __future__ import print_function

import copy
import heapq
import itertools
import operator
import math
from collections import deque
from functools import wraps
import block  # removed from blockchain
from blockchain_utils import BlockchainUtils as BU  # removed from blockchain
import time
import csv

__author__ = u'Stefan Kögl <stefan@skoegl.net>'
__version__ = '0.16'
__website__ = 'https://github.com/stefankoegl/kdtree'
__license__ = 'ISC license'


class BaseNode(dict, object):
    """
    A Node in a kd-tree.

    A tree is represented by its root node, and every node represents
    its subtree.
    """
    def __init__(self, data=None, left=None, right=None):
        super().__init__()
        self.data = data
        self.left = left
        self.right = right

    # def __repr__(self):
    #     return f'Node({self.data}, {self.left}, {self.right})'

    @property
    def is_leaf(self):
        """
        Returns True if a Node has no subnodes.
        eg:
                >>> BaseNode().is_leaf
                True

                >>> BaseNode( 1, left=BaseNode(2) ).is_leaf
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


class KDNode(BaseNode):
    """A Node that contains kd-tree specific data and methods."""

    def __init__(self, data=None, left=None, right=None, axis=None,
                 sel_axis=None, dimensions=None, size=0, right_size=0, left_size=0):
        """
        Construct a new node for a kd-tree.

        Parameters:
            data (Point): any point-like object with coordinates
            left (KDNode): left child
            right (KDNode): right child
            axis (int): dimension index of this KDNode
            sel_axis: a function used when creating child nodes
            dimensions (int): total dimensions of the tree
        Data Parameters:
            size (int): size of the tree
            right_size (int): size of the right branch
            left_size(int): size of the left branch
        """
        super(KDNode, self).__init__(data, left, right)
        self.axis = axis
        self.sel_axis = sel_axis
        self.dimensions = dimensions
        self.size = size
        self.right_size = right_size
        self.left_size = left_size
        self.subtree_hash = BU.hash(self.data.coords).hexdigest()

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
        for _ in self.inorder():
            i += 1
        return i

    def __repr__(self):
        return f'KDNode({self.data}, {self.axis}, {self.dimensions}, {self.subtree_hash}'

    @require_axis
    def update_subtree_hash(self, tree2):
        """
        Update the subtree hash of a Merkle KD-tree.

        Hashes are updated in bottom-up fashion. The hash
        of the parent is included in the concatenation.
            (left + right + parent)
        Parameters:
            tree2 (KDNode): node of a KD-Tree
        """
        tree2_hash = tree2.subtree_hash
        node_list = self.search_node_parent(tree2)

        # First entry if second tree's parent is found
        while node_list:
            if node_list[1] is None:
                if node_list[2]:
                    node_list[0].subtree_hash = copy.deepcopy(concat_hashes("None", tree2_hash, node_list[0].subtree_hash))
                else:
                    node_list[0].subtree_hash = copy.deepcopy(concat_hashes(tree2_hash, "None", node_list[0].subtree_hash))
            else:
                if node_list[2]:
                    node_list[0].subtree_hash = copy.deepcopy(
                        concat_hashes(node_list[1].subtree_hash, tree2_hash, node_list[0].subtree_hash))
                else:
                    node_list[0].subtree_hash = copy.deepcopy(
                        concat_hashes(tree2_hash, node_list[1].subtree_hash, node_list[0].subtree_hash))
            tree2_hash = node_list[0].subtree_hash
            node_list = self.search_node_parent(node_list[0])

    @require_axis
    def add(self, point):  # point refers to a block for our purposes
        """
        Adds a point to the current node or iteratively descends to one of its children.

        Users should call add() only to the topmost tree.
        """

        current = self
        # print(f'current.left: {current.left} current.right: {current.right}')

        start_time = time.time()

        while True:
            check_dimensionality([point], dimensions=current.dimensions)
            # Adding has hit an empty leaf-node, add here
            if current.data is None:
                current.data = point
                current.size += 1

                time_taken = time.time() - start_time

                with open(f'time_to_add_block.csv', mode='a') as time_add:
                    time_add_writer = csv.writer(time_add, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    time_add_writer.writerow([self.size, self.left_size, self.right_size, time_taken])
                return current

            # split on self.axis, recurse either left or right
            if point[current.axis] < current.data[current.axis]:
                if current.left is None:
                    self.left_size += 1
                    current.size += 1
                    parent = current.data
                    current.left = current.create_subnode(point)
                    with open(f'time_to_add_block.csv', mode='a') as time_add:
                        time_add_writer = csv.writer(time_add, delimiter='.', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        time_taken = time.time() - start_time
                        time_add_writer.writerow([self.size, self.left_size, self.right_size, time_taken])
                    self.update_subtree_hash(current.left)
                    return current.left, parent
                else:
                    current.size += 1
                    current = current.left
            else:
                if current.right is None:
                    self.right_size += 1
                    current.size += 1
                    parent = current.data
                    current.right = current.create_subnode(point)
                    with open(f'time_to_add_block.csv', mode='a') as time_add:
                        time_add_writer = csv.writer(time_add, delimiter='.', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        time_taken = time.time() - start_time
                        time_add_writer.writerow([self.size, self.left_size, self.right_size, time_taken])
                        self.update_subtree_hash(current.right)
                    return current.right, parent
                else:
                    current.size += 1
                    current = current.right

    # I don't think .size is working now
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


    # # def aggregate(self, other_tree):
    # @require_axis
    # def merge(self, other_tree):
    #     if self.subtree_hash != other_tree.subtree_hash:
    #         merging_tree = other_tree if other_tree.size < self.size else self
    #         merged_into_tree = other_tree if other_tree.size >= self.size else self
    #
    #         for kdn in merging_tree.level_order():
    #             merged_into_tree.add_node(kdn)
    #
    #         return merged_into_tree
    #     else:
    #         return self


    @require_axis
    def add_node(self, node):
        """ Adds a node to the tree and re-hashes it. """
        if node != self:
            if node.data[self.axis] < self.data[self.axis]:
                if not self.left:
                    self.left = node
                    self.left.subtree_hash = BU.hash(self.left.data).hexdigest()
                    right_hash = self.right.subtree_hash if self.right is not None else ''
                    self.subtree_hash = concat_hashes(self.left.subtree_hash, right_hash)
                    self.left.data.parent_hash = BU.hash(self.data.to_json).hexdigest()
                    return self.left
                else:
                    return self.left.add_node(node)
            else:
                if not self.right:
                    self.right = node
                    self.right.subtree_hash = BU.hash(self.right.data.to_json).hexdigest()
                    left_hash = self.left.subtree_hash if self.left is not None else ''
                    self.subtree_hash = concat_hashes(left_hash, self.right.subtree_hash)
                    self.right.data.parent_hash = BU.hash(self.data.to_json).hexdigest()
                    return self.right
                else:
                    return self.right.add_node(node)
        else:
            print("Node already in tree")

    @require_axis
    def node_in_tree(self, node):
        """Returns: True if node in tree, False otherwise"""
        return True if self.search_node(node) is not None else False

    @require_axis
    def search_node(self, node):
        """
        Search the tree for a given node.

        KD search is an implementation of binary search. Instead of
        only comparing one type of value, it cycles through the tree's
        dimensions at each layer of the tree in order to compare a
        different dimensional value with every step.
        Each KD node in the tree stores its dimensional axis to make
        it easier to know which level it resides in.

        Returns:
            self (KDNode): the node found in the tree
            None: nothing if the node was not found
        """
        if node.data.coords == self.data.coords:
            return self
        else:
            if node.data.coords[self.axis] < self.data.coords[self.axis]:
                return self.left.search_node(node) if self.left is not None else None
            else:
                return self.right.search_node(node) if self.right is not None else None

    @require_axis
    def search_node_parent(self, node):
        """
        Search the tree for a given node's parent.

        Same search mechanism as search_node, but returns
        the found parent along with its children.

        Returns:
            search_info (list): list that contains a found parent and its children
        """
        search_info = []
        if node.data.coords == self.data.coords:
            return search_info
        else:
            if node.data.coords[self.axis] < self.data.coords[self.axis]:
                if self.left is not None:
                    if self.left.data.coords == node.data.coords:
                        search_info.append(self)
                        search_info.append(self.right)
                        search_info.append(0)
                        return search_info
                    return self.left.search_node_parent(node)
                else:
                    return search_info
            else:
                if self.right is not None:
                    if self.right.data.coords == node.data.coords:
                        search_info.append(self)
                        search_info.append(self.left)
                        search_info.append(1)
                        return search_info
                    return self.right.search_node_parent(node)
                else:
                    return search_info

    def search_parent(self, node):
        if node.data.coords == self.data.coords:
            return 0
        else:
            if node.data.coords[self.axis] < self.data.coords[self.axis]:
                # print(f'{node.data.coords[self.axis]} < {self.data.coords[self.axis]}')
                if self.left is not None:
                    if self.left.data.coords == node.data.coords:
                        return self
                    return self.left.search_parent(node)
                else:
                    # print('failed search')
                    return 0
            else:
                # print(f'{node.data.coords[self.axis]} >= {self.data.coords[self.axis]}')
                if self.right is not None:
                    if self.right.data.coords == node.data.coords:
                        return self
                    return self.right.search_parent(node)
                else:
                    # print('failed search')
                    return 0

    @require_axis
    def search_by_coords(self, coordinates):
        # print(f'coords: {coordinates}, self: {self.data}')
        if set(coordinates) == set(self.data):
            return self
        else:
            if coordinates[self.axis] < self.data[self.axis]:
                # print(f'{coordinates[self.axis]} < {self.data[self.axis]}')
                if self.left is not None:
                    return self.left.search_by_coords(coordinates)
                else:
                    # print('failed search')
                    return
            else:
                # print(f'{coordinates[self.axis]} >= {self.data[self.axis]}')
                if self.right is not None:
                    return self.right.search_by_coords(coordinates)
                else:
                    # print('failed search')
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


def rec_merge(agg_node, tree1, tree2):
    """
    Recursively merge two MKD-trees while ignoring identical subtrees.

    Searches the first tree for each of the second tree's nodes recursively.
    If a node is not found, it is published to all members, including the publisher.
    Ignores identical subtrees by comparing the subtree hashes.

    Parameters:
        agg_node (Node): network node performing the aggregation (NOT BaseNode)
        tree1 (KDNode): first tree involved in the merge
        tree2 (KDNode): second tree involved in the merge
    """
    test_node = tree1.search_node(tree2)
    if test_node:
        # Identical subtree detection
        if test_node.subtree_hash == tree2.subtree_hash:
            return
        # Only identical node, not identical subtree.
        if tree2.left:
            rec_merge(agg_node, tree1, tree2.left)
        if tree2.right:
            rec_merge(agg_node, tree1, tree2.right)
        return
    agg_node.publish(tree2.data)

    # After publishing, keep traversing if possible.
    if tree2.left:
        rec_merge(agg_node, tree1, tree2.left)
    if tree2.right:
        rec_merge(agg_node, tree1, tree2.right)
    return


def mergerr_no_iden(agg_node, tree1, tree2):
    """
    Recursively merge two MKD-trees.

    Searches the first tree for each of the second tree's nodes recursively.
    If a node is not found, it is published to all other members.

    Parameters:
        agg_node (Node): network node performing the aggregation (NOT BaseNode)
        tree1 (KDNode): first tree involved in the merge
        tree2 (KDNode): second tree involved in the merge
    """
    nodee = tree1.search_node(tree2)
    if nodee:
        if tree2.left:
            rec_merge(agg_node, tree1, tree2.left)
        if tree2.right:
            rec_merge(agg_node, tree1, tree2.right)
        return
    agg_node.publish(tree2.data)
    if tree2.left:
        rec_merge(agg_node, tree1, tree2.left)
    if tree2.right:
        rec_merge(agg_node, tree1, tree2.right)
    return


def identical_subtrees(test_num, root1, root2):
    """
    Detect identical subtrees between two Merkle KD-trees.

    Parameters:
        test_num (int): version of the test that is running
        root1 (KDNode): root node of the first tree
        root2 (KDNode): root node of the second tree
    """
    cw = csv.writer(open(f'duplicates_{test_num}.csv', 'a'))
    temp1 = ["B1 Size", "B2 Size", "B1,B2 Total", "No. Identical Subtrees", "Blocks in Id Trees", "total Id Blocks",
             "Identical / B1.size", "Identical / B2.size", "Ratio Id Blocks/ Id tree Blocks"]
    cw.writerow(list(temp1))
    temp1.clear()

    temp = list(comp_info(root1, root2))
    temp[1] = count_size(root2)
    temp2 = [temp[0], temp[1], (temp[0] + temp[1] - temp[4]), temp[2], temp[3], temp[4],
             float(temp[4]/(temp[0])), float(temp[4]/(temp[1]))]
    cw.writerow(list(temp2))
    cx = csv.writer(open(f"blockchains_{test_num-100}.csv", 'a'))
    cx.writerow(list(temp2))


def count_size(root):
    """Count the number of nodes in a Merkle KD-tree."""
    i = 1
    if root.left is not None:
        i += count_size(root.left)
    if root.right is not None:
        i += count_size(root.right)
    return i


def comp_info(root1, root2):
    """
    Compile Merkle KD-tree information.

    Traverses in post-order (left, right, parent).
    An info array is used to collect all relevant data points.
        info[0]: size of first tree
        info[1]: size of second tree
        info[2]: number of identical subtrees
        info[3]: amount of blocks in identical subtrees
        info[4]: total number of identical blocks

    Parameters:
        root1 (KDNode): root of first tree
        root2 (KDNode): root of second tree

    Returns:
        info (list): compiled list of information
    """
    info = [0, 0, 0, 0, 0]
    if not root1:
        return info
    else:
        info[0] = 1
    nodee = root2.search_node(root1)
    # If the node is found, test for identical subtree
    if nodee:
        # Identical subtree information is recorded
        if nodee.subtree_hash == root1.subtree_hash:
            info[0] = count_size(root1)
            info[2] = 1
            info[3] = info[0]
            info[4] = info[0]
            return info
        else:
            info[4] = 1

    # The node is not found, check children
    if root1.left:
        temp = comp_info(root1.left, root2)
        info[0] += temp[0]
        info[2] += temp[2]
        info[3] += temp[3]
        info[4] += temp[4]
    if root1.right:
        temp = comp_info(root1.right, root2)
        info[0] += temp[0]
        info[2] += temp[2]
        info[3] += temp[3]
        info[4] += temp[4]
    return info


def collect_hash(root):
    hashset = set()
    temp = []
    duplicates = set()
    temp.append(hashset)
    temp.append(duplicates)
    if ((root.left is None) & (root.right is None)):
        hashset.add(root.subtree_hash)
        temp[0] = hashset
        temp[1] = duplicates
        return temp
    elif root.right is None:
        temp = collect_hash(root.left)
        if root.subtree_hash in temp[0]:
            duplicates.add(root.subtree_hash)
            temp[1] = temp[1].union(duplicates)
            return temp
        else:
            hashset.add(root.subtree_hash)
            temp[0] = temp[0].union(hashset)
            return temp
    elif root.left is None:
        temp = collect_hash(root.right)
        if root.subtree_hash in temp[0]:
            duplicates.add(root.subtree_hash)
            temp[1] = temp[1].union(duplicates)
            return temp
        else:
            hashset.add(root.subtree_hash)
            temp[0] = temp[0].union(hashset)
            return temp
    else:
        temp = collect_hash(root.right)
        hashset = hashset.union(temp[0])
        duplicates = duplicates.union(temp[1])
        temp.clear()
        temp = collect_hash(root.left)
        duplicates = duplicates.union(hashset & temp[0])
        duplicates = duplicates.union(temp[1])
        hashset = hashset.union(temp[0])
        if root.subtree_hash in hashset:
            duplicates.add(root.subtree_hash)
            temp.clear()
            temp.append(hashset)
            temp.append(duplicates)
            return temp
        else:
            hashset.add(root.subtree_hash)
            temp.clear()
            temp.append(hashset)
            temp.append(duplicates)
            return temp

# # !!
# # TODO: Check
# def verify_subtree_hash(root, kdnode):
#     if kdnode != root:
#         if concat_hashes(root.left.subtree_hash, root.right.subtree_hash) == root.subtree_hash:
#             if kdnode.data[root.axis] < root.data[root.axis]:
#                 if kdnode == root.left:
#                     if concat_hashes(root.left.left.subtree_hash, root.left.right.subtree_hash) == kdnode.subtree_hash:
#                         return print('Valid')
#                     else:
#                         return print('Invalid')
#                 else:
#                     return verify_subtree_hash(root.left, kdnode)
#             else:
#                 if kdnode == root.right:
#                     if concat_hashes(root.right.left.subtree_hash, root.right.right.subtree_hash) == kdnode.subtree_hash:
#                         return print('Valid')
#                     else:
#                         return print('Invalid')
#                 else:
#                     return verify_subtree_hash(root.right, kdnode)
#         else:
#             print('Invalid')
#     elif kdnode.subtree_hash == root.subtree_hash:
#         print('Valid')
#     else:
#         print('Invalid')


def concat_hashes(hash1, hash2, parent_hash):
    """Concatenate the hash of a parent with those of its children."""
    return BU.hash(hash1 + hash2 + parent_hash).hexdigest()


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

    return KDNode(loc, left, right, axis=axis, sel_axis=sel_axis, dimensions=dimensions)


def create_root(dimensions, genesis_node_id, forger, sel_axis=None):
    """"""
    sel_axis = sel_axis or (lambda prev_axis: (prev_axis + 1) % dimensions)
    g_block = block.Block.genesis(genesis_node_id, forger)
    return KDNode(g_block, left=None, right=None, axis=0, sel_axis=sel_axis, size=1)


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


def bfprint(tree):
    temp = []
    temp2 = []
    temp.append(tree)

    temp.append("root")
    while temp:
        i = 1
        for x in temp:
            if i % 2:
                print("| ", end='')
                print(x.subtree_hash, end='')
                print("  /  ", end='')
                print(x.data, end='')
                print(" |", end='')
                if x.left:
                    temp2.append(x.left)
                    temp2.append(x.subtree_hash)
                if x.right:
                    temp2.append(x.right)
                    temp2.append(x.subtree_hash)
            else:
                print(x)
                print("| ", end='')
                print("| ", end='')
            i += 1

        print("\n")
        temp.clear()
        temp = copy.deepcopy(temp2)
        temp2.clear()
        temp2 = []


def csv_bfprint(root, test_num):
    """
    Print an MKD-tree into a .csv file.

    A breadth-first search algorithm modified in order to
    write the node, its parent, and its subtree hash to a .csv file.

    Parameters:
        root (KDNode): node representing the root of the tree
        test_num (int): number of the test that is running the function
    """
    temp = [root, "root"]
    cw = csv.writer(open(f"blockchains_{test_num}.csv", 'a'))

    temp1 = []
    temp2 = []
    while temp:
        i = 1
        for x in temp:
            if i % 2 == 1:
                temp1.append(str(x.data))
                temp1.append(x.subtree_hash)
                if x.left:
                    temp2.append(x.left)
                    temp2.append(x.data)
                if x.right:
                    temp2.append(x.right)
                    temp2.append(x.data)
            else:
                temp1.append(str(x))
                cw.writerow(list(temp1))
                temp1.clear()
            i += 1
        temp.clear()
        temp = copy.deepcopy(temp2)
        temp2.clear()
        i += 1


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

