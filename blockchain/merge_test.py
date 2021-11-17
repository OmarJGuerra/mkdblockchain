import kdtree
from block import Block
from wallet import Wallet

import copy
import random

TOTAL_NODES = 80
GENESIS_ID = int((TOTAL_NODES/2))
WALLETS = dict(zip([i+1 for i in range(TOTAL_NODES)], [Wallet() for _ in range(TOTAL_NODES)]))
# BASE_TREE1 = kdtree.create_root(4, GENESIS_ID, WALLETS[GENESIS_ID])
# BASE_TREE2 = copy.deepcopy(BASE_TREE1)


def random_block():
    temp_id = random.randint(0, TOTAL_NODES) + 1
    return Block([random.randint(0, 999) for _ in range(10)],
                 temp_id,
                 random.random() * 100, random.random() * 100,
                 WALLETS[temp_id].public_key_string(),
                 random.uniform(1635030000.000000, 1635035000.999999))


def generate_block_list(list_size):
    return [random_block() for _ in range(list_size)]


def generate_subtree(st_size):
    start_node = kdtree.KDNode(random_block(), axis=random.randint(0, 4), dimensions=4)
    for _ in range(0, st_size-1):
        start_node.add(random_block())
    return start_node


def add_identical(tree1, tree2, st_size):
    subtree = generate_subtree(st_size)
    tree1.add_node(subtree)
    tree2.add_node(subtree)


def merge_test(first=None, second=None):
    first = first or kdtree.create_root(4, GENESIS_ID, WALLETS[GENESIS_ID])
    second = second or copy.deepcopy(first)

    blocks1 = generate_block_list(30) if first.size == 1 else []
    blocks2 = generate_block_list(30) if second.size == 1 else []

    for b1, b2 in zip(blocks1, blocks2):
        first.add(b1)
        second.add(b2)

    kdtree.collect_dt(first, second)

