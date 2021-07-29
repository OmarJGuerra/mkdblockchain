from block import Block
from blockchain_utils import BlockchainUtils
from account_model import AccountModel
from proof_of_stake import ProofOfStake
from pydag.dag import DAG
from blockchain import block
from matplotlib import pyplot as plt

# for visualize
import itertools as itrt
from collections import deque
from ordered_set import OrderedSet
from typing import Iterator, AbstractSet, Dict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import networkx as nx
import numpy as np


class DAGBlockchain:
    def __init__(self, dimensions, gnode_id, genesis_forger):
        # self.blocks = kdtree.KDNode(Block.genesis(), left=None, right=None, axis=0, sel_axis=1, )
        # self.blocks = kdtree.create_root(dimensions, gnode_id, genesis_forger)
        dag = DAG()
        g_block = block.Block.genesis(gnode_id, genesis_forger)
        dag.add_node(g_block)
        self.blocks = [g_block]
        self.size = dag.size
        self.account_model = AccountModel()
        # self.pos = ProofOfStake()
        self.chain_id = 0
        self.gnode_id = gnode_id
        self.dag = dag

    def add_block(self, new_block):
        self.execute_transactions(new_block.transactions)
        self.blocks.append(new_block)
        self.dag.add_node(new_block)
        # self.dag.add_edge(new_block.node_id, self.gnode_id)
        self.dag.add_edge(new_block.node_id, self.dag.leaf_selection(new_block, self.gnode_id))
        print('block add successful')
        # self.blocks.append(block)03

    def merge(self, blockchain2):
        for first_nodes in blockchain2.dag.predecessors(blockchain2.dag.all_leaves()[0]):
            blockchain2.dag.graph[first_nodes].edges.remove(blockchain2.gnode_id)
            self.dag.add_node_chain(first_nodes, blockchain2.dag.graph)
            self.dag.add_edge(
                first_nodes,
                self.dag.leaf_selection(blockchain2.dag.graph[first_nodes].block, self.gnode_id))
            # blockchain2.dag.delete_node(blockchain2.dag.graph[blockchain2.gnode_id].block)

    def to_json(self):
        data = {}
        json_blocks = []
        for block in self.blocks:
            json_blocks.append(block.to_json())
        data['blocks'] = json_blocks
        return data

    # NOTE: Might not need blockcount
    def block_count_valid(self, block):
        if self.blocks[-1].block_count == block.block_count - 1:
            return True
        else:
            return False

    def parent_block_hash_valid(self, block):
        latest_blockchain_block_hash = BlockchainUtils.hash(
            self.blocks[-1].payload()).hexdigest()
        if latest_blockchain_block_hash == block.last_hash:
            return True
        else:
            return False

    def get_covered_transaction_set(self, transactions):
        covered_transactions = []
        for transaction in transactions:
            if self.transaction_covered(transaction):
                covered_transactions.append(transaction)
            else:
                print('transaction is not covered by sender')
        return covered_transactions

    def transaction_covered(self, transaction):
        if transaction.tr_type == 'EXCHANGE':
            return True
        sender_balance = self.account_model.get_balance(
            transaction.sender_public_key)
        if sender_balance >= transaction.amount:
            return True
        else:
            return False

    def execute_transactions(self, transactions):
        for transaction in transactions:
            self.execute_transaction(transaction)

    def execute_transaction(self, transaction):
        if transaction.tr_type == 'STAKE':
            sender = transaction.sender_public_key
            receiver = transaction.receiver_public_key
            if sender == receiver:
                amount = transaction.amount
                self.pos.update(sender, amount)
                self.account_model.update_balance(sender, -amount)
        else:
            sender = transaction.sender_public_key
            receiver = transaction.receiver_public_key
            amount = transaction.amount
            self.account_model.update_balance(sender, -amount)
            self.account_model.update_balance(receiver, amount)

    def next_forger(self):
        parent_block_hash = BlockchainUtils.hash(
            self.blocks.latest_point.payload()).hexdigest()
        next_forger = self.pos.forger(parent_block_hash)
        return next_forger

    def create_block(self, transactions_from_pool, forger_wallet, node_id):
        covered_transactions = self.get_covered_transaction_set(
            transactions_from_pool)
        self.execute_transactions(covered_transactions)
        new_block = forger_wallet.create_block(covered_transactions, node_id)

        # check if we need add here or in separate function

        return_data = self.blocks.add(new_block)
        parent = return_data[1]
        traversed_kdnodes = return_data[2]
        return new_block, parent, traversed_kdnodes

    def transaction_exists(self, transaction):
        for node in self.blocks.inorder():
            for block_transaction in node.data.transactions:
                if transaction.id.equals(block_transaction.id):
                    return True
        return False

    def forger_valid(self, block):
        forger_public_key = self.pos.forger(block.last_hash)
        proposed_block_forger = block.forger
        if forger_public_key == proposed_block_forger:
            return True
        else:
            return False

    def transactions_valid(self, transactions):
        covered_transactions = self.get_covered_transaction_set(transactions)
        if len(covered_transactions) == len(transactions):
            return True
        return False

    def merkle_root(self):
        return self.blocks.subtree_hash

    # TODO: loop through bc to add each node to self instead of the entire bc tree
    # somewhat done ?
    # def merge(self, bc):
    #     for b in bc.levelorder():
    #         self.blocks.add_node(b)

    def visualize(self):
        visual = nx.DiGraph()
        for element in self.dag.graph.items():
            node_id = element[0]
            visual.add_node(node_id)
            for edge in self.dag.graph[node_id].edges:
                visual.add_edge(node_id, edge)
        nx.draw_networkx(visual, arrows=True)
