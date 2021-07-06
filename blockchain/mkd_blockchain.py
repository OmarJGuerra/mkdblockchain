from block import Block
from blockchain_utils import BlockchainUtils
from account_model import AccountModel
from proof_of_stake import ProofOfStake
from mkdtree import kdtree


BLOCKCHAIN_DIMENSIONS = 4
GENESIS_NODE_ID = 40

# TODO: link code so that blockchain is treated as mkd tree
class MKDBlockchain:
    def __init__(self):
        # self.blocks = kdtree.KDNode(Block.genesis(), left=None, right=None, axis=0, sel_axis=1, )
        self.blocks = kdtree.create_root(BLOCKCHAIN_DIMENSIONS, GENESIS_NODE_ID)
        self.account_model = AccountModel()
        self.pos = ProofOfStake()

    def add_block(self, block):
        self.execute_transactions(block.transactions)
        self.blocks.add(block)
        print('block add successful')
        # self.blocks.append(block)03

    # modified loop to traverse the tree in order
    def to_json(self):
        data = {}
        json_blocks = []
        for block in self.blocks.inorder():
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

    # TODO: Implement kdtree version of parent hash - replace indexing with tree traversal
    def create_block(self, transactions_from_pool, forger_wallet):
        covered_transactions = self.get_covered_transaction_set(
            transactions_from_pool)
        self.execute_transactions(covered_transactions)
        new_block = forger_wallet.create_block(covered_transactions, len(self.blocks), 9)
        self.blocks.add(new_block)
        return new_block

    def transaction_exists(self, transaction):
        for block in self.blocks.inorder():
            for block_transaction in block.transactions:
                if transaction.equals(block_transaction):
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
    def merge(self, bc):
        for b in bc.levelorder():
            self.blocks.add_node(b)