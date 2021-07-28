from blockchain_utils import BlockchainUtils
from account_model import AccountModel
from proof_of_stake import ProofOfStake
import kdtree  # removed from blockchain


class MKDBlockchain:
    def __init__(self, dimensions, gnode_id, genesis_forger):
        # self.blocks = kdtree.KDNode(Block.genesis(), left=None, right=None, axis=0, sel_axis=1, )
        self.blocks = kdtree.create_root(dimensions, gnode_id, genesis_forger)
        self.account_model = AccountModel()
        self.pos = ProofOfStake()

    def add_block(self, block):
        self.execute_transactions(block.transactions)
        self.blocks.add(block)
        print('block add successful')
        # self.blocks.append(block)03

    # modified loop to traverse the tree in order
    def to_json(self):
        j_data = {}
        json_blocks = []
        for block in self.blocks.inorder():
            json_blocks.append(block.to_json())
        j_data['blocks'] = json_blocks
        j_data['pos'] = self.pos.to_json()
        return j_data

    # NOTE: Might not need blockcount
    def block_count_valid(self, block):
        if self.blocks[-1].block_count == block.block_count - 1:
            return True
        else:
            return False

    def parent_block_hash_valid(self, block):
        parent_block_hash = BlockchainUtils.hash(block).hexdigest()
        if parent_block_hash == block.parent_hash:
            return True
        else:
            return False

    def get_parent(self, node):
        current = self.blocks
        #print(f'node being passed into get_parent: {node}')
        #print(f'current get_parent initial, before while loop: {current}')
        for kd_node in kdtree.level_order(current):
            #print(f'kd_node: {kd_node}')
            #print(f'kd_node.children(): {kd_node.children}')
            for tup in kd_node.children:
                if tup[0].data.parent_hash == node.data.parent_hash:
                    return kd_node
        '''
        
        if node.data.parent_hash == '0':
            return node
        while current is not None:
            print(f'current from get_parent: {current}')
            # TODO: GET CORRECT PHASH FOR GENESIS OR DEAL WITH THE ERROR
            if current.left is None and current.right is None:
                if current.data.parent_hash == node.datad
            if current.left is not None:
                if node.data == current.left.data:
                    return current
                elif node.data[current.axis] < current.data[current.axis]:
                    current = current.left
            else:

            elif current.right is not None:
                if node.data == current.right.data:
                    return current









            if node.data[current.axis] < current.data[current.axis]:
                current = current.left
            else:
                current = current.right
    '''

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

    def create_block(self, node_coords, transactions_from_pool, forger_wallet, node_id):
        # covered_transactions = self.get_covered_transaction_set(
        #     transactions_from_pool)
        #self.execute_transactions(transactions_from_pool)
        new_block = forger_wallet.create_block(transactions_from_pool, node_id, node_coords)

        # check if we need add here or in separate function

        return_data = self.blocks.add(new_block)
        parent = return_data[1]
        traversed_kdnodes = return_data[2]
        return new_block, parent, traversed_kdnodes

    def transaction_exists(self, transaction):
        for kd_node in kdtree.level_order(self.blocks):
            for block_transaction in kd_node.data.transactions:
                if transaction.id == block_transaction.id:
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
