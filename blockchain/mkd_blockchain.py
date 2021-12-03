from blockchain_utils import BlockchainUtils
from account_model import AccountModel
from proof_of_stake import ProofOfStake
import kdtree


class MKDBlockchain:
    """
    A Merkle KD-Blockchain.

    Attributes:
        blocks (KDNode): Root node of the Merkle KD-Tree
        account_model (AccountModel): Model of all transactors involved in the blockchain
        pos (ProofOfStake): Consensus mechanism

    Methods:
        add_block(block)
        get_parent()
        execute_transactions()
        execute_transaction()
        next_forger()
        create_block()
        transaction_exists()
        to_json()
    """

    def __init__(self, dimensions, gen_node_id, genesis_forger):
        """
        Construct an MKDBlockchain object.

        Parameters:
            dimensions (int): Dimensionality of the KD-Tree
            gen_node_id (int): ID of the node that forges the genesis block (median of total nodes)
            genesis_forger (str): Public key string of the wallet that forged the genesis block
        """
        self.blocks = kdtree.create_root(dimensions, gen_node_id, genesis_forger)
        self.account_model = AccountModel()
        self.pos = ProofOfStake()

    def add_block(self, block):
        """
        Add a block to the blockchain.

        Parameters:
            block (Block): the block to be inserted
        """
        self.execute_transactions(block.transactions)
        self.blocks.add(block)
        print('block add successful')
        # self.blocks.append(block)

    def get_parent(self, node):
        """Get the parent of a node in the blockchain."""
        current = self.blocks
        for kd_node in kdtree.level_order(current):
            for tup in kd_node.children:
                if tup[0].data.parent_hash == node.data.parent_hash:
                    return kd_node

    def execute_transactions(self, transactions):
        """Execute multiple transactions"""
        for transaction in transactions:
            self.execute_transaction(transaction)

    def execute_transaction(self, transaction):
        """Execute a transaction"""
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
        """
        Determine the node that will forge the next block

        Returns:
            next_forger (str): public key of the forging node
        """
        parent_block_hash = BlockchainUtils.hash(
            self.blocks.latest_point.payload()).hexdigest()
        next_forger = self.pos.forger(parent_block_hash)
        return next_forger

    def create_block(self, node_coords, transactions_from_pool, forger_wallet, node_id):
        """
        Create a block.

        Parameters:
            node_coords (list): coordinates of the forger node
            transactions_from_pool (list): list of transactions currently in the pool
            forger_wallet (Wallet): wallet that will forge the block
            node_id (int): id of the forger node

        Returns:
              new_block (Block): the forged block
              parent (KDNode): parent of the node that was created with the block
        """
        new_block = forger_wallet.create_block(transactions_from_pool, node_id, node_coords)
        return_data = self.blocks.add(new_block)
        parent = return_data[1]
        return new_block, parent, False, return_data[0]

    def transaction_exists(self, transaction):
        """Check if a given transaction is already in the tree."""
        for kd_node in kdtree.level_order(self.blocks):
            for block_transaction in kd_node.data.transactions:
                if transaction.id == block_transaction.id:
                    return True
        return False

# modified loop to traverse the tree in order
    def to_json(self):
        """
        Represent the MKD Blockchain as json data.

        Returns:
            j_data (dict): A json representation of the MKD Blockchain
        """
        j_data = {}
        json_blocks = []
        for block in self.blocks.inorder():
            json_blocks.append(block.to_json())
        j_data['blocks'] = json_blocks
        j_data['pos'] = self.pos.to_json()
        return j_data
