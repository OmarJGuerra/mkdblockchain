from mkd_blockchain import MKDBlockchain
from transaction import Transaction
from transaction_pool import TransactionPool
from wallet import Wallet
from socket_communication import SocketCommunication
from node_api import NodeAPI
from message import Message
from block import Block
from blockchain_utils import BlockchainUtils
from pubsub import pub
import copy


class Node:

    def __init__(self, node_id, cluster_id, key=None):
        self.p2p = None
        self.node_id = node_id
        self.cluster_id = cluster_id
        # self.port = port
        self.blockchain = None
        self.transaction_pool = TransactionPool()
        self.wallet = Wallet()
        if key is not None:
            self.wallet.from_key(key)

    # def startP2P(self):
    #     self.p2p = SocketCommunication(self.ip, self.port)
    #     self.p2p.startSocketCommunication(self)

    # def startAPI(self, apiPort):
    #     self.api = NodeAPI()
    #     self.api.injectNode(self)
    #     self.api.start(apiPort)

    def start_listener(self, cluster_topic):
        pub.subscribe(self.node_listener, cluster_topic)

    def node_listener(self, arg):
        print(f'c{self.node_id} received payload: {arg}')
        t = type(arg)
        if t is Block:
            self.handle_block(arg)
        elif t is MKDBlockchain:
            self.handle_blockchain(arg)
        else:
            self.handle_transaction(arg)

    def move_listener(self, old_topic, new_topic):
        pub.unsubscribe(self.node_listener, old_topic)  # core.TopicManager.getTopicsSubscribed(listener))
        pub.subscribe(self.node_listener, new_topic)

    # TODO: fix handlers to use publish instead of p2p
    def publish(self, message):
        cluster = 'c'+str(self.cluster_id).strip()
        pub.sendMessage(cluster, arg=message)

    def handle_transaction(self, transaction):
        data = transaction.payload()
        signature = transaction.signature
        signer_public_key = transaction.sender_public_key
        signature_valid = Wallet.signature_valid(
            data, signature, signer_public_key)
        transaction_exists = self.transaction_pool.transaction_exists(transaction)
        transaction_in_block = self.blockchain.transaction_exists(transaction)
        if not transaction_exists and not transaction_in_block and signature_valid:
            self.transaction_pool.add_transaction(transaction)
            message = Message(self.p2p.socketConnector,
                              'TRANSACTION', transaction)
            encoded_message = BlockchainUtils.encode(message)
            self.p2p.broadcast(encoded_message)
            forging_required = self.transaction_pool.forging_required()
            if forging_required:
                self.forge()

    def handle_block(self, block):
        forger = block.forger
        block_hash = block.payload()
        signature = block.signature

        block_count_valid = self.blockchain.block_count_valid(block)
        last_block_hash_valid = self.blockchain.parent_block_hash_valid(block)
        forger_valid = self.blockchain.forger_valid(block)
        transactions_valid = self.blockchain.transactions_valid(
            block.transactions)
        signature_valid = Wallet.signature_valid(block_hash, signature, forger)
        if not block_count_valid:
            self.request_chain()
        if last_block_hash_valid and forger_valid and transactions_valid and signature_valid:
            self.blockchain.add_block(block)
            self.transaction_pool.remove_from_pool(block.transactions)
            message = Message(self.p2p.socketConnector, 'BLOCK', block)
            self.p2p.broadcast(BlockchainUtils.encode(message))

    def handle_blockchain_request(self, requesting_node):
        message = Message(self.p2p.socketConnector,
                          'BLOCKCHAIN', self.blockchain)
        self.p2p.send(requesting_node, BlockchainUtils.encode(message))

    # TODO: Possibly add some of the old functionality back if needed.
    def handle_blockchain(self, blockchain):
        if len(blockchain.blocks) == 1:
            self.blockchain = blockchain
        """for block in blockchain.blocks.inorder():
            print(block)
        local_blockchain_copy = copy.deepcopy(self.blockchain)
        print(local_blockchain_copy)
        if local_blockchain_copy is not None:
            local_block_count = len(local_blockchain_copy.blocks)
        else:
            local_block_count = 0
        received_chain_block_count = len(blockchain.blocks)
        if local_block_count < received_chain_block_count:
            for block_number, block in enumerate(blockchain.blocks):
                if block_number >= local_block_count:
                    local_blockchain_copy.add_block(block)
                    self.transaction_pool.remove_from_pool(block.transactions)
            self.blockchain = local_blockchain_copy"""

    # TODO: Modify forger (remove extraneous operations, add necessary behavior)
    def forge(self):
        forger = self.blockchain.next_forger()
        if forger == self.wallet.public_key_string():
            print('i am the forger')
            block = self.blockchain.create_block(
                self.transaction_pool.transactions, self.wallet)
            self.transaction_pool.remove_from_pool(
                self.transaction_pool.transactions)
            message = Message(self.p2p.socketConnector, 'BLOCK', block)
            self.p2p.broadcast(BlockchainUtils.encode(message))
        else:
            print('i am not the forger')

    def request_chain(self):
        message = Message(self.p2p.socketConnector, 'BLOCKCHAINREQUEST', None)
        self.p2p.broadcast(BlockchainUtils.encode(message))
