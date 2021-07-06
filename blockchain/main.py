from transaction import Transaction
from wallet import Wallet
from transaction_pool import TransactionPool
from block import Block
from mkd_blockchain import MKDBlockchain
import pprint
from blockchain_utils import BlockchainUtils
from account_model import AccountModel
from node import Node
import sys

if __name__ == '__main__':
    node_num = int(sys.argv[1])
    cluster_num = int(sys.argv[2])
    total_time = int(sys.argv[3])

    with open('dataset.txt') as infile:
        for s in range(0, total_time):
            for n in range(0, node_num):
                line = infile.readline().split()
                if len(sys.argv) > 4:
                    keyFile = sys.argv[4]

                for i in range(0, cluster_num):
                    node = Node(node_num, cluster_num, keyFile)
                    # node.startP2P()
                    # node.startAPI(apiPort)
