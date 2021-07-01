from Transaction import Transaction
from Wallet import Wallet
from TransactionPool import TransactionPool
from Block import Block
from MKDBlockchain import MKDBlockchain
import pprint
from BlockchainUtils import BlockchainUtils
from AccountModel import AccountModel
from Node import Node
import sys

if __name__ == '__main__':
    node_num = int(sys.argv[1])
    cluster_num = int(sys.argv[2])

    if len(sys.argv) > 4:
        keyFile = sys.argv[4]

    for i in range(0, node_num):
        node = Node(ip, port, keyFile)
        node.startP2P()
        node.startAPI(apiPort)
