import time
import copy
from blockchain_utils import BlockchainUtils as BU


# Block class temporarily inherits dict to make JSON serialization easy.
# Should be changed to be more robust at a later time.
class Block(dict):
    # def __init__(self, transactions, parent_hash, x, y, forger, block_count):
    def __init__(self, transactions, node_id, x, y, forger, t, parent_hash=None, signature=''):
        dict.__init__(self)
        self.coords = [node_id, x, y, t]
        self.transactions = transactions
        self.parent_hash = parent_hash
        self.forger = forger
        self.signature = signature

    def __getitem__(self, item):
        return self.coords[item]

    # block[0] -> return self.coords[0]
    def __setitem__(self, key, value):
        self.coords[key] = value

    def __len__(self):
        return len(self.coords)

    def __repr__(self):
        # return f'Block({self.coords}, {self.forger}, {self.parent_hash}, {self.transactions})'
        # return f'Block({self.coords}, {self.transactions},\n' \
        #        f'{self.parent_hash},\n' \
        #        f'{self.forger},\n' \
        #        f'{self.signature})'
        return f'Block({self.coords})'

    def __hash__(self):
        return BU.hash(self)

    @staticmethod
    def genesis(genesis_node_id, forger):
        t = time.time()
        genesis_block = Block([], genesis_node_id, x=0, y=0, forger=forger, t=t, parent_hash='0')
        return genesis_block

    def to_json(self):
        j_data = {'coordinates': self.coords,
                  'signature': self.signature,
                  'parent_hash': self.parent_hash}

        json_transactions = []
        for transaction in self.transactions:
            json_transactions.append(transaction.to_json())
        j_data['transactions'] = json_transactions

        return j_data

    def payload(self):
        json_representation = copy.deepcopy(self.to_json())
        json_representation['signature'] = ''
        return json_representation

    def sign(self, signature):
        self.signature = signature
