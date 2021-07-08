import time
import copy


# Block class temporarily inherits dict to make JSON serialization easy.
# Should be changed to be more robust at a later time.
class Block(dict):
    # def __init__(self, transactions, parent_hash, x, y, forger, block_count):
    def __init__(self, transactions, parent_hash, node_id, x, y, forger):
        dict.__init__(self)
        # self.block_count = block_count
        self.transactions = transactions
        self.parent_hash = parent_hash
        self.coords = [node_id, x, y, time.time()]
        # self.x = x
        # self.y = y
        # self.timestamp = time.time()
        self.forger = forger
        self.signature = ''

    def __getitem__(self, item):
        return self.coords[item]

    # block[1] -> return self.coors[1]
    def __setitem__(self, key, value):
        self.coords[key] = value

    def __len__(self):
        return len(self.coords)

    @staticmethod
    def genesis(genesis_node_id, forger):
        genesis_block = Block([], '0', genesis_node_id, x=0, y=0, forger=forger)
        # genesis_block.timestamp = 0
        return genesis_block

    def to_json(self):
        data = {'parent_hash': self.parent_hash,
                'signature': self.signature,
                'forger': self.forger}
        # data['block_count'] = self.block_count
        # data['x'] = self.x
        # data['y'] = self.y
        # data['timestamp'] = self[2]
        json_transactions = []
        for transaction in self.transactions:
            json_transactions.append(transaction.to_json())
        data['transactions'] = json_transactions
        return data

    def payload(self):
        json_representation = copy.deepcopy(self.to_json())
        json_representation['signature'] = ''
        return json_representation

    def __repr__(self):
        return f'Block({self.coords}, {self.transactions})'

    def sign(self, signature):
        self.signature = signature
