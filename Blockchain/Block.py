import time
import copy


class Block:
    # def __init__(self, transactions, last_hash, x, y, forger, block_count):
    def __init__(self, transactions, last_hash, node_id, x, y, forger, block_count):
        # self.block_count = block_count
        self.transactions = transactions
        self.last_hash = last_hash
        # self.lastBlock = lastBlock
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
    def genesis():
        # TODO: change genesis hash
        genesis_block = Block([], '0', 'genesis', x=0, y=0, forger=None, block_count=1)
        genesis_block[2] = 0
        # genesis_block.timestamp = 0
        return genesis_block

    def to_json(self):
        data = {'last_hash': self.last_hash,
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