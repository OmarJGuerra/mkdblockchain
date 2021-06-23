import time
import copy


class Block:
    # def __init__(self, transactions, lastHash, x, y, forger, blockCount):
    def __init__(self, transactions, lastHash, lastBlock, nodeID, x, y, forger, blockCount):
        # self.blockCount = blockCount
        self.transactions = transactions
        self.lastHash = lastHash
        self.lastBlock = lastBlock
        self.coords = [nodeID, x, y, time.time()]
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
        genesisBlock = Block([], 'genesisHash', 'genesis', x=0, y=0)
        genesisBlock[2] = 0
        # genesisBlock.timestamp = 0
        return genesisBlock

    def toJson(self):
        data = {}
        # data['blockCount'] = self.blockCount
        data['lastHash'] = self.lastHash
        data['signature'] = self.signature
        data['forger'] = self.forger
        # data['x'] = self.x
        # data['y'] = self.y
        # data['timestamp'] = self[2]
        jsonTransactions = []
        for transaction in self.transactions:
            jsonTransactions.append(transaction.toJson())
        data['transactions'] = jsonTransactions
        return data

    def payload(self):
        jsonRepresentation = copy.deepcopy(self.toJson())
        jsonRepresentation['signature'] = ''
        return jsonRepresentation

    def __repr__(self):
        return f'Block({self.coords}, {self.transactions}, {hash(self)})'

    def __hash__(self):
        return hash(self.coords)

    def sign(self, signature):
        self.signature = signature
