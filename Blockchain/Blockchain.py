from Block import Block
from BlockchainUtils import BlockchainUtils
from AccountModel import AccountModel
from ProofOfStake import ProofOfStake
from mkdtree import kdtree


# TODO: link code so that blockchain is treated as mkd tree
class Blockchain():
    def __init__(self):
        self.blocks = kdtree.KDNode(Block.genesis(), left=None, right=None, axis=0)
        # self.blocks = kdtree.KDNode
        self.accountModel = AccountModel()
        self.pos = ProofOfStake()

    def addBlock(self, block):
        self.executeTransactions(block.transactions)
        self.blocks.add(block)
        # self.blocks.append(block)

    # modified loop to traverse the tree in order
    def toJson(self):
        data = {}
        jsonBlocks = []
        for block in self.blocks.inorder():
            jsonBlocks.append(block.toJson())
        data['blocks'] = jsonBlocks
        return data

    # NOTE: Might not need blockcount
    def blockCountValid(self, block):
        if self.blocks[-1].blockCount == block.blockCount - 1:
            return True
        else:
            return False

    def lastBlockHashValid(self, block):
        latestBlockchainBlockHash = BlockchainUtils.hash(
            self.blocks[-1].payload()).hexdigest()
        if latestBlockchainBlockHash == block.lastHash:
            return True
        else:
            return False

    def getCoveredTransactionSet(self, transactions):
        coveredTransactions = []
        for transaction in transactions:
            if self.transactionCovered(transaction):
                coveredTransactions.append(transaction)
            else:
                print('transaction is not covered by sender')
        return coveredTransactions

    def transactionCovered(self, transaction):
        if transaction.type == 'EXCHANGE':
            return True
        senderBalance = self.accountModel.getBalance(
            transaction.senderPublicKey)
        if senderBalance >= transaction.amount:
            return True
        else:
            return False

    def executeTransactions(self, transactions):
        for transaction in transactions:
            self.executeTransaction(transaction)

    def executeTransaction(self, transaction):
        if transaction.type == 'STAKE':
            sender = transaction.senderPublicKey
            receiver = transaction.receiverPublicKey
            if sender == receiver:
                amount = transaction.amount
                self.pos.update(sender, amount)
                self.accountModel.updateBalance(sender, -amount)
        else:
            sender = transaction.senderPublicKey
            receiver = transaction.receiverPublicKey
            amount = transaction.amount
            self.accountModel.updateBalance(sender, -amount)
            self.accountModel.updateBalance(receiver, amount)

    def nextForger(self):
        lastBlockHash = BlockchainUtils.hash(
            self.blocks[-1].payload()).hexdigest()
        nextForger = self.pos.forger(lastBlockHash)
        return nextForger

    # TODO: Implement kdtree version of parent hash - replace indexing with tree traversal
    def createBlock(self, transactionsFromPool, forgerWallet):
        coveredTransactions = self.getCoveredTransactionSet(
            transactionsFromPool)
        self.executeTransactions(coveredTransactions)
        newBlock = forgerWallet.createBlock(
            coveredTransactions, BlockchainUtils.hash(self.blocks[-1].payload()).hexdigest(), len(self.blocks))
        self.blocks.add(newBlock)
        return newBlock

    def transactionExists(self, transaction):
        for block in self.blocks:
            for blockTransaction in block.transactions:
                if transaction.equals(blockTransaction):
                    return True
        return False

    def forgerValid(self, block):
        forgerPublicKey = self.pos.forger(block.lastHash)
        proposedBlockForger = block.forger
        if forgerPublicKey == proposedBlockForger:
            return True
        else:
            return False

    def transactionsValid(self, transactions):
        coveredTransactions = self.getCoveredTransactionSet(transactions)
        if len(coveredTransactions) == len(transactions):
            return True
        return False

    # TODO: loop through bc to add each node to self instead of the entire bc tree
    def merge(self, bc):
        self.blocks.add_node(bc)
