from Transaction import Transaction
from Wallet import Wallet
from TransactionPool import TransactionPool
from Block import Block
from Blockchain import Blockchain
import pprint
from BlockchainUtils import BlockchainUtils




if __name__ == '__main__':

    sender = 'sender'
    receiver = 'receiver'
    amount = 1
    type = 'TRANSFER'

    # ---------------------------------
    # this was as practice or intro
    # ---------------------------------
    # transaction = Transaction(sender, receiver, amount, type)
    #
    # print(transaction.toJson())   #prints transaction in readable format using toJason()
    #
    # testing if sig works
    #
    # wallet = Wallet() #must import Wallet
    # signature = wallet.sign(transaction.toJson()) #sing trans in a jason format, returns a sign
    #
    # transaction.sign(signature)  #gives false transaction verification (it should be true)
    # print(transaction.toJson())
    #
    # validate signature
    # signatureValid = Wallet.signatureValid(transaction.payload(), transaction.signature, wallet.publicKeyString())
    # print(signatureValid)

    # ---------------------------------


    wallet = Wallet() #create wallet
    fraudulentWallet = Wallet() # to test only with fake wallet

    pool = TransactionPool() #create transaction pool

    transaction = wallet.createTransaction(receiver, amount, type) #create transaction
    print("--------------------------------------------------")
    #print(transaction.toJpason())
    print(transaction.payload())

    # validate signature - 1  (intro/basic method)
    # signatureValid = Wallet.signatureValid(transaction.payload(), transaction.signature, wallet.publicKeyString())
    # signatureValid = Wallet.signatureValid(transaction.payload(), transaction.signature, fraudulentWallet.publicKeyString()) #to test fake wallet
    # print(signatureValid)

    #to check if it detects duplicate transaction, only one must be in the transaction pool
    if pool.transaction_exists(transaction) == False:
        pool.add_transaction(transaction)

    #to check if it detects duplicate transaction, only one must be in the transaction pool
    #if pool.transaction_exists(transaction) == False:
        #pool.add_transaction(transaction)

    print("--------------------------------------------------")
    print("Transaction pool size = ",len(pool.transactions),'\n',pool.transactions)

    print("--------------------------------------------------")
    print("Creating blockchain object")
    #Create a blockchain as an object
    blockchain = Blockchain()

    print("--------------------------------------------------")
    print("Getting lastHash from the blockchain")
    #get the lastHash in the blockchain
    lastHash = BlockchainUtils.hash(blockchain.blocks[-1].payload()).hexdigest() #we first hash the blockchain, get lastHash in binary, trasfer it to hex format. must import BlockchainUtils class,
    print("lastHash = ", lastHash)


    print("--------------------------------------------------")
    print("Calculating the new blockCount by accessing the last blockCount in the blockchain")
    #Calculating the new blockCount by accessing the last blockCount in the blockchain
    blockCount = blockchain.blocks[-1].blockCount + 1 
    #blockCount = blockchain.blocks[-1].blockCount + 3333 #to test what happen if BlockCount is not valid
    print("previous blockCount = ", blockchain.blocks[-1].blockCount, " new blockCount = ",blockCount)

    print("--------------------------------------------------")
    print("Creating a blcok and adding it to the blockchain")
    block = wallet.createBlock(pool.transactions, lastHash, blockCount)
    


    #test to create randome block
    #block = Block(pool.transactions, 'lastHash', 'forger', 1) #random (lastHash,forger,blockCount) because no consensus,lastHash etc

    #print("--------------------------------------------------")
    #print("Block as Jason represntation = ",block.toJson())

    #test to create randome block
    #block = wallet.createBlock(pool.transactions, 'lastHash', 1) # random lastHash,blockCount. not set yet only for test


    print("--------------------------------------------------")
    pprint.pprint(block.toJson())

    #validate signature - 2  (proper method)
    signatureValid = Wallet.signatureValid(block.payload(), block.signature, wallet.publicKeyString()) #test valid wallet
    #signatureValid = Wallet.signatureValid(block.payload(), block.signature, fraudulentWallet.publicKeyString()) #test invalid wallet
    print("--------------------------------------------------")
    print("Signature validity is ",signatureValid)    


    if not blockchain.lastBlockHashValid(block):
        print('lastBlockHash is not valid')

    if not blockchain.blockCountValid(block):
        print('BlockCount is not valid')

    if blockchain.lastBlockHashValid(block) and blockchain.blockCountValid(block):
        blockchain.addBlock(block)

    #add block to the blockchain
    #blockchain.addBlock(block)
    print("--------------------------------------------------")
    print("Full Blockchain = ")
    pprint.pprint(blockchain.toJson())