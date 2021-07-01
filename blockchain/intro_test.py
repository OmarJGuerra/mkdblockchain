from transaction import Transaction
from wallet import Wallet
from transaction_pool import TransactionPool
from block import Block
from mkd_blockchain import MKDBlockchain
import pprint
from blockchain_utils import BlockchainUtils




if __name__ == '__main__':

    sender = 'sender'
    receiver = 'receiver'
    amount = 1
    type = 'TRANSFER'

    # ---------------------------------
    # this was as practice or intro
    # ---------------------------------
    # transaction = Transaction(sender, receiver, amount, tr_type)
    #
    # print(transaction.to_json())   #prints transaction in readable format using toJason()
    #
    # testing if sig works
    #
    # wallet = Wallet() #must import Wallet
    # signature = wallet.sign(transaction.to_json()) #sing trans in a jason format, returns a sign
    #
    # transaction.sign(signature)  #gives false transaction verification (it should be true)
    # print(transaction.to_json())
    #
    # validate signature
    # signature_valid = Wallet.signature_valid(transaction.payload(), transaction.signature, wallet.public_key_string())
    # print(signature_valid)

    # ---------------------------------

    wallet = Wallet()  # create wallet
    fraudulentWallet = Wallet()  # to test only with fake wallet

    pool = TransactionPool()  # create transaction pool

    transaction = wallet.create_transaction(receiver, amount, type) #create transaction
    print("--------------------------------------------------")
    # print(transaction.toJpason())
    print(transaction.payload())

    # validate signature - 1  (intro/basic method)
    # signature_valid = Wallet.signature_valid(transaction.payload(), transaction.signature, wallet.public_key_string())
    # signature_valid = Wallet.signature_valid(transaction.payload(), transaction.signature, fraudulentWallet.public_key_string()) #to test fake wallet
    # print(signature_valid)

    # Check if it detects duplicate transaction, only one must be in the transaction pool
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
    blockchain = MKDBlockchain()

    print("--------------------------------------------------")
    print("Getting last_hash from the blockchain")
    #get the last_hash in the blockchain
    lastHash = BlockchainUtils.hash(blockchain.blocks[-1].payload()).hexdigest() #we first hash the blockchain, get last_hash in binary, trasfer it to hex format. must import BlockchainUtils class,
    print("last_hash = ", lastHash)


    print("--------------------------------------------------")
    print("Calculating the new block_count by accessing the last block_count in the blockchain")
    #Calculating the new block_count by accessing the last block_count in the blockchain
    blockCount = blockchain.blocks[-1].blockCount + 1 
    #block_count = blockchain.blocks[-1].block_count + 3333 #to test what happen if BlockCount is not valid
    print("previous block_count = ", blockchain.blocks[-1].blockCount, " new block_count = ",blockCount)

    print("--------------------------------------------------")
    print("Creating a block and adding it to the blockchain")
    block = wallet.create_block(pool.transactions, lastHash, blockCount)
    


    #test to create randome block
    #block = Block(pool.transactions, 'last_hash', 'forger', 1) #random (last_hash,forger,block_count) because no consensus,last_hash etc

    #print("--------------------------------------------------")
    #print("Block as Jason represntation = ",block.to_json())

    #test to create randome block
    #block = wallet.create_block(pool.transactions, 'last_hash', 1) # random last_hash,block_count. not set yet only for test


    print("--------------------------------------------------")
    pprint.pprint(block.to_json())

    #validate signature - 2  (proper method)
    signatureValid = Wallet.signature_valid(block.payload(), block.signature, wallet.public_key_string()) #test valid wallet
    #signature_valid = Wallet.signature_valid(block.payload(), block.signature, fraudulentWallet.public_key_string()) #test invalid wallet
    print("--------------------------------------------------")
    print("Signature validity is ",signatureValid)    


    if not blockchain.last_block_hash_valid(block):
        print('last_block_hash is not valid')

    if not blockchain.block_count_valid(block):
        print('BlockCount is not valid')

    if blockchain.last_block_hash_valid(block) and blockchain.block_count_valid(block):
        blockchain.add_block(block)

    #add block to the blockchain
    #blockchain.add_block(block)
    print("--------------------------------------------------")
    print("Full blockchain = ")
    pprint.pprint(blockchain.to_json())