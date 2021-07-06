import random

from Crypto.PublicKey import RSA
from transaction import Transaction
from block import Block
from blockchain_utils import BlockchainUtils
from Crypto.Signature import PKCS1_v1_5


class Wallet:
    def __init__(self):
        self.keyPair = RSA.generate(2048)

    def from_key(self, file):
        key = ''
        with open(file, 'r') as keyfile:
            key = RSA.importKey(keyfile.read())
        self.keyPair = key

    def sign(self, data):
        data_hash = BlockchainUtils.hash(data)
        signature_scheme_object = PKCS1_v1_5.new(self.keyPair)
        signature = signature_scheme_object.sign(data_hash)
        return signature.hex()

    @staticmethod
    def signature_valid(data, signature, public_key_string):
        signature = bytes.fromhex(signature)
        data_hash = BlockchainUtils.hash(data)
        public_key = RSA.importKey(public_key_string)
        signature_scheme_object = PKCS1_v1_5.new(public_key)

        signature_scheme_object.verify(data_hash, signature)
        return True

    def public_key_string(self):
        public_key_string = self.keyPair.publickey().exportKey(
            'PEM').decode('utf-8')
        return public_key_string

    def create_transaction(self, receiver, amount, tr_type):
        transaction = Transaction(
            self.public_key_string(), receiver, amount, tr_type)
        signature = self.sign(transaction.payload())
        transaction.sign(signature)
        return transaction

    def create_block(self, transactions, last_hash, block_count):
        block = Block(transactions, last_hash, self.public_key_string(),
                      x=random.randint(0, 1000), y=random.randint(0, 1000), forger=None, block_count=block_count)
        signature = self.sign(block.payload())
        block.sign(signature)
        return block
