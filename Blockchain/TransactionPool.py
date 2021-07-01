

class TransactionPool:

    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

    def transaction_exists(self, transaction):
        for pool_transaction in self.transactions:
            if pool_transaction.equals(transaction):
                return True
        return False

    def remove_from_pool(self, transactions):
        new_pool_transactions = []
        for poolTransaction in self.transactions:
            insert = True
            for transaction in transactions:
                if poolTransaction.equals(transaction):
                    insert = False
            if insert:
                new_pool_transactions.append(poolTransaction)
        self.transactions = new_pool_transactions

    def __repr__(self):
        return f'{self.transactions}'

    def forging_required(self):
        if len(self.transactions) >= 3:
            return True
        else:
            return False