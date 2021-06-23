

class TransactionPool:

    def __init__(self):
        self.transactions = []

    def add_transaction(self, transaction):
        self.transactions.append(transaction)

    def transaction_exists(self, transaction):
        for poolTransaction in self.transactions:
            if poolTransaction.equals(transaction):
                return True
        return False

    def remove_from_pool(self, transactions):
        newPoolTransactions = []
        for poolTransaction in self.transactions:
            insert = True
            for transaction in transactions:
                if poolTransaction.equals(transaction):
                    insert = False
            if insert == True:
                newPoolTransactions.append(poolTransaction)
        self.transactions = newPoolTransactions

    def forgingRequired(self):
        if len(self.transactions) >= 3:
            return True
        else:
            return False
