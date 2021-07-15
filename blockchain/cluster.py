from proof_of_stake import ProofOfStake


class Cluster:

    def __init__(self, c_id=0):
        self.cluster_id = c_id
        self.member_nodes = []
        self.pos = ProofOfStake()
