from cluster import Cluster
from mkd_blockchain import MKDBlockchain
from node import Node
from sensor_transaction import SensorTransaction

import csv
import multiprocessing
import random
import time

MAX_NODES = 9999


def run_sim(test_num, num_clusters, num_nodes, forge_interval, dimensions):
    """
    Run an instance of the simulation.

    The simulation will read through the given dataset and perform node
    movement and blockchain operations accordingly. All data gathered will
    be recorded in appropriate .csv files in the project directory.

    Parameters:
        test_num (int): number belonging to this run
        num_clusters (int): amount of clusters in the network
        num_nodes (int): amount of nodes in the network
        forge_interval (int): interval of time units (cycles) at which blocks will be forged
        dimensions (int): dimensionality of the blockchain
    """

    print(f'START TEST {test_num}')
    # Initialize nodes with a genesis block.

    # The median of the number of nodes is chosen as the
    # genesis forger in order to keep the tree balanced.
    genesis_node_id = int(num_nodes / 2)
    nodes = []
    for i in range(num_nodes):
        new_node = Node(test_num, i + 1, f'{test_num}.c0')
        # Each node first subscribes to the 'genesis cluster' (c0).
        new_node.start_listener(f'{test_num}.c0')
        nodes.append(new_node)

    clusters = []
    for i in range(num_clusters):
        new_cluster = Cluster(i + 1, test_num)
        new_cluster.start_listener(f'{test_num}.c{i+1}')
        clusters.append(new_cluster)

    # The blockchain is initialized
    genesis_forger = nodes[genesis_node_id-1].wallet.public_key_string()
    mkd_blockchain = MKDBlockchain(dimensions, genesis_node_id, genesis_forger)
    publisher = nodes[genesis_node_id-1]
    publisher.publish(mkd_blockchain)

    # %%

    # Opens the data set file and generates nodes based on the provided data.
    # file is comma delimited in the following format: time-nodeID-x_coord-y_coord-miner-region-prev_region
    with open(f'./time-nodeID-xcoor-ycoor-miner-region-previousRegion_{test_num}.txt') as f:
        lines = f.readlines()  # list containing lines of file
        i = 0
        cycles = 0

        nodes_transacting = []

        for line in lines:
            j = i % num_nodes  # j will represent the increments of nodes
            parts = line.split(',')

            moving_node = nodes[j]  # choosing specific node to manipulate
            # assigning new coordinates to the moving node
            moving_node.coords = [float(parts[2]), float(parts[3])]

            # identifying old and new cluster and calling the move function to perform the needed changes
            old_cluster_id = moving_node.cluster_id
            new_cluster_id = f'{test_num}.c{parts[5]}'
            moving_node.move_node(old_cluster_id, new_cluster_id)

            #  generates list of nodes that need to publish a transaction to their cluster
            if int(parts[4]) == 1:
                nodes_transacting.append(moving_node)

            #  Once it reaches the end of num_nodes for each time increment submit transactions
            if j == (num_nodes - 1):
                cycles += 1
                print(f'Number of cycles for test {test_num}: {cycles}')
                for node in nodes:
                    with open(f'branch_size_left_right_{test_num}.csv', mode='a') as branch_size:
                        branch_size_writer = csv.writer(branch_size, delimiter='.', quotechar='"',
                                                        quoting=csv.QUOTE_MINIMAL)
                        branch_size_writer.writerow([cycles, node.cluster_id, node.node_id, node.blockchain.blocks.size,
                                                     node.blockchain.blocks.left_size, node.blockchain.blocks.right_size,
                                                     node.blockchain.blocks.height()])
                for node in nodes_transacting:
                    transaction = SensorTransaction(node.wallet.public_key_string(), random.randint(0, 1000))
                    node.publish(transaction)

                # if time to forge then forge and broadcast, needs to scan and perform all clusters
                if int(parts[1]) % forge_interval == 0:
                    for cluster in clusters:
                        #  choose a forger
                        forger = cluster.next_forger()
                        #  if transaction pool not empty then forge and broadcast
                        for node in cluster.member_nodes:
                            if node.wallet.public_key_string() == forger and \
                               node.transaction_pool.transactions is not []:
                                forge_begin = time.time()
                                node.mkd_forge()
                                forge_end = time.time() - forge_begin
                                with open(f'cluster_block_forging_{test_num}.csv', mode='a') as cluster_block_forging:
                                    cluster_forging_writer = csv.writer(cluster_block_forging, delimiter=',',
                                                                        quotechar='"', quoting=csv.QUOTE_MINIMAL)
                                    cluster_forging_writer.writerow([node.cluster_id, node.node_id,
                                                                     node.blockchain.blocks.size,
                                                                     node.blockchain.blocks.left_size,
                                                                     node.blockchain.blocks.right_size, forge_end])
            i += 1

    # now = time.time() - then
    # print(f'Complete movement simulation @ {num_nodes} nodes: {now}')


# run_sim(test_num, num_clusters, num_nodes, forge_interval, dimensions):
if __name__ == '__main__':
    # run_sim(3, 16, 80, 80, 3)

    """
    Process Method
    args = test_num, num_clusters, num_nodes, forge_interval, dimensions
    """
    test_0 = multiprocessing.Process(target=run_sim, args=(0, 16, 80, 10, 4))
    test_1 = multiprocessing.Process(target=run_sim, args=(1, 16, 80, 10, 4))
    test_2 = multiprocessing.Process(target=run_sim, args=(2, 16, 80, 10, 4))
    test_3 = multiprocessing.Process(target=run_sim, args=(3, 16, 80, 10, 4))
    test_4 = multiprocessing.Process(target=run_sim, args=(4, 16, 80, 10, 4))
    test_5 = multiprocessing.Process(target=run_sim, args=(5, 16, 80, 10, 4))
    test_6 = multiprocessing.Process(target=run_sim, args=(6, 16, 80, 10, 4))
    test_7 = multiprocessing.Process(target=run_sim, args=(7, 16, 80, 10, 4))
    test_8 = multiprocessing.Process(target=run_sim, args=(8, 16, 80, 10, 4))
    test_9 = multiprocessing.Process(target=run_sim, args=(9, 16, 80, 10, 4))
    test_10 = multiprocessing.Process(target=run_sim, args=(10, 16, 80, 10, 4))

    test_0.start()
    test_1.start()
    test_2.start()
    test_3.start()
    test_4.start()
    test_5.start()
    test_6.start()
    test_7.start()
    test_8.start()
    test_9.start()
    test_10.start()

    test_0.join()
    test_1.join()
    test_2.join()
    test_3.join()
    test_4.join()
    test_5.join()
    test_6.join()
    test_7.join()
    test_8.join()
    test_9.join()
    test_10.join()


