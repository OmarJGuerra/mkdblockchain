from mkd_blockchain import MKDBlockchain
from cluster import Cluster
from node import Node
from sensor_transaction import SensorTransaction

import random
import time
import csv

if __name__ == '__main__':

    cluster_block_forging = open('cluster_block_forging.csv', mode='w')
    cluster_forging_writer = csv.writer(cluster_block_forging, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Initialize Nodes with Genesis Block
    print('START')
    MAX_NODES = 9999
    num_nodes = 80
    num_clusters = 16
    forging_interval = 10  # Time interval.  10 units of time
    nodes = []  # list of all nodes
    blockchain_dimensions = 4
    genesis_node_id = int(num_nodes / 2)

    # iterate and initialize each node
    then = time.time()
    for i in range(num_nodes):
        new_node = Node(i + 1, 0)
        # new_node.blockchain = mkd_blockchain
        new_node.start_listener('c0')
        nodes.append(new_node)
    now = time.time() - then
    print(f'block init time @ {num_nodes} nodes: {now}')

    # initialize each cluster
    clusters = []
    for i in range(num_clusters):
        new_cluster = Cluster(i + 1)
        new_cluster.start_listener(('c' + str(i + 1)))
        clusters.append(new_cluster)
        #print(f'cluster {new_cluster.cluster_id} initialized')

    genesis_forger = nodes[genesis_node_id - 1].wallet.public_key_string()
    mkd_blockchain = MKDBlockchain(blockchain_dimensions, genesis_node_id, genesis_forger)

    publisher = nodes[genesis_node_id - 1]

    start_broadcast = time.time()
    publisher.publish(mkd_blockchain)
    completed_broadcast = time.time() - start_broadcast
    print(f'time to complete broadcast to {num_nodes} nodes: {completed_broadcast}')

    # %%

    # Opens the data set file and generates nodes based on the provided data.
    # file is comma delimited in the following format: time-nodeID-x_coord-y_coord-miner-region-prev_region
    then = time.time()
    with open('blockchain/node_data.txt') as f:  # added blockchain to file path
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
            old_cluster_id = int(moving_node.cluster_id)
            new_cluster_id = int(parts[5])
            moving_node.move_node(old_cluster_id, new_cluster_id)

            #  generates list of nodes that need to publish a transaction to their cluster
            if int(parts[4]) == 1:
                nodes_transacting.append(moving_node)

            #  Once it reaches the end of num_nodes for each time increment submit transactions
            if j == (num_nodes - 1):
                cycles += 1
                print(f'Number of cycles: {cycles}')
                for node in nodes_transacting:
                    transaction = SensorTransaction(node.wallet.public_key_string(), random.randint(0, 1000))
                    node.publish(transaction)

                block_num = 0
                # if time to forge then forge and broadcast, needs to scan and perform all clusters
                if int(parts[1]) % forging_interval == 0:
                    block_num +=1
                    for cluster in clusters:
                        cluster_id = 1
                        #  choose a forger
                        forger = cluster.next_forger()
                        #  if transaction pool not empty then forge and broadcast
                        for node in cluster.member_nodes:
                            if node.wallet.public_key_string() == forger and node.transaction_pool.transactions is not []:
                                forge_begin = time.time()
                                node.mkd_forge()
                                forge_end = time.time()-forge_begin
                        cluster_forging_writer.writerow([cluster_id, block_num, forge_end])
                        cluster_id += 1
                    #print(f'Time to forge block {block_num}: {forge_end}')
            i += 1

    now = time.time() - then
    cluster_block_forging.close()
    print(f'Complete movement simulation @ {num_nodes} nodes: {now}')


    # with open('dataset.txt') as infile:
    #     for s in range(0, total_time):
    #         for n in range(0, node_num):
    #             line = infile.readline().split()
    #             if len(sys.argv) > 4:
    #                 keyFile = sys.argv[4]
    #
    #             for i in range(0, cluster_num):
    #                 node = Node(node_num, cluster_num, keyFile)
    #                 # node.startP2P()
    #                 # node.startAPI(apiPort)
