import threading

from cluster import Cluster
from mkd_blockchain import MKDBlockchain
from node import Node
from sensor_transaction import SensorTransaction

import concurrent.futures
import csv
import random
import time

MAX_NODES = 9999


def run_sim_thread(test_num, num_clusters, num_nodes, forge_interval, dimensions):
    cluster_block_forging = open(f'cluster_block_forging_{test_num}.csv', mode='w')
    cluster_forging_writer = csv.writer(cluster_block_forging, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    # Initialize Nodes with Genesis Block
    print(f'START TEST {test_num}')
    # num_nodes = 80
    # num_clusters = 16
    # forging_interval = 20  # Time interval units of time
    # blockchain_dimensions = 4
    nodes = []  # list of all nodes
    genesis_node_id = int(num_nodes / 2)

    # iterate and initialize each node
    for i in range(num_nodes):
        new_node = Node(test_num, i + 1, 0)
        # new_node.blockchain = mkd_blockchain
        new_node.start_listener(f'{test_num}.c0')
        nodes.append(new_node)

    # initialize each cluster
    clusters = []
    for i in range(num_clusters):
        new_cluster = Cluster(i + 1)
        new_cluster.start_listener(f'{test_num}.c{i+1}')
        clusters.append(new_cluster)

    genesis_forger = nodes[genesis_node_id - 1].wallet.public_key_string()
    mkd_blockchain = MKDBlockchain(dimensions, genesis_node_id, genesis_forger)

    publisher = nodes[genesis_node_id - 1]

    # start_broadcast = time.time()
    publisher.publish(mkd_blockchain)
    # completed_broadcast = time.time() - start_broadcast
    # print(f'time to complete broadcast to {num_nodes} nodes: {completed_broadcast}')

    # %%

    # Opens the data set file and generates nodes based on the provided data.
    # file is comma delimited in the following format: time-nodeID-x_coord-y_coord-miner-region-prev_region
    # then = time.time()
    with open(f'blockchain/node_data_{test_num}.txt') as f:  # added blockchain to file path
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
                print(f'Number of cycles for test {test_num}: {cycles}')
                for node in nodes_transacting:
                    transaction = SensorTransaction(node.wallet.public_key_string(), random.randint(0, 1000))
                    node.publish(transaction)

                block_num = 0
                # if time to forge then forge and broadcast, needs to scan and perform all clusters
                if int(parts[1]) % forge_interval == 0:
                    block_num += 1
                    cluster_id = 1
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
                            left_size, right_size = node.blockchain.blocks.get_left_right_size()
                            with open(f'branch_size_left_right_{test_num}.csv', mode='a') as branch_size:
                                branch_size_writer = csv.writer(branch_size, delimiter='.', quotechar='"',
                                                                quoting=csv.QUOTE_MINIMAL)
                                branch_size_writer.writerow([node.cluster_id, node.node_id, left_size, right_size])
                            cluster_forging_writer.writerow([cluster_id, block_num, forge_end])
                        cluster_id += 1
            i += 1

    # now = time.time() - then
    cluster_block_forging.close()
    # print(f'Complete movement simulation @ {num_nodes} nodes: {now}')


# run_sim_thread(test_num, num_clusters, num_nodes, forge_interval, dimensions):
if __name__ == '__main__':
    run_sim_thread(1, 16, 80, 100, 4)

    '''
        Regular Threading Method
    test_1 = threading.Thread(target=run_sim_thread, args=(1, 16, 80, 20, 4))
    test_2 = threading.Thread(target=run_sim_thread, args=(2, 16, 80, 30, 4))
    test_3 = threading.Thread(target=run_sim_thread, args=(3, 16, 80, 40, 4))

    test_1.start()
    test_2.start()
    test_3.start()

    test_1.join()
    test_2.join()
    test_3.join()
    '''

    '''
        Executor Method
    
    # def run_sim_thread(test_num, num_clusters, num_nodes, forge_interval, dimensions):
    test_amount = 3
    cluster_test_numbers = [16, 16, 16]
    node_test_numbers = [80, 80, 80]
    forge_interval_test_numbers = [20, 30, 40]
    dimensions_test_numbers = [4, 4, 4]

    with concurrent.futures.ThreadPoolExecutor(thread_name_prefix='test:') as executor:
        executor.map(run_sim_thread, range(1, test_amount+1), cluster_test_numbers, node_test_numbers,
                     forge_interval_test_numbers, dimensions_test_numbers)
    '''
    # cluster_block_forging = open('cluster_block_forging.csv', mode='w')
    # cluster_forging_writer = csv.writer(cluster_block_forging, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #
    # # Initialize Nodes with Genesis Block
    # print('START')
    # MAX_NODES = 9999
    # num_nodes = 80
    # num_clusters = 16
    # forging_interval = 20  # Time interval.  10 units of time
    # nodes = []  # list of all nodes
    # blockchain_dimensions = 4
    # genesis_node_id = int(num_nodes / 2)
    #
    # # iterate and initialize each node
    # then = time.time()
    # for i in range(num_nodes):
    #     new_node = Node(i + 1, 0)
    #     # new_node.blockchain = mkd_blockchain
    #     new_node.start_listener('c0')
    #     nodes.append(new_node)
    # now = time.time() - then
    # print(f'block init time @ {num_nodes} nodes: {now}')
    #
    # # initialize each cluster
    # clusters = []
    # for i in range(num_clusters):
    #     new_cluster = Cluster(i + 1)
    #     new_cluster.start_listener(('c' + str(i + 1)))
    #     clusters.append(new_cluster)
    #
    # genesis_forger = nodes[genesis_node_id - 1].wallet.public_key_string()
    # mkd_blockchain = MKDBlockchain(blockchain_dimensions, genesis_node_id, genesis_forger)
    #
    # publisher = nodes[genesis_node_id - 1]
    #
    # start_broadcast = time.time()
    # publisher.publish(mkd_blockchain)
    # completed_broadcast = time.time() - start_broadcast
    # print(f'time to complete broadcast to {num_nodes} nodes: {completed_broadcast}')
    #
    # # %%
    #
    # # Opens the data set file and generates nodes based on the provided data.
    # # file is comma delimited in the following format: time-nodeID-x_coord-y_coord-miner-region-prev_region
    # then = time.time()
    # with open('blockchain/node_data.txt') as f:  # added blockchain to file path
    #     lines = f.readlines()  # list containing lines of file
    #     i = 0
    #     cycles = 0
    #
    #     nodes_transacting = []
    #
    #     for line in lines:
    #         j = i % num_nodes  # j will represent the increments of nodes
    #         parts = line.split(',')
    #
    #         moving_node = nodes[j]  # choosing specific node to manipulate
    #         # assigning new coordinates to the moving node
    #         moving_node.coords = [float(parts[2]), float(parts[3])]
    #
    #         # identifying old and new cluster and calling the move function to perform the needed changes
    #         old_cluster_id = int(moving_node.cluster_id)
    #         new_cluster_id = int(parts[5])
    #         moving_node.move_node(old_cluster_id, new_cluster_id)
    #
    #         #  generates list of nodes that need to publish a transaction to their cluster
    #         if int(parts[4]) == 1:
    #             nodes_transacting.append(moving_node)
    #
    #         #  Once it reaches the end of num_nodes for each time increment submit transactions
    #         if j == (num_nodes - 1):
    #             cycles += 1
    #             print(f'Number of cycles: {cycles}')
    #             for node in nodes_transacting:
    #                 transaction = SensorTransaction(node.wallet.public_key_string(), random.randint(0, 1000))
    #                 node.publish(transaction)
    #
    #             block_num = 0
    #             # if time to forge then forge and broadcast, needs to scan and perform all clusters
    #             if int(parts[1]) % forging_interval == 0:
    #                 block_num +=1
    #                 for cluster in clusters:
    #                     cluster_id = 1
    #                     #  choose a forger
    #                     forger = cluster.next_forger()
    #                     #  if transaction pool not empty then forge and broadcast
    #                     for node in cluster.member_nodes:
    #                         if node.wallet.public_key_string() == forger and node.transaction_pool.transactions is not []:
    #                             forge_begin = time.time()
    #                             node.mkd_forge()
    #                             forge_end = time.time()-forge_begin
    #                         left_size, right_size = node.blockchain.blocks.get_left_right_size()
    #                         with open('branch_size_left_right.csv', mode='a') as branch_size:
    #                             branch_size_writer = csv.writer(branch_size, delimiter='.', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #                             branch_size_writer.writerow([node.cluster_id, node.node_id, left_size, right_size])
    #                       cluster_forging_writer.writerow([cluster_id, block_num, forge_end])
    #                     cluster_id += 1
    #         i += 1
    #
    # now = time.time() - then
    # cluster_block_forging.close()
    # print(f'Complete movement simulation @ {num_nodes} nodes: {now}')


