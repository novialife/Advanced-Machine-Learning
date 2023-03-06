""" This file is created as a suggested solution template for question 1.2 in DD2434 - Assignment 1A.

    We encourage you to keep the function templates.
    However, this is not a "must" and you can code however you like.
    You can write helper functions etc. however you want.

    If you want, you can use the class structures provided to you (Node and Tree classes in Tree.py
    file), and modify them as needed. In addition to the data files given to you, it is very important for you to
    test your algorithm with your own simulated data for various cases and analyse the results.

    For those who do not want to use the provided structures, we also saved the properties of trees in .txt and .npy
    format. Let us know if you face any problems.

    Also, we are aware that the file names and their extensions are not well-formed, especially in Tree.py file
    (i.e example_tree_mixture.pkl_samples.txt). We wanted to keep the template codes as simple as possible.
    You can change the file names however you want.

    For this task, we gave you three different trees (q1A_2_small_tree, q1A_2_medium_tree, q1A_2_large_tree).
    Each tree has 5 samples (the inner nodes' values are masked with np.nan).
    We want you to calculate the likelihoods of each given sample and report it.

    Note:   The alphabet "K" is K={0,1,2,3,4}.

    Note:   A VERY COMMON MISTAKE is to use incorrect order of nodes' values in CPDs.
            theta is a list of lists, whose shape is approximately (num_nodes, K, K).
            For instance, if node "v" has a parent "u", then p(v=Zv | u=Zu) = theta[v][Zu][Zv].

            If you ever doubt your useage of theta, you can double-check this marginalization:
            \sum_{k=1}^K p(v = k | u=Zu) = 1
"""

import numpy as np
from Tree import Tree
import bebektree


def calculate_likelihood(tree_topology, theta, beta, tree: Tree):
    """
    This function calculates the likelihood of a sample of leaves.
    :param: tree_topology: A tree topology. Type: numpy array. Dimensions: (num_nodes, )
    :param: theta: CPD of the tree. Type: list of numpy arrays. Dimensions (approximately): (num_nodes, K, K)
    :param: beta: A list of node assignments. Type: numpy array. Dimensions: (num_nodes, )
                Note: Inner nodes are assigned to np.nan. The leaves have values in [K]
    :return: likelihood: The likelihood of beta. Type: float.

    This is a suggested template. You don't have to use it.
    """
    inherent_prob_array = tree.get_inherent_prob_array()
    likelihood = 1.0
    log_likelihood = 0

    for i, node_value in enumerate(beta):
        if not np.isnan(node_value):
            inherent_prob_array_node = inherent_prob_array[i]
            log_likelihood += np.log(inherent_prob_array_node[int(node_value)])
            likelihood *= inherent_prob_array_node[int(node_value)]

    # This is the likelihood via summing log of likelihoods. Same result, though.
    # Could differ with full log sum implementation, if k is large, not in this case, however.
    # np.exp(log_likelihood)
    return likelihood


def main():
    print("Hello World!")
    print("This file is the solution template for question 1.2.")

    print("\n1. Load tree data from file and print it\n")
    from_pickle = False # Open tree from pickle file, uses original node and will not work.
    bebek = False # Run tree designed in bebektree.py

    filename = "data/q1A_2/q2_2_large_tree.pkl" # "./data/q1A_2/q1A_2_small_tree.pkl"  # "data/q1A_2/q2_2_medium_tree.pkl", "data/q1A_2/q2_2_large_tree.pkl"
    print("filename: ", filename)

    topology_array_filename = filename + '_topology.npy'
    theta_array_filename = filename + '_theta.npy'
    sample_array_filename = filename + "_filtered_samples.npy"

    t = Tree()

    if from_pickle:
        t.load_tree(filename)
    elif bebek:
        t = bebektree.get_bebek()
    else:
        t.load_tree_from_arrays(topology_array_filename, theta_array_filename)
        filtered_samples = np.load(sample_array_filename)
        t.filtered_samples = filtered_samples
        t.num_samples = len(filtered_samples)

    t.print()
    print("K of the tree: ", t.k, "\talphabet: ", np.arange(t.k))

    print("\n2. Calculate likelihood of each FILTERED sample\n")
    # These filtered samples already available in the tree object.
    # Alternatively, if you want, you can load them from corresponding .txt or .npy files

    for sample_idx in range(t.num_samples):
        beta = t.filtered_samples[sample_idx]
        print("\n\tSample: ", sample_idx, "\tBeta: ", beta)
        sample_likelihood = calculate_likelihood(t.get_topology_array(), t.get_theta_array(), beta, t)
        print("\tLikelihood: ", sample_likelihood)


if __name__ == "__main__":
    main()
