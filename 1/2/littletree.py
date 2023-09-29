from numpy import array, nan
import Tree


def get_bebek():

    # Small tree structure with only one leaf, for experimentation

    # tree_struct = [nan,  0,  0,  1,  1]
    # 
    # thetas = [[0.5, 0.5],
    # 
    #           [array([0.7, 0.3]),
    #            array([0.2, 0.8])],
    # 
    #           [array([0.5, 0.5]),
    #            array([0.5, 0.5])],
    # 
    #           [array([0.6, 0.4]),
    #            array([0.8, 0.2])],
    # 
    #           [array([0.5, 0.5]),
    #            array([0.5, 0.5])]
    #           ]
    # 
    # sample = [[nan, nan, nan, 1., nan],
    #           [nan, nan, nan, 0., nan],
    #           #[nan, nan, 1., 1., 1.],
    #           #[nan, nan, 1., 0., 1.],
    #           #[nan, nan, 0., 0., 1.],
    #           #[nan, nan, 0., 0., 0.]
    #           ]

    # Tree like in notes with two children per level, in three levels.

    tree_struct = [nan, 0, 0, 1, 1, 2, 2]

    thetas = [

        # Root theta_0
        [0.7, 0.3],

        # theta_1
        [array([0.6, 0.4]),
         array([0.2, 0.8])],

        # theta_2
        [array([0.9, 0.1]),
         array([0.3, 0.7])],

        # theta_3
        [array([0.6, 0.4]),
         array([0.1, 0.9])],

        # theta_4
        [array([0.8, 0.2]),
         array([0.7, 0.3])],

        # theta_5
        [array([0.9, 0.1]),
         array([0.9, 0.1])],

        # theta_6
        [array([0.6, 0.4]),
         array([0.8, 0.2])]
    ]

    sample = [
        [
            nan,    # beta_0
            nan,    # beta_1
            nan,    # beta_2
            0.,     # beta_3
            1.,     # beta_4
            1.,     # beta_5
            0.      # beta_6
        ]
    ]

    t = Tree.Tree()
    t.load_tree_from_direct_arrays(tree_struct, theta_array=thetas)
    t.filtered_samples = sample
    t.num_samples = len(sample)
    return t
