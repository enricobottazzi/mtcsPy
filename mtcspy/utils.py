import pandas as pd

def shuffle_matrix(matrix, pi):
    """
    Shuffle a matrix on a random permutation `pi`
    """

    # shuffle the columns of the matrix
    matrix = matrix[pi]

    # shuffle the rows of the matrix
    matrix = matrix.reindex(pi)

    return matrix

def unshuffle_matrix(matrix, pi):
    """
    Unshuffle a shuffle matrix to restore original order based on the permutation `pi` used for shuffling
    """

    # Create inverse permutation
    n = len(pi)
    inv_pi = pd.Series(range(n), index=pi)
    
    # unshuffle the columns using inverse permutation
    matrix = matrix[inv_pi]

    # unshuffle the rows using inverse permutation
    matrix = matrix.reindex(inv_pi)

    return matrix