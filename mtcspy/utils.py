import pandas as pd

def shuffle_matrix(matrix: pd.DataFrame, pi):
    """
    Shuffle a matrix on a random permutation `pi`
    """

    # shuffle the columns of the matrix and reset column names
    matrix = matrix.iloc[:, pi]
    matrix.columns = range(len(matrix.columns))

    # shuffle the rows of the matrix and reset the index
    matrix = matrix.iloc[pi].reset_index(drop=True)

    return matrix

def unshuffle_matrix(matrix: pd.DataFrame, pi):
    """
    Unshuffle a shuffle matrix to restore original order based on the permutation `pi` used for shuffling
    """

    # Create inverse permutation
    n = len(pi)
    inv_pi = list(range(n))
    for i in range(n):
        inv_pi[pi[i]] = i
    
    # unshuffle the columns using inverse permutation and reset column names
    matrix = matrix.iloc[:, inv_pi]
    matrix.columns = range(len(matrix.columns))

    # unshuffle the rows using inverse permutation and reset the index
    matrix = matrix.iloc[inv_pi].reset_index(drop=True)

    return matrix
