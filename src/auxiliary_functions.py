import autograd.numpy as np


def fd(matrix, option="column"):
    """Guarantees that autograd.numpy aarray is 2d.

    Args:
        matrix: np.ndarray
            An array.
        option: string, Optional
            Either "column" or "row", determining whether we get a row or column vector. (Default is column.)

    Returns:
        A 2d version of matrix as a column or row vector.
    """
    if matrix.ndim == 1:
        if option == "column":
            matrix = matrix[:, np.newaxis]
        elif option == "row":
            matrix = matrix[np.newaxis, :]
    return matrix

def get_non_zero_indices(x: np.ndarray):
    return np.where(x.flatten())[0].tolist()


def create_sublist(input_list, k, i):
    """
    Creates a sublist from a given list based on a specific condition.

    Args:
        input_list (list): The input list to be split into a sublist.
        k (int): The total number of sublists that could be created.
        i (int): The index of the sublist to be created (0 <= i < k).

    Returns:
        list: A sublist containing elements that satisfy the condition 'i + r' is divisible by 'k',
              where 'i' is the element index and 'r' is a non-negative integer.

    Example:
        >>> original_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> k = 3
        >>> i = 1
        >>> sublist = create_sublist(original_list, k, i)
        >>> print(sublist)
        [2, 5, 8]

        In this example, the input list is split into 3 possible sublists, and the function returns
        the sublist where 'i + r' is divisible by 'k' (in this case, 'i' is 1).
    """
    sublist = []

    for j, item in enumerate(input_list):
        if (j + i) % k == 0:
            sublist.append(item)

    return sublist