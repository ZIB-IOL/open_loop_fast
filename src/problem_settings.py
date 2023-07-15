
import pandas as pd
from src.feasible_region import NuclearNormBall
from src.objective_function import  HuberLossCollaborativeFiltering

import os

def movielens(radius: int = 5000):
    """Create the feasible region and objective function for collaborative filtering for the movielens dataset.
    The matrix storing the movie reviews is of dimension 943 x 1682.


    References:
        [1] F. M. Harper and J. A. Konstan. The MovieLens datasets: History and context. ACM Transactions on Interactive
        Intelligent Systems, 5(4):19:1–19:19, 2015.
    """


    # Load the data files into numpy arrays
    data = pd.read_csv(os.path.dirname(__file__) + '/../datasets/movielens100k.csv',
                       names=['user id', 'item id', 'rating', 'timestamp'])
    A = pd.pivot_table(data, values='rating', index='user id', columns='item id').values
    m, n = A.shape
    print("Dimensions: ", (m, n))

    objective_function = HuberLossCollaborativeFiltering(A=A)
    feasible_region = NuclearNormBall(m, n, radius=radius)



    return feasible_region, objective_function
