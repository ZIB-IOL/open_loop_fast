import autograd.numpy as np
from scipy.sparse.linalg import svds

from src.auxiliary_functions import fd
from src.objective_function import SquaredLoss


def lpnorm(vector, p):
    """Computes the lp norm of a vector"""
    if p == 1:
        return np.linalg.norm(vector, ord=1)
    elif p == 2:
        return np.linalg.norm(vector, ord=2)
    elif p == -1:
        return np.linalg.norm(vector, ord=np.inf)
    else:
        vector = vector.flatten()
        absolute_vector = np.abs(vector)
        power_vector = absolute_vector ** p
        solution = float(np.sum(power_vector) ** (1 / p))
        return solution


class LpBall:
    """A class used to represent the lp ball of radius 1.

    Args:
        dimension: integer, Optional
            The number of data points.
        p: float, Optional
            Set p = -1 for L infinity ball. (Default is 1.0.)
        radius: float, Optional
            (Default is 1.0.)

    Methods:
        linear_minimization_oracle(v: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in lp <v, g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, dimension: int = 400, p: float = 1.0, radius: float = 1.0):
        self.dimension = dimension
        self.p = p
        if self.p > 1:
            self.q = 1 / (1 - 1 / self.p)
        self.radius = radius
        self.diameter = 2 * self.radius

    def linear_minimization_oracle(self,
                                   v: np.ndarray,
                                   x: np.ndarray):
        """Solves the linear minimization problem min_g in lp <v, g>.

        Args:
            v: np.ndarray
            x: np.ndarray

        Returns:
            fw_vertex: np.ndarray
                The solution to the linear minimization problem.
            fw_gap: float
                The FW gap.
            distance_iterate_fw_vertex: float
                The distance between the iterate x and the FW vertex fw_vertex.
        """
        if self.p == 1:
            v = v.flatten()
            tmp_pos = np.abs(v).argmax()
            sign = np.sign(v[tmp_pos])
            fw_vertex = np.zeros(self.dimension)
            fw_vertex[tmp_pos] = - sign * self.radius
            assert np.linalg.norm(fw_vertex, ord=1) <= self.radius, "p is not in the feasible region."
        elif self.p == -1:
            v = v.flatten()
            fw_vertex = -np.sign(v) * self.radius
            assert (np.abs(fw_vertex) <= self.radius).all(), "p is not in the feasible region."

        else:
            # The solution to min_||f||_p <= 1 <f,g> is given by f_i = g_i^{q-1}/||g||_q^{q-1}.
            x = x.flatten()
            v = v.flatten()
            fw_vertex = -self.radius * np.sign(v) * np.abs(v) ** (self.q - 1) / (
                    (lpnorm(v, self.q)) ** (self.q - 1))
            fw_vertex = self.radius * fw_vertex / lpnorm(fw_vertex, self.p)
            assert abs(lpnorm(fw_vertex, self.p) - self.radius) < 10e-10, "p is not in the feasible region."
        fw_gap = float(fd(v).T.dot(fd(x)) - fd(v).T.dot(fd(fw_vertex)))
        distance_iterate_fw_vertex = np.linalg.norm(x.flatten() - fw_vertex.flatten())
        return fw_vertex, fw_gap, distance_iterate_fw_vertex

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""
        norm = lpnorm(x, self.p)
        if abs(self.radius - norm) <= epsilon:
            return "boundary"
        elif (self.radius - norm) > epsilon:
            return "interior"
        elif (self.radius - norm) < epsilon:
            return "exterior"

    def initial_point(self):
        """Returns the initial vertex."""
        x = np.zeros((self.dimension, 1))
        x[0] = self.radius
        return x


class ProbabilitySimplex:
    """A class used to represent the probability simplex.

    Args:
        dimension: integer, Optional
            The number of data points. (Default is 400.)

    Methods:
        linear_minimization_oracle(v: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in probability simplex <v, g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, dimension: int = 400):
        self.dimension = dimension

    def linear_minimization_oracle(self,
                                   v: np.ndarray,
                                   x: np.ndarray):
        """Solves the linear minimization problem min_g in probability simplex <v, g>.

        Args:
            v: np.ndarray
            x: np.ndarray

        Returns:
            fw_vertex: np.ndarray
                The solution to the linear minimization problem.
            fw_gap: float
                The FW gap.
            distance_iterate_fw_vertex: float
                The distance between the iterate x and the FW vertex fw_vertex.
        """
        tmp_pos = v.argmin()
        fw_vertex = np.zeros(self.dimension)
        fw_vertex[tmp_pos] = 1
        assert self.membership_oracle(fw_vertex) in ["boundary", "interior"], "fw_vertex is not in the feasible region."
        fw_gap = float(v.T.dot(fd(x)) - v.T.dot(fd(fw_vertex)))

        pt_xt = np.linalg.norm(x.flatten() - fw_vertex.flatten())
        return fw_vertex, fw_gap, pt_xt

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""
        norm = lpnorm(x, 1)
        if (x >= -epsilon).all():
            if abs(1 - norm) <= epsilon:
                return "boundary"
            elif (1 - norm) > epsilon:
                return "interior"
        elif (1 - norm) < epsilon:
            return "exterior"

    def initial_point(self):
        """Returns the initial vertex."""
        x = np.zeros((self.dimension, 1))
        x[0] = 1
        return x


def away_oracle(active_vertices: np.ndarray, direction: np.ndarray):
    """Solves the maximization problem max_{i} in C <direction, active_vertices[:, i]>.

        Args:
            active_vertices: np.ndarray
            direction: np.ndarray

        Returns:
            active_vertices_idx: int
                Reference to the column in the active train_or_test corresponding to the away vertex.
            away_vertex: np.ndarray
                The away vertex.
    """
    tmp = active_vertices.T.dot(direction)
    active_vertices_idx = np.argmax(tmp)
    away_vertex = active_vertices[:, active_vertices_idx]
    return away_vertex, active_vertices_idx


def vertex_among_active_vertices(active_vertices: np.ndarray, fw_vertex: np.ndarray):
    """Checks if the fw_vertex is in the set of active vertices for l1 ball or probability simplex

    Args:
        active_vertices: np.ndarray
            A matrix whose column vectors are vertices of the l1 ball.
        fw_vertex: np.ndarray
            The Frank-Wolfe vertex.

    Returns:
        active_vertex_index:
            Returns the position of fw_vertex in active_vertices as an int. If fw_vertex is not a column of
            active_vertices, this value is None.
    """
    active_vertices = fd(active_vertices)
    num_cols = active_vertices.shape[1]
    fw_vertex = fd(fw_vertex)
    # Loop through the columns of active_vertices
    for i in range(num_cols):
        # Check if the ith column is identical to x
        if np.array_equal(active_vertices[:, i], fw_vertex):
            # Return the index of the identical column
            return i
    # If no identical column was found, return None
    return None

class NuclearNormBall:
    """A class used to represent the nuclear norm ball in R^{m x n}.

    Args:
        m: integer
        n: integer
        radius: float, Optional
            (Default is 1.0.)

    Methods:
        linear_minimization_oracle(v: np.ndarray, x: np.ndarray)
            Solves the linear minimization problem min_g in nuclear norm ball <v, g>.
        membership_oracle(x,epsilon: float):
            Determines whether x is in the feasible region, on the boundary, or exterior the feasible region.
        initial_point()
            Returns the initial vertex.
    """

    def __init__(self, m: int, n: int, radius: float = 1.0):
        self.m = m
        self.n = n
        self.radius = radius
        self.diameter = 2 * self.radius

    def linear_minimization_oracle(self, v: np.ndarray, x: np.ndarray):
        """Solves the linear minimization problem min_g in nuclear norm ball <v, g>.

        Args:
            v: np.ndarray
            x: np.ndarray

        Returns:
            fw_vertex: np.ndarray
                The solution to the linear minimization problem.
            fw_gap: float
                The FW gap.
            distance_iterate_fw_vertex: float
                The distance between the iterate x and the FW vertex fw_vertex.
        """

        G = np.reshape(-v, (self.m, self.n))
        u1, s, u2 = svds(G, k=1, which='LM')
        fw_vertex = np.reshape(self.radius * np.outer(u1, u2), len(v))

        assert self.membership_oracle(fw_vertex) in ["boundary", "interior"], "fw_vertex is not in the feasible region."
        fw_gap = float(v.T.dot(fd(x)) - v.T.dot(fd(fw_vertex)))

        distance_iterate_fw_vertex = np.linalg.norm(x.flatten() - fw_vertex.flatten())
        return fw_vertex, fw_gap, distance_iterate_fw_vertex

    def membership_oracle(self, x: np.ndarray, epsilon: float = 10e-10):
        """Determines whether x is in the interior, boundary, or exterior of the feasible region."""

        x_mat = np.reshape(x, (self.m, self.n))
        norm = np.linalg.norm(x_mat, 'nuc')
        if abs(self.radius - norm) <= epsilon:
            return "boundary"
        elif (self.radius - norm) > epsilon:
            return "interior"
        elif (self.radius - norm) < epsilon:
            return "exterior"

    def initial_point(self):
        """Returns the initial vertex."""
        x = self.radius * np.reshape(np.outer(np.identity(self.m)[0], np.identity(self.n)[0]), self.m * self.n)
        return x
