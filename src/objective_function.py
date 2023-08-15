import numpy

import autograd.numpy as np
from autograd import grad

from scipy.special import huber


class SquaredLoss:
    """Represents the loss function f(x) = 1/2||Ax - b||^2 + 1/2 lmbda ||x||^2.

    Attributes:
        A: np.ndarray
            A np.ndarray of dimension (m, n).
        b: np.ndarray
            A np.ndarray of dimension (m, 1).
        lmbda: float, Optional
            Regularization parameter. (Default is 0.0.)

    Methods:
        evaluate_loss(x: np.ndarray)
            Evaluates the loss of f at x.
        evaluate_gradient(x: np.ndarray)
            Evaluates the gradient of f at x.
        compute_step_size(iteration: int, x: np.ndarray, direction: np.ndarray, step: dict, max_step: float = 1)
            Computes the step-size for iterate x in a certain direction.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray, lmbda: float = 0.0):

        self.A = A
        self.Asquared = self.A.T.dot(self.A)
        self.b = b.flatten()
        self.Ab = self.A.T.dot(self.b[:, np.newaxis]).flatten()
        self.m, self.n = self.A.shape
        eigenvalues, _ = np.linalg.eigh(self.Asquared)
        self.L = float(np.max([np.max(eigenvalues), 1]))
        self.min_eigenvalue = float(np.min([np.max(eigenvalues), 1]))

        assert self.b.shape[0] == self.m, "Arrays not of correct dimensions."
        self.lmbda = lmbda

    def evaluate_loss(self, x: np.ndarray):
        """Evaluates the loss at x."""
        x = x.flatten()
        return float(
            1 / 2 * (float(self.lmbda) * np.linalg.norm(x) ** 2 + np.linalg.norm(
                self.A.dot(x[:, np.newaxis]).flatten() - self.b) ** 2))

    def evaluate_gradient(self, x: np.ndarray):
        """Evaluates the gradient of f at x."""
        x = x.flatten()
        gradient = self.Asquared.dot(x[:, np.newaxis]).flatten() - self.Ab + self.lmbda * x
        return gradient

    def compute_step_size(self,
                          iteration: int,
                          x: np.ndarray,
                          direction: np.ndarray,
                          gradient: np.array,
                          step: dict,
                          max_step: float = 1):
        """Computes the step-size for iterate x in a certain direction.

        Args:
            iteration: integer
                The current iteration of the algorithm. Needed for "open-loop".
            x: np.ndarray
            direction: np.ndarray
                FW vertex.
            gradient: np.ndarray
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop", "open-loop constant", "line-search", "line-search afw",
                    "short-step afw", "line-search difw probability simplex", "short-step", and
                    "short-step difw probability simplex".
                Additional Arguments:
                    For "open-loop", provide float values for the keys "a", "b", "c", "d" that affect the step type
                    as follows: a / (b * iteration**d + c)
            max_step: float, Optional
                Maximum step-size. (Default is 1.)

        Returns:
            optimal_distance: float
                The step-size computed according to the chosen method.
        """
        x = x.flatten()
        direction = direction.flatten()
        gradient = gradient.flatten()
        step_type = step["step type"]
        if step_type == "open-loop":
            a = step["a"]
            b = step["b"]
            c = step["c"]
            d = step["d"]
            optimal_distance = a / (b * iteration ** d + c)
        elif step_type == "open-loop constant":
            optimal_distance = step["cst"]

        elif step_type == "line-search":
            p_x = direction - x
            optimal_distance = float(-gradient.T.dot(p_x)) / (p_x.T.dot(self.Asquared).dot(p_x))

        elif step_type == "short-step":
            optimal_distance = gradient.dot(x - direction) / (self.L * np.linalg.norm(x - direction) ** 2)

        elif step_type == "line-search afw":
            optimal_distance = float(-gradient.T.dot(direction)) / (direction.T.dot(self.Asquared).dot(direction))

        elif step_type == "short-step afw":
            optimal_distance = float(-gradient.T.dot(direction)) / (self.L * np.linalg.norm(direction) ** 2)

        elif step_type == "line-search difw probability simplex":
            y_mod = direction.copy()[(direction < 0)]
            x_mod = x.copy()[(direction < 0)]
            y_mod[(y_mod == 0)] = 1
            max_step = min(max_step, np.max(-x_mod / y_mod))
            optimal_distance = float(-gradient.T.dot(direction)) / (direction.T.dot(self.Asquared).dot(direction))

        elif step_type == "short-step difw probability simplex":
            y_mod = direction.copy()[(direction < 0)]
            x_mod = x.copy()[(direction < 0)]
            y_mod[(y_mod == 0)] = 1

            try:
                max_step = min(max_step, np.max(-x_mod / y_mod))
            except ValueError:  # raised if `y` is empty.
                pass
            optimal_distance = float(-gradient.T.dot(direction)) / (self.L * np.linalg.norm(direction) ** 2)

        if optimal_distance > max_step:
            optimal_distance = max_step
        return float(optimal_distance)


class LogisticLoss:
    """Represents the logistic loss function f(x) = (1/m)sum_i log(1 + exp(-b_i * a_i^T * x)).

    Attributes:
        A: np.ndarray
            A np.ndarray of dimension (m, n).
        b: np.ndarray
            A np.ndarray of dimension (m, 1).

    Methods:
        evaluate_loss(x: np.ndarray)
            Evaluates the loss of f at x.
        evaluate_gradient(x: np.ndarray)
            Evaluates the gradient of f at x.
        compute_step_size(iteration: int, x: np.ndarray, direction: np.ndarray, step: dict, max_step: float = 1)
            Computes the step-size for iterate x in a certain direction.
    """

    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b.flatten()
        self.m, self.n = A.shape
        self.objective = lambda x: np.mean(np.log(1 + np.exp(-self.b * np.dot(self.A, x))))
        self.L = numpy.linalg.eigvalsh(1 / self.m * self.A.T.dot(self.A))[-1]

    def evaluate_loss(self, x: np.ndarray):
        """Evaluates the loss at x."""
        return float(self.objective(x))

    def evaluate_gradient(self, x: np.ndarray):
        """Evaluates the gradient of f at x."""
        grad_objective = grad(self.objective)
        return grad_objective(x)

    def compute_step_size(self,
                          iteration: int,
                          x: np.ndarray,
                          direction: np.ndarray,
                          gradient: np.array,
                          step: dict,
                          max_step: float = 1):
        """Computes the step-size for iterate x in a certain direction.

        Args:
            iteration: integer
                The current iteration of the algorithm. Needed for "open-loop".
            x: np.ndarray
            direction: np.ndarray
                FW vertex.
            gradient: np.ndarray
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop", "open-loop constant"
                Additional Arguments:
                    For "open-loop", provide float values for the keys "a", "b", "c", "d" that affect the step type
                    as follows: a / (b * iteration**d + c)
            max_step: float, Optional
                Maximum step-size. (Default is 1.)

        Returns:
            optimal_distance: float
                The step-size computed according to the chosen method.
        """
        step_type = step["step type"]
        if step_type == "open-loop":
            a = step["a"]
            b = step["b"]
            c = step["c"]
            d = step["d"]
            optimal_distance = a / (b * iteration ** d + c)
        elif step_type == "open-loop constant":
            optimal_distance = step["cst"]

        if optimal_distance > max_step:
            optimal_distance = max_step

        return float(optimal_distance)


class HuberLossCollaborativeFiltering:
    """Represents the Huber loss function for collaborative filtering f(X) = 1/|I| sum_{(i,j)in I} h_rho (A_ij - X_ij),
    where h_rho is the Huber loss with parameter rho > 0: h_rho = t^2/2 if |t|<= rho and rho(|t| - rho/2) if |t|>rho

    Attributes:
        A: np.ndarray
            A np.ndarray of dimension (m, n).
        rho: float, Optional
            (Default is 1.0.)

    Methods:
        evaluate_loss(x: np.ndarray)
            Evaluates the loss of f at x.
        evaluate_gradient(x: np.ndarray)
            Evaluates the gradient of f at x.
        compute_step_size(iteration: int, x: np.ndarray, direction: np.ndarray, step: dict, max_step: float = 1)
            Computes the step-size for iterate x in a certain direction.
    """

    def __init__(self, A: np.ndarray, rho: float = 1.0):
        self.A = A
        self.m, self.n = A.shape
        self.mn = self.m * self.n
        self.a = np.reshape(A, self.mn)
        self.N = np.sum(~np.isnan(self.a))  # number of observed entries
        self.rho = rho
        self.objective = lambda x: np.sum(huber(self.rho, np.nan_to_num(self.a - x))) / self.N
        self.derivative = np.vectorize(lambda t: t if abs(t) <= self.rho else self.rho if t > self.rho else -self.rho)

    def evaluate_loss(self, x: np.ndarray):
        """Evaluates the loss at x."""
        return float(self.objective(x))

    def evaluate_gradient(self, x: np.ndarray):
        """Evaluates the gradient of f at x."""
        der_huber = np.vectorize(lambda t: t if abs(t) <= self.rho else self.rho if t > self.rho else -self.rho)
        grad_objective = lambda x: - der_huber(np.nan_to_num(self.a - x)) / self.N
        return grad_objective(x)

    def compute_step_size(self,
                          iteration: int,
                          x: np.ndarray,
                          direction: np.ndarray,
                          gradient: np.array,
                          step: dict,
                          max_step: float = 1):
        """Computes the step-size for iterate x in a certain direction.

        Args:
            iteration: integer
                The current iteration of the algorithm. Needed for "open-loop".
            x: np.ndarray
            direction: np.ndarray
                FW vertex.
            gradient: np.ndarray
            step: dict
                A dictionnary containing the information about the step type. The dictionary can have the following arg-
                uments:
                    "step type": Choose from "open-loop", "open-loop constant"
                Additional Arguments:
                    For "open-loop", provide float values for the keys "a", "b", "c", "d" that affect the step type
                    as follows: a / (b * iteration**d + c)
            max_step: float, Optional
                Maximum step-size. (Default is 1.)

        Returns:
            optimal_distance: float
                The step-size computed according to the chosen method.
        """
        step_type = step["step type"]
        if step_type == "open-loop":
            a = step["a"]
            b = step["b"]
            c = step["c"]
            d = step["d"]
            optimal_distance = a / (b * iteration ** d + c)
        elif step_type == "open-loop constant":
            optimal_distance = step["cst"]

        if optimal_distance > max_step:
            optimal_distance = max_step

        return float(optimal_distance)
