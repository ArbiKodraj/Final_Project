import numpy as np
import matplotlib.pyplot as plt


class PlotNodes:
    """Class that visualizes *Uniform* and *Chebychev* nodes that have the following form:

    *Uniform:*

    .. math::
        x_i = a + \\frac{i-1}{n-1} (b-a) ~ \\forall i = 1,2,..,n

    *Chebychev:*

    .. math::
        x_i = \\frac{a + b}{2} + \\frac{b-a}{2} cos((\\frac{n-i+0.5}{n})\\times\pi) ~ \\forall i = 1,2,..,n.

    Args:
        n (int): Number of nodes.
        a (int): Lower bound of interval.
        b (int): Upper bound of interval.
    """
    def __init__(self, n, a, b):
        """Constructor method.
        """
        self.n = n
        self.uni_nodes = np.linspace(a, b, n)
        self.cheb_nodes = np.cos(
            (n - np.arange(1, n + 1) + 0.5) * np.pi / n
        )

    def plot_nodes(self, fs=(10, 6)):
        """Plots Uniform and Chebychev nodes.

        Args:
            fs (tuple, optional): Figure size. Defaults to (10, 6).
        """
        i, j = [], []
        [i.append(1) for l in range(len(self.uni_nodes))]
        [j.append(1) for k in range(len(self.cheb_nodes))]

        plt.figure(figsize=fs)
        plt.subplot(2, 1, 1)
        plt.scatter(self.uni_nodes, i, label=str(self.n) + " Uniform Nodes")
        plt.title("Figure 1: Plot of Nodes")
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.scatter(self.cheb_nodes, j, label=str(self.n) + " Chebychev Nodes")
        plt.legend()

        plt.show()
