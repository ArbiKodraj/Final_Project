import numpy as np
import matplotlib.pyplot as plt


class PlotNodes:

    """Plots the Uniform and Chebychev Nodes

    Prameters
    ---------
    n : number of nodes
    a : lower bound
    b : upper bound

    Methods
    -------
    uniform_nodes  : creates uniform nodes

    chebychev_nodes: creates chebychev nodes

    plot_nodes(fs) : plot both nodes to compare
        fs: figure size

    """

    def __init__(self, n, a, b):

        self.n = n
        self.a = a
        self.b = b

    def uniform_nodes(self):
        self.uni_nodes = np.linspace(self.a, self.b, self.n)

    def chebychev_nodes(self):
        self.cheb_nodes = np.cos(
            (self.n - np.arange(1, self.n + 1) + 0.5) * np.pi / self.n
        )

    def plot_nodes(self, fs):

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
