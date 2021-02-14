import numpy as np
import matplotlib.pyplot as plt


class PlotNodes:
    """Plots the Uniform and Chebychev Nodes

       Prameters
       ---------
           n: (int) number of nodes
           a: (int) lower bound
           b: (int) upper bound

       Methods
       -------
           uniform_nodes  : creates uniform nodes
           chebychev_nodes: creates chebychev nodes
           plot_nodes     : plot both nodes to compare
    """

    def __init__(self, n, a, b):
        """Initiliaze Object"""
        self.n = n
        self.a = a
        self.b = b

    def uniform_nodes(self):
        """Defines Uniform nodes"""
        self.uni_nodes = np.linspace(self.a, self.b, self.n)

    def chebychev_nodes(self):
        """Defines Chebychev nodes"""
        self.cheb_nodes = np.cos(
            (self.n - np.arange(1, self.n + 1) + 0.5) * np.pi / self.n
        )

    def plot_nodes(self, fs=(10, 6)):
        """Plot nodes
        
           Parameters
           ----------
               fs: (float) size of figure, by default=(10,6)
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
