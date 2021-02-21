import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy.polynomial import Polynomial as P 
from numpy.polynomial import Chebyshev as C  

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from matplotlib import gridspec

def k(x):
    """``Benchmark function``.

    Args:
        x (int, float): Input.

    Returns:
        float: Output. :math:`-sin(x\\pi)`
    """
    return np.exp(-np.sin(x * np.pi))   

# -------------------------------------------------------------------------------- 3.1.1 Benchmark : evenly spaced interpolation

class PMethod:
    """This object interplates a function using Uniform nodes and the numpy library.
    as well as monomial polnomials, i.e.,

    .. math::
        x_i = a + \\frac{i-1}{n-1} (b-a) \\forall i = 1,2,..,n

    Args:
        a (int): Lower bound interval.
        b (int): Upper bound interval.
        n (int): Number of nodes.
        degree (int): Degree of Approximation.
        func (function): Benchmark function.
    """
    def __init__(self, a, b, n, degree, func):
        """Constructor method.
        """
        self.a = a
        self.b = b
        self.n = n
        self.degree = degree
        self.func = func

    def naive_poly(self):
        """Evenly spaced node interpolation.
        """
        x = np.linspace(self.a, self.b, self.n)
        self.poly = P.fit(x, self.func(x), self.degree)

    def simple_plot_appro(self, N, fs):  
        """Plots benchmark function, approximation as well as approximation error.

        Args:
            N (int): Number of interpolation nodes.
            fs (tuple): Figuresize.
        """
        self.x = np.linspace(self.a, self.b, N)
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])
        ax0 = plt.subplot(gs[0])
        ax0.plot(self.x, self.func(self.x), label="Real Function")
        ax0.plot(self.x, self.poly(self.x), label="Approximation")
        ax0.set_title(
            "Figure 2.1: Naive Approximation Output " + str(self.degree) + " Degree"
        )
        plt.grid()
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", shadow=True, fancybox=True
        )

        plt.setp(ax0.get_xticklabels(), visible=False)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.plot(
            self.x, self.poly(self.x) - self.func(self.x), label="Error", color="r"
        )
        ax1.set_title(
            "Figure 2.2: Naive Approximation Error " + str(self.degree) + " Degree"
        )
        plt.subplots_adjust(hspace=0.0)
        plt.grid()
        plt.legend(
            bbox_to_anchor=(1.05, 1), loc="upper left", shadow=True, fancybox=True
        )

        plt.tight_layout()
        plt.show()

    def increasing_degree(self, inc_factor=1):
        """Increases approximation degree.

        Args:
            inc_factor (int, optional): Factor by which the degree will be decreases. 
                Defaults to 1.
        """
        self.degree = self.degree + inc_factor

    def increasing_nodes(self):
        """Increases number of nodes.
        """
        self.na = self.n * 3
        self.nb = self.n * 9

    def naive_poly_inc_nodes(self):
        """Applies interpolation using Uniform nodes. 
        """
        xa = np.linspace(self.a, self.b, self.na)
        xb = np.linspace(self.a, self.b, self.nb)

        self.polya = P.fit(xa, self.func(xa), self.degree)
        self.polyb = P.fit(xb, self.func(xb), self.degree)

    def plot_appro_inc_nodes(self, fs):
        """Plots the naive approximation using updated Uniform nodes.

        Args:
            fs (tuple): Figuresize.
        """
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])

        ax0 = plt.subplot(gs[0])
        ax0.plot(self.x, self.func(self.x), label="Real Function")
        ax0.plot(self.x, self.poly(self.x), label=str(self.n) + " Nodes Approximation")
        ax0.plot(
            self.x, self.polya(self.x), label=str(3 * self.n) + " Nodes Approximation"
        )
        ax0.plot(
            self.x, self.polyb(self.x), label=str(9 * self.n) + " Nodes Approximation"
        )
        ax0.set_title(
            "Figure 3.1: Naive Approximation Output " + str(self.degree) + " Degree"
        )
        plt.grid()
        plt.legend(
            title="Approximation for different Nodes",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )

        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.plot(
            self.x,
            self.poly(self.x) - self.func(self.x),
            label=str(self.n) + " Nodes Error",
        )
        ax1.plot(
            self.x,
            self.polya(self.x) - self.func(self.x),
            label=str(3 * self.n) + " Nodes Error",
        )
        ax1.plot(
            self.x,
            self.polyb(self.x) - self.func(self.x),
            label=str(9 * self.n) + " Nodes Error",
        )
        ax1.set_title(
            "Figure 3.2: Naive Approximation Error " + str(self.degree) + " Degree"
        )
        plt.subplots_adjust(hspace=0.0)
        plt.grid()
        plt.legend(
            title="Error for different Nodes",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )

        plt.tight_layout()
        plt.show()

    def table_error(self):
        """Returns dataframe of approximation accuracy for different numbers of nodes.

        Returns:
            pd.DataFrame: Approximation accuracy.
        """
        mae = mean_absolute_error(self.poly(self.x), self.func(self.x))
        maea = mean_absolute_error(self.polya(self.x), self.func(self.x))
        maeb = mean_absolute_error(self.polyb(self.x), self.func(self.x))

        mse = mean_squared_error(self.poly(self.x), self.func(self.x))
        msea = mean_squared_error(self.polya(self.x), self.func(self.x))
        mseb = mean_squared_error(self.polyb(self.x), self.func(self.x))

        ev = explained_variance_score(self.poly(self.x), self.func(self.x))
        eva = explained_variance_score(self.polya(self.x), self.func(self.x))
        evb = explained_variance_score(self.polyb(self.x), self.func(self.x))

        r2 = r2_score(self.poly(self.x), self.func(self.x))
        r2a = r2_score(self.polya(self.x), self.func(self.x))
        r2b = r2_score(self.polyb(self.x), self.func(self.x))

        df = pd.DataFrame(
            [mae, mse, ev, r2],
            index=[
                "Mean Squared Error",
                "Mean Absolute Error",
                "Explained Variance",
                "$R^2$ Score",
            ],
            columns=[str(self.n) + " Nodes"],
        )
        dfa = pd.DataFrame(
            [maea, msea, eva, r2a],
            index=[
                "Mean Squared Error",
                "Mean Absolute Error",
                "Explained Variance",
                "$R^2$ Score",
            ],
            columns=[str(3 * self.n) + " Nodes"],
        )
        dfb = pd.DataFrame(
            [maeb, mseb, evb, r2b],
            index=[
                "Mean Squared Error",
                "Mean Absolute Error",
                "Explained Variance",
                "$R^2$ Score",
            ],
            columns=[str(9 * self.n) + " Nodes"],
        )

        rslt = pd.concat([df, dfa, dfb], axis=1).style.set_caption(
            "Table 1: Accuracy of Naive Approximation for "
            + str(self.degree)
            + " Degrees"
        )
        return rslt

# -------------------------------------------------------------------------------- 3.1.2 Benchmark : chebychev interpolation

class CMethod:
    """This object refers to the :class:`PMethod` class. Instead of using *monomial*
    polynomials, it uses *Chebychev* polnomials for the interpolation of a function

    Args:
        a (int): Lower bound of interval.
        b (int): Upper bound of interval.
        n (int): Number of interpolation nodes.
        degree (int): Degree of approximation.
        func (function): Unknown function.
    """
    def __init__(self, a, b, n, degree, func):
        """Constructor method. It uses the exact same arguments as the :class:`PMethod`
        constructor method. This could also be achieved by ``inheritance``.
        """
        self.a = a
        self.b = b
        self.n = n
        self.degree = degree
        self.func = func

    def increase_degree(self, inc_factor=1):
        """Increases number of degrees by certain factor.

        Args:
            inc_factor (int): Factor by which the degree is increased.
                Default to 1.
        """
        self.degree += inc_factor

    def extending_nodes(self, ext_factor):
        """Multiplies number of nodes by certain factor. 

        Args:
            ext_factor (int): Factor by which the degree is multiplied.
        """
        self.n *= ext_factor

    def cheb_poly(self):
        """Applies Chebychev interpolation.  
        """
        xa = np.linspace(self.a, self.b, self.n)
        xb = np.linspace(self.a, self.b, 3 * self.n)
        xc = np.linspace(self.a, self.b, 9 * self.n)

        self.polya = C.fit(xa, self.func(xa), self.degree)
        self.polyb = C.fit(xb, self.func(xb), self.degree)
        self.polyc = C.fit(xc, self.func(xc), self.degree)

    def plot_cheb_int(self, fs, N):
        """Plots Chebychev interpolation for different nodes.

        Args:
            fs (tuple): Figuresize.
            N (int): Number of nodes final interpolation.
        """
        self.x = np.linspace(self.a, self.b, N)

        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])

        ax0 = plt.subplot(gs[0])
        ax0.plot(self.x, self.func(self.x), label="Real Function")
        ax0.plot(self.x, self.polya(self.x), label=str(self.n) + " Nodes Approximation")
        ax0.plot(
            self.x, self.polyb(self.x), label=str(3 * self.n) + " Nodes Approximation"
        )
        ax0.plot(
            self.x, self.polyc(self.x), label=str(9 * self.n) + " Nodes Approximation"
        )
        ax0.set_title(
            "Figure 4.1: Chebychev Approximation Output " + str(self.degree) + " Degree"
        )
        plt.grid()
        plt.legend(
            title="Chebychev Interpolation for different Nodes",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )

        plt.setp(ax0.get_xticklabels(), visible=False)

        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.plot(
            self.x,
            self.polya(self.x) - self.func(self.x),
            label=str(self.n) + " Nodes Error",
        )
        ax1.plot(
            self.x,
            self.polyb(self.x) - self.func(self.x),
            label=str(3 * self.n) + " Nodes Error",
        )
        ax1.plot(
            self.x,
            self.polyc(self.x) - self.func(self.x),
            label=str(9 * self.n) + " Nodes Error",
        )
        ax1.set_title(
            "Figure 4.2: Chebychev Approximation Error " + str(self.degree) + " Degree"
        )
        plt.subplots_adjust(hspace=0.0)
        plt.grid()
        plt.legend(
            title="Chebychev Error for different Nodes",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )

        plt.tight_layout()
        plt.show()

    def table_error(self):
        """Returns approximation accuracy as styled dataframe.

        Returns:
            pd.DataFrame: Approximation accuracy.
        """
        maea = mean_absolute_error(self.polya(self.x), self.func(self.x))
        maeb = mean_absolute_error(self.polyb(self.x), self.func(self.x))
        maec = mean_absolute_error(self.polyc(self.x), self.func(self.x))

        msea = mean_squared_error(self.polya(self.x), self.func(self.x))
        mseb = mean_squared_error(self.polyb(self.x), self.func(self.x))
        msec = mean_squared_error(self.polyc(self.x), self.func(self.x))

        eva = explained_variance_score(self.polya(self.x), self.func(self.x))
        evb = explained_variance_score(self.polyb(self.x), self.func(self.x))
        evc = explained_variance_score(self.polyc(self.x), self.func(self.x))

        r2a = r2_score(self.polya(self.x), self.func(self.x))
        r2b = r2_score(self.polyb(self.x), self.func(self.x))
        r2c = r2_score(self.polyc(self.x), self.func(self.x))

        dfa = pd.DataFrame(
            [maea, msea, eva, r2a],
            index=[
                "Mean Squared Error",
                "Mean Absolute Error",
                "Explained Variance",
                "$R^2$ Score",
            ],
            columns=[str(self.n) + " Nodes"],
        )
        dfb = pd.DataFrame(
            [maeb, mseb, evb, r2b],
            index=[
                "Mean Squared Error",
                "Mean Absolute Error",
                "Explained Variance",
                "$R^2$ Score",
            ],
            columns=[str(3 * self.n) + " Nodes"],
        )
        dfc = pd.DataFrame(
            [maec, msec, evc, r2c],
            index=[
                "Mean Squared Error",
                "Mean Absolute Error",
                "Explained Variance",
                "$R^2$ Score",
            ],
            columns=[str(9 * self.n) + " Nodes"],
        )

        rslt = pd.concat([dfa, dfb, dfc], axis=1).style.set_caption(
            "Table 2: Accuracy of Chebychev Approximation for "
            + str(self.degree)
            + " Degrees"
        )
        return rslt
