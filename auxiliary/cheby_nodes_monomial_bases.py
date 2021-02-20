import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import gridspec
from scipy.optimize import curve_fit
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from auxiliary.scipy_implementation import five_interp
from auxiliary.scipy_implementation import six_interp
from auxiliary.scipy_implementation import seven_interp
from auxiliary.scipy_implementation import eight_interp
from auxiliary.scipy_implementation import nine_interp
from auxiliary.scipy_implementation import ten_interp


# ---------------------------- Attempt Chebychev approximation via Chebychev interpolation, fails with curve fit method of scipy


def cheb_nodes(a, b, n):
    """Returns Chebychev nodes

    :param a: Lower bound of interpolation interval
    :type a: int
    :param b: Upper bound of interpolation interval
    :type b: int
    :param n: Number of interpolation nodes
    :type n: int
    :return: Chebychev nodes
    :rtype: np.array
    """
    ccn = np.cos((n - np.arange(1, n + 1) + 0.5) * np.pi / n)
    return ccn


def poly(a, b, n, x):
    """Returns Chebychev Polynomials

    :param a: Lower bound of interpolation interval
    :type a: int
    :param b: Upper bound of interpolation interval
    :type b: int
    :param n: Number of interpolation nodes
    :type n: int
    :param x: Evaluation point
    :type x: float, optional
    :return: Chebychev Polynomial
    :rtype: np.array
    """
    z = 2 * (x - a) / (b - a) - 1
    T = 2 * z * np.cos(np.arccos(z) * (np.arange(0, n) - 1)) - np.cos(
        np.arccos(z) * (np.arange(0, n) - 2)
    )
    return T


def ccaproximation(x, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    """ Returns Chebychev approximation

    :param x: Evaluation point
    :type x: float, optional
    :param c*: Optimal weights
    :type c*: float
    :return: Chebychev Approximation
    :rtype: np.array                          
    """
    T = poly(-1, 1, 10, x)
    weights = np.array([c0, c1, c2, c3, c4, c5, c6, c7, c8, c9])
    return sum(T * weights)


# ------------------------------------------------------------------------------------ Chebychev Interpolations


def show_cheb_polyn(a, b, n):
    """ Plots Chebychev polynomials

    :param a: Lower bound of interpolation interval
    :type a: int
    :param b: Upper bound of interpolation interval
    :type b: int
    :param n: Number of interpolation nodes
    :type n: int   
    """
    x = np.linspace(a, b, n)

    plt.figure(figsize=(12, 4))
    [
        plt.plot(x, np.cos(np.arccos(x) * j), label=f"Polynomial for {j} degree")
        for j in range(1, 7)
    ]
    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        title="Polynomial Degree",
        shadow=True,
        fancybox=True,
        title_fontsize=12,
    )
    plt.title("Figure 7: Chebychev Polynomials")
    plt.grid()
    plt.show()


# ----------------------------------------------------- 3.3.1 Benchmark: Monomial Interpolation with Chebychev Nodes


class ChebyNodeInterpolation:
    """This object implements the Chebychev interpolation using Chebychev nodes
    and polynomials. The implementation with Uniform nodes is done by
    the :class:`CMethod` class

    :param n: Number of interpolation nodes
    :type n: int   
    :param func: Unknown function that 
    :type n: function
    :param degree: Degree of interpolation 
    :type n: int
    """

    def __init__(self, n, func, degree):
        """Constructor method
        """
        self.n = n
        self.func = func
        self.degree = degree

        self.ccna = np.cos((n - np.arange(1, n + 1) + 0.5) * np.pi / n)
        self.ccnb = np.cos((n - np.arange(1, n + 1 + 10) + 0.5) * np.pi / (n + 10))

    def choose_approx(self):
        """Chooses approximation degree based on degree input
        """
        assert 5 <= self.degree <= 10, "Degree must be between 5 and 10!"

        if self.degree == 5:
            self.approx = five_interp
        elif self.degree == 6:
            self.approx = six_interp
        elif self.degree == 7:
            self.approx = seven_interp
        elif self.degree == 8:
            self.approx = eight_interp
        elif self.degree == 9:
            self.approx = nine_interp
        elif self.degree == 10:
            self.approx = ten_interp

    def increase_degree(self, factor):
        """Increases degree by specific unit

        :param factor: Factor that is added to degree
        :type factor: int
        """
        if self.degree < 10:
            self.degree += factor

    def fit_curve(self):
        """Interpolation using scipy curved fit method
        """
        self.popta = curve_fit(self.approx, self.ccna, self.func(self.ccna))[0]
        self.poptb = curve_fit(self.approx, self.ccnb, self.func(self.ccnb))[0]

    def plot_cn_interp(self, N, fs, number_1, number_2):
        """Plots interpolation and true function as well as approximation error

        :param N: Number of nodes
        :type N: int
        :param fs: Figuresize
        :type fs: tuple
        :param number_1: Figure number of interpolation
        :type number_1: float, optional
        :param number_2: Figure number of error
        :type number_2: float, optional
        """
        a = -1
        b = 1

        self.x = np.linspace(a, b, N)
        fig = plt.figure(figsize=fs)
        gs = gridspec.GridSpec(2, 1, height_ratios=[1.8, 1])
        ax0 = plt.subplot(gs[0])
        ax0.plot(self.x, self.func(self.x), label="Real Function")
        ax0.plot(
            self.x,
            self.approx(self.x, *self.popta),
            label=str(self.n) + " Nodes Approximation",
        )
        ax0.plot(
            self.x,
            self.approx(self.x, *self.poptb),
            label=str(10 + self.n) + " Nodes Approximation",
        )

        ax0.set_title(
            f"Figure {number_1}: Naive Approximation Output "
            + str(self.degree)
            + " Degree"
        )
        plt.grid()
        plt.legend(
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
            shadow=True,
            fancybox=True,
        )

        plt.setp(ax0.get_xticklabels(), visible=False)
        ax1 = plt.subplot(gs[1], sharex=ax0)
        ax1.plot(
            self.x,
            self.approx(self.x, *self.popta) - self.func(self.x),
            label=str(self.n) + " Nodes Error",
        )
        ax1.plot(
            self.x,
            self.approx(self.x, *self.poptb) - self.func(self.x),
            label=str(10 + self.n) + " Nodes Error",
        )
        ax1.set_title(
            f"Figure {number_2}: Naive Approximation Error "
            + str(self.degree)
            + " Degree"
        )
        plt.subplots_adjust(hspace=0.0)
        plt.grid()
        plt.legend(
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            borderaxespad=0,
            shadow=True,
            fancybox=True,
        )
        plt.tight_layout()
        plt.show()

    def table_error(self, number):
        """Creates dataframe of approximation accuracy

        :param number: Number of table
        :type number: int
        :return: Approximation Accuracy
        :rtype: pd.DataFrame
        """
        maea = mean_absolute_error(self.approx(self.x, *self.popta), self.func(self.x))
        maeb = mean_absolute_error(self.approx(self.x, *self.poptb), self.func(self.x))
        msea = mean_squared_error(self.approx(self.x, *self.popta), self.func(self.x))
        mseb = mean_squared_error(self.approx(self.x, *self.poptb), self.func(self.x))
        eva = explained_variance_score(
            self.approx(self.x, *self.popta), self.func(self.x)
        )
        evb = explained_variance_score(
            self.approx(self.x, *self.poptb), self.func(self.x)
        )
        r2a = r2_score(self.approx(self.x, *self.popta), self.func(self.x))
        r2b = r2_score(self.approx(self.x, *self.poptb), self.func(self.x))

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
            columns=[str(10 + self.n) + " Nodes"],
        )
        rslt = pd.concat([dfa, dfb], axis=1).style.set_caption(
            f"Table {number}: Accuracy of Monomial Interpolation with Chebychev-nodes for "
            + str(self.degree)
            + " Degrees"
        )
        return rslt
