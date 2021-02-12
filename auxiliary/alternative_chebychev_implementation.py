import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error


f = lambda x: x * np.sin(x)


# -----------------------------------------------------------------------------


class CCMethod:

    """Chebyshev(a, b, n, func)

    Parameters:
    -------------
        a:    lower limit
        b:    upper limit
        n:    maximum degrees
        func: function that shall be approximated


    Method:
    ---------
        increase_degree : increases degree of approxiation by input
            factor      : increasing factor

        eval : yields the approximated function value.
            x: evaluation point
    """

    def __init__(self, a, b, n, func):

        self.a = a
        self.b = b
        self.n = n
        self.func = func

        bma = 0.5 * (b - a)
        bpa = 0.5 * (b + a)

        cheb_nodes = [
            np.cos(np.pi * (k + 0.5) / n) * bma + bpa for k in range(n)
        ]  # Formula of nodes

        f = list(map(func, cheb_nodes))  # y values of nodes
        self.c = [
            (2.0 / n)
            * sum([f[k] * np.cos(np.pi * j * (k + 0.5) / n) for k in range(n)])
            for j in range(n)
        ]

    def eval(self, x):

        assert (
            self.a <= x <= self.b
        ), "x is not in defined interval, choose x between a and b"  # x picked from interval

        z = (
            2.0 * (x - self.a) / (self.b - self.a) - 1
        )  # define z to normalize domain to [-1, 1]
        z2 = 2.0 * z

        (d, dd) = (self.c[-1], 0)  # Special case first step for efficiency
        for cj in self.c[-2:0:-1]:  # Clenshaw's recurrence
            (d, dd) = (z2 * d - dd + cj, d)
        return z * d - dd + 0.5 * self.c[0]  # Last step is different


def approximation_error(func, approx, figure, degree):

    a = -3  # self.a
    b = 4  # self.b + 1, just do use class interval! Did not want to extend or even change the method significantly

    error_term = [func(x) - approx.eval(x) for x in range(a, b)]

    plt.figure(figsize=(8, 4))
    plt.plot(range(a, b), error_term, label="Error")
    plt.hlines(0, a, b, label="Without error")
    plt.ylabel("Error")
    plt.xlabel("$x$ - values")
    plt.title(
        f" Figure {figure}: Error Term of Chebyshev Approximation for {degree} Degrees"
    )
    plt.legend(
        bbox_to_anchor=(1.04, 0.5),
        loc="center left",
        shadow=True,
        fancybox=True,
        borderaxespad=0,
    )
    plt.grid()
    plt.show()

    print(
        f"The maximal error of the Chebyshev interplation equals: {max(error_term):.5f}"
    )


def data_frame_rslt(chep_approx, x, function):

    approx = [chep_approx.eval(n) for n in x]
    real_values = function(x)
    error = abs(approx - real_values)

    result = pd.DataFrame(
        [approx, real_values, error],
        index=["Approximation", "Real Value", "Absolut Error Term"],
        columns=list(x),
    )
    return result.rename_axis("$Y$-Values / $X$-Values").style.set_caption(
        "Table 6: Absolut Error of Interpolation for different values"
    )  # Style DataFrame


def error_for_diff_orders(orders, func, r, figure):

    """Plot the Error Term of multiple Approximations via Chebyshev Method

     Parameters
    -----------

        orders: How many degrees shall the approximation have
        func:   Which function is going to be approximated
        r:      Range of approximation
    """

    plt.figure(figsize=(12, 5))

    for d in orders:

        cheb_approx_weird_function = CCMethod(-10, 10, d, func)
        error_term = [func(x) - cheb_approx_weird_function.eval(x) for x in r]
        plt.plot(r, error_term, label="Degree: " + str(d))
        plt.legend(
            title="Degree of Approximation",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            shadow=True,
            fancybox=True,
            title_fontsize=12,
        )
        plt.xlabel("x")
        plt.title(f"Figure {figure}: Approximation for different Degrees")

    plt.grid()
    plt.show()
