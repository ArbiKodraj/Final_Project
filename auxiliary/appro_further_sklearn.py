import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    r2_score,
)

class Regressor:
    """This class approximates a discontinuous function using ML ``KNN Regression``,
    DTR and Linear Regression with various degrees.

    Args:
        a (int, optional): Lower bound of interpolation interval. Defaults to -1.
        b (int, optional): Upper bound of interpolation interval. Defaults to 1.
        nodes (int, optional): Number of interpolation nodes. Defaults to 100.
        ts (float, optional): Size of testing data. Defaults to 0.33.
    """
    def __init__(self, a=-1, b=1, nodes=100, ts=0.33):
        """Constructor method.
        """
        self.a = a
        self.b = b
        self.nodes = nodes
        self.ts = ts

        x = np.linspace(a, b, nodes).reshape(-1, 1)
        y = np.piecewise(
            x,
            [x < 0, (x >= 0) & (x < 0.5), (x >= 0.5) & (x < 0.8), x >= 0.8],
            [lambda x: -x + 5, 0, lambda x: x - 5, lambda x: np.exp(x)],
        ).reshape(-1, 1)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=ts
        )

    def prep_regr(self, depth=None, weight="uniform", k_neigh=1):
        """Prepares further ML methods.

        Args:
            depth (int, optional): Deepness of DTR. Defaults to None.
            weight (str, optional): Weights of KNN Regression. Defaults to "uniform".
            k_neigh (int, optional): K-Neighbors of KNN Regression. Defaults to 1.
        """
        dtr = DecisionTreeRegressor(max_depth=depth)
        hist1 = dtr.fit(self.X_train, self.y_train.ravel())
        self.pred1 = dtr.predict(self.X_test)

        neigh = KNeighborsRegressor(n_neighbors=k_neigh, weights=weight)
        hist2 = neigh.fit(self.X_train, self.y_train.ravel())
        self.pred2 = neigh.predict(self.X_test)

    def plot_pred(self, num, fitting=None):
        """Plots training, testing and predicted data.

        Args:
            num (int, float): Number of figure.
            fitting (str, optional): Overfitting methods. Defaults to None.
        """
        plt.figure(figsize=(12, 5))
        plt.plot(self.X_train, self.y_train, "o", ms=4, label="Training Data")
        plt.plot(self.X_test, self.y_test, "o", ms=4, color="y", label="Testing Data")
        plt.plot(
            self.X_test,
            self.pred2,
            "*",
            ms=5,
            color="c",
            label="Prediction KNN Regressor",
        )
        plt.plot(
            self.X_test,
            self.pred1,
            "x",
            ms=4,
            color="r",
            label="Prediction DTR",
            alpha=0.7,
        )
        plt.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Data",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )
        plt.title(
            f"Figure {num}: Approximation of discontinuous functions via KNN Regressor and DTR"
            )
        if fitting == "over":
            plt.title(
            f"Figure {num}: Over-fitted Approximation of discontinuous functions via KNN Regressor and DTR"
            )
        plt.grid()
        plt.show()

    def create_error_df(self, num, fitting=None):
        """Creates summary of prediction accuracy as table.

        Args:
            num (int, float): Number of table.
            fitting (str, optional): Overfitting methods. Defaults to None.

        Returns:
            pd.DataFrame: Prediction accuracy.
        """
        l = []
        for p in [self.pred1, self.pred2]:
            l.append(mean_absolute_error(p, self.y_test))
            l.append(mean_squared_error(p, self.y_test))
            l.append(explained_variance_score(p, self.y_test))
            l.append(r2_score(p, self.y_test))
            df = pd.DataFrame(
                [l[:4], l[4:8]],
                columns=[
                    "Mean Absolute Error",
                    "Mean Squared Error",
                    "Explained Variance Score",
                    "$R^2$ Score",
                ],
                index=["DTR", "KNN Regressor"]
            )
        if fitting=="over":
            return df.style.set_caption(
                f"Table {num}: Approximation Accuracy ML Methods"
            )
        else:
            return df.style.set_caption(
                f"Table {num}: Approximation Accuracy ML Methods"
            )
