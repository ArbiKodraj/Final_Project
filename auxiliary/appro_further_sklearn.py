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
    """This class approximates a discontinuous function using ML KNN Regression,
    DTR and Linear Regression with various degrees
    
    :param a: Lower bound of interpolation interval, defaults to -1
    :type a: int
    :param b: Upper bound of interpolation interval, defaults to 1
    :type b: int
    :param nodes: Number of interpolation nodes, defaults to 100
    :type nodes: int
    :param ts: Size of testing data, defaults to 0.33
    :type a: float
    """
 
    def __init__(self, a=-1, b=1, nodes=100, ts=0.33):
        """Constructor method
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
        """Prepares further ML Methods 

        :param depth: Deepness of DTR, defaults to None
        :type depth: int
        :param weight: Weights of KNN Regression, defaults to "uniform"
        :type weight: int
        :param k_neigh: K-Neighbors of KNN Regression, defaults to 1
        :type k_neigh: int
        """
        dtr = DecisionTreeRegressor(max_depth=depth)
        hist1 = dtr.fit(self.X_train, self.y_train.ravel())
        self.pred1 = dtr.predict(self.X_test)

        neigh = KNeighborsRegressor(n_neighbors=k_neigh, weights=weight)
        hist2 = neigh.fit(self.X_train, self.y_train.ravel())
        self.pred2 = neigh.predict(self.X_test)

    def plot_pred(self, num, fitting=None):
        """Plots Training, Testing and predicted Data

        :param num: Number of figure
        :type num: float, optional
        :param fitting: Overfitting methods, default to None
        :type fitting: str
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
        """Creates summary of prediction accuracy as table

        :param num: Number of table
        :type num: float, optional
        :param fitting: Overfitting methods, default to None
        :type fitting: str

        :return: A dataframe of the prediction accuracy
        :rtype: pd.DataFrame
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
