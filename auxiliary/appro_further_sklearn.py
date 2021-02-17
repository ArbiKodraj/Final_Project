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

    """Estimation of a discontinuous function via three different sklearn methods

    Parameters
    ----------

    a    : lower bound
    b    : upper bound
    nodes: amount of known points
    ts   : test size

    Methods
    ----------

    prep_regr: prepared all regressions/methods in order to approximate test data

    plot_pred: plots data (train, test and prediction)
               - Note: SVR Prediction is excluded because of its bad performance
          num: number of figure

    create_error_df: creates accuracy dataframe of each method in order to compare their performance
                     and in general their goodness
                num: number of table
    """

    def __init__(self, a=-1, b=1, nodes=100, ts=0.33):
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

    def prep_regr(self):

        dtr = DecisionTreeRegressor()
        hist1 = dtr.fit(self.X_train, self.y_train.ravel())
        self.pred1 = dtr.predict(self.X_test)

        neigh = KNeighborsRegressor(n_neighbors=1)
        hist2 = neigh.fit(self.X_train, self.y_train.ravel())
        self.pred2 = neigh.predict(self.X_test)

    def plot_pred(self, num):
        plt.figure(figsize=(12, 5))

        plt.plot(self.X_train, self.y_train, "o", ms=4, label="Train Data")
        plt.plot(self.X_test, self.y_test, "o", ms=4, color="y", label="Test Data")
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
        plt.grid()
        plt.show()

    def create_error_df(self, num):
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
                index=["KNN Regressor", "DTR"],
            )

        return df.style.set_caption(
            f"Table {num}: Approximation Accuracy ML Methods"
        )
