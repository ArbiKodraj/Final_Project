# Basic imports

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# Advanced imports

from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn2pmml import PMMLPipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    explained_variance_score,
    r2_score,
)


# ------------------------------------------------------------------------------------------- 3.1 MPL Regression, Benchmark1


def highlight_min(s):
    # highlights the minimum in a Series s yellow
    is_min = s == s.min()
    return ["background-color: yellow" if v else "" for v in is_min]


h = lambda x: 5 * np.sin(np.pi * x) - np.exp(np.sin(5 * np.pi * x))  # Function Example
h1 = lambda x: 1 / (x ** 2 + 1)


def split_method(x, func, ts):
    """Split data in test and train in order to check approximation goodness

       Parameters
       ----------
           x:    (np.array) x values
           func: (function) true function
           ts:   (float) testing size = (1-alpha)
                 how much of the data shall be assigned to train data

       Returns
       -------
           (np.array) training and testing data
    """

    y = func(x)
    y = y / y.max()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts)

    x_train = x_train.reshape(-1, 1)
    x_test = x_test.reshape(-1, 1)

    return x_train, x_test, y_train, y_test


def change_data(method, x1, x2, y1, y2):
    """Normalize or standarize train and test data

       Parameters
       ----------
           method: (str) standardize or normalize method
           x1, x2: (np.arrays) training and testing data x
           y1, y2: (np.arrays) training and testing data y

       Returns
       --------
            (np.array) transformed train and test data
    """
    
    if method == "standardize":
        return scale(x1), scale(x2), scale(y1), scale(y2)
    elif method == "normalize":
        x1 /= np.max(x1)
        x2 /= np.max(x2)
        y1 /= np.max(y1)
        y2 /= np.max(y2)
        return x1, x2, y1, y2
    else:
        return x1, x2, y1, y2


def plot_train_test(
    x_train, x_test, y_train, y_test, prediction, xlabel, ylabel, num, method
):
    """Plot Test and Train Data as well as final MPL Prediction

       Parameters
       -----------
           x_train    : (np.array) x training data
           x_test     : (np.array) x testing data
           y_train    : (np.array) y training data
           y_test     : (np.array) y testing data
           prediction : (np.array) final prediction of test data via MPL-Method
           xlabel     : (str) name of the x label
           ylabel     : (str) name of the y label
           num        : (float) number of figure
           method     : (str) used method for data transformation

       Returns
       ---------
           Plots training, testing data and (optional) prediction
    """

    train_marker = mlines.Line2D(
        [],
        [],
        color="orange",
        marker="x",
        linestyle="None",
        markersize=4,
        label="Test Data",
    )
    test_marker = mlines.Line2D(
        [],
        [],
        color="blue",
        marker="o",
        linestyle="None",
        markersize=4,
        label="Train Data",
    )
    approx_marker = mlines.Line2D(
        [],
        [],
        color="red",
        marker="*",
        linestyle="None",
        markersize=4,
        label="MLP Output",
    )
    plt.figure(figsize=(12, 5))
    if isinstance(prediction, np.ndarray) == False:
        plt.plot(x_train, y_train, "o", ms=3)
        plt.plot(x_test, y_test, "x", ms=3)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(
            handles=[train_marker, test_marker],
            title="Data",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )
        plt.title(f"Figure {num}: {method} Training and Testing Data of $h(x)$")

    else:
        plt.plot(x_train, y_train, "o", ms=3)
        plt.plot(x_test, y_test, "x", ms=3)
        plt.plot(x_test, prediction, "*", ms=3, color="r")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend(
            handles=[train_marker, test_marker, approx_marker],
            title="Data",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )
        plt.title(
            f"Figure {num}: Approximation of Testing Data of $h(x)$ using MLP and {method} Data"
        )
    plt.grid()
    plt.show()


def mlp_approximation(
    xtrain,
    xtest,
    ytrain,
    ytest,
    layer_sizes=(30, 40, 50, 30),
    max_iter=2000,
    solver="lbfgs",
):

    """Approximation using MLP Regressor

       Parameters
       ----------
           layer_sizes: (tuple) number of neurons
           max_iter:    (int) number of epochs (how many times each data point will be used)
           solver:      (str) weight optimization, default = lbfgs (quasi newton method)
           *train:      (np.arrays) training data x, y
           *trest:      (np.arrays) testing data x, y

       Returns
       -------
           (np.array) prediction of neural system represented by MLP Regression
    """

    mlp = MLPRegressor(hidden_layer_sizes=layer_sizes, max_iter=max_iter, solver=solver)
    scaler = StandardScaler()
    pipeline = PMMLPipeline([("scaler", scaler), ("regressor", mlp)])

    mlp.fit(
        xtrain, ytrain.ravel()
    )  # instead of mlp one can also use pipeline for StandardScaler
    pred = mlp.predict(
        xtest
    )  # instead of mlp one can also use pipeline for StandardScaler

    return pred


def prediction_report(N_iter, xtrain, xtest, ytrain, ytest, num, method="Unchanged"):

    """Computes Error Terms for different iterations

       Parameters
       ----------
           N_iter : (int) List of iteration, note that length of list must equal 5!!
           *train : (np.arrays) training data x, y
           *test  : (np.arrays) testing data x, y
           num    : (float) number of table
           method : (str) normalize or standardized, default = unchanged

       Returns
       -------
           (pd.DataFrame) returns styled dataframe of prediction accuracy
    """

    if not isinstance(N_iter, list):
        raise TypeError("N_iter object has to be list type!")
    assert (
        len(N_iter) == 3
    ), "Not correct length. Input of list N_iter has to have length of 3!"

    predictions = [
        mlp_approximation(xtrain, xtest, ytrain, ytest, max_iter=n) for n in N_iter
    ]

    i = 0
    MAS, MSE, EVS, R2 = [], [], [], []
    while i < len(N_iter):
        MAS.append(mean_absolute_error(ytest, predictions[i]))
        MSE.append(mean_squared_error(ytest, predictions[i]))
        EVS.append(explained_variance_score(ytest, predictions[i]))
        R2.append(r2_score(ytest, predictions[i]))
        i = i + 1

    errors = np.transpose(
        pd.DataFrame(
            [MAS, MSE, EVS, R2],
            columns=[
                str(N_iter[0]) + " Iterations",
                str(N_iter[1]) + " Iterations",
                str(N_iter[2]) + " Iterations",
            ],
        )
    )

    errors.columns = [
        "Absolute Mean Error",
        "Mean Squared Error",
        "Explained Variance Score",
        "$R^2$ Score",
    ]
    return errors.style.set_caption(
        f"Table {num}: Accuracy of MPL Approximation of $h(x)$ using {method} Training Data"
    )


# ------------------------------------------------------------------------------------------- 3.1 MPL Regression, Dataset


def get_data(path, iv, dv, ts):

    """Get Dataset and Train-Test Sample

       Parameters
       ----------
           path : (str) path of the dataset - as string
           iv   : (str) independent variable - as string
           dv   : (str) dependent variable - as string
           ts   : (float) test size of train data

       Returns
       -------
           (np.arrays) training and testing data
    """

    df = pd.read_csv(path)

    x = df[iv]
    y = df[dv]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=ts)

    x_train = x_train.values.reshape(-1, 1)
    x_test = x_test.values.reshape(-1, 1)

    return x_train, x_test, y_train, y_test


# --------------------------------------------------------------------------------- 3.1 MPL Regression, Benchmark2 3DFunction


z = lambda x, y: 0.2 * np.sin(5 * x) * np.cos(5 * y)  # 3D Function


def smooth_fuction(x, y, num):

    """Plots the smooth 3D Function

       Parameters
       ----------
            x : (np.array) x values
            y : (np.array) y values

       Returns
       --------
           plots function z
    """

    X, Y = np.meshgrid(x, y)
    Z = 0.2 * np.sin(5 * X) * np.cos(5 * Y)

    fig = plt.figure(figsize=(15, 7))
    ax = fig.gca(projection="3d")
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_zlabel("$z(x, y)$")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title(f"Figure {num}: Function $z(x, y)$ in the interval [-1, 1]")
    plt.show()


class MLP3D:

    """Approximation of the 3D Function splitting into train and test data

       Parameters
       ----------
           func: three dimensional function that shall be approximated


       Methods
       -------
           train_test_method: splits data into train and test data
           mpl_regr         : prepares models, fits data and predicts test data
           error            : computes errors and some other quantitative measures
           plt_rslt         : plots train, test and predicted data for 1600 Dots
    """

    def __init__(self, func):
        """Initializes Object"""

        self.func = func
        self.x = [np.arange(-1, 1, n) for n in [0.25, 0.15, 0.1, 0.05, 0.02]]

        self.xy = []
        for i in range(len(self.x)):
            self.xy.append([(j, k) for j in self.x[i] for k in self.x[i]])

        self.zvals = []
        for j in range(len(self.xy)):
            self.zvals.append([z(p[0], p[1]) for p in self.xy[j]])

    def train_test_method(self, ts):
        """Prepares training and testing data

           Parameters
           ----------
               ts: (float) testing size
        """

        self.x_train0, self.x_test0, self.y_train0, self.y_test0 = train_test_split(
            self.xy[0], self.zvals[0], test_size=ts
        )
        self.x_train1, self.x_test1, self.y_train1, self.y_test1 = train_test_split(
            self.xy[1], self.zvals[1], test_size=ts
        )
        self.x_train2, self.x_test2, self.y_train2, self.y_test2 = train_test_split(
            self.xy[2], self.zvals[2], test_size=ts
        )
        self.x_train3, self.x_test3, self.y_train3, self.y_test3 = train_test_split(
            self.xy[4], self.zvals[4], test_size=ts
        )
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.xy[3], self.zvals[3], test_size=ts
        )

    def mpl_regr(self, hidden_layer, max_iter=2000):
        """Prepares MLP Regressor as neural network

           Parameters
           ----------
               hidden_layer: (tuple) number of neurons in hidden layers
               max_iter    : (int) number of backpropagations
        """

        mlp0 = mlp1 = mlp2 = mlp3 = mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer,
            max_iter=max_iter,
            tol=0,
        )
        # train
        mlp0.fit(self.x_train0, self.y_train0)
        mlp1.fit(self.x_train1, self.y_train1)
        mlp2.fit(self.x_train2, self.y_train2)
        mlp3.fit(self.x_train3, self.y_train3)
        mlp.fit(self.x_train, self.y_train)

        # test
        self.predictions0 = mlp.predict(self.x_test0)
        self.predictions1 = mlp.predict(self.x_test1)
        self.predictions2 = mlp.predict(self.x_test2)
        self.predictions3 = mlp.predict(self.x_test3)
        self.predictions = mlp.predict(self.x_test)

    def error(self, caption):
        """Tables accuracy of prediction in terms of errors and explained variance

           Parameters
           ----------
               caption: (str) caption of table

           Returns
           -------
               (pd.DataFrame) styled accuracy dataframe of prediction
        """

        columns = [
            "Mean Absolute Error",
            "Mean Squared Error",
            "Explained Variance Score",
            "$R^2$ Score",
        ]
        ind = [
            str(len(self.xy[0])),
            str(len(self.xy[1])),
            str(len(self.xy[2])),
            str(len(self.xy[3])),
            str(len(self.xy[4])),
        ]
        Y = [self.y_test0, self.y_test1, self.y_test2, self.y_test3, self.y_test]
        P = [
            self.predictions0,
            self.predictions1,
            self.predictions2,
            self.predictions3,
            self.predictions,
        ]

        mae = mse = evs = r2 = []
        for y, p in zip(Y, P):
            mae.append(mean_absolute_error(y, p))
            mse.append(mean_squared_error(y, p))
            evs.append(explained_variance_score(y, p))
            r2.append(r2_score(y, p))
        rslt = pd.DataFrame(
            [mae[i : i + 4] for i in range(0, len(mae), 4)],
            columns=columns,
            index=[i + " Dots" for i in ind],
        )

        return rslt.style.set_caption(caption)

    def pltres(self, num, fs=(14, 7)):
        """Plots training, testing and predicted data

           Paramaters
           ----------
               num: (float) number of figure
               fs : (tuple) size of figure, default = (14,7)
        """

        fig = plt.figure(figsize=fs)
        ax = fig.gca(projection="3d")

        # plot train data points
        x1_vals = np.array([p[0] for p in self.x_train])
        x2_vals = np.array([p[1] for p in self.x_train])

        ax.scatter(x1_vals, x2_vals, self.y_train, label="Training Data", alpha=0.5)

        # plot test data points
        x1_vals = np.array([p[0] for p in self.x_test])
        x2_vals = np.array([p[1] for p in self.x_test])

        ax.scatter(x1_vals, x2_vals, self.y_test, label="Testing Data")

        # plot approximation
        ax.scatter(
            x1_vals,
            x2_vals,
            self.predictions,
            c="red",
            marker="x",
            label="Approximation",
        )

        # style Graph
        ax.set_xlabel("x")
        ax.set_xlabel("y")
        ax.set_xlabel("z")
        plt.title(f"Figure {num}: 3D Approximation using 1600 Dots")
        plt.grid()
        plt.legend(
            title="Data",
            bbox_to_anchor=(1.04, 0.5),
            loc="center left",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )
        plt.show()
