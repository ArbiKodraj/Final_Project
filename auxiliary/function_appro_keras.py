import numpy as np
import pandas as pd
import matplotlib as mpl

# mpl.use('tkagg')

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.model_selection import train_test_split
from itertools import product

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

# ---------------------------------------------------------------------- Functions & Styling


def highlight_min(s):  
    """ highlights the minimum in a Series yellow
    """
    is_max = s == s.min()
    return ["background-color: yellow" if v else "" for v in is_max]


first_function = lambda x: np.exp(-x)
second_function = lambda x: abs(x)
third_function = lambda x: x * np.sin(np.pi * x)

# ---------------------------------------------------------------------- Keras Sequential Method


class SeqMethod:

    """Generate Keras Model and prepares it for the approximation of missing values

    Parameters
    ----------
    func   : true function
    nodes  : how many nodes are known
    epochs : amount of epochs of fitting method

    Methods
    ----------
    seq_init : initialize sequential method
        activator_in : which activator for input neuronal network (relu, sigmoid, etc.)
        activator_out: which activator for ouput of neuronal network (same options as above)

    predict_test : predicts test data via initialized sequential model
        a : lower bound
        b : upper bound
        ts: test size

    return_data: returns all relevant data
    """

    def __init__(self, func, nodes=300, epochs=1000):
        self.func = func
        self.nodes = nodes
        self.epochs = epochs

    def seq_init(self, activator_out,  activator_in="relu", n1_layer=20, n2_layer=18, n3_init=False, n3_layer=10):
        self.model = Sequential()
        self.model.add(
            Dense(
                n1_layer, input_dim=1, activation=activator_in, kernel_initializer="he_uniform"
            )
        )  
        self.model.add(
            Dense(n2_layer, activation=activator_in, kernel_initializer="he_uniform")
        )  
        if n3_init == True:
            self.model.add(
                Dense(n3_layer, activation=activator_in, kernel_initializer="he_uniform")
            )  
        self.model.add(Dense(1, activation=activator_out))
        self.model.compile(optimizer="adam", loss="mae")

    def predict_test(self, a=-1, b=1, ts=0.4):
        
        X = np.linspace(a, b, self.nodes)
        y = self.func(X)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=ts
        )
        callback = EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True
        )
        self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=self.epochs,
            callbacks=[callback],
            verbose=False,
        )
        self.pred = self.model.predict(self.x_test)

    def return_data(self):
        return self.x_train, self.x_test, self.y_train, self.y_test, self.pred


class _Outcome(SeqMethod):
    def plt_approximation(self, num):
        train_marker = Line2D(
            [],
            [],
            color="orange",
            marker="o",
            linestyle="None",
            markersize=4,
            label="Test Data",
        )
        test_marker = Line2D(
            [],
            [],
            color="blue",
            marker="v",
            linestyle="None",
            markersize=4,
            label="Train Data",
        )
        approx_marker = Line2D(
            [],
            [],
            color="red",
            marker="*",
            linestyle="None",
            markersize=4,
            label="Keras Output",
        )

        plt.figure(figsize=(15, 5))
        plt.plot(self.x_train, self.y_train, "o", ms=3, color="b", label="Training Data")
        plt.plot(self.x_test, self.y_test, "v", ms=3, color="y", label="Testing Data")
        plt.plot(self.x_test, self.pred, "*", ms=3, color="r", label="Prediction")
        plt.xlabel("x")
        plt.title(f"Figure {num}: Approximation of missing Values")
        plt.legend(handles=[train_marker, test_marker, approx_marker])
        plt.grid()
        plt.show()

    def return_y_test(self):
        return self.y_test

    def return_predict(self):
        return self.pred


def plt_side_by_side(
    num1,
    num2,
    num3,
    x_train1,
    y_train1,
    x_test1,
    y_test1,
    pred1,
    x_train2,
    y_train2,
    x_test2,
    y_test2,
    pred2,
    x_train3,
    y_train3,
    x_test3,
    y_test3,
    pred3,
):

    """Plots Approximation results of three different functions, where inputs are in this case already globally defined

    Parameters:
    ------------
    num1, num2, num3     : Number of Graphs
    *arg1, *arg2, arg3*  : data of first, second and third function
    """

    train_marker = Line2D(
        [],
        [],
        color="orange",
        marker="o",
        linestyle="None",
        markersize=4,
        label="Test Data",
    )
    test_marker = Line2D(
        [],
        [],
        color="blue",
        marker="v",
        linestyle="None",
        markersize=4,
        label="Train Data",
    )
    approx_marker = Line2D(
        [],
        [],
        color="red",
        marker="*",
        linestyle="None",
        markersize=4,
        label="Keras Output",
    )

    figure, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.plot(x_train1, y_train1, "o", ms=3, color="b")
    ax1.plot(x_test1, y_test1, "v", ms=3, color="y")
    ax1.plot(x_test1, pred1, "*", ms=3, color="r")
    ax1.set_title(f"Figure {num1}: Approximation of " + "$e^{-x}$")
    ax1.grid()

    ax2.plot(x_train2, y_train2, "o", ms=3, color="b")
    ax2.plot(x_test2, y_test2, "v", ms=3, color="y")
    ax2.plot(x_test2, pred2, "*", ms=3, color="r")
    ax2.set_title(f"Figure {num2}: Approximation of $|x|$")
    ax2.grid()

    ax3.plot(x_train3, y_train3, "o", ms=3, color="b")
    ax3.plot(x_test3, y_test3, "v", ms=3, color="y")
    ax3.plot(x_test3, pred3, "*", ms=3, color="r")
    ax3.set_title(f"Figure {num3}: Approximation of $x sin(\pi x)$")
    ax3.grid()
    ax3.legend(
        handles=[train_marker, test_marker, approx_marker],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        title="Data",
        shadow=True,
        fancybox=True,
        borderaxespad=0,
        title_fontsize=12,
    )
    plt.show()


def gen_frame(true_list, pred_list, num):

    """Generates a DataFrame of Errors for the three funcions

    Parameters:
    -----------
    true_list: test values of functions as a list
    pred_list: predicted values of functions as a list
    num      : number of table
    """
    accuracy = []
    for t, p in zip(true_list, pred_list):
        accuracy.append(mean_absolute_error(t, p))
        accuracy.append(mean_squared_error(t, p))
        accuracy.append(explained_variance_score(t, p))
        accuracy.append(r2_score(t, p))

    rslt = pd.DataFrame(
        [accuracy[i : i + 4] for i in range(0, len(accuracy), 4)],
        columns=[
            "Mean Absolute Error",
            "Mean Squared Error",
            "Explained Variance Score",
            "$R^2$ Score",
        ],
        index=["$e^{-x}$", "$|x|$", "$x sin(\pi x)$"],
    )
    return rslt.style.set_caption(
        f"Table {num}: Approximation-Accuracy of Keras Method for different functions"
    ).apply(highlight_min, subset=["Mean Absolute Error", "Mean Squared Error"])


# ---------------------------------------------------------------------- Keras Regressor


def _baseline_model():

    opt = Adam(learning_rate=1e-4)
    model = Sequential()
    model.add(Dense(9, input_dim=6, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(6, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(1, activation="relu", kernel_initializer="he_uniform"))
    model.compile(loss="mean_absolute_error", optimizer=opt)

    return model


class MultidimApprox:
    def __init__(self, path, out):
        self.path = path
        self.out = out

        if path.split(".")[-1] == "csv":
            self.data = pd.read_csv(path)
            data = self.data.select_dtypes(float)
        else:
            print("Data must be in csv format!")

        for i in data.isna().sum():
            if i != 0:
                data.dropna()
                print("There were NaN Values!")

        y = data[out].values
        X = data.drop(out, axis=1).values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.33
        )

    def return_data(self):
        """ Returns dataset as pd.DataFrame 
        """
        return self.data

    def estimator(self, bs=5, vs=.1, epoch=1000):
        estimator = KerasRegressor(
            build_fn=_baseline_model, batch_size=bs, verbose=False
        )
        self.hist = estimator.fit(
            self.X_train, self.y_train, epochs=epoch, validation_split=vs
        )
        self.prediction = estimator.predict(self.X_test)

    def plt_first_rslt(self, num):

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.plot(self.hist.history["loss"], label="Mean Absolute Error")
        ax1.plot(self.hist.history["val_loss"], label="Validation Loss")
        ax1.set_ylim([0, 2.5])
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Error")
        ax1.set_title(f"Figure {num}: Epoch-Loss Trade off", y=1.105)
        ax1.legend(
            title="Loss",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=3,
            title_fontsize=12,
            fancybox=True,
            shadow=True,
        )
        ax1.grid()

        ax2.plot(self.y_test, "o", label="True Values")
        ax2.plot(self.prediction, "x", label="Prediction")
        ax2.set_title(f"Figure {num + 0.1}: Prediciton", y=1.105)
        ax2.legend(
            title="Data",
            loc="upper center",
            bbox_to_anchor=(0.5, 1.1),
            ncol=3,
            fancybox=True,
            shadow=True,
            title_fontsize=12,
        )
        ax2.grid()

        plt.show()

    def plt_second_rslt(self, num1, num2):

        accu = [
            mean_absolute_error(self.prediction, self.y_test),
            mean_squared_error(self.prediction, self.y_test),
            explained_variance_score(self.prediction, self.y_test),
            r2_score(self.prediction, self.y_test),
        ]
        error_df = pd.DataFrame(
            accu,
            index=[
                "Mean Absolute Error",
                "Mean Squared Error",
                "Explained Variance",
                "$R^2$ Score",
            ],
            columns=["Score"],
        )
        error = self.y_test - self.prediction

        fig = plt.figure(figsize=(17, 5))

        ax1 = fig.add_subplot(121)
        ax1.plot(error, "o", ms=4, label="Error Term")
        ax1.hlines(error.mean(), 0, 52, ls="--", lw=2, color="y", label="Mean Error")
        ax1.hlines(
            abs(error).mean(),
            0,
            52,
            ls="-.",
            lw=2,
            color="c",
            label="Mean Absolute Error",
        )
        ax1.hlines(
            (error ** 2).mean(),
            0,
            52,
            ls="dotted",
            lw=2,
            color="r",
            label="Mean Squared Error",
        )
        ax1.set_xlim([0, 50])
        ax1.set_ylim([-1.9, 1.6])
        ax1.grid()
        ax1.set_title(f"Figure {num1}: Error of multidimensional Approximation")
        ax1.legend(
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            title="Data",
            shadow=True,
            fancybox=True,
            borderaxespad=0,
            title_fontsize=12,
        )

        ax2 = fig.add_subplot(122)
        font_size = 10
        bbox = [0.4, 0, 0.2, 0.7]
        ax2.axis("off")
        ax2.set_title(f"Table {num2}: Accuracy Score Approximation", y=0.72)
        mpl_table = ax2.table(
            cellText=error_df.values.round(5),
            rowLabels=error_df.index,
            bbox=bbox,
            colLabels=error_df.columns,
        )
        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        plt.show()
