import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

import requests
from bs4 import BeautifulSoup

from numba import njit
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor as knn
from sklearn.neural_network import MLPRegressor


class StockPricePredictor(object):
    """Object that predicts Stock Price using KNN Regressor and MLP Regressor

       Idea
       -----
           1)    0   1   2   ..  T  - time periode (pd.datetime)
                P0  P1  P2  ..  PT  - stock price (float)
           2)  randomly assignment to train and test
           3)  predict test data and test predictions' precision
           4)   0   1   2   ..  T  =>    T+n     T+2n  ..  future time periode
               P0  P1  P2  ..  PT  =>  P_T+n   P_T+2n  ..  future stock price
               predict future price

       Parameters
       ----------
           ticker: (str) represents the Ticker Symbol of the stock

       Methods
       --------
           _data_prep        : Splits Data into Test and Train
           plot_data         : plots train, test and predicted data
           knn_accuracy      : returns accuracy of prediction using test data
           _full_train_data  : Prepares Train Data for future Prediction using whole price history
           _predict_price_knn: predicts future stock price for certain days and periods
           predict_price_knn : returns the peviously defined predicted price
           _nn_model         : prepares the MLP Regressor Object
           nn_accuracy       : tests accuracy of MLP prediction
           plot_pred_data    : plots train and predicted data
    """

    def __init__(self, ticker):
        """ Initiliaze Object """
        if not isinstance(ticker, str):
            raise TypeError("Ticker Symbol must be a String Type!")
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def __repr__(self):
        """ Representation of object using stocks history """
        self.history = pd.DataFrame(self.stock.history(period="max"))
        return repr(self.history)

    def _data_prep(self, variable="Close", **kwargs):
        """Prepares Train and Test Data

           Parameters
           ----------
               variable : (str) price to look at (Open, High, Low, Close)
                           by default = Close
                **kwargs: further arguments for train test split method

            Returns
            --------
                train and test data (pd.Series)
        """
        if not isinstance(variable, str):
            raise TypeError("Variable must be a String Type!")
        time = self.history.index
        price = self.history[variable]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            time, price, **kwargs
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot_data(self, neigh=2, fs=(15, 7), variable="Close", num=18.1, **kwargs):
        """Prepares Model and Plots Data

           Parameters
           ----------
               neigh   : (int) number of neighbors, by default = 2
               fs      : (tuple) figure sizre, by default (14, 8)
               variable: (str) price to look at (Open, High, Low, Close)
                          by default = Close
               **kwargs: further arguments for train test split method

           Returns
           ----------
               plot of train and test data
               (float) prepares model and make prediction accuracy self variable
        """
        X_train, X_test, y_train, y_test = self._data_prep(variable="Close", **kwargs)
        X_train, X_test, y_train, y_test = (
            X_train.values.reshape(-1, 1),
            X_test.values.reshape(-1, 1),
            y_train.values.reshape(-1, 1),
            y_test.values.reshape(-1, 1),
        )
        assert (X_train.shape == y_train.shape) and (
            X_test.shape == y_test.shape
        ), "Shapes are not correct!"
        self.mod = knn(
            n_neighbors=neigh,
            weights="uniform",
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
        )
        hist = self.mod.fit(X_train, y_train)
        y_pred = self.mod.predict(X_test)
        self.X_train = np.array(X_train)
        self.X_test = np.array(X_test)
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        y_pred = np.array(y_pred)
        self.rsme = mean_squared_error(self.y_test, y_pred, squared=False)
        assert y_pred.shape == X_test.shape
        data = ["Train", "Test", "Prediction"]
        colors = ["y", "g", "r"]
        markers = ["o", "o", "x"]
        fig, axis = plt.subplots(1, 1, figsize=fs)
        for i, (train, test) in enumerate(
            zip(
                [self.X_train, self.X_test, self.X_test],
                [self.y_train, self.y_test, y_pred],
            )
        ):
            axis.plot(train, test, markers[i], label=data[i], color=colors[i], ms=2)
            axis.grid(True)
            plt.legend()
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.title(f"Figure {num}: Splitted Data Apple Stock Price")
        plt.show()

    def knn_accuracy(self):
        """ Returns Accuracy of Test Data Prediction """
        return print(f"Accuracy of KNN Regressor in terms of RSME: {self.rsme}")

    def _full_train_data(self, variable="Close"):
        """Prepares Train Data for future Prediction using whole price history

           Returns
           -------
               (np.array) train and test data using whole price and time data
        """
        train_price = self.history[variable]
        train_time = self.history.index
        return train_price.values.reshape(-1, 1), train_time.values.reshape(-1, 1)

    def _predict_price_knn(self, weight="distance", neigh=2, days=3, periods=30):
        """Predicts Future Price of Stock

           Parameters
           ----------
               weight : (str) weight of knn regression, by default = distance
               neigh  : (int) number of neighbors of knn regression, by default = 2
               days   : (int) number of future days, by default = 3
               periods: (int) how many periods to consider within 3 days, by default = 30

           Returns
           --------
               (np.arrays) train, test and predicted data
        """
        train_price, train_time = self._full_train_data()
        predict_model = knn(
            n_neighbors=neigh,
            weights=weight,
            algorithm="auto",
            leaf_size=30,
            p=2,
            metric="minkowski",
        )
        predict_model.fit(train_time, train_price)
        today = datetime.today().strftime("%Y-%m-%d")
        last_day = (datetime.today() + timedelta(days)).strftime("%Y-%m-%d")
        rng = pd.date_range(start=today, end=last_day, periods=periods)
        predict_price = predict_model.predict(rng.values.reshape(-1, 1))
        return train_time, train_price, rng, predict_price

    def predict_price_knn(self, **kwargs):
        """ Returns Prediction of KNN Regression """
        train_time, train_price, rng, predict_price = self._predict_price_knn(**kwargs)
        return print(predict_price.ravel())

    def _nn_model(
        self,
        inputs_layers=1,
        output_layers=1,
        activation="identity",
        solver="adam",
        iterations=5000,
        learning_rate=1e-4,
    ):
        """Prepares MLP Method

           Parameters
           ----------
               inputs_layers: (int) number of inputs features, by default = 1
               n_layer      : (int) number of layers, by default = 3
               output_layers: (int) number of ouputs, by default = 1
               running_var  : (int) adding variable for layers in hidden layer, by default = 0
               activation   : (str) method used for activating the network, by default = identity
               solver       : (str) used optimizer, by default = adam
               iterations   : (int) number of iterations, by default = 5000
               learning_rate: (int) by default = 1e-4

           Returns
           --------
               initiliazed mlp model
        """
        mlp = MLPRegressor(
            hidden_layer_sizes=(inputs_layers, output_layers),
            activation=activation,
            solver=solver,
            max_iter=iterations,
            learning_rate_init=learning_rate,
        )
        return mlp

    def nn_accuracy(self, **kwargs):
        """ Tests Accuracy of MLP Prediction and returns result """
        mlp = self._nn_model(**kwargs)
        X_train = np.array(
            pd.to_datetime(pd.Series(self.X_train.ravel())).map(datetime.toordinal)
        )
        X_test = np.array(
            pd.to_datetime(pd.Series(self.X_test.ravel())).map(datetime.toordinal)
        )
        mlp.fit(X_train.reshape(-1, 1), self.y_train.reshape(-1, 1).ravel())
        pred = mlp.predict(X_test.reshape(-1, 1))
        return print(
            f"Accuracy of MLP Regressor in terms of RSME: {mean_squared_error(self.y_test, pred, squared=False)}"
        )

    def plot_pred_data(self, num=18.2, **kwargs):
        """ Plots whole Train Data and Future Prediction """
        train_time, train_price, rng, predict_price = self._predict_price_knn(**kwargs)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 7))
        ax0.plot(train_time, train_price)
        ax1.plot(rng, predict_price)
        ax0.set_xlabel("Time")
        ax1.set_xlabel("Time")
        ax0.set_ylabel("Stock Price")
        ax1.set_ylabel("Stock Price")
        ax0.grid(True)
        ax1.grid(True)
        ax0.set_title(f"Figure {num}: Training Data")
        ax1.set_title(f"Figure {num + .1}: KNN Future Stock Prediction")
        plt.show()


class NetworkPricePredictor(StockPricePredictor):
    """Object that predicts Stock Price using MLP regressor and Keras neuronal network
       Makes use of Recrussive equation with one period, i.e.,
           > P_t+1 = a_t*P_t + e_t

        Idea
        ----
            1)  train data X -    P_0, P_1, P_2, .., P_k-1 (np.array)
                train data y -    P_1, P_2, P_3, .., P_k   (np.array)
                test data X  -    P_k, P_k+1,   .., P_T-1  (np.array)
                test data y  -    P_k+1, P_k+2, .., P_T    (np.array)
            2)  fit model with train data and predict test data X
                check accuracy of prediction
            3)  train data X -  P_0, P_1, .., P_T-1
                train data y -  P_1, P_2, .., P_T
            4)  predict P_T+1 (y) using P_T (X)

        Parameters
        ----------
            ticker: (str) represents the Ticker Symbol of the stock

        Methods
        -------
            _split_data      : splits data to trest and train
            assign_data      : plots train test data and returns them as arrays
            mlp_prediction   : prepares mlp regression model
            _whole_data      : returns price of history
            future_prediction: predicts future stock prices
    """

    def __init__(self, ticker):
        """ Initiliaze Object, inheritance """
        super().__init__(ticker)

    def __name__(self):
        """ Returns name of the object """
        return print("Neuronal Network Object for Stock Prediction")

    def _split_data(self, train=0.9, price="Close"):
        """ Splits data into Training and Testing Data """
        info = pd.DataFrame(self.stock.history(period="max"))
        n_train = int(len(info) * train)
        price = info[price]
        time = info.index
        self.time_train = np.array(time[:n_train])
        self.time_test = np.array(time[n_train:])
        train = np.array(price[:n_train])
        test = np.array(price[n_train:])
        return train, test

    def assign_data(self, plot=True, num=18.3):
        """ Plots Train and Test Data or Returns them as arrays """
        if not isinstance(plot, bool):
            raise TypeError("Plot argument has to be boolean!")
        train, test = self._split_data()
        X_train, X_test, y_train, y_test = _assign_data(train, test)
        assert len(X_train) == len(y_train), "Train data are not compatible!"
        assert len(X_test) == len(y_test), "Test data are not compatible!"
        if plot == True:
            plt.figure(figsize=(15, 7))
            plt.plot(self.time_train, train, label="Train Data")
            plt.plot(self.time_test, test, label="Test Data")
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.legend()
            plt.grid(True)
            plt.title(
                f"Figure {num}: Visualization of Train and Test Data for MLP Regression"
            )
            plt.show()
        elif plot == False:
            return (
                np.array(X_train),
                np.array(X_test),
                np.array(y_train),
                np.array(y_test),
            )

    def mlp_prediction(self, plot=True, num=18.5, **kwargs):
        """ MLP Model Preperation """
        mlp = self._nn_model(**kwargs)
        X_train, X_test, y_train, y_test = self.assign_data(plot=False)
        mlp.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1).ravel())
        prediction = mlp.predict(X_test.reshape(-1, 1))
        plt.figure(figsize=(15, 7))
        plt.plot(self.time_train[1:], y_train, label="Training Data", lw=2)
        plt.plot(self.time_test[1:], y_test, label="Testing Data", lw=2)
        plt.plot(self.time_test[1:], prediction, label="Prediction", lw=2)
        plt.xlabel("Time")
        plt.ylabel("Stock Price")
        plt.legend()
        plt.grid(True)
        plt.title(f"Figure {num}: Training, Testing and predicted Data Nio Stock Price")
        plt.show()
        return print(
            f"RMSE of prediction equals: {np.mean(np.abs(prediction - y_test))}"
        )

    def _whole_data(self, price="Close"):
        """ Returns whole data for training the method """
        info = pd.DataFrame(self.stock.history(period="max"))
        prices = info[price].values
        self.time = info.index
        return prices

    def future_prediction(self, todays_price, **kwargs):
        """ Predicts P_t+1 (tomorrows close price) by P_t (todays close price) """
        prices = self._whole_data()
        if not isinstance(prices, np.ndarray):
            raise TypeError("Variable prices must be an array!")
        train, test = _assignment_for_future_prediction(prices)
        mlp = self._nn_model(**kwargs)
        mlp.fit(np.array(train).reshape(-1, 1), np.array(test).reshape(-1, 1).ravel())
        prediction = mlp.predict(todays_price)
        return print(f"Tomorrows Price will be : {prediction[0]}")


@njit
def _assign_data(train, test):
    """Recursive Sequence : P_t+1 = a P_t + e
       for Train and Test Data

       Parameters
       ----------
           train: (np.array) train data
           test : (np.array) test data

       Returns
       -------
           (ls) P_t and P_t+1 for test and train data
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(train) - 1):
        X_train.append(train[i])
    for j in range(len(train) - 1):
        y_train.append(train[j + 1])
    for k in range(len(test) - 1):
        X_test.append(test[k])
    for h in range(len(test) - 1):
        y_test.append(test[h + 1])
    return X_train, X_test, y_train, y_test


@njit
def _assignment_for_future_prediction(data):
    """Recursive Sequence : P_t+1 = a_t P_t + e
       for whole Data

       Parameters
       ----------
           data: (np.array) data set

       Returns
       -------
           (ls) P_t and P_t+1
    """
    x = []
    y = []
    for i in range(len(data) - 1):
        x.append(data[i])
        y.append(data[i + 1])
    return x, y


def get_actual_price_stock(link):
    """Returns close price of specific stock

       Paramaters
       ----------
           link: (str) Stock can be selected via link
                  For this, use Yahoo Finance and lock for the stock
                  you would like to predict
       Returns
       --------
           (np.array) current stock price
    """
    crawler = requests.get(link)
    hist = BeautifulSoup(crawler.text, "html.parser")
    price = float(
        hist.find_all("div", {"class": "D(ib) Va(m) Maw(65%) Ov(h)"})[0]
        .find("span")
        .text
    )
    return np.array(price)
