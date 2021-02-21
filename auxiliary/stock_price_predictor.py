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

class StockPricePredictor:
    """This object estimates missing stock prices using training data. Besides,
    it predicts furture stock prices using the whole stock price history of a
    company. To do so, it makes use of KNN Regression. The mathematical formulation 
    is given below, where f is to be approximated.

    .. math::
        P_t = f(t) + \\epsilon_t,

    Args:
        ticker (str): Ticker symbol of stock.

    Raises:
        TypeError: Ticker symbol must be passed as a string.
    """
    
    def __init__(self, ticker):
        """Constructor method.
        """
        if not isinstance(ticker, str):
            raise TypeError("Ticker Symbol must be a String Type!")
        self.ticker = ticker
        self.stock = yf.Ticker(ticker)

    def __repr__(self):
        """Representation method of object as stock history.
        """
        self.history = pd.DataFrame(self.stock.history(period="max"))
        return repr(self.history)

    def _data_prep(self, variable="Close", **kwargs):
        """Prepares training and testing data by assigning stock price history
        randomly.

        Args:
            variable (str, optional): Price to look at (Open, High, Low, Close). 
                Defaults to "Close".
            **kwargs: Further arguments for ``train_test_split`` method.

        Raises:
            TypeError: Argument variable must be a string.

        Returns:
            pd.Series: Training and testing data.
        """
        if not isinstance(variable, str):
            raise TypeError("Variable must be a String Type!")
        time = self.history.index
        price = self.history[variable]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            time, price, **kwargs
        )
        return self.X_train, self.X_test, self.y_train, self.y_test

    def plot_data(self, neigh=2, fs=(14, 6), variable="Close", num=18.1, **kwargs):
        """Prepares KNN regression model and plots data.

        Args:
            neigh (int, optional): k-nearest neighbors. Defaults to 2.
            fs (tuple, optional): Figuresize. Defaults to (14, 6).
            variable (str, optional): Price to look at (Open, High, Low, Close). 
                Defaults to "Close".
            num (float, optional): Number of Figure. Defaults to 18.1.
            **kwargs: Further arguments for ``train_test_split`` method.
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
        data = ["Training", "Testing", "Prediction"]
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
        """Returns accuracy of testing data prediction.
        """
        return print(f'Accuracy of KNN Regressor in terms of RSME: {self.rsme}')
 
    def _full_train_data(self, variable="Close"):
        """Prepares training data for future prediction using whole price history.

        Args:
            variable (str, optional): Price to look at (Open, High, Low, Close). 
                Defaults to "Close".

        Returns:
            np.array: Training and testing data using whole price and time data.
        """
        train_price = self.history[variable]
        train_time = self.history.index
        return train_price.values.reshape(-1, 1), train_time.values.reshape(-1, 1)

    def _predict_price_knn(self, weight="distance", neigh=2, days=3, periods=30):
        """Predicts future stock price.

        Args:
            weight (str, optional): Weight of knn regression. Defaults to "distance".
            neigh (int, optional): Number of k-nearest neighbors of regression models. 
                Defaults to 2.
            days (int, optional): Number of future days. Defaults to 3.
            periods (int, optional): periods to consider within selected days. 
                Defaults to 30.

        Returns:
            np.array: Training, testing and predicted data.
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
        """Returns prediction of KNN regression.

        Args:
            **kwargs: Further arguments for train_test_split method. Refers
                to :func:`_predict_price_knn` method.

        Returns:
            float: Predicted price.
        """
        train_time, train_price, rng, predict_price = self._predict_price_knn(**kwargs)
        return print(predict_price.ravel())

    def _nn_model(
        self,
        h1_layer=12,
        h2_layer=3,
        activation="relu",
        solver="adam",
        iterations=5000,
        learning_rate=1e-4,
    ):
        """Prepares MLP method. Particular interesting for the :class:`NetworkPricePredictor`
        class. 

        Args:
            h1_layer (int, optional): Number of neurons in first hidden layer. Defaults to 12.
            h2_layer (int, optional): Number of neurons in second hidden layer. Defaults to 3.
            activation (str, optional): Method used for network activation. Defaults to "relu".
            solver (str, optional): Used optimizer. Defaults to "adam".
            iterations (int, optional): Number of iterations. Defaults to 5000.
            learning_rate (float, optional): Learning rate of method. Defaults to 1e-4.

        Returns:
            function: Trained MLP model.
        """
        mlp = MLPRegressor(
            hidden_layer_sizes=(h1_layer, h2_layer),
            activation=activation,
            solver=solver,
            max_iter=iterations,
            learning_rate_init=learning_rate,
        )
        return mlp

    def nn_accuracy(self, **kwargs):
        """Returns accuracy of MLP prediction.

        Args: 
            **kwargs: Arguments for MLP method. Refers to :func:`_nn_model` for 
                neural system

        Returns:
            print: Accuracy of prediction in terms of Root Squared Mean Error.
        """
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
            f'Accuracy of MLP Regressor in terms of RSME: {mean_squared_error(self.y_test, pred, squared=False)}'
        )

    def plot_pred_data(self, num=18.2, **kwargs):
        """Plots whole training data and future prediction.

        Args:
            num (float, optional): Figure number. Defaults to 18.2.
            **kwargs: Further arguments for ``train_test_split`` method. Refers
                to :func:`_predict_price_knn`
        """
        train_time, train_price, rng, predict_price = self._predict_price_knn(**kwargs)
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 6))
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
    """Object that predicts stock price using the ``MLP regressor``. Instead of just using 
    time and price as presented in the :class:`StockPricePredictor` class, this object works 
    recrussive. It identifies patterns by using today's price :math:`P_t` as independent variable 
    and tomorrows price :math:`P_{t+1}` as dependent variable. It aims to approximate 
    :math:`\\xi{(P_t)}`, where: 

    .. math::
        P_{t+1} = \\xi{(P_t)} + \\epsilon_{t+1}

    Args:
        StockPricePredictor (object): Prepares neural system.
        ticker (str): Ticker symbol of the stock
    """
    def __init__(self, ticker):
        """Constructur method. Inheritance of :class:`StockPricePredictor`.
        """
        super().__init__(ticker)

    def __name__(self):
        """Returns name of the object.
        """
        return print("Neuronal Network Object for Stock Prediction")

    def _split_data(self, train=0.9, price="Close"):
        """Splits dataset as stock histroy randomly into training and testing data.

        Args:
            train (float, optional): Size of training data = α. Defaults to 0.9.
            price (str, optional): Price to look at (Open, High, Low, Close). 
                Defaults to "Close".

        Returns:
            np.array: Training and testing data.
        """
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
        """Plots training and testing data. Returns them as arrays, if plot argument False.

        Args:
            plot (bool, optional): Plots data if True. Returns training and testing data
                if False. Defaults to True.
            num (float, optional): Number of figure. Defaults to 18.3.

        Raises:
            TypeError:  Argument plot has to be bool.
            AssertionError: Length of training data must be equal.
            AssertionError: Length of testing data must be equal.

        Returns:
            np.array: Training and testing data if plot argument False.
        """
        if not isinstance(plot, bool):
            raise TypeError("Plot argument has to be boolean!")
        train, test = self._split_data()
        X_train, X_test, y_train, y_test = _assign_data(train, test)
        assert len(X_train) == len(y_train), "Train data are not compatible!"
        assert len(X_test) == len(y_test), "Test data are not compatible!"
        if plot == True:
            plt.figure(figsize=(14, 6))
            plt.plot(self.time_train, train, label="Training Data")
            plt.plot(self.time_test, test, label="Testing Data")
            plt.xlabel("Time")
            plt.ylabel("Stock Price")
            plt.legend()
            plt.grid(True)
            plt.title(
                f"Figure {num}: Visualization of Training and Testing Data for MLP Regression"
            )
            plt.show()
        elif plot == False:
            return (
                np.array(X_train),
                np.array(X_test),
                np.array(y_train),
                np.array(y_test),
            )

    def mlp_prediction(self, num=18.4, **kwargs):
        """MLP model initialization.

        Args:
            num (float, optional): Number of figure. Defaults to 18.4.
            **kwargs: Arguments for neural network.

        Returns:
            print: Root Mean Squared Error of prediction.
        """
        mlp = self._nn_model(**kwargs)
        X_train, X_test, y_train, y_test = self.assign_data(plot=False)
        mlp.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1).ravel())
        prediction = mlp.predict(X_test.reshape(-1, 1))
        plt.figure(figsize=(14, 6))
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
        """Returns whole price history training data for future price prediction.

        Args:
            price (str, optional): Price to look at (Open, High, Low, Close). 
                Defaults to "Close".

        Returns:
            pd.Series: Price vector.
        """
        info = pd.DataFrame(self.stock.history(period="max"))
        prices = info[price].values
        self.time = info.index
        return prices

    def future_prediction(self, todays_price, **kwargs):
        """Predicts future price (tomorrows close price) by todays close price.

        Args:
            todays_price (np.array): Current stock price. Crawled by :func:`get_actual_price_stock`.
            **kwargs: Arguments for neural network.

        Raises:
            TypeError: Argument todays_price must be an array.

        Returns:
            np.array: Tomorrows predicted (close) price.
        """
        prices = self._whole_data()
        if not isinstance(prices, np.ndarray):
            raise TypeError("Variable prices must be an array!")
        train, test = _assignment_for_future_prediction(prices)
        mlp = self._nn_model(**kwargs)
        mlp.fit(np.array(train).reshape(-1, 1), np.array(test).reshape(-1, 1).ravel())
        prediction = mlp.predict(todays_price)
        return print(f"Tomorrows Close Price will be : {prediction[0]:.2f}")

@njit
def _assign_data(train, test):
    """Prepares recursive sequence for training and testing data.
    Constructs two vectors, :math:`P_{t+1}` and :math:`P_t`.
    
    .. math::
        P_{t+1} = \\xi{(P_t)} + \\epsilon_{t+1}

    Args:
        train (np.array): Training data.
        test (np.array): Testing data.

    Returns:
        list: Training data :math:`P_t` and testing data :math:`P_{t+1}`.
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
    """Prepares recursive Sequence for whole data.
    Uses the same logic as :func:`_assign_data`.

    Args:
        data (pd.DataFrame): Stock price history.

    Returns:
        list: Training data as two vectors.
    """
    x = []
    y = []
    for i in range(len(data) - 1):
        x.append(data[i])
        y.append(data[i + 1])
    return x, y

def get_actual_price_stock(link):
    """Returns close price of specific stock.

    Args:
        link (str): Yahoo finance link of the interesting stock.

    Returns:
        np.array: Current stock price.
    """
    crawler = requests.get(link)
    hist = BeautifulSoup(crawler.text, "html.parser")
    price = float(
        hist.find_all("div", {"class": "D(ib) Va(m) Maw(65%) Ov(h)"})[0]
        .find("span")
        .text
    )
    return np.array(price)
