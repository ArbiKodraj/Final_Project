import unittest
import sys
import numpy as np
import pandas as pd

sys.path.append("../auxiliary/")

from stock_price_predictor import _assign_data, _assignment_for_future_prediction
from stock_price_predictor import StockPricePredictor
from stock_price_predictor import NetworkPricePredictor

class Test_StockPricePredictor(unittest.TestCase):
    """Object that aims to test some specifications of the 
    :class:`StockPricePredictor` class. It assures that methods 
    do what they are supposed to do. It is created by subclassing 
    :class:`unittest.TestCase`.
    
    Args:
        unittest (object): Base class, which may be used to create new test cases.
    """
    def test_construction(self):
        """Tests constructor method.
        """
        self.assertIsInstance(
            StockPricePredictor('aapl').ticker, str
            )
        
    def test_prep(self):
        """Tests training and testing data preperation method.
        """
        aapl = StockPricePredictor('aapl')
        aapl.__repr__() 
        for _ in range(4):
            if _ < 2:
                self.assertEqual(
                    type(aapl._data_prep()[_]).__name__, 'DatetimeIndex'
                ) 
            else:
                self.assertEqual(
                    type(aapl._data_prep()[_]).__name__, 'Series'
                )      
    
    def test_data_and_mlp(self):
        """Tests data preperation and mlp preperation method.
        """
        aapl = StockPricePredictor('aapl')
        aapl.__repr__()
        for i in range(4):
            if i != 2:
                self.assertIs(
                    type(aapl._predict_price_knn()[i]),
                    np.ndarray
                )
            else:
                self.assertIs(
                    type(aapl._predict_price_knn()[i]),
                    pd.DatetimeIndex
                )
        self.assertRaises(
            AssertionError, aapl._nn_model, h1_layer=12.3, h2_layer=10.1
        )
        self.assertRaises(
            AssertionError, aapl._nn_model, iterations=10.1
        )
        self.assertRaises(
            ValueError, aapl._nn_model, learning_rate=1
        )
        self.assertRaises(
            ValueError, aapl._nn_model, learning_rate=-1
        )
        self.assertRaises(
            ValueError, aapl._nn_model, h1_layer=-10
        )
        self.assertRaises(
            ValueError, aapl._nn_model, iterations=-1000
        )

class Test_NetworkPricePredictor(unittest.TestCase):
    """Object that aims to test some specifications of the 
    :class:`NetworkPricePredictor` class. The :class:`NetworkPricePredictor` 
    class is based on the :class:`StockPricePredictor` class.
    It also assures that methods do what they are supposed to do. 
    It is created by subclassing :class:`unittest.TestCase`.
    
    Args:
        unittest (object): Base class, which may be used to create new test cases.
    """
    def test_split_data(self):
        """Tests split data method.
        """
        nio = NetworkPricePredictor('nio')
        for j in range(2):
            self.assertIsInstance(
                nio._split_data()[j],
                np.ndarray
            )

    def test_assign_data(self):
        """Tests assign data method.
        """
        nio = NetworkPricePredictor('nio')
        self.assertRaises(
            TypeError,
            nio.assign_data,
            plot=10
        )

    def test__assign_data(self):
        """Tests _assign data method.
        """
        nio = NetworkPricePredictor('nio')
        train, test = nio._split_data()
        for _ in range(4):
            self.assertIsInstance(
                _assign_data(train, test)[_],
                list
            )
        self.assertEqual(
            len(_assign_data(train, test)[0]),
            len(_assign_data(train, test)[2])
        )
        self.assertEqual(
            len(_assign_data(train, test)[1]),
            len(_assign_data(train, test)[3])
        )
        self.assertEqual(
            _assign_data(train, test)[0][1],   # P_t   (training data x), assert P_t+1 = P_t
            _assign_data(train, test)[2][0]    # P_t+1 (training data y)
        )
        self.assertEqual(
            _assign_data(train, test)[1][1],   # P_t   (testing data x), assert P_t+1 = P_t
            _assign_data(train, test)[3][0]    # P_t+1 (testing data y)
        )

    def test_whole_data(self):
        """Tests whole data method and remaining ones.
        """
        nio = NetworkPricePredictor('nio')
        prices = nio._whole_data()
        self.assertIsInstance(
            prices,
            np.ndarray
        )
        self.assertEqual(
            len(_assignment_for_future_prediction(prices)[0]),
            len(_assignment_for_future_prediction(prices)[1])
        )
        for n in range(2):
            self.assertIsInstance(
                _assignment_for_future_prediction(prices)[n], 
                list
            )
        self.assertEqual(
            _assignment_for_future_prediction(prices)[0][1], # P_t (training)
            _assignment_for_future_prediction(prices)[1][0]  # P_t+1 (testing) => assert reccursive form
        )
            
if __name__ == '__main__':
    unittest.main()

