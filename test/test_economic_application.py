import unittest
import sys
import numpy as np

sys.path.append("../auxiliary/")

from economic_application import demand, supply
from economic_application import PolynomialDS
from economic_application import AISupplyDemandApprox


class Test_demand_supply(unittest.TestCase):
    """This object aims to test the demand and supply function.
    It is created by subclassing :class:`unittest.TestCase.`.

    Args:
        unittest (object):  Base class, which may be used to create new test cases.
    """

    def test_demand(self):
        """Tests demand function."""
        self.assertRaises(TypeError, demand, [1, 2, 3])
        self.assertRaises(TypeError, demand, [-1, -3])
        self.assertRaises(TypeError, demand, 2)
        self.assertEqual(
            np.array([np.nan, np.nan, np.nan]).all(),
            demand(np.array([0.1, 1, 0.2])).all(),
        )

    def test_supply(self):
        """Tests supply function."""
        self.assertRaises(TypeError, supply, [1, 2, 3])
        self.assertRaises(TypeError, supply, [-1, -3])
        self.assertRaises(TypeError, supply, 2)
        self.assertEqual(
            np.array([1.0, 1.0, 1.5, 3.0, 4.2, 12.0, 20.0]).all(),
            supply(np.array([8, 3, 12, 20, 44, 79, 99])).all(),
        )


class Test_polynomial(unittest.TestCase):
    """This object aims to test the :class:`PolynomialDS` object.

    Args:
        unittest (object):  Base class, which may be used to create new test cases.
    """

    def test_construction(self):
        """Tests constructor."""
        self.assertRaises(AssertionError, PolynomialDS, -1, 10, 10, demand, supply)
        self.assertRaises(AssertionError, PolynomialDS, 10, 200, 10, demand, supply)

    def test_length(self):
        """Tests length method."""
        nodes = 100
        self.assertEqual(nodes, len(PolynomialDS(1, 50, nodes, demand, supply)))


class Test_AI(unittest.TestCase):
    """This object aims to test the :class:`AISupplyDemandApprox` object.

    Args:
        unittest (object):  Base class, which may be used to create new test cases.
    """

    def test_construction(self):
        """Tests object construction."""
        self.assertRaises(
            AssertionError, AISupplyDemandApprox, 100, supply, demand, a=-1
        )
        obj = AISupplyDemandApprox(100, supply, demand)
        self.assertNotIn(np.nan, obj.pd_train)
        self.assertNotIn(np.nan, obj.pd_test)

    def test_name(self):
        """Tests object's name."""
        obj = AISupplyDemandApprox(200, supply, demand)
        self.assertIsInstance(obj.__name__(), str)

    def test_plot(self):
        """Tests plot method."""
        obj = AISupplyDemandApprox(200, supply, demand)
        self.assertRaises(AssertionError, obj.plots, degrees=[3, 6, 10])


if __name__ == "__main__":
    unittest.main()
