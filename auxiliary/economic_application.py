import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from scipy.interpolate import interp1d
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline



# -------------------------------------------------------------------------------- 5.1 Approximation Demand and Supply

# ---------- Demand and Supply Functions ----------
def demand(p):
    ''' Vectorized Function to determine demand
    
    Parameter
    ---------
    p : (array) price vector for demand curve
    
    '''
    if not isinstance(p, np.ndarray):
        raise ValueError('Price vector has to be an array!')
    r = np.random.rand() * 2
    n = abs(np.random.randn()) * 2
    q = 40 / (p + n) + 1 / (1 + np.exp(p - 75 + r)) + 2 / (1 + np.exp(p - 50 + r)) + 3 / (1 + np.exp(p - 25 + r))
    q[q > 20] = np.nan
    assert type(q) == type(p), 'Type of output does not equals type of input!'
    
    return q

def supply(p):
    ''' Vectorized Function to determine supply
    
    Parameter
    ---------
    p : (array) price vector for demand curve
    
    '''
    if not isinstance(p, np.ndarray):
        raise ValueError('Price vector has to be an array!')
    q = np.zeros(p.shape)
    
    for i,c in enumerate(p):   
        if (c > 0) and (c < 10):
            q[i] = 1.0
        elif (c >= 10) and (c < 20):
            q[i] = 1.5 
        elif (c >= 20) and (c < 25):
            q[i] = 3.0 
        elif (c >= 25) and (c < 35):
            q[i] = 3.6 
        elif (c >= 35) and (c < 45):
            q[i] = 4.2
        elif (c >= 45) and (c < 60):
            q[i] = 5.0
        elif (c >= 60) and (c < 75):
            q[i] = 8.0
        elif (c >= 75) and (c < 85):
            q[i] = 12.0
        elif (c >= 85) and (c < 90):
            q[i] = 16.5
        elif (c >= 90) and (c < 95):
            q[i] = 18.5
        elif (c >= 95):
            q[i] = 20.0
    assert type(q) == type(p), 'Type of output does not equals type of input!'

    return q

# ---------- Approximation using scipy ----------
class PolynomialDS:
    
    ''' Object that approximate certain supply and demand functions via sicpy 
        interpolate, includes dunder methods and assertion errors
    
    Parameter
    ---------
    a      : lower bound of prices
    b      : upper bound of prices
    nodes  : known values (price, demand, supply)
    demand : real demand function
    supply : real supply function
    
    Methods
    --------
    plt_approx - Plots approximation of demand and supply and nodes 
        ******
        fs     : (tuple) figuresize
        *num   : (float) number of figures
        
    close_intersection - Computes the point where true as well as approximated demand and supply 
                         are closest to each other, interesection 
        ******
        nodes  :  (int)  amount of nodes used for computation of intersection
    '''
    
    def __init__(self, a, b, nodes, demand, supply):
        ''' Initializer of the PolynomialDS object 
        '''
        self.a = a
        self.b = b
        assert a >= 0, 'Price cannot be negative!'
        assert (b > a) and (b <= 100), 'By Assumption: Price cannot exceed 100!'
        self.nodes = nodes 
        self.demand = demand
        self.supply = supply
        self.p = np.linspace(a, b, nodes)
        self.qd = demand(self.p)
        self.qs = supply(self.p)
    
    def __len__(self):
        ''' Amount of Nodes/known Points
        '''
        return len(self.p)

    def __repr__(self):
        ''' String representation of object
        '''
        p = np.around(self.p, decimals=2)
        qd = np.around(self.qd, decimals=2)
        qs = np.around(self.qs, decimals=2)
        return f'{len(self)} known values for Demand and Supply:\n\nPrices={p} \n\nDemand={qd} \nSupply={qs}'
    
    def __call__(self, p):
        ''' Returns approximated value of demand and supply for a given price 
        '''
        self.apprx_qd = interp1d(self.p, self.qd)
        self.apprx_qs = interp1d(self.p, self.qs)

        return f'-- Real value -- at price {p}: \n\nDemand = {self.demand(p)} \nSupply = {self.supply(p)} \n\n-- Approximated value -- at price {p}: \n\nDemand = {self.apprx_qd(p)} \nSupply = {self.apprx_qs(p)}'
    
    def __name__(self):
        ''' Returns the name of the object
        '''
        return 'Demand and Supply Interpolator'
        
    def plt_approx(self, fs=(14,7), num1=16.1, num2=16.2, num3=16.3, num4=16.4):
        
        prices = np.linspace(self.a,self.b,self.nodes * 150)
        apprx_qd = self.apprx_qd(prices)
        apprx_qs = self.apprx_qs(prices)
        qd = self.demand(prices)
        qs = self.supply(prices)
                
        fig, (ax1, ax2) = plt.subplots(2,2,figsize=fs)

        ax1[0].plot(self.qd, self.p, 'o', label='Nodes Demand', color='#4B045D')
        ax1[0].plot(apprx_qd, prices, label='Interpolation Demand', ls='--', color='#8E0C08')
        ax1[0].plot(qd, prices, label='Real Demand', alpha=.7, color='#D98D08')
        ax1[0].set_title(f'Figure {num1}: Approximation of Demand')
        ax1[0].legend(loc='center right')
        ax1[0].grid()
        
        ax1[1].plot(self.qs, self.p, 'o', label='Nodes Supply', color='#4B045D')
        ax1[1].plot(apprx_qs, prices, label='Interpolation Supply', ls='--', color='#0C5BCD')
        ax1[1].plot(qs, prices, label='Real Supply', alpha=.7, color='#67853E')
        ax1[1].set_title(f'Figure {num2}: Approximation of Supply')
        ax1[1].legend(loc='center right')
        ax1[1].grid()
        
        ax2[0].plot(apprx_qd, prices, label='Interpolation Demand', ls='--', color='#8E0C08')
        ax2[0].plot(apprx_qs, prices, label='Interpolation Supply', ls='--', color='#0C5BCD')
        ax2[0].set_title(f'Figure {num3}: Approximated Demand and Supply')
        ax2[0].legend(loc='center right')
        ax2[0].grid()
        
        ax2[1].plot(qd, prices, label='Real Demand', color='#D98D08')
        ax2[1].plot(qs, prices, label='Real Supply', color='#67853E')
        ax2[1].set_title(f'Figure {num4}: True Demand and Supply')
        ax2[1].legend(loc='center right')
        ax2[1].grid()
        
        plt.show()
        
        abs_error_qd = np.array(abs(qd - apprx_qd))
        abs_error_qd = abs_error_qd[~np.isnan(abs_error_qd)]
        abs_error_qs = np.array(abs(qs - apprx_qs))
        print(f'Mean Absolute Error: \n\nDemand = {abs_error_qd.mean():.4f} \nSupply = {abs_error_qs.mean():.4f}')
        
    def close_intersection(self, nodes=1000000):
        
        prices = np.linspace(self.a, self.b, nodes)

        f = lambda p: self.demand(p) - self.supply(p)
        abs_sd = f(prices)
        abs_sd = abs_sd[~np.isnan(abs_sd)]
        argmin = abs(abs_sd).argmin()
        pe = prices[argmin]
        qe_demand = np.around(demand(np.array([pe])), decimals=3)
        qe_supply = np.around(supply(np.array([pe])), decimals=3)
        
        g = lambda p: self.apprx_qd(p) - self.apprx_qs(p)
        abs_asd = f(prices)
        abs_asd = abs_asd[~np.isnan(abs_asd)]
        argmin_a = abs(abs_asd).argmin()
        pea = prices[argmin_a]
        aqe_demand = np.around(self.apprx_qd(np.array([pea])), decimals=3)
        aqe_supply = np.around(self.apprx_qs(np.array([pea])), decimals=3)
        print(f'Equilibrium True (Quantity, Price) \n*** *** *** *** \nDemand: {(qe_demand[0], np.around(pe, decimals=3))} \nSupply: {(qe_supply[0], np.around(pe, decimals=3))}\n')
        print(f'Equilibrium Approximation (Quantity, Price) \n*** *** *** *** \nDemand: {(aqe_demand[0], np.around(pea, decimals=3))} \nSupply: {(aqe_supply[0], np.around(pea, decimals=3))}')
     
    
# ---------- Approximation using ML ----------
class AISupplyDemandApprox:
    
    ''' Approximated Supply and Demand via different Modern Machine Learning Methods 
    
    Parameters
    ----------
    
    nodes  : Nodes of data (train and test)
    supply : Supply function
    demand : demand function
    a      : lower bound, equals 0 by default
    b      : upper bound, equals 100 by default
    ts     : test size, 0.33 by default
    rs     : random size, 42 by default
        
    Methods
    ---------
    
    plots - Plots approximation results as well as train and test data
        *******
        colors      : (list)  colors of approximation as list
        labels      : (list)  labels of train and test data
        markers     : (list)  list of markers for approximation
        n_neighbors : (int)   neighbors for KNN method
        degrees     : (list)  degrees of sklearn poly method 
        weight      : (str)   weight of decision tree method 
        fs          : (tuple) figure size of plot
        **num       : (float) number of figures
        
    reslts_as_frame - Plots approximation's accuracy (error) as da data frame
        *******
        num : (int) number of data frame 
    '''
    
    def __init__(self, 
                 nodes, 
                 supply, 
                 demand,
                 a = 0, 
                 b = 100, 
                 ts = 0.33,
                 rs = 42):
        ''' Initializer of the AISupplyDemandApprox object 
        '''

        assert a >= 0, 'Price must be Non Negative!'
        if not nodes < b:
            raise ValueError('Nodes ouf of Range!')
        
        p = np.linspace(a, b, nodes)
        q = supply(p)
        qd = demand(p)
        
        p_train, p_test, q_train, q_test = train_test_split(p, q, test_size=ts, random_state=rs)
        pd_train, pd_test, qd_train, qd_test = train_test_split(p, qd, test_size=ts, random_state=rs)
        
        self.p_train = p_train.reshape(-1,1)         # reshape data 
        self.p_test = p_test.reshape(-1,1)           # reshape data 
        self.q_train = q_train.reshape(-1,1)         # reshape data 
        self.q_test = q_test.reshape(-1,1)           # reshape data 
        
        nan_ind = np.argwhere(np.isnan(qd_train))    # select index of nan values
        qd_train_mod = np.delete(qd_train, nan_ind)  # delete nan index value
        pd_train_mod = np.delete(pd_train, nan_ind)

        self.pd_train = pd_train_mod.reshape(-1,1)
        self.pd_test = pd_test.reshape(-1,1)
        self.qd_train = qd_train_mod.reshape(-1,1)
        self.qd_test = qd_test.reshape(-1,1)
        
        assert np.isnan(self.pd_train).any() == False, 'There are nan Values!'
        assert np.isnan(self.pd_test).any() == False, 'There are nan Values!'
        
    def __name__(self):
        ''' Name of the AISupplyDemandApprox object
        '''
        return 'Moder-ML Demand and Supply Interpolator'
        
    def plots(self, 
              colors = ['teal', 'yellowgreen', 'gold'],
              label = ['Train Values', 'Test Values'] * 2,
              markers = ['x', '*', 'v'],
              n_neighbors = 3,
              degrees = [2, 6],
              weight = 'distance',
              fs = (15,10),
              num1 = 17.1,
              num2 = 17.2,
              num3 = 17.3,
              num4 = 17.4):
        
        assert len(degrees) == 2, 'List out of range!'
                
        qsup, psup = [self.q_train, self.q_test], [self.p_train, self.p_test]
        qdem, pdem = [self.qd_train, self.qd_test], [self.pd_train, self.pd_test]
        
        fig, (ax1, ax2) = plt.subplots(2,2,figsize=fs)
        
        for i, (qs, ps, qd, pd) in enumerate(zip(qsup, psup, qdem, pdem)):
            for ax in [ax1[0], ax1[1]]:
                ax.plot(qs, ps, 'o', ms=4, label=label[i])
            for ax in [ax2[0], ax2[1]]:
                ax.plot(qd, pd, 'o', ms=4, label=label[i])
                
        self.maes, self.maed = [], []
        self.mses, self.msed = [], []
        self.evss, self.evsd = [], []
        self.r2s, self.r2d   = [], [] 
        
        for i, ax in enumerate([ax1, ax2]):
            for j, d in enumerate(degrees):
                model = make_pipeline(PolynomialFeatures(d), Ridge())

                if i == 0:
                    model.fit(self.p_train, self.q_train)
                    pred = model.predict(self.p_test)
                    ax[i].plot(pred, self.p_test, markers[j], color=colors[j], ms=5, label=f'Approximation Degree {d}')
                    indexs_to_order_by = pred.ravel().argsort()
                    pred_ordered = pred[indexs_to_order_by]
                    ptest_ordered = self.p_test.ravel()[indexs_to_order_by]
                    ax[i].plot(pred_ordered, ptest_ordered, color= colors[j], alpha=.5)
                    ax[i].set_title(f'Figure {num1}: Polynomial Features Approximation Supply')
                    ax[i].grid(True)
                    ax[i].legend(loc='center right')

                    self.maes.append(mean_absolute_error(pred, self.q_test))
                    self.mses.append(mean_squared_error(pred, self.q_test))
                    self.evss.append(explained_variance_score(pred, self.q_test))
                    self.r2s.append(r2_score(pred, self.q_test))

                elif i == 1:
                    model.fit(self.pd_train, self.qd_train)
                    pred = model.predict(self.pd_test)
                    ax[i-1].plot(pred, self.pd_test, markers[j], color=colors[j], ms=5, label=f'Approximation Degree {d}')
                    indexs_to_order_by = pred.ravel().argsort()
                    pred_ordered = pred[indexs_to_order_by]
                    ptest_ordered = self.pd_test.ravel()[indexs_to_order_by]
                    ax[i-1].plot(pred_ordered, ptest_ordered, color= colors[j], alpha=.5)
                    ax[i-1].set_title(f'Figure {num3}: Polynomial Features Approximation Demand')
                    ax[i-1].grid(True)
                    ax[i-1].legend(loc='center right')

                    self.maed.append(mean_absolute_error(pred, self.qd_test))
                    self.msed.append(mean_squared_error(pred, self.qd_test))
                    self.evsd.append(explained_variance_score(pred, self.qd_test))
                    self.r2d.append(r2_score(pred, self.qd_test))
                
        methods = ['KNN', 'DecisionTree']
        knn = KNeighborsRegressor(n_neighbors, weights=weight)
        tree = DecisionTreeRegressor()
        for i, ax in enumerate([ax1, ax2]):
            for j, m in enumerate([knn, tree]):
                if i == 0:
                    m.fit(self.p_train, self.q_train)
                    pred = m.predict(self.p_test)
                    ax[i+1].plot(pred, self.p_test, markers[j], color=colors[j], ms=4, label=f'Approximation using {methods[j]}')
                    indexs_to_order_by = pred.ravel().argsort()
                    pred_ordered = pred[indexs_to_order_by]
                    ptest_ordered = self.pd_test.ravel()[indexs_to_order_by]
                    ax[i+1].plot(pred_ordered, ptest_ordered, color= colors[j], alpha=.5)
                    ax[i+1].set_title(f'Figure {num2}: KNN and DT Approximation Supply')
                    ax[i+1].grid(True)
                    ax[i+1].legend(loc='center right')
                    
                    self.maes.append(mean_absolute_error(pred, self.q_test))
                    self.mses.append(mean_squared_error(pred, self.q_test))
                    self.evss.append(explained_variance_score(pred, self.q_test))
                    self.r2s.append(r2_score(pred, self.q_test))

                elif i == 1:
                    m.fit(self.pd_train, self.qd_train)
                    pred = m.predict(self.pd_test)
                    ax[i].plot(pred, self.pd_test, markers[j], color=colors[j], ms=4, label=f'Approximation using {methods[j]}')
                    indexs_to_order_by = pred.ravel().argsort()
                    pred_ordered = pred[indexs_to_order_by]
                    ptest_ordered = self.pd_test.ravel()[indexs_to_order_by]
                    ax[i].plot(pred_ordered, ptest_ordered, color= colors[j], alpha=.5)
                    ax[i].set_title(f'Figure {num4}: KNN and DT Approximation Demand')
                    ax[i].grid(True)
                    ax[i].legend(loc='center right')

                    self.maed.append(mean_absolute_error(pred, self.qd_test))
                    self.msed.append(mean_squared_error(pred, self.qd_test))
                    self.evsd.append(explained_variance_score(pred, self.qd_test))
                    self.r2d.append(r2_score(pred, self.qd_test))

        plt.show()
        
    def reslts_as_frame(self, num=14):
        index_as_array_sup = [
            np.array(['Supply'] * 4),
            np.array(['PolynomialFeatures']*2 +['Unsupervised', 'Supervised']),
            np.array(['2 Degrees', '6 Degrees', 'KNN', 'DT'])
        ]
        index_as_array_dem = [
            np.array(['Demand'] * 4),
            np.array(['PolynomialFeatures']*2 + ['Unsupervised', 'Supervised']),
            np.array(['2 Degrees', '6 Degrees', 'KNN', 'DT'])
        ]
        col = ['Mean Absolute Error', 'Mean Squared Error', 'Explained Variance Score', '$R^2$-Score']
        
        data_supply = pd.concat(
            [pd.DataFrame(self.maes, index=index_as_array_sup),
             pd.DataFrame(self.mses, index=index_as_array_sup),
             pd.DataFrame(self.evss, index=index_as_array_sup),
             pd.DataFrame(self.r2s, index=index_as_array_sup)],
             axis=1)
        data_demand = pd.concat(
            [pd.DataFrame(self.msed, index=index_as_array_dem),
             pd.DataFrame(self.msed, index=index_as_array_dem),
             pd.DataFrame(self.evsd, index=index_as_array_dem),
             pd.DataFrame(self.r2d, index=index_as_array_dem)],
             axis=1)

        data = pd.concat([data_supply, data_demand])
        data.columns = col
        
        return data.style.set_caption(f'Table {num}: Accuracy Approximation Demand and Supply using Modern ML-Methods')
    

# -------------------------------------------------------------------------------- 5.2 Cake Problem











