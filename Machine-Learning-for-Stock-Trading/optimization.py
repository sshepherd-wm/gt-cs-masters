""""""  		  	   		   	 			  		 			 	 	 		 		 	
"""MC1-P2: Optimize a portfolio.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		   	 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		   	 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		   	 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		   	 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		   	 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		   	 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		   	 			  		 			 	 	 		 		 	
or edited.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		   	 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		   	 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		   	 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
Student Name: Stephen Shepherd 		  	   		   	 			  		 			 	 	 		 		 	
GT User ID: sshepherd35		  	   		   	 			  		 			 	 	 		 		 	
GT ID: 903659366  		  	   		   	 			  		 			 	 	 		 		 	
"""  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import datetime as dt  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import numpy as np  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
import matplotlib.pyplot as plt  		  	   		   	 			  		 			 	 	 		 		 	
import pandas as pd  		  	   		   	 			  		 			 	 	 		 		 	
from util import get_data, plot_data

import scipy.optimize as spo

### Helper functions
def _get_sharpe(samp_freq, avg_daily_return, std_daily_return, daily_rf_rate):
    ## calculate sharpe ratio
    sr = np.sqrt(samp_freq) * (avg_daily_return - daily_rf_rate) / std_daily_return
    #sr = (avg_daily_return - daily_rf_rate) / std_daily_return
    return sr

def _min_func(x, normed_data):
    ## function for optimization to minimize
    
    ## allocations based on the optimization of the array x
    alloc = normed_data * x
    
    start_val = 1
    pos_vals = alloc * start_val
    
    port_val = pos_vals.sum(axis=1)
    
    #daily_rets = port_val / port_val[0] - 1
    #daily_rets = daily_rets[1:]
    
    daily_returns = port_val.copy()
    daily_returns[1:] = (daily_returns[1:] / daily_returns[:-1].values) - 1
    daily_returns.iloc[0] = 0
    daily_rets = daily_returns.copy()
    
    ## sharpe ratio
    avg_daily_returns = daily_rets.mean()
    std_daily_returns = daily_rets.std()
    
    sample_freq = 252 ## daily
    risk_free_daily_return = 0.0 ## per project docs
    sr = _get_sharpe(sample_freq, avg_daily_returns, std_daily_returns, risk_free_daily_return)
    
    ## negate the sharpe ratio we return here as we actually want to maximize it
    return -1 * sr	  	   		   	 			  		 			 	 	 		 		 	
###		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
# This is the function that will be tested by the autograder  		  	   		   	 			  		 			 	 	 		 		 	
# The student must update this code to properly implement the functionality  		  	   		   	 			  		 			 	 	 		 		 	
def optimize_portfolio(  		  	   		   	 			  		 			 	 	 		 		 	
    sd=dt.datetime(2008, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
    ed=dt.datetime(2009, 1, 1),  		  	   		   	 			  		 			 	 	 		 		 	
    syms=["GOOG", "AAPL", "GLD", "XOM"],  		  	   		   	 			  		 			 	 	 		 		 	
    gen_plot=False,  		  	   		   	 			  		 			 	 	 		 		 	
):  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    This function should find the optimal allocations for a given set of stocks. You should optimize for maximum Sharpe  		  	   		   	 			  		 			 	 	 		 		 	
    Ratio. The function should accept as input a list of symbols as well as start and end dates and return a list of  		  	   		   	 			  		 			 	 	 		 		 	
    floats (as a one-dimensional numpy array) that represents the allocations to each of the equities. You can take  		  	   		   	 			  		 			 	 	 		 		 	
    advantage of routines developed in the optional assess portfolio project to compute daily portfolio value and  		  	   		   	 			  		 			 	 	 		 		 	
    statistics.  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    :param sd: A datetime object that represents the start date, defaults to 1/1/2008  		  	   		   	 			  		 			 	 	 		 		 	
    :type sd: datetime  		  	   		   	 			  		 			 	 	 		 		 	
    :param ed: A datetime object that represents the end date, defaults to 1/1/2009  		  	   		   	 			  		 			 	 	 		 		 	
    :type ed: datetime  		  	   		   	 			  		 			 	 	 		 		 	
    :param syms: A list of symbols that make up the portfolio (note that your code should support any  		  	   		   	 			  		 			 	 	 		 		 	
        symbol in the data directory)  		  	   		   	 			  		 			 	 	 		 		 	
    :type syms: list  		  	   		   	 			  		 			 	 	 		 		 	
    :param gen_plot: If True, optionally create a plot named plot.png. The autograder will always call your  		  	   		   	 			  		 			 	 	 		 		 	
        code with gen_plot = False.  		  	   		   	 			  		 			 	 	 		 		 	
    :type gen_plot: bool  		  	   		   	 			  		 			 	 	 		 		 	
    :return: A tuple containing the portfolio allocations, cumulative return, average daily returns,  		  	   		   	 			  		 			 	 	 		 		 	
        standard deviation of daily returns, and Sharpe ratio  		  	   		   	 			  		 			 	 	 		 		 	
    :rtype: tuple  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    ###### 1) GET OPTIMAL ALLOCATIONS ######
    
    ## get data
    dates = pd.date_range(sd, ed)
    data = get_data(syms, dates)
    
    ## normalize on first day's value, remove S&P500 from portfolio, remove first day
    normed = data / data.iloc[0]
    normed = normed[[c for c in normed.columns if c != 'SPY']]
    
    ## run optimization to determine portfolio's stock allocations
    min_result = spo.minimize(
        fun = _min_func,
        x0 = [1 / len(syms) for n in syms],
        #x0 = [1 for n in syms],
        args = (normed,),
        bounds = [(0,1) for n in syms],
        constraints = spo.LinearConstraint([1 for n in syms], 1, 1),
        #constraints = ({ 'type': 'ineq', 'fun': lambda inputs: 1 - np.sum(inputs) }),
        method='SLSQP',
        options={'disp': True}
    )
    
    ## allocations, result from optimization
    allocs = min_result.x

    ###### 2) CALCULATE SUMMARY METRICS ######

    ## position values
    start_val = 1 ## assuming a one dollar portfolio....

    #print(allocs)
    #display(normed.head(1))

    alloc = normed * allocs
    pos_vals = alloc * start_val

    ## portfolio values
    port_val = pos_vals.sum(axis=1)

    ## daily returns
    daily_returns = port_val.copy()
    daily_returns[1:] = (daily_returns[1:] / daily_returns[:-1].values) - 1
    daily_returns.iloc[0] = 0
    daily_rets = daily_returns.copy()

    ## metrics
    cum_ret = port_val[-1] / port_val[0] - 1
    avg_daily_ret = daily_rets.mean()
    std_daily_ret = daily_rets.std()
    sr = _get_sharpe(
        samp_freq = 252, ## daily frequency
        avg_daily_return = avg_daily_ret,
        std_daily_return = std_daily_ret,
        daily_rf_rate = 0.0
    )

    ## OPTIONAL PLOT
    if gen_plot == True:
        for_plot = data[['SPY']] / data[['SPY']].iloc[0]
        for_plot['Portfolio'] = port_val
        fig = (
            for_plot
            .plot(
                figsize=[10,6], style='-',
                title="Daily Portfolio Value and SPY",
            )
        )
        fig.set(xlabel='Date', ylabel="Price")   
        fig = fig.get_figure()
        fig.savefig('plot.png')

    return allocs, cum_ret, avg_daily_ret, std_daily_ret, sr	   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
def test_code():  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
    This function WILL NOT be called by the auto grader.  		  	   		   	 			  		 			 	 	 		 		 	
    """  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    start_date = dt.datetime(2009, 1, 1)  		  	   		   	 			  		 			 	 	 		 		 	
    end_date = dt.datetime(2010, 1, 1)  		  	   		   	 			  		 			 	 	 		 		 	
    symbols = ["GOOG", "AAPL", "GLD", "XOM", "IBM", "KO"]  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Assess the portfolio  		  	   		   	 			  		 			 	 	 		 		 	
    allocations, cr, adr, sddr, sr = optimize_portfolio(  		  	   		   	 			  		 			 	 	 		 		 	
        sd=start_date, ed=end_date, syms=symbols, gen_plot=True  		  	   		   	 			  		 			 	 	 		 		 	
    )  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
    # Print statistics  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Start Date: {start_date}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"End Date: {end_date}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Symbols: {symbols}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Allocations:{allocations}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Sharpe Ratio: {sr}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Volatility (stdev of daily returns): {sddr}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Average Daily Return: {adr}")  		  	   		   	 			  		 			 	 	 		 		 	
    print(f"Cumulative Return: {cr}")  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
  		  	   		   	 			  		 			 	 	 		 		 	
if __name__ == "__main__":  		  	   		   	 			  		 			 	 	 		 		 	
    # This code WILL NOT be called by the auto grader  		  	   		   	 			  		 			 	 	 		 		 	
    # Do not assume that it will be called  		  	   		   	 			  		 			 	 	 		 		 	
    test_code()  		  	   		   	 			  		 			 	 	 		 		 	
