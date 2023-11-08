
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


class ConBacktester():
    ''' Class for the vectorized backtesting of simple contrarian trading strategies.
    '''    
    
    def __init__(self, symbol, start, end, tc):
        '''
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        tc: float
            proportional transaction/trading costs per trade
        '''
        self.symbol = symbol
        self.start = start # start date tange
        self.end = end # end date tange
        self.tc = tc # trading cost
        self.results = None
        self.get_data()
        
    def __repr__(self):
        return "ConBacktester(symbol = {}, start = {}, end = {})".format(self.symbol, self.start, self.end)

    ## Function for get the data 
    def get_data(self):
        ''' Imports the data from intraday_pairs.csv (source can be changed).
        '''
        raw = pd.read_csv("../Data/intraday_pairs.csv", parse_dates = ["time"], index_col = "time") 
        raw = raw[self.symbol].to_frame().dropna() # take the data of the symbol, then deop NaN
        raw = raw.loc[self.start:self.end].copy() # extract the data from the date ramge
        raw.rename(columns={self.symbol: "price"}, inplace=True) # change the name of the column
        raw["returns"] = np.log(raw / raw.shift(1)) # compute the log return
        self.data = raw 

    ## Function for test (run) the strategy    
    def test_strategy(self, window = 1):
        ''' Backtests the simple contrarian trading strategy.
        
        Parameters
        ----------
        window: int
            time window (number of bars) to be considered for the strategy.
        '''
        self.window = window # set the window
        data = self.data.copy().dropna()
        
        # take positions:
        # if the rolling mean return is (-), we take long position,
        # and the rolling mean return is (+), we take short position,
        data["position"] = -np.sign(data["returns"].rolling(self.window).mean())

        # then, compute the return of our strategy
        data["strategy"] = data["position"].shift(1) * data["returns"]
        data.dropna(inplace=True)
        
        # determine the number of trades in each bar
        # (how much transactions we made?)
        data["trades"] = data.position.diff().fillna(0).abs()
        
        # subtract transaction/trading costs from pre-cost return
        data.strategy = data.strategy - data.trades * self.tc
        
        # compute the dumelative return of buy and hold strategy
        data["creturns"] = data["returns"].cumsum().apply(np.exp)
        # compute the dumelative return of our strategy
        data["cstrategy"] = data["strategy"].cumsum().apply(np.exp)
        self.results = data
        
        perf = data["cstrategy"].iloc[-1] # absolute performance of the strategy
        outperf = perf - data["creturns"].iloc[-1] # out-/underperformance of strategy
        
        return round(perf, 6), round(outperf, 6)
    
    ## Function for plot the result of the strategy 
    def plot_results(self):
        ''' Plots the performance of the trading strategy and compares to "buy and hold".
        '''
        if self.results is None:
            print("Run test_strategy() first.")
        else:
            title = "{} | Window = {} | TC = {}".format(self.symbol, self.window, self.tc)
            # Plot the cumelative return of our strategy and of buy and hold.
            self.results[["creturns", "cstrategy"]].plot(title=title, figsize=(12, 8))

    ## Function for find the best hyperparameters for our strategy.         
    def optimize_parameter(self, window_range):
        ''' Finds the optimal strategy (global maximum) given the window parameter range.

        Parameters
        ----------
        window_range: tuple
            tuples of the form (start, end, step size)
        '''
        
        windows = range(*window_range)
            
        results = []
        for window in windows:
            results.append(self.test_strategy(window)[0])
        
        best_perf = np.max(results) # best performance
        opt = windows[np.argmax(results)] # optimal parameter
        
        # run/set the optimal strategy
        self.test_strategy(opt)
        
        # create a df with many results
        many_results =  pd.DataFrame(data = {"window": windows, "performance": results})
        self.results_overview = many_results
        
        return opt, best_perf
                               
        