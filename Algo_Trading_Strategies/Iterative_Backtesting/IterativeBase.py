
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")


class IterativeBase():
    '''
      Base class for iterative (event-driven) backtesting of trading strategies.
     Its sutble for any backtesting. 
      
      This class create the general trading mechanizem:
        - Take a position
        - close position
        - Update amount
        - import the data
        - prepare the data
    
    '''

    def __init__(self, symbol, start, end, amount, use_spread = True):
        '''            
        Parameters
        ----------
        symbol: str
            ticker symbol (instrument) to be backtested
        start: str
            start date for data import
        end: str
            end date for data import
        amount: float
            initial amount to be invested per trade
        use_spread: boolean (default = True) 
            whether trading costs (bid-ask spread) are included
        '''
        self.symbol = symbol
        self.start = start
        self.end = end
        self.initial_balance = amount
        self.current_balance = amount
        self.units = 0
        self.trades = 0
        self.position = 0
        self.use_spread = use_spread
        self.get_data()
    
    ## Function for get the data 
    def get_data(self):
        ''' Imports the data from detailed.csv (source can be changed).
        '''
        # read the file
        raw = pd.read_csv("../Data/detailed.csv", parse_dates = ["time"], index_col = "time").dropna()
        # Grab the dates range
        raw = raw.loc[self.start:self.end].copy()
        #  compute the log returns
        raw["returns"] = np.log(raw.price / raw.price.shift(1))
        # store the data. 
        self.data = raw
    
    ## Function for plot the data
    def plot_data(self, cols = None):  
        ''' 
        Plots the closing price for the symbol.

        cols: which coloumn to plot ? 
        '''

        if cols is None:
            cols = "price"
        # Plot the price column 
        self.data[cols].plot(figsize = (12, 8), title = self.symbol)
    
    ## Function for rechive information for a given bar. 
    def get_values(self, bar):
        ''' Returns the date, the price and the spread for the given bar.
            bar -> represent a single row in the data. (e.g day.)
        '''
        
        # extract the data of the specific bar
        date = str(self.data.index[bar].date()) 
        # extract the price
        price = round(self.data.price.iloc[bar], 5)
        # extract the spread
        spread = round(self.data.spread.iloc[bar], 5)
        return date, price, spread
    
    ## Function for print the current cash balance.
    def print_current_balance(self, bar):
        ''' Prints out the current (cash) balance.
        '''
        # extract values of a given bar
        date, price, spread = self.get_values(bar)
        # print the balance.
        print("{} | Current Balance: {}".format(date, round(self.current_balance, 2)))
        
    ## Function for buy an instrument.    
    def buy_instrument(self, bar, units = None, amount = None):
        ''' 
        Places and executes a buy order (market order).
        
        :bar: -> the point that you intrested to buy (contain the price, date, ask, spread, and so on)
        :units: -> how many units you want to buy ?  (option 1)
        :amount: -> you can specify the amount of investment for calculate how many units. (option 2)    
        '''
        # extract values for a given bar
        date, price, spread = self.get_values(bar)

        if self.use_spread:
            # ask price
            price += spread / 2  # include costs
        
        # use units if units are passed, otherwise calculate units
        if amount is not None:  
            units = int(amount / price)
       
        # re-compute the current balance (becase we buy stock new stock now!)
        self.current_balance -= units * price # reduce cash balance by "purchase price"
        # update the units (we bugth new units!)
        self.units += units
        # update the trade (we take a position now!)
        self.trades += 1

        # Print a nicly massaage.  
        print("{} |  Buying {} for {}".format(date, units, round(price, 5)))
    
    ## Function for selling an instument
    def sell_instrument(self, bar, units = None, amount = None):
        '''
          Places and executes a sell order (market order).

          :bar: -> the current information on the instrument that we want to buy.
          :units: -> how mant units you want to buy? (option 1) 
          :amount: -> alternativetly, you can defind an amount and wi'll calculate for you hoe many units. (option 2)
        '''
        # get the current information about the instrument
        date, price, spread = self.get_values(bar)

        if self.use_spread:
            # bid price
            price -= spread/2 
        
        # use units if units are passed, otherwise calculate units
        if amount is not None: 
            units = int(amount / price)
        
        # update the current balance
        self.current_balance += units * price # increases cash balance by "purchase price"
        # update the units
        self.units -= units
        # update the trades 
        self.trades += 1
        print("{} |  Selling {} for {}".format(date, units, round(price, 5)))
    

    ## Function for return information about the current position 
    def print_current_position_value(self, bar):
        ''' Prints out the current position value.
        '''

        # Grab the most recent bar (the most resent instrument price).
        date, price, spread = self.get_values(bar)
        # multiply that by unuts
        cpv = self.units * price
        
        # inform about the current position value.
        print("{} |  Current Position Value = {}".format(date, round(cpv, 2)))
    

    ## Function for print the current net asset value
    def print_current_nav(self, bar):
        ''' Prints out the current net asset value (nav).
        '''
        # extract information from the bar
        date, price, spread = self.get_values(bar)
        # compute the nab
        nav = self.current_balance + self.units * price
        
        print("{} |  Net Asset Value = {}".format(date, round(nav, 2)))

    ## Function for close a position
    def close_pos(self, bar):
        ''' Closes out a long or short position (go neutral).
        '''
        # extract information from a bar
        date, price, spread = self.get_values(bar)
        print(75 * "-")
        print("{} | +++ CLOSING FINAL POSITION +++".format(date))

        # closing final position (works with short and long!)
        self.current_balance += self.units * price 
        # substract half-spread costs
        self.current_balance -= (abs(self.units) * spread/2 * self.use_spread) 
        
        print("{} | closing position of {} for {}".format(date, self.units, price))
        # setting position to neutral
        self.units = 0 
        # update the trades
        self.trades += 1
        
        # compute the performance 
        # How mant % we gain/loss? 10% profit? 
        perf = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        self.print_current_balance(bar)
       
        print("{} | net performance (%) = {}".format(date, round(perf, 2) ))
        print("{} | number of trades executed = {}".format(date, self.trades))
        print(75 * "-")