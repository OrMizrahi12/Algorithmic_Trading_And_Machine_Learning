## run the trader with a script ##


### Import libraries
from ib_insync import * 
import pandas as pd
import numpy as np
import datetime as dt
#from IPython.display import display, clear_output
import os # new
#util.startLoop()


### Connection to IBKR
ib = IB()
ib.connect()

### strategy parameters
sma_s = 2
sma_l = 5
freq = "1 min"
units = 1000
end_time = dt.time(16, 30, 0) # stop condition 
contract = Forex('EURUSD') 
ib.qualifyContracts(contract)
cfd = CFD("EUR", currency = "USD")
ib.qualifyContracts(cfd)
conID = cfd.conId



### Callback function for updata the data and run the strategy on the driven data 
def onBarUpdate(bars, hasNewBar):  
    global df, last_bar
    
    if bars[-1].date > last_bar: 
        last_bar = bars[-1].date
    
        # Data Processing
        df = pd.DataFrame(bars)[["date", "open", "high", "low", "close"]].iloc[:-1] 
        df.set_index("date", inplace = True)
        
        ####################### Trading Strategy ###########################
        df = df[["close"]].copy()
        df["sma_s"] = df.close.rolling(sma_s).mean()
        df["sma_l"] = df.close.rolling(sma_l).mean()
        df.dropna(inplace = True)
        df["position"] = np.where(df["sma_s"] > df["sma_l"], 1, -1 )
        ####################################################################
        
        # Trading
        target = df["position"][-1] * units
        execute_trade(target = target)
        
        # Display
        os.system('cls')
        print(df)
    else:
        try:
            trade_reporting()
        except:
            pass

### Function for execute a trade
def execute_trade(target):
    global current_pos
    
    # 1. get current Position
    try:
        current_pos = [pos.position for pos in ib.positions() if pos.contract.conId == conID][0]
    except:
        current_pos = 0
         
    # 2. identify required trades
    trades = target - current_pos
        
    # 3. trade execution
    if trades > 0:
        side = "BUY"
        order = MarketOrder(side, abs(trades))
        trade = ib.placeOrder(cfd, order)  
    elif trades < 0:
        side = "SELL"
        order = MarketOrder(side, abs(trades))
        trade = ib.placeOrder(cfd, order)
    else:
        pass


### Function for give information on the trades
def trade_reporting():
    global report
    
    fill_df = util.df([fs.execution for fs in ib.fills()])[["execId", "time", "side", "cumQty", "avgPrice"]].set_index("execId")
    profit_df = util.df([fs.commissionReport for fs in ib.fills()])[["execId", "realizedPNL"]].set_index("execId")
    report = pd.concat([fill_df, profit_df], axis = 1).set_index("time").loc[session_start:]
    report = report.groupby("time").agg({"side":"first", "cumQty":"max", "avgPrice":"mean", "realizedPNL":"sum"})
    report["cumPNL"] = report.realizedPNL.cumsum()
        
    os.system('cls')
    print(df, report)




### if you run trader.py as python script (run the file from the terminal)
if __name__ == "__main__": 
   
   
    ## start trading session
    session_start = pd.to_datetime(dt.datetime.utcnow()).tz_localize("utc")
    bars = ib.reqHistoricalData(
            contract, # the instrument
            endDateTime='',
            durationStr='1 D', # time span
            barSizeSetting=freq, # the frequency (of driven data)
            whatToShow='MIDPOINT',
            useRTH=True,
            formatDate=2,
            keepUpToDate=True)
    

    last_bar = bars[-1].date # store the last bar
    bars.updateEvent += onBarUpdate # subscribe the function to event


    ## stop trading session
    while True:
        ib.sleep(5) # check every 5 seconds
        if dt.datetime.utcnow().time() >= end_time: # if stop conditions has been met
            execute_trade(target = 0) # close open position 
            ib.cancelHistoricalData(bars) # stop stream
            ib.sleep(10)
            try:
                trade_reporting() # final reporting
            except:
                pass
            print("Session Stopped.")
            ib.disconnect()
            break
        else:
            pass