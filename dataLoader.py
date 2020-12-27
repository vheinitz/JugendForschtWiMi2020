import pandas as pd

from settings import BASE_PATH
from settings import TICKERS

def load_stocks( load_path, tickers ):
    stock_map = {}
    for t in tickers:

        stockdata = pd.read_csv("%s/%s.csv" % (load_path,t))
        stock_map[t] = stockdata
        #print(stockdata)

    return stock_map

if __name__ == '__main__':

    stocks = load_stocks(BASE_PATH, TICKERS)
    print( stocks )
