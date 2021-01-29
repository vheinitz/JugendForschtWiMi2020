import yfinance as yf
import os.path
import requests
import urllib3
import settings
from settings import BASE_PATH
from settings import TICKERS


def update_stocks( save_path, tickers ):
    for t in tickers:
        if not os.path.isfile("%s/%s.csv" % (save_path, t)):
            try:
                hist = yf.download(t,
                        start='2019-01-01',
                        end='2021-1-29',
                        progress=True,
                        interval="1d")

                hist.to_csv("%s/%s.csv" % (save_path,t))
                print("Downloaded %s" % (t))
            except:
            #except requests.exceptions.ConnectionError as ex:
                print("Failed to download %s. Exception: %s" % (t,ex) )



if __name__ == '__main__':

    update_stocks(BASE_PATH, TICKERS)
