import numpy as np

import dataLoader
from settings import BASE_PATH
from settings import TICKERS
from settings import MODEL_ROW_TICKS

def get_diff_response_of(hist, futureTicks=0):
    response_data = []

    pdColumn = hist["Close"]
    np_data = pdColumn.to_numpy()

    for d in range(MODEL_ROW_TICKS,len(np_data)-futureTicks):  #Fuer jede Zeile (Tick - Tag, Stunde, etc)
        nextValue = np_data[d+futureTicks]
        prevValue = np_data[d+futureTicks-1]
        resultValue = prevValue - nextValue
        response_data.append(resultValue)

    y = np.array(response_data)
    return y


def get_data_as_np(hist):
    model_data = []
    np_data = np.c_[
        hist["Open"].to_numpy(),
        hist["High"].to_numpy(),
        hist["Low"].to_numpy(),
        hist["Close"].to_numpy(),
        #hist["Volume"].to_numpy(),
    ]

    for d in range(0,len(np_data)-(MODEL_ROW_TICKS)):  #Fuer jede Zeile (Tick - Tag, Stunde, etc)
        model_row = []
        for row in np_data[d:d+MODEL_ROW_TICKS]:     #Ab der aktueller Zeile N-Ticks fuer ML-Data Zeile
            model_row.extend(row)

        if len(model_data) == 0:    #Wenn Modell leer ->
            model_data = model_row  # erste Zeile im Modell
        else:
            model_data = np.vstack([model_data, model_row]) # eine neue Zeile hinzufuegen

    return model_data


def get_many(stocks):

    model_data = []
    for k in stocks:
        histdata = get_data_as_np(stocks[k])
        #print (k)
        if len(model_data) == 0:    #Wenn Modell leer ->
            model_data = histdata  # model_data = historical ticker Data
        else:
            model_data = np.hstack([model_data, histdata]) # Daten von neuen Ticker rechts hinzufügen

    return model_data


def diff_response_to_class(diffs):
    classes = []
    for v in diffs:
        if ( v <= 0 ):
            classes.append(1)
        else:
            classes.append(2)

    return classes


if __name__ == '__main__':

    stocks = dataLoader.load_stocks(BASE_PATH, TICKERS)

    ticker = stocks["CSCO"]
    y = get_diff_response_of(ticker)
    x = get_data_as_np(ticker)
    x = get_many(stocks)
    print(x,x.shape)
    pass