def get_diff_response_of(hist, futureTicks=0):
    response_data = []

    pdColumn = hist["Close"]
    np_data = pdColumn.to_numpy()

    for d in range(MODEL_ROW_TICKS,len(np_data)-futureTicks):  #Fuer jede Zeile (Tick - Tag, Stunde, etc)
        nextValue = np_data[d+futureTicks]
        prevValue = np_data[d+futureTicks-1]
        resultValue = nextValue - prevValue
        response_data.append(resultValue)

    y = np.array(response_data)
    return y