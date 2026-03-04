import yfinance as yf
import pandas as pd
import numpy as np
import time

from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import argrelextrema

symbol = "PYTH-USD"


def get_data():

    df = yf.download(
        symbol,
        interval="4h",
        period="120d"
    )

    df.dropna(inplace=True)

    return df


def add_indicators(df):

    df["rsi"] = RSIIndicator(df["Close"]).rsi()

    macd = MACD(df["Close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()

    df["ma20"] = df["Close"].rolling(20).mean()
    df["ma50"] = df["Close"].rolling(50).mean()

    return df


def support_resistance(df):

    order = 5

    local_min = argrelextrema(df["Low"].values, np.less_equal, order=order)[0]
    local_max = argrelextrema(df["High"].values, np.greater_equal, order=order)[0]

    supports = df.iloc[local_min]["Low"]
    resistances = df.iloc[local_max]["High"]

    return supports.tail(3), resistances.tail(3)


def build_dataset(df):

    df["future"] = df["Close"].shift(-3)
    df["target"] = (df["future"] > df["Close"]).astype(int)

    features = df[[
        "Close",
        "rsi",
        "macd",
        "macd_signal",
        "ma20",
        "ma50"
    ]]

    df = df.dropna()

    X = features.loc[df.index]
    y = df["target"]

    return X, y


def predict_reversal(df):

    X, y = build_dataset(df)

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X[:-1], y[:-1])

    proba = model.predict_proba([X.iloc[-1]])[0]

    return proba


def analyze():

    df = get_data()

    df = add_indicators(df)

    supports, resistances = support_resistance(df)

    prob = predict_reversal(df)

    last_price = df["Close"].iloc[-1]

    print("\n------ MARKET ANALYSIS ------")
    print("Preço atual:", last_price)

    print("\nSuportes:")
    print(supports.values)

    print("\nResistências:")
    print(resistances.values)

    print("\nProbabilidade próxima candle:")

    print("Alta:", round(prob[1]*100,2), "%")
    print("Baixa:", round(prob[0]*100,2), "%")

    result = f"""
    ------ MARKET ANALYSIS ------
    Preço: {last_price}

    Suportes: {supports.values}

    Resistências: {resistances.values}

    Alta: {round(prob[1]*100,2)} %
    Baixa: {round(prob[0]*100,2)} %
    """

    print(result)

    with open("signals.log", "a") as f:
        f.write(result + "\n")


def main():

    while True:

        try:
            analyze()

        except Exception as e:
            print("Erro:", e)

        print("\nPróxima análise em 4h")
        time.sleep(14400)


if __name__ == "__main__":
    main()
