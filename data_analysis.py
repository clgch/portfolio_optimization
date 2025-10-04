import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns 
from skfolio.preprocessing import prices_to_returns

tickers = ["CW8.PA", "PSP5.PA", "MFEC.PA", "ETSZ.DE"] 

ohlc = yf.download(tickers, period="max")
prices = ohlc["Close"].dropna(how="any")
print(prices.tail())

prices_cleaned = prices[prices.index >= "2018-01-01"].interpolate()

prices_cleaned.plot(figsize=(15,10))
plt.show()

X = prices_to_returns(prices_cleaned)

X_train = X[X.index <= "2023-09-20"].dropna(how="any")
X_test = X[X.index >= "2023-09-20"].dropna(how="any")

sns.pairplot(X_train)
plt.show()

corr = X_train.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Matrix Heatmap')
plt.show()
