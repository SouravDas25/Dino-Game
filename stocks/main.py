import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# %matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense

from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("archive/ADANIPORTS.csv")
df.head()

df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']
plt.figure(figsize=(16, 8))
plt.plot(df["Close"], label='Close Price history')
