import pandas as pd
import datetime
import numpy as np
from pandas_datareader import data as web
from matplotlib import pyplot as pp

zscore = lambda x:(x -x.mean())/x.std() # zscore: normalization of log returns

start = datetime.datetime(2012,1,1)
end = datetime.datetime(2016,1,1)

df = web.DataReader("GOOGL", "google", start, end)

def _load_data(data, n_prev):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        zscore(data.iloc[i:i+n_prev+1])
        docX.append(zscore(data.iloc[i:i+n_prev]).as_matrix())
        docY.append(zscore(data.iloc[i:i+n_prev+1]).Close[30])
    alsX = np.array(docX)
    alsY = np.array(docY)
    
    return alsX, alsY


def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = int(round(len(df) * (1 - test_size)))

    X_train, y_train = _load_data(df.iloc[0:ntrn],30)
    X_test, y_test = _load_data(df.iloc[ntrn:],30)

    return (X_train, y_train), (X_test, y_test)

(X_train, y_train), (X_test, y_test) = train_test_split(df)

from keras.models import Sequential  
from keras.layers.core import Dense, Activation, Dropout 
from keras.layers.recurrent import LSTM


hidden_neurons = 300

model = Sequential()
model.add(LSTM(300, input_shape=(30, 5), return_sequences=False))       
model.add(Dropout(0.2))
model.add(Dense(1))
model.add(Activation("tanh"))

model.compile(loss="mean_squared_error", optimizer="Adam")  

model.summary()

model.fit(X_train, y_train, batch_size=100, epochs=10, validation_split=0.05)

predicted = model.predict(X_test) 

outcome = pd.DataFrame()
outcome['actual'] = y_test
outcome['predicted'] = predicted
outcome.to_csv("outcome.csv")  