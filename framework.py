""" 
RNN-stocks-prediction | Alessandro Solbiati

https://github.com/SolbiatiAlessandro/RNN-stocks-prediction

This is the main framework for the prediction of financial assets' behaviour using binary classification [p1,p2]

The python script does 

	- download financial data according to TICKER_LIST and train the NN model with these data
	- save the model under SAVE_NAME in hdf5 format
	- save the accuracy and loss csv under NAME_CSV
	- output a basic performance evaluation on the console

Currently the framework just implements Convolutional Neural Network model 

NOTE: the performance of the CNN is largely influenced by the hyperparameters below

"""

START_YEAR = 1990
END_YEAR = 2016
WINDOW = 30
EMB_SIZE = 5
STEP = 1
FORECAST = 5
NUMBER_EPOCHS = 100
SAVE_NAME = "classification_model.hdf5"
ENABLE_CSV_OUTPUT = 1
NAME_CSV = "classification.csv"
TRAIN_TEST_PERCENTAGE = 0.9
TICKER_LIST = ['ABT', 'ABBV', 'ACN', 'ACE', 'ADBE', 'ADT', 'AAP', 'AES', 'AET', 'AFL', 'AMG', 'A', 'GAS', 'APD', 'ARG', 'AKAM', 'AA', 'AGN', 'ALXN', 'ALLE', 'ADS', 'ALL', 'ALTR', 'MO', 'AMZN', 'AEE', 'AAL', 'AEP', 'AXP', 'AIG', 'AMT', 'AMP', 'ABC', 'AME', 'AMGN', 'APH', 'APC', 'ADI', 'AON', 'APA', 'AIV', 'AMAT', 'ADM', 'AIZ', 'T', 'ADSK', 'ADP', 'AN', 'AZO', 'AVGO', 'AVB', 'AVY', 'BHI', 'BLL', 'BAC', 'BK', 'BCR', 'BXLT', 'BAX', 'BBT', 'BDX', 'BBBY', 'BRK-B', 'BBY', 'BLX', 'HRB', 'BA', 'BWA', 'BXP', 'BSK', 'BMY', 'BRCM', 'BF-B', 'CHRW', 'CA', 'CVC', 'COG', 'CAM', 'CPB', 'COF', 'CAH', 'HSIC', 'KMX', 'CCL', 'CAT', 'CBG', 'CBS', 'CELG', 'CNP', 'CTL', 'CERN', 'CF', 'SCHW', 'CHK', 'CVX', 'CMG', 'CB', 'CI', 'XEC', 'CINF', 'CTAS', 'CSCO', 'C', 'CTXS', 'CLX', 'CME', 'CMS', 'COH', 'KO', 'CCE', 'CTSH', 'CL', 'CMCSA', 'CMA', 'CSC', 'CAG', 'COP', 'CNX', 'ED', 'STZ', 'GLW', 'COST', 'CCI', 'CSX', 'CMI', 'CVS', 'DHI', 'DHR', 'DRI', 'DVA', 'DE', 'DLPH', 'DAL', 'XRAY', 'DVN', 'DO', 'DTV', 'DFS', 'DISCA', 'DISCK', 'DG', 'DLTR', 'D', 'DOV', 'DOW', 'DPS', 'DTE', 'DD', 'DUK', 'DNB', 'ETFC', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'EMC', 'EMR', 'ENDP', 'ESV', 'ETR', 'EOG', 'EQT', 'EFX', 'EQIX', 'EQR', 'ESS', 'EL', 'ES', 'EXC', 'EXPE', 'EXPD', 'ESRX', 'XOM', 'FFIV', 'FB', 'FAST', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FSIV', 'FLIR', 'FLS', 'FLR', 'FMC', 'FTI', 'F', 'FOSL', 'BEN', 'FCX', 'FTR', 'GME', 'GPS', 'GRMN', 'GD', 'GE', 'GGP', 'GIS', 'GM', 'GPC', 'GNW', 'GILD', 'GS', 'GT', 'GOOGL', 'GOOG', 'GWW', 'HAL', 'HBI', 'HOG', 'HAR', 'HRS', 'HIG', 'HAS', 'HCA', 'HCP', 'HCN', 'HP', 'HES', 'HPQ', 'HD', 'HON', 'HRL', 'HSP', 'HST', 'HCBK', 'HUM', 'HBAN', 'ITW', 'IR', 'INTC', 'ICE', 'IBM', 'IP', 'IPG', 'IFF', 'INTU', 'ISRG', 'IVZ', 'IRM', 'JEC', 'JBHT', 'JNJ', 'JCI', 'JOY', 'JPM', 'JNPR', 'KSU', 'K', 'KEY', 'GMCR', 'KMB', 'KIM'. 'KMI', 'KLAC', 'KSS', 'KRFT', 'KR', 'LB', 'LLL', 'LH', 'LRCX', 'LM', 'LEG', 'LEN', 'LVLT', 'LUK', 'LLY', 'LNC', 'LLTC', 'LMT', 'L', 'LOW', 'LYB', 'MTB', 'MAC', 'M', 'MNK', 'MRO', 'MPC', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MAT', 'MKC', 'MCD', 'MHFI', 'MCK', 'MJN', 'MMV', 'MDT', 'MRK', 'MET', 'KORS', 'MCHP', 'MU', 'MSFT', 'MHK', 'TAP', 'MDLZ', 'MON', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MUR', 'MYL', 'NDAQ', 'NOV', 'NAVI', 'NTAP', 'NFLX', 'NWL', 'NFX', 'NEM', 'NWSA', 'NEE', 'NLSN', 'NKE', 'NI', 'NE', 'NBL', 'JWN', 'NSC', 'NTRS', 'NOC', 'NRG', 'NUE', 'NVDA', 'ORLY', 'OXY', 'OMC', 'OKE', 'ORCL', 'OI', 'PCAR', 'PLL', 'PH', 'PDCO', 'PAYX', 'PNR', 'PBCT', 'POM', 'PEP', 'PKI', 'PRGO', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PXD', 'PBI', 'PCL', 'PNC', 'RL', 'PPG', 'PPL', 'PX', 'PCP', 'PCLN', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PSA', 'PHM', 'PVH', 'QRVO', 'PWR', 'QCOM', 'DGX', 'RRC', 'RTN', 'O', 'RHT', 'REGN', 'RF', 'RSG', 'RAI', 'RHI', 'ROK', 'COL', 'ROP', 'ROST', 'RLC', 'R', 'CRM', 'SNDK', 'SCG', 'SLB', 'SNI', 'STX', 'SEE', 'SRE', 'SHW', 'SIAL', 'SPG', 'SWKS', 'SLG', 'SJM', 'SNA', 'SO', 'LUV', 'SWN', 'SE', 'STJ', 'SWK', 'SPLS', 'SBUX', 'HOT', 'STT', 'SRCL', 'SYK', 'STI', 'SYMC', 'SYY', 'TROW', 'TGT', 'TEL', 'TE', 'TGNA', 'THC', 'TDC', 'TSO', 'TXN', 'TXT', 'HSY', 'TRV', 'TMO', 'TIF', 'TWX', 'TWC', 'TJK', 'TMK', 'TSS', 'TSCO', 'RIG', 'TRIP', 'FOXA', 'TSN', 'TYC', 'UA', 'UNP', 'UNH', 'UPS', 'URI', 'UTX', 'UHS', 'UNM', 'URBN', 'VFC', 'VLO', 'VAR', 'VTR', 'VRSN', 'VZ', 'VRTX', 'VIAB', 'V', 'VNO', 'VMC', 'WMT', 'WBA', 'DIS', 'WM', 'WAT', 'ANTM', 'WFC', 'WDC', 'WU', 'WY', 'WHR', 'WFM', 'WMB', 'WEC', 'WYN', 'WYNN', 'XEL', 'XRX', 'XLNX', 'XL', 'XYL', 'YHOO', 'YUM', 'ZBH', 'ZION', 'ZTS']

print 'starting framework..'
import numpy as np
from numpy import log
import pandas as pd
import matplotlib.pylab as plt
from pandas_datareader import data as web
import datetime
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Convolution1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *

start = datetime.datetime(START_YEAR,1,1)
end = datetime.datetime(END_YEAR,1,1)

_ddf = []
for ticker in TICKER_LIST:
    print 'downloading ',ticker,' data..'
    ddf = web.DataReader(ticker, "google", start, end) #read data with panda_datareader
    _ddf.append(ddf)
df = pd.concat(_ddf)
print 'download finished: dataframe size [', len(df), ']'

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    # used in train_test split fun
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def remove_nan_examples(data):
    #some basic util functions
    newX = []
    for i in range(len(data)):
        if np.isnan(data[i]).any() == False:
            newX.append(data[i])
    return newX

def window_sizing(O,H,L,C,V):
    for i in range(0, len(O), STEP): 
        try:
            o = O[i:i+WINDOW]
            h = H[i:i+WINDOW]
            l = L[i:i+WINDOW]
            c = C[i:i+WINDOW]
            v = V[i:i+WINDOW]
            #zscore on time window interval
            o = (np.array(o) - np.mean(o)) / np.std(o)
            h = (np.array(h) - np.mean(h)) / np.std(h)
            l = (np.array(l) - np.mean(l)) / np.std(l)
            c = (np.array(c) - np.mean(c)) / np.std(c)
            v = (np.array(v) - np.mean(v)) / np.std(v)

            x_i = C[i:i+WINDOW]
            y_i = C[i+WINDOW+FORECAST]  

            last_close = x_i[-1]
            next_close = y_i
    
            if last_close < next_close:
                y_i = [1, 0]
            else:
                y_i = [0, 1] 

            x_i = np.column_stack((o, h, l, c, v))
            
            if i%5000==0:
                print i, 'windows sized'
                
        except Exception as e:
            break

        X.append(x_i)
        Y.append(y_i)

def format_data(df):
    O = df.Open.tolist()
    C = df.Close.tolist()
    H = df.High.tolist()
    L = df.Low.tolist()
    V = df.Volume.tolist()

    window_sizing(O,H,L,C,V)
    
def train_test(X, y, percentage=TRAIN_TEST_PERCENTAGE):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
     
    X_train, Y_train = shuffle_in_unison(X_train, Y_train)
 
    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

#DATA PREPARATION

X , Y = [], []
format_data(df)
X , Y = np.array(X) , np.array(Y)

X_train, X_test, y_train, y_test = train_test(X, Y)

X_train, X_test = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE)), np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))

#MODEL DEFINITION
print 'initializing model..'
model = Sequential()
model.add(Convolution1D(input_shape = (WINDOW, EMB_SIZE),
                        nb_filter=16,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Convolution1D(nb_filter=8,
                        filter_length=4,
                        border_mode='same'))
model.add(BatchNormalization())
model.add(LeakyReLU())
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(64))
model.add(BatchNormalization())
model.add(LeakyReLU())


model.add(Dense(2))
model.add(Activation('softmax'))

opt = Nadam(lr=0.002)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath=SAVE_NAME, verbose=1, save_best_only=True)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])


#MODEL TRAINING
print 'start training..'
history = model.fit(X_train, y_train, 
          nb_epoch = NUMBER_EPOCHS, 
          batch_size = 128, 
          verbose=1, 
          validation_data=(X_test, y_test),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

#MODEL PREDICTIONS
print 'performance computation..'
model.load_weights(SAVE_NAME)
pred = model.predict(np.array(X_test))

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
C = confusion_matrix([np.argmax(y) for y in y_test], [np.argmax(y) for y in pred])

print C / C.astype(np.float).sum(axis=1)

if ENABLE_CSV_OUTPUT:
	_df1 = pd.DataFrame()
	_df1['acc']=history.history['acc']
	_df1['val_acc']=history.history['val_acc']
	_df1['loss']=history.history['loss']
	_df1['val_loss']=history.history['val_loss']
	_df1.to_csv(NAME_CSV)

right, wrong = 0, 0

for x in xrange(len(pred)):
    if pred[x][1]>0.5:
        action = 1
    else:
        action = 0

    if y_test[x][1]==action:
        right = right+1
    else:
        wrong = wrong+1
    
print "RIGHT: ", right,"| WRONG: ", wrong, "| RIGHT PERCENTAGE:", ((right*100)/len(pred)),"%"

