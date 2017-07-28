TICKER = "PG"
START_YEAR = 1990
END_YEAR = 2016
WINDOW = 30
EMB_SIZE = 5
STEP = 1
FORECAST = 1
SAVE_NAME = "classification_model.hdf5"
ENABLE_CSV_OUTPUT = 1
NAME_CSV_ACCURACY = "classification_acc.csv"
NAME_CSV_LOSS = "classification_loss.csv"

print 'Initializing libraries..'
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

start = datetime.datetime(START_YEAR,1,1)
end = datetime.datetime(END_YEAR,1,1)
print 'reading data of', TICKER ,' from Google Finance..'
df = web.DataReader(TICKER, "google", start, end) #read data with panda_datareader

O = df.Open.tolist()
C = df.Close.tolist()
H = df.High.tolist()
L = df.Low.tolist()
V = df.Volume.tolist()

print 'preformatting data..'
X, Y = [], []
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

    except Exception as e:
        break

    X.append(x_i)
    Y.append(y_i)

X, Y = np.array(X), np.array(Y)

print 'formatting training set..'
def train_test(X, y, percentage=0.9):
    p = int(len(X) * percentage)
    X_train = X[0:p]
    Y_train = y[0:p]
     
    X_train, Y_train = shuffle_in_unison(X_train, Y_train)
 
    X_test = X[p:]
    Y_test = y[p:]

    return X_train, X_test, Y_train, Y_test

X_train, X_test, y_train, y_test = train_test(X, Y)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], EMB_SIZE)) #not really necessary
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], EMB_SIZE))

#CNN MODEL

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
print 'building model..'
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath=SAVE_NAME, verbose=1, save_best_only=True)
model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print 'the training will start in 3 2 1..'
history = model.fit(X_train, y_train, 
          nb_epoch = 100, 
          batch_size = 64, 
          verbose=1, 
          validation_data=(X_test, y_test),
          callbacks=[reduce_lr, checkpointer],
          shuffle=True)

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
	_df1.to_csv(NAME_CSV_ACCURACY)

	_df2 = pd.DataFrame()
	_df2['loss']=history.history['loss']
	_df2['val_loss']=history.history['val_loss']
	_df2.to_csv(NAME_CSV_LOSS)

pred_action = []
right = 0
wrong = 0
for x in xrange(len(pred)):
    
    if pred[x][1]>0.5:
        action = 1
    else:
        action = 0
    pred_action.append(action)
    if y_test[x][1]==pred_action[x]:
        right = right+1
    else:
        wrong = wrong+1
    
print "RIGHT: ", right,"| WRONG: ", wrong, "| RIGHT PERCENTAGE:", ((right*100)/len(pred)),"%"
