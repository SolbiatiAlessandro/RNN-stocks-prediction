import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
import datetime
import locale
locale.setlocale(locale.LC_NUMERIC, "")


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

ASK_FNAME = "rtf/eurusd002-ask.rtf"
BID_FNAME = "rtf/eurusd002-BID.rtf"
WINDOW=90
FORECAST=45
EMB_SIZE=10
STEP=1  #best is 75
TRAIN_TEST_PERCENTAGE=0.9
SAVE_NAME = "classification_model.hdf5"
LOAD_NAME = "hdf5/alg08-alfa0.495.hdf5"
ENABLE_CSV_OUTPUT = 1
NAME_CSV = "classification"
TRAINING = 1
TESTING = 0
NUMBER_EPOCHS = 10
TRADING_DAYS = 14
def ternary_tensor(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC,bV):
    count=0
    X,Y = [],[]
    i=0
    try:
        count=count+1
        try:
            #ask open, ask high.. bid close, bid volume
            ao = aO[i:i+WINDOW]
            ah = aH[i:i+WINDOW]
            al = aL[i:i+WINDOW]
            ac = aC[i:i+WINDOW]
            av = aV[i:i+WINDOW]
            #zscore on time window interval
            ao = (np.array(ao) - np.mean(ao)) / np.std(ao)
            ah = (np.array(ah) - np.mean(ah)) / np.std(ah)
            al = (np.array(al) - np.mean(al)) / np.std(al)
            ac = (np.array(ac) - np.mean(ac)) / np.std(ac)

            
            bo = bO[i:i+WINDOW]
            bh = bH[i:i+WINDOW]
            bl = bL[i:i+WINDOW]
            bc = bC[i:i+WINDOW]
            bv = bV[i:i+WINDOW]
            #zscore on time window interval
            bo = (np.array(bo) - np.mean(bo)) / np.std(bo)
            bh = (np.array(bh) - np.mean(bh)) / np.std(bh)
            bl = (np.array(bl) - np.mean(bl)) / np.std(bl)
            bc = (np.array(bc) - np.mean(bc)) / np.std(bc)        

            x_i = np.column_stack((ao,ah,al,ac,av,bo,bh,bl,bc,bv))
    
                
        except Exception as e:
            print e
            pass

    except Exception as e:
        print e
        pass
    return x_i
    def format_data(df,ternary=1,binary=0):
    Time = df.Datetime
    aO = df.Open_x.tolist()
    aH = df.High_x.tolist()
    aL = df.Low_x.tolist()
    aC = df.Close_x.tolist()
    aV = df.Volume_x.tolist()
    bO = df.Open_y.tolist()
    bH = df.High_y.tolist()
    bL = df.Low_y.tolist()
    bC = df.Close_y.tolist()
    bV = df.Volume_y.tolist()
    
    #print(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC)
    
    if(ternary==1):
        return ternary_tensor(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC,bV)

    elif(binary==1):
        return binary_tensor(Time,aO,aH,aL,aC,aV,bO,bH,bL,bC,bV)


def spread(Y):
    still = 0
    for vec in Y:
        if vec[2]==1:
            still=still+1
    spread =still*100/len(Y)
    print spread,"%"
    return spread

def shuffle_in_unison(a, b):
    # courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
    # shuffling of training data
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


model.add(Dense(3))
model.add(Activation('softmax'))

opt = Nadam(lr=0.002)
reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9, patience=30, min_lr=0.000001, verbose=1)
checkpointer = ModelCheckpoint(filepath=SAVE_NAME, verbose=1, save_best_only=True)

model.compile(optimizer=opt, 
              loss='categorical_crossentropy',
              metrics=['accuracy'])
              import httplib
import json
from Tkinter import *
import datetime
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.select import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pymouse import PyMouse
CHROME_PATH= '/Users/alex/Desktop/Coding/WWW/chromedriver'
URL = "https://demo-login.dukascopy.com/binary/?_ga=2.86701976.1253522851.1502011850-1942799576.1501349679"
USR = "Leipert75EU"
PSW = "913724d5"

class DukascopyBinary(object):
    """docstring for ClassName"""
    def __init__(self, URL):
        self.URL = URL
        self.driver = webdriver.Chrome(CHROME_PATH)
        self.driver.get(URL)
        #datafeed values
        self.minute=60
        self.vec=[]
        self.data=[]
        
    def set_type(self, ASK=0, BID=1):
        global USR
        global PSW
        
        
        time.sleep(10)
        self.login(USR,PSW)
        time.sleep(15)
        self.driver.find_elements_by_css_selector(".S-T-U.a-rk-sk-tk-bb.a-rk-sk-tk-bb-cb.a-rk-sk-tk-bb-W")[0].click()
        time.sleep(0.5)
        self.driver.find_element_by_id(":1u").click()
        time.sleep(0.5)
        self.driver.find_elements_by_css_selector(".S-T-U.a-rk-sk-tk-bb.a-rk-sk-tk-bb-cb.a-rk-sk-tk-bb-W")[2].click()
        time.sleep(0.5)
        self.driver.find_element_by_id(":4f").click()
        time.sleep(0.5)
        
        if(ASK):
            self.type = "ASK"
            self.driver.find_element_by_id(":2e").click()
            time.sleep(0.5)
            self.driver.find_element_by_id(":0").click()
            
        elif(BID):
            self.type = "BID"
        
        for i in xrange(100):
            self.driver.find_element_by_css_selector(".S-T-U.a-pb-ek-O").click()
            
    def set_mouse(self,m):
        print 'put mouse in',self.type,' position.. (5s)'
        time.sleep(5)
        self.x,self.y=m.position()
        print '-> position captured'
        
    
    def login(self,USR,PSW):
        self.driver.find_element_by_id("textfield-1020-inputEl").send_keys(USR)
        self.driver.find_element_by_id("textfield-1021-inputEl").send_keys(PSW)
        self.driver.find_element_by_id("button-1035-btnEl").click()
    
    def call(self):
        self.driver.find_element_by_class_name("call").click()
        time.sleep(0.2)
        self.driver.find_element_by_id("button-1014").click()
    
    def put(self):
        self.driver.find_element_by_class_name("put").click()
        time.sleep(0.2)
        self.driver.find_element_by_id("button-1014").click()
        class App(object):
    def __init__(self):
        global model
        model.load_weights(LOAD_NAME)
        
        global URL
        self.ask = DukascopyBinary(URL)
        self.ask.set_type(ASK=1,BID=0)
        self.bid = DukascopyBinary(URL)
        self.bid.set_type(ASK=0,BID=1)
        self.running = False
        self.m = PyMouse()
        
        self.data = []
        self.temp = ""
        
        self.datetime1,self.datetime2 = "",""
        
        self.aOpen,self.aHigh,self.aLow,self.aClose,self.aVolume=0,0,0,0,0
        self.bOpen,self.bHigh,self.bLow,self.bClose,self.bVolume=0,0,0,0,0
    
    def mouse_config(self):
        self.ask.set_mouse(self.m)
        self.bid.set_mouse(self.m)
        
    def start(self):
        self.running = True

    def stop(self):
        self.running = False
    
    def __tensor__(self):
        print 'checking tensor'
        if(len(self.data)>WINDOW):
            df = pd.DataFrame(self.data[-WINDOW:])
            df = df.rename(columns={ df.columns[0] : 'Datetime', df.columns[1] : 'Open_x', df.columns[2] : 'High_x', df.columns[3] : 'Low_x', df.columns[4]: 'Close_x', df.columns[5] : 'Volume_x', df.columns[6] : 'Open_y', df.columns[7] : 'High_y', df.columns[8] : 'Low_y', df.columns[9]: 'Close_y', df.columns[10] : 'Volume_y' })
            self.X = format_data(df,1,0)
            self.X = np.array(self.X)
            return 1
        else: 
            return 0
        
    def __alg08__(self,pred,alfa):
        if pred[0][0]-pred[0][2]>alfa:
            self.ask.call()
            print 'put executed'
        
        elif pred[0][1]-pred[0][2]>alfa:
            self.ask.put()
            print 'call executed'
        
        else:
            print 'no signal'
    
    def trading(self):
        if self.running:
            #print 'RUNNING'

            try:
                
                now = app.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[0].get_attribute("innerHTML")+" "+app.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[1].get_attribute("innerHTML")
        
                if now!=self.temp:
                    #print 'NEW VALUE'

                    self.ask.driver.find_element_by_css_selector(".S-T-U.a-pb-ek-O").click()
                    self.bid.driver.find_element_by_css_selector(".S-T-U.a-pb-ek-O").click()
                    self.m.move(self.ask.x,self.ask.y)
                    self.m.move(self.bid.x,self.bid.y)
            
                    try:
                        print self.aOpen,self.aHigh,self.aLow,self.aClose,self.aVolume,self.bOpen,self.bHigh,self.bLow,self.bClose,self.bVolume
                        print 'NEW VALUE APPENDED',self.aClose
                        
                        _time = datetime.datetime.strptime(app.datetime1+" "+app.datetime2,"%Y-%m-%d %H:%M:%S.%f") 
                        
                        self.data.append([_time,float(self.aOpen),float(self.aHigh),float(self.aLow),float(self.aClose),float(self.aVolume),float(self.bOpen),float(self.bHigh),float(self.bLow),float(self.bClose),float(self.bVolume)])
                        self.temp=now
                        
                        if(self.__tensor__()):
                            pred =model.predict(np.reshape(self.X, (1,self.X.shape[0],self.X.shape[1])))
                            self.text.insert(1.0, pred)
                            self.__alg08__(pred, 0.495)
                        
                        
                    except Exception,e:
                        print 'inner loop exception:',str(e)
                        pass
                    
                    
        
                #print 'START SCANNING', self.aClose
                self.datetime1 = self.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[0].get_attribute("innerHTML")
                self.datetime2= self.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[1].get_attribute("innerHTML")
                self.aOpen = self.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[2].get_attribute("innerHTML")
                self.aHigh = self.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[3].get_attribute("innerHTML")
                self.aLow = self.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[4].get_attribute("innerHTML")
                self.aClose = self.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[5].get_attribute("innerHTML")
                self.aVolume = self.ask.driver.find_elements_by_class_name("a-b-G-Li-Mi")[6].get_attribute("innerHTML")
                self.bOpen = self.bid.driver.find_elements_by_class_name("a-b-G-Li-Mi")[2].get_attribute("innerHTML")
                self.bHigh = self.bid.driver.find_elements_by_class_name("a-b-G-Li-Mi")[3].get_attribute("innerHTML")
                self.bLow = self.bid.driver.find_elements_by_class_name("a-b-G-Li-Mi")[4].get_attribute("innerHTML")
                self.bClose = self.bid.driver.find_elements_by_class_name("a-b-G-Li-Mi")[5].get_attribute("innerHTML")
                self.bVolume = self.bid.driver.find_elements_by_class_name("a-b-G-Li-Mi")[6].get_attribute("innerHTML")
                #print 'FINISH SCANNING', self.aClose
                
            except Exception,e:
                
                #print 'outer loop exception:',str(e)
                pass
            
        self.root.after(100, self.trading)
        
        
    def run(self):
        self.root = Tk()
        self.root.title("v.0.4.0-framework")
        self.root.geometry("300x100")
        
        self.root2 = Tk()
        self.text=Text(self.root2, width = 40, height=4, font=("Helvetica",32))
        self.text.pack()
        self.text.insert(1.0, "PREDIZIONI:")

        

        app = Frame(self.root)
        app.grid()

        start = Button(app, text="Start", command=self.start)
        stop = Button(app, text="Stop", command=self.stop)
        start.grid(row=0, column=0, padx=(40,40), pady=(40,40))
        stop.grid(row=0, column=1, padx=(40,40), pady=(40,40))

        start.grid()
        stop.grid()

        self.root.after(1000, self.trading)  # After 1 second, call scanning
        self.root.mainloop()
        self.root2.mainloop()
        
        
        
app = App()

app.mouse_config()

app.run()
