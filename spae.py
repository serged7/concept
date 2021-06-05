import keras.callbacks as cb
import keras.backend as K
import numpy as np
import random as rd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D #@UnresolvedImport
import tensorflow as tf
#keras
from keras.layers import Input, Dense, AlphaDropout, GaussianNoise, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model, load_model
from keras import regularizers, activations
from keras.optimizers import SGD, Adadelta, Adam
from keras.utils import plot_model, to_categorical
from keras import backend as K

#sklearn
from sklearn import neighbors, tree
from sklearn.externals import joblib
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from keras.callbacks import ModelCheckpoint

# def relumx(x):
#     return K.relu(x, max_value=10)

class Esl(cb.Callback): #early stopping loss
    def __init__(self, monitor='loss', tv = 0.60):
        super(cb.Callback, self).__init__()
        self.monitor = monitor
        self.tv = tv
        
    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        l = logs.get('loss')
        #print("loss:",epoch,l)
        if epoch > 3 and l > self.tv:
            print("stopped on loss")
            self.model.stop_training = True
        if epoch > 6 and l > 0.55:
            print("stopped on loss")
            self.model.stop_training = True
        if epoch > 14 and l > 0.45:
            #print("stopped on loss")
            self.model.stop_training = True
        if epoch > 19 and l > 0.42:
            #print("stopped on loss")
            self.model.stop_training = True
        if epoch > 24 and l > 0.4:
            #print("stopped on loss")
            self.model.stop_training = True
class Sem(object):
    def __init__(self, obs):
        self.obs = obs
        self.sym = {}
        self.prot = None
        self.ms = None
        
    def fit(self, dat, q=0):
        en = self.obs.encode(dat)
        if q > 0:
            bw = estimate_bandwidth(en, quantile=q, n_samples=len(en))
            ms = MeanShift(bandwidth=bw, bin_seeding=True)
            ms.fit(en)
            self.pl = ms.cluster_centers_
            self.po = self.obs.generate(self.pl)
            self.ms = ms
        else:
            l = np.asarray([0,0,0,1,1,1,2,2,2])
            self.ms = neighbors.KNeighborsClassifier(2, weights='distance')
            self.ms.fit(en,l)
            self.pl = [np.mean(en[:3],axis=0), np.mean(en[3:6],axis=0), np.mean(en[6:],axis=0)]
            #print(self.pl)
            self.po = self.obs.generate(np.asarray(self.pl))
        
    def ident(self, sam):
        ens = self.obs.encode(sam)
        return int(self.ms.predict(ens))
        

class Saen(object):
    def __init__(self, size=64, ed=3, hd=100, stride=1, type =3, r="sparse"):
        # type = 3 rgb, 1 = greyscale
        self.size = size
        self.hd = hd
        self.ed = ed
        self.stride = stride
        self.type = type # color: 3
        rs = 16 # 32
        acc = 26
        if size == 32:
            rs = 4
        depth = -8 #flat: -13 sparse: -12

        inp = Input(shape=(size,size,type))
        
        print("input", inp.shape)
        x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', strides=(stride,1), padding='same')(inp) #16
        x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) #8
#         # conv2
        x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', strides=(stride,1), padding='same')(x)
        enc1 = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
#         #print(x.shape)
#         # conv3
#         x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', strides=(stride,1), padding='same')(x)
#         enc1 = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
        print("enc1", enc1.shape)
        
        x = Reshape((rs*rs*8,))(enc1)  # 32: (4*4*8)  64: (8*8*8)
        print("flattened", x.shape)
        x = Dense(hd, activation='relu')(x) #
#         x = LeakyReLU()(x)
#         x = Dense(ed)(x) #flat activation #
        #encoded = LeakyReLU()(x) #flat
        if r == "sparse":
            encoded = Dense(ed, activation='relu', activity_regularizer=regularizers.l1(10e-6))(x) #sparse activation
            print("sparse mode")
        else:
            #relumx = lambda x : activations.relu(x, max_value=acc)
            encoded = Dense(ed, activation='relu')(x)
            print("flat mode", acc)
        print("encoded:", r, encoded.shape)
        y = encoded
        y = Dense(hd, activation='relu')(y)
#         y = LeakyReLU()(y)
        y = Dense(rs*rs*8, activation='relu')(y) #activation='relu'
        y = Reshape((rs,rs,8))(y)
            # DeConv1
        y = Conv2D(8, (3, 3), activation='relu', strides=(stride,1), padding='same')(y)
        y = UpSampling2D((2, 2))(y)
        
        y = Conv2D(8, (3, 3), activation='relu', strides=(stride,1), padding='same')(y)
        y = UpSampling2D((2, 2))(y)
        #print(x.shape)

#         x = Conv2D(8, (3, 3), activation='relu', strides=(stride,1), padding='same')(x)
#         x = UpSampling2D((2, 2))(x)
#         #print(x.shape)
# 
#         x = Conv2D(16, (3, 3), activation='relu', strides=(stride,1), padding='same')(x)
#         x = UpSampling2D((2, 2))(x)
        #print(x.shape)
        decoded = Conv2D(type, (3, 3), activation='sigmoid', padding='same')(y)
        print("decoded", decoded.shape)
        
        self.mod = Model(inp, decoded)
        self.mod.summary()
        #encoder
        self.encoder = Model(inp, encoded) 
        
        #generator
        inp_enc = Input((ed,))
        l = inp_enc
        for k in range(depth,-1,1): #-13 flat -12 sparse
            l = self.mod.layers[k](l)            
        decoded_layer = self.mod.layers[-1]
        #print(decoded_layer, l)
        self.gen = Model(inp_enc, decoded_layer(l))     
        self.mod.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
        
    def train(self, train=None, eps=50, bs=64, val=0.2, shuffle=True):
        if train is None:
            train = self.rand
        print(train.shape)
        h = self.mod.fit(train, train, epochs=eps, batch_size=bs, shuffle=True,
                            validation_split=val, verbose = True, callbacks=[Esl(tv = 0.60)])
        return h   
        
    def encode(self, x_in): 
        return self.encoder.predict(x_in)
    
    def generate(self, e_in):
        return self.gen.predict(e_in)

    def save(self, filename=""):
        self.mod.save(filename+"mod_cae.h5")
        self.encoder.save(filename+"mod_cenc.h5")
        self.gen.save(filename+"mod_cgen.h5")
        
def load(filename="fn", hd=100, ed=3, t=1, sz=32):
    pref = filename if filename[0] == "_" else ""
    enc = load_model(pref + "mod_cenc.h5")
    cae = load_model(pref + "mod_cae.h5")
    gen = load_model(pref + "mod_cgen.h5")
    #ae.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['accuracy'])

    saen = Saen(ed=ed, hd=hd, size = sz)
    saen.mod= cae
    saen.encoder = enc
    saen.gen = gen
    return saen


