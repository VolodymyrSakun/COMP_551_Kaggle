from keras.applications import VGG16
from keras.applications import VGG19
from keras.applications import ResNet50
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K
from keras import models
from keras import layers
from keras import optimizers
import utils
from keras.models import load_model

# Global variables
frameSize = 32

imagesForecast = np.load('test_images32x32.npy', encoding='latin1')

x = np.zeros(shape=(len(imagesForecast), frameSize * frameSize), dtype=float)
for i in range(0, len(imagesForecast), 1):
    x[i, :] = imagesForecast[i, 1].reshape(-1) # for regular neural

x /= 255.0

model = load_model('model.h5')
labelMap = utils.loadObject('labelMap.dat')

im_shape = (frameSize, frameSize, 1)
x = x.reshape(x.shape[0], *im_shape)

yForecast = model.predict_proba(x)
results = pd.DataFrame(yForecast, columns=labelMap.keys(), dtype=float)

# To submission
resultsOut = pd.DataFrame(results.idxmax(axis=1))
resultsOut['Id'] = resultsOut.index
resultsOut.rename(columns={0: 'Category'}, inplace=True)
resultsOut = resultsOut[['Id', 'Category']]
resultsOut.to_csv('Forecast.csv', sep=',', index=False)










