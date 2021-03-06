
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dropout
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import utils 
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import load_model

# Global variables
batch_size = 1000 # 16 or 32
epochs = 5
img_width_cols = 32
img_height_rows = 32
input_shape = (img_height_rows, img_width_cols, 1)

images = np.load('train_images32x32Extended.npy', encoding='latin1')
labelsTrain = pd.read_csv('train_labels_extended32.csv')

labelsTrain.drop(columns=['Id'], inplace=True)
nClasses = labelsTrain.Category.nunique()

labelencoder_X=LabelEncoder()
labelsTrain['Numeric'] = labelencoder_X.fit_transform(labelsTrain.Category.values)
y = labelsTrain['Numeric'].values.astype(np.uint8)

labels = list(labelencoder_X.classes_) # from 0
labelMap = {}
for i in range(0, len(labels), 1):
    labelMap[labels[i]] = i

x = np.zeros(shape=(len(images), img_width_cols*img_height_rows), dtype=float)
for i in range(0, len(images), 1):
    x[i, :] = images[i, 1].reshape(-1) 

x /= 255.0

# Let's split the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

# Encode the categories
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
y_train = to_categorical(y_train, nClasses)
y_test = to_categorical(y_test, nClasses)

im_shape = (img_height_rows, img_width_cols, 1)
x_train = x_train.reshape(x_train.shape[0], *im_shape) # Python TIP :the * operator unpacks the tuple
x_test = x_test.reshape(x_test.shape[0], *im_shape)

model = load_model('model.h5')

history = model.fit(x_train, y_train,
                  batch_size=batch_size, epochs=epochs,
                  validation_data=(x_test, y_test))
				  
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

# Save the model
model.save('model.h5')
model.save_weights('weights.h5') 
#utils.saveObject('history.dat', history)
utils.saveObject('labelMap.dat', labelMap)

fig2 = plt.figure(2, figsize=(15, 12))
plt.subplot(2, 1, 1)
plt.plot(np.arange(0, epochs), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), history.history["val_loss"], label="val_loss")
plt.legend()
plt.ylabel('Loss')
plt.xlabel('Number of epochs')
plt.title('Loss')

plt.subplot(2, 1, 2)
plt.plot(np.arange(0, epochs), history.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), history.history["val_acc"], label="val_acc")
plt.legend()
plt.ylabel('Accuracy')
plt.xlabel('Number of epochs')
plt.title('{} {}'.format('Accuracy:', scores[1]*100))

F = '{}{}'.format('Accuracy and loss versus number of epochs', '.png')
plt.savefig(F, bbox_inches='tight')
				  
