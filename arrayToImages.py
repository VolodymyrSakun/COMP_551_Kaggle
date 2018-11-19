#!!! save cleaned images from numpy array to png files
from PIL import Image
import numpy as np
import os
import pandas as pd

images = np.load('train_images30x30Clean.npy', encoding='latin1')
imagesTest = np.load('test_images30x30Clean.npy', encoding='latin1')
labelsTrain = pd.read_csv('train_labels.csv')
labelsTrain.drop(columns=['Id'], inplace=True)


# create subdirs for training set
workingDir = os.getcwd()
trainSubdir = os.path.join(workingDir, "training_set")
try:
    os.mkdir(trainSubdir)
except:
    pass
os.chdir(trainSubdir)
categories = list(labelsTrain.Category.unique())
for category in categories:
    newDir = os.path.join(trainSubdir, str(category))
    try:
        os.mkdir(newDir)
    except:
        pass    
os.chdir(workingDir)   
 
# store training images    
for i in range(0, 10000, 1):
    if i%100 == 0:
        print(i)
    category = str(labelsTrain.Category.loc[i])
    image = (images[i][1]).reshape(30, 30)
    im = Image.fromarray(image)
    newDir = os.path.join(trainSubdir, category)
    os.chdir(newDir)
    im.save('{}{}{}{}'.format(category, '.', i, '.png'))

os.chdir(workingDir)

# test
testSubdir = os.path.join(workingDir, "test_set")
try:
    os.mkdir(testSubdir)
except:
    pass  
os.chdir(testSubdir) 
    
for i in range(0, 10000, 1):
    if i%100 == 0:
        print(i)
    image = (imagesTest[i][1]).reshape(30, 30)
    im = Image.fromarray(image)
    im.save('{}{}'.format(i, '.png'))
    
os.chdir(workingDir)



