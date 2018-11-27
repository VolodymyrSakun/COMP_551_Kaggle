#!!! save cleaned images from numpy array to png files
from PIL import Image
import numpy as np
import os
import pandas as pd

def arrayToImages(fileTrain, fileTest, fileLabelsTrain, trainOutDir, testOutDir,\
    originalShape, newShape=None, smoothing=Image.BICUBIC, imageFormat='png', verbose=True):
    """
    """
    imagesTrain = np.load(fileTrain, encoding='latin1')
    labelsTrain = pd.read_csv(fileLabelsTrain)
    labelsTrain.drop(columns=['Id'], inplace=True)

    # create subdirs for training set
    workingDir = os.getcwd()
    trainSubdir = os.path.join(workingDir, trainOutDir)
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
#    for i in range(0, len(imagesTrain), 1):
    for i in range(0, 1000, 1):
        if i%100 == 0 and verbose:
            print(i)
        category = str(labelsTrain.Category.loc[i])
        image = (imagesTrain[i][1]).reshape(originalShape[0], originalShape[1])
        im = Image.fromarray(image.astype(np.uint8))
        if newShape is not None:
            im = im.resize((newShape[0], newShape[1]), smoothing)
        newDir = os.path.join(trainSubdir, category)
        os.chdir(newDir)
        im.save('{}{}{}{}{}'.format(category, '.', i, '.', imageFormat))
    
    os.chdir(workingDir)
    del(imagesTrain)
    # test
    imagesTest = np.load(fileTest, encoding='latin1')
    testSubdir = os.path.join(workingDir, testOutDir)
    try:
        os.mkdir(testSubdir)
    except:
        pass  
    os.chdir(testSubdir)
    testSubdir2 = os.path.join(testSubdir, 'Unknown')
    try:
        os.mkdir(testSubdir2)
    except:
        pass     
    os.chdir(testSubdir2)
#    for i in range(0, len(imagesTest), 1):
    for i in range(0, 1000, 1):
        if i%100 == 0 and verbose:
            print(i)
        image = (imagesTest[i][1]).reshape(originalShape[0], originalShape[1])
        im = Image.fromarray(image.astype(np.uint8))
        if newShape is not None:
            im = im.resize((newShape[0], newShape[1]), smoothing)
        im.save('{}{}{}'.format(i, '.', imageFormat))
        
    os.chdir(workingDir)

arrayToImages('train_images50x50.npy', 'test_images50x50.npy',\
    'train_labels.csv', 'training_set224', 'test_set224',  (50, 50),\
    newShape=(224, 224), smoothing=Image.BICUBIC, imageFormat='png', verbose=True)

#arrayToImages('train_images.npy', 'test_images.npy',\
#    'train_labels.csv', 'training', 'test',  (100, 100),\
#    newShape=(224, 224), smoothing=Image.BICUBIC, imageFormat='png', verbose=True)
