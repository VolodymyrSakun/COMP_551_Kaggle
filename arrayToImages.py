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
    imagesTest = np.load(fileTest, encoding='latin1')
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
    for i in range(0, 10000, 1):
        if i%100 == 0 and verbose:
            print(i)
        category = str(labelsTrain.Category.loc[i])
        image = (imagesTrain[i][1]).reshape(originalShape[0], originalShape[1])
        im = Image.fromarray(image)
        if newShape is not None:
            im = im.resize((newShape[0], newShape[1]), smoothing)
        newDir = os.path.join(trainSubdir, category)
        os.chdir(newDir)
        im.save('{}{}{}{}{}'.format(category, '.', i, '.', imageFormat))
    
    os.chdir(workingDir)
    
    # test
    testSubdir = os.path.join(workingDir, testOutDir)
    try:
        os.mkdir(testSubdir)
    except:
        pass  
    os.chdir(testSubdir) 
        
    for i in range(0, 10000, 1):
        if i%100 == 0 and verbose:
            print(i)
        image = (imagesTest[i][1]).reshape(originalShape[0], originalShape[1])
        im = Image.fromarray(image)
        if newShape is not None:
            im = im.resize((newShape[0], newShape[1]), smoothing)
        im.save('{}{}{}'.format(i, '.', imageFormat))
        
    os.chdir(workingDir)

arrayToImages('train_images30x30Clean.npy', 'test_images30x30Clean.npy',\
    'train_labels.csv', 'training_set50', 'test_set50',  (30, 30),\
    newShape=(50, 50), smoothing=Image.BICUBIC, imageFormat='png', verbose=True)

