
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import copy

def centerImage(image, frameSize, com=None):
    image = image.reshape(frameSize, frameSize)
    image2 = image.astype(bool)
    image3 = image2.astype(np.uint8)    
    x, y = ndimage.measurements.center_of_mass(image3)
    if com is None:
        imageOnes = np.ones(shape=(frameSize, frameSize), dtype=int)
        imageOnes = imageOnes.astype(bool)
        imageOnes = imageOnes.astype(np.uint8)    
        xCenter, yCenter = ndimage.measurements.center_of_mass(imageOnes)
    else:
        xCenter = com[0]
        yCenter = com[1]
        
    yShift = int(round(x - xCenter)) # positive = shift left
    xShift = int(round(y - yCenter)) # positive = shift up

    imageCopy = copy.deepcopy(image)
    centered = image
    if xShift > 0:
        centered = np.zeros(shape=(frameSize, frameSize), dtype=np.uint8)
        centered[:, 0:frameSize-xShift] = imageCopy[:, xShift:] # shiftLeft
        imageCopy = copy.deepcopy(centered)
    if xShift < 0:
        xShift = abs(xShift)
        centered = np.zeros(shape=(frameSize, frameSize), dtype=np.uint8)
        centered[:, xShift:] = imageCopy[:, 0:frameSize-xShift:] # shiftRight
        imageCopy = copy.deepcopy(centered)
    if yShift > 0:
        centered = np.zeros(shape=(frameSize, frameSize), dtype=np.uint8)
        centered[0:frameSize-yShift, :] = imageCopy[yShift:, :] # shiftUp
        imageCopy = copy.deepcopy(centered)
    if yShift < 0:
        yShift = abs(yShift)
        centered = np.zeros(shape=(frameSize, frameSize), dtype=np.uint8)
        centered[yShift:, :] = imageCopy[0:frameSize-yShift:, :] # shiftDown

    return centered

def centerImages(fileIn, fileOut, encoding='latin1', frameSize=30, asVector=True, com=None):
    """
    if asVector all images are vectors
    """
    images = np.load(fileIn, encoding='latin1')
    imagesCentered = np.zeros(shape=images.shape, dtype=object)
    imagesCentered[:, 0] = images[:, 0]
    for i in range(0, len(images), 1):  
        centered = centerImage(images[i][1], frameSize, com=com)
        if asVector:
            imagesCentered[i, 1] = centered.reshape(-1)
        else:
            imagesCentered[i, 1] = centered
    
    np.save(fileOut, imagesCentered, fix_imports=False)

centerImages('train_images30x30Clean.npy', 'train_images30x30CleanCentered.npy',\
             encoding='latin1', frameSize=30, asVector=True, com=(14.5, 14.5))
centerImages('test_images30x30Clean.npy', 'test_images30x30CleanCentered.npy',\
             encoding='latin1', frameSize=30, asVector=True, com=(14.5, 14.5))

centerImages('train_imagesBW30x30Clean.npy', 'train_imagesBW30x30CleanCentered.npy',\
             encoding='latin1', frameSize=30, asVector=True, com=(14.5, 14.5))
centerImages('test_imagesBW30x30Clean.npy', 'test_imagesBW30x30CleanCentered.npy',\
             encoding='latin1', frameSize=30, asVector=True, com=(14.5, 14.5))

