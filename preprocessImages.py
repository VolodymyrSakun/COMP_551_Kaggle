import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import pandas as pd
import utils

# frame if square
def findFrame(image, xStart, yStart, frameSize):
    """
    image must contain 0 or 1
    """
    def checkBoundary(image, x, y, frameSize):
        if ((frameSize + x) > image.shape[0]) or ((frameSize + y) > image.shape[1]):
            return False # out of boundary
        return True # fits
            
    if not checkBoundary(image, xStart, yStart, frameSize):
        return None
    x = xStart
    y = yStart
    scoreBest = np.sum(image[x : x + frameSize, y : y + frameSize])
    foundBetter = True
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    scores = np.zeros(shape=(4,), dtype=int)
    while foundBetter:
        foundBetter = False
        for i, move in enumerate(moves, 0):
            xNew = x + move[0]
            yNew = y + move[1]          
            if not checkBoundary(image, xNew, yNew, frameSize):
                scores[i] = 0
            else:
                scores[i] = np.sum(image[xNew : xNew + frameSize, yNew : yNew + frameSize])
        maxScoreIdx = np.argmax(scores)
        if scores[maxScoreIdx] > scoreBest:
            scoreBest = scores[maxScoreIdx]
            x = x + moves[maxScoreIdx][0]
            y = y + moves[maxScoreIdx][1]
            foundBetter = True
    return x, y, scoreBest

def countHighestScores(sortedList):
    """
    count number of top scores in sorted list
    """
    i = 0
    count = 1
    value = sortedList[0]
    for i in range(1, len(sortedList), 1):
        if sortedList[i] != value:
            break
        else:
            count += 1
    return count

def findImage(image, frameSize, outSize, maxOccurence=10):
    """
    return shape frameSize x frameSize
    """
    image2 = image.astype(bool)
    image3 = image2.astype(np.uint8)
    list1 = []
    list2 = []    
    for i in range(0, 1000, 1):
        x = np.random.randint(0, 100-frameSize)
        y = np.random.randint(0, 100-frameSize)
        xBest, yBest, scoreBest = findFrame(image3, x, y, frameSize)
        list1.append(scoreBest)
        list2.append((xBest, yBest))    
        list1, list2 = (list(t) for t in zip(*sorted(zip(list1, list2), reverse=True)))    
        count = countHighestScores(list1)
        if count >= maxOccurence:
            break    
    x=list2[0][0]
    y=list2[0][1]
    shift = int((outSize - frameSize) / 2)
    x = x - shift
    y = y - shift
    if x < 0:
        x = 0
    if y < 0:
        y = 0
    if x + outSize > 100:
        x = 100 - outSize
    if y + outSize > 100:
        y = 100 - outSize        
    return image[x : x + outSize, y : y + outSize].astype(np.uint8)

def reduceIntensity(image, lowest):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if image[i, j] < lowest:
                image[i, j] = 0
    return image
    
def cropImage(img, tol=0):
    # img is image data
    # tol  is tolerance
    mask = img > tol
    return img[np.ix_(mask.any(1),mask.any(0))]

def getBigContour2(contours):
    c = contours[0]
    y = c[:, 0, 0]
    x = c[:, 0, 1]
    size = len(x)
    for i in range(1, len(contours), 1):
        newC = contours[i]
        xNew = newC[:, 0, 1]
        newSize = len(xNew)
        if newSize > size:
            y = newC[:, 0, 0]
            x = newC[:, 0, 1]
            size = newSize
    return x, y, size

def getBigContour(contours):
    c = contours[0]
    y = c[:, 0, 0]
    x = c[:, 0, 1]
    area = float(cv.contourArea(c))
    perimeter = float(cv.arcLength(c,True))
    product = area * perimeter
    for i in range(1, len(contours), 1):
        newC = contours[i]
        newArea = float(cv.contourArea(newC))
        newPerimeter = float(cv.arcLength(newC, True))
        newProduct = newArea * newPerimeter
        if newProduct > product:
            y = newC[:, 0, 0]
            x = newC[:, 0, 1]
            area = newArea
            perimeter = newPerimeter
            product = area * perimeter
    return x, y, int(perimeter), int(area)

def fillFromPoint(image, X, Y):
    newImage = np.zeros(shape=image.shape, dtype=np.uint8)
    for i in range(0, len(X), 1):
        x = X[i]
        y = Y[i]
        xOld = x
        yOld = y
        newImage[x, y] = image[x, y]
    
        while image[x, y] != 0:
            x += 1
            if x >= image.shape[0]:
                break
            if image[x, y] != 0:
                newImage[x, y] = image[x, y]
        x = xOld
        y = yOld
        while image[x, y] != 0:
            x -= 1
            if x < 0:
                break
            if image[x, y] != 0:
                newImage[x, y] = image[x, y]
        x = xOld
        y = yOld
        while image[x, y] != 0:
            y += 1
            if y >= image.shape[1]:
                break
            if image[x, y] != 0:
                newImage[x, y] = image[x, y]
        x = xOld
        y = yOld
        while image[x, y] != 0:
            y -= 1
            if y < 0:
                break
            if image[x, y] != 0:
                newImage[x, y] = image[x, y]
    return newImage

def centerImage2(image, frame=(50, 50)):
 
    newImage = np.zeros(shape=frame, dtype=np.uint8)
    firstRow = int((frame[0] - image.shape[0]) / 2)
    firstColumn = int((frame[1] - image.shape[1]) / 2)
    newImage[firstRow : firstRow + image.shape[0], firstColumn : firstColumn + image.shape[1]] = image[:, :]
    
    return newImage

def centerImage(image, frame=(32, 32)):
 
    newImage = np.zeros(shape=frame, dtype=np.uint8)
    
    if image.shape[0] > frame[0]:
        firstRow = int((image.shape[0] - frame[0]) / 2)
        image = image[firstRow : firstRow + frame[0], :]
    if image.shape[1] > frame[1]:
        firstColumn = int((image.shape[1] - frame[1]) / 2)
        image = image[:, firstColumn : firstColumn + frame[1]]    
    
    firstRow = int((frame[0] - image.shape[0]) / 2)
    firstColumn = int((frame[1] - image.shape[1]) / 2)
    newImage[firstRow : firstRow + image.shape[0], firstColumn : firstColumn + image.shape[1]] = image[:, :]
    
    return newImage

def validImage(image):
    if max(image.shape[0], image.shape[1]) <= 13:
        return False
    return True

def cleanImages(fileIn, fileOut, labels=None, labelsOut=None,\
                encoding='latin1', frame=(32, 32), reshape=True):
    
    images = np.load(fileIn, encoding='latin1')
    if labels is not None:
        labels = pd.read_csv(labels)
        
    imagesReduced = np.zeros(shape=images.shape, dtype=object)
    imagesReduced[:, 0] = images[:, 0]
    empty = np.zeros(shape=frame, dtype=np.uint8)    
    perimeters = [] # estimating
    areas = [] # estimating
    for i in range(0, len(images), 1):
#    for i in range(0, 10000, 1):
        if i%100 == 0:
            print(i)  
        if labels is not None:        
            if labels.Category.loc[i] == 'empty':
                imagesReduced[i, 1] = empty
                perimeters.append(0)
                areas.append(0)
                continue
   
        image = images[i, 1]
        image = image.reshape(100, 100)
        image2 = image.astype(bool)
        image3 = image2.astype(np.uint8)   
        _, contours, _ = cv.findContours(image3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)    
        x, y, perimeter, area = getBigContour(contours)        
        newImage = fillFromPoint(image, x, y)
        im = cropImage(newImage, tol=0)
        if not validImage(im):
            im1 = findImage(image, frame[0], frame[0], maxOccurence=10)
            im2 = im1.astype(bool)
            im3 = im2.astype(np.uint8)   
            _, contours, _ = cv.findContours(im3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)            
            x, y, perimeter, area = getBigContour(contours)
            newImage = fillFromPoint(im1, x, y)
            im = cropImage(newImage, tol=0)
        perimeters.append(perimeter)
        areas.append(area)
        centered = centerImage(im, frame=frame)
        if perimeter < 42 or area < 30:
            imagesReduced[i, 1] = empty
            if labels is not None:
                labels.Category.loc[i] = 'empty'
        else:
            imagesReduced[i, 1] = centered        
        if reshape:
            imagesReduced[i, 1] = imagesReduced[i, 1].reshape(-1)
    
    np.save(fileOut, imagesReduced, fix_imports=False)
    if labelsOut is not None and labels is not None:
        labels.to_csv(labelsOut, sep=',', index=False)

    return perimeters, areas
    

perimetersTrain, areasTrain = cleanImages('train_images.npy', 'train_images32x32.npy',\
    labels='train_labels.csv', labelsOut='train_labels_new.csv',\
    encoding='latin1', frame=(32, 32), reshape=True)    
    
perimetersTest, areasTest = cleanImages('test_images.npy', 'test_images32x32.npy',\
    labels=None, labelsOut=None, encoding='latin1', frame=(32, 32), reshape=True)    

labelsTrain = pd.read_csv('train_labels_new.csv')


df = pd.DataFrame(index=list(range(0, len(areasTrain), 1)),\
    columns=['perimeters', 'areas', 'labels'])
df['perimeters'] = perimetersTrain
df['areas'] = areasTrain
df['labels'] = labelsTrain['Category']

df.sort_values(['perimeters'], inplace=True)
df.sort_values(['areas'], inplace=True)

utils.saveObject('df.dat', df)


