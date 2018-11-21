#!!! Image preprocessing; 100x100 to 30x30
#!!! Works with RGB or BW
 
import numpy as np

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

def findImage(image, frameSize, maxOccurence=10):
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
    return image[x : x + frameSize, y : y + frameSize].astype(np.uint8)

def zoomImage(fileIn, fileOut, encoding='latin1', frameSize=30, maxOccurence=10, reshape=True):
    """
    if reshape all images are vectors
    """
    images = np.load(fileIn, encoding='latin1')
    imagesReduced = np.zeros(shape=images.shape, dtype=object)
    imagesReduced[:, 0] = images[:, 0]
    for i in range(0, len(images), 1):
        if i%100 == 0:
            print(i)    
        image = (images[i][1]).reshape(100, 100)
        imageReduced = findImage(image, frameSize=frameSize, maxOccurence=maxOccurence)
        if reshape:
            imagesReduced[i, 1] = imageReduced.reshape(-1)
        else:
            imagesReduced[i, 1] = imageReduced
    np.save(fileOut, imagesReduced, fix_imports=False)
    return
    
zoomImage('train_images.npy', 'train_images30x30.npy', encoding='latin1', frameSize=30, maxOccurence=10, reshape=True)
zoomImage('test_images.npy', 'test_images30x30.npy', encoding='latin1', frameSize=30, maxOccurence=10, reshape=True)    






