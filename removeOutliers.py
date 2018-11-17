import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import copy

def transformToFeatures(image, dimension):
    image1 = image.reshape(dimension, dimension)
    image2 = image1.astype(bool)
    image3 = image2.astype(np.uint8)
    nPoints = np.sum(image3)
    x = np.zeros(shape=(nPoints, 2), dtype=np.uint8)
    k = 0
    for i in range(0, image3.shape[0], 1):
        for j in range(0, image3.shape[1], 1):
            if image3[i, j] != 0:
                x[k, 0] = i
                x[k, 1] = j
                k += 1
    return x

def removeNoise(image, x, y, dimension):
    """
    output size = dimension x dimension
    """
    image1 = copy.deepcopy(image.reshape(dimension, dimension))
    if len(x) != len(y):
        return None
    for i in range(0, len(y), 1):
        if y[i] == -1:
            image1[x[i, 0], x[i, 1]] = 0
    return image1

def outliersDetection(X, contamination=0.1):
    """
    """
    # Return 1 = inlier, -1 = outlier    
    
    y = np.zeros(shape=(X.shape[0], 5), dtype=float)
# Outliers detection IsolationForest
#    print('IsolationForest')
    clf = IsolationForest(n_estimators=100, max_samples='auto', contamination=contamination,\
        max_features=1.0, bootstrap=False, n_jobs=1, behaviour='new')
    clf.fit(X)
    y[:, 0] = clf.predict(X) # 1 = inlier, -1 = outlier
# Outliers detection LocalOutlierFactor    
#    print('LocalOutlierFactor')
    # metric='minkowski', p=2 = euclidian distance
    clf = LocalOutlierFactor(n_neighbors=20, algorithm='auto', leaf_size=30,\
        metric='minkowski', p=2, metric_params=None, contamination=contamination, n_jobs=1)
    y[:, 1] = clf.fit_predict(X)
# Outliers detection EllipticEnvelope    
#    print('EllipticEnvelope')
    ee = EllipticEnvelope(store_precision=False, assume_centered=False,\
        support_fraction=None, contamination=contamination)
    ee.fit(X)
    y[:, 2] = ee.predict(X)
# Outliers detection OneClassSVM    
#    print('OneClassSVM')
    ocSvm = OneClassSVM(kernel='linear', gamma='auto', coef0=0.0, tol=0.001, nu=contamination,\
        shrinking=True, cache_size=200, max_iter=-1)
    ocSvm.fit(X)
    y[:, 3] = ocSvm.predict(X)
   
# Determine outliers based on algorithms results and store to original set
    y[:, 4] = np.sum(y, axis=1, dtype=int, out=None)    
    return y, np.where(y[:, 4] > 0, 1, -1)

def showOutlierProgress(image, x, yFull, y, dimension):
    aNew = []
    aNew.append(removeNoise(image, x, yFull[:, 0], dimension))
    aNew.append(removeNoise(image, x, yFull[:, 1], dimension))
    aNew.append(removeNoise(image, x, yFull[:, 2], dimension))
    aNew.append(removeNoise(image, x, yFull[:, 3], dimension))
    aNew.append(removeNoise(image, x, y, dimension))
    aNew.append(image)
    
    fig = plt.figure(figsize=(15, 9))
    ax1 = fig.add_subplot(331)
    ax2 = fig.add_subplot(332)
    ax3 = fig.add_subplot(333)
    ax4 = fig.add_subplot(334)
    ax5 = fig.add_subplot(335)
    ax6 = fig.add_subplot(336)
    
    ax1.imshow(aNew[0])
    ax2.imshow(aNew[1])
    ax3.imshow(aNew[2])
    ax4.imshow(aNew[3])
    ax5.imshow(aNew[4])
    ax6.imshow(aNew[5])
        
    ax1.title.set_text('IsolationForest')
    ax2.title.set_text('LocalOutlierFactor')
    ax3.title.set_text('EllipticEnvelope')
    ax4.title.set_text('OneClassSVM')
    ax5.title.set_text('>0')
    ax6.title.set_text('Original')
    plt.show()
    return

def cleanImages(fileIn, fileOut, encoding='latin1', frameSize=30, contamination=0.3, reshape=True):
    """
    if reshape all images are vectors
    """
    images = np.load(fileIn, encoding='latin1')
    imagesClean = np.zeros(shape=images.shape, dtype=object)
    imagesClean[:, 0] = images[:, 0]
    for i in range(0, 10000, 1):
        if i%100 == 0:
            print(i)    

        x = transformToFeatures(images[i][1], dimension=frameSize)
        yFull, y = outliersDetection(x, contamination=contamination)   
        imageClean = removeNoise(images[i][1], x, y, frameSize)
        
        if reshape:
            imagesClean[i, 1] = imageClean.reshape(-1)
        else:
            imagesClean[i, 1] = imageClean
    np.save(fileOut, imagesClean, fix_imports=False)
    return

cleanImages('train_images30x30.npy', 'train_images30x30Clean.npy',\
    encoding='latin1', frameSize=30, contamination=0.3, reshape=True)

cleanImages('test_images30x30.npy', 'test_images30x30Clean.npy',\
    encoding='latin1', frameSize=30, contamination=0.3, reshape=True)










