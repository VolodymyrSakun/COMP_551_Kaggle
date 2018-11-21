from scipy.ndimage import rotate
from scipy.misc import face
from matplotlib import pyplot as plt
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import random 
import pandas as pd

def clipped_zoom(img, zoom_factor, **kwargs):

    h, w = img.shape[:2]

    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:

        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        # Zero-padding
        out = np.zeros_like(img)
        out[top:top+zh, left:left+zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:

        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2

        out = zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs)
        
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        out = out[trim_top:trim_top+h, trim_left:trim_left+w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


rotationBounds = [-40, 40]
zoomBounds = [0.7, 1.2]
nRotations = 20
nZoom = 20

images = np.load('train_images30x30CleanCentered.npy', encoding='latin1')
newSize = len(images) * (nRotations + nZoom)

labelsTrain = pd.read_csv('train_labels.csv')
labelsNew = pd.DataFrame(index=list(range(0, newSize, 1)), columns=labelsTrain.columns)
labelsNew['Id'] = labelsNew.index

k = 0 # new images idx
newImages = np.zeros(shape=(newSize, 2), dtype=object)
newImages[:, 0] = list(range(0, newSize, 1))
categories = []
for m in range(0, len(images), 1): # original images idx
    if m % 100 == 0:
        print(m)
    image = images[m, 1]
    category = labelsTrain.Category.iloc[m]
    image = image.reshape(30, 30)
    for i in range(0, nRotations, 1):
        angle = random.uniform(rotationBounds[0], rotationBounds[1])
        newImage = rotate(image, angle, reshape=False)
        categories.append(category)
        newImages[k, 1] = newImage.reshape(-1)
        k += 1
    for i in range(0, nZoom, 1):
        Good = False
        fails = 0
        while not Good:            
            zoomRate = random.uniform(zoomBounds[0], zoomBounds[1])
            newImage = clipped_zoom(image, zoomRate)
            newImages[k, 1] = newImage.reshape(-1)
            if len(newImages[k, 1]) == 900:
                Good = True
                categories.append(category)
                k += 1    
            else:
                if fails > 10:
                    print("Crash")
                    break
                fails += 1
    
labelsNew['Category'] = categories
newImages = np.concatenate((images, newImages), axis=0)
labelsNew = pd.concat([labelsTrain, labelsNew], axis=0)
np.save('train_images30x30Extended.npy', newImages, fix_imports=False)
labelsNew.to_csv('train_labels_extended.csv', sep=',', index=False)
    
