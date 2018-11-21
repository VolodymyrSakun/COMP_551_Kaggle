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

def shiftImage(image, shift, direction, frameSize):
    if shift >= frameSize:
        print("shift must be less than frameSize")
        return None
    centered = np.zeros(shape=(frameSize, frameSize), dtype=np.uint8)
    if direction == 'left':        
        centered[:, 0:frameSize-shift] = image[:, shift:] # shiftLeft        
    elif direction == 'right':   
        centered[:, shift:] = image[:, 0:frameSize-shift:] # shiftRight
    elif direction == 'up':   
        centered[0:frameSize-shift, :] = image[shift:, :] # shiftUp      
    elif direction == 'down':  
        centered[shift:, :] = image[0:frameSize-shift:, :] # shiftDown
    else:
        print('Wrong direction')
        return None
    return centered
 
directions = ['left', 'right', 'up', 'down']
shiftBounds = [0, 6]
rotationBounds = [-40, 40]
zoomBounds = [0.7, 1.2]
nClones = 40
frameSize = 30
vectorLength = frameSize * frameSize

images = np.load('train_images30x30CleanCentered.npy', encoding='latin1')
newSize = len(images) * nClones

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
    image = image.reshape(frameSize, frameSize)
    for i in range(0, nClones, 1):
        direction = random.choice(directions)
        shift = random.randrange(shiftBounds[0], shiftBounds[1])
        newImage1 = shiftImage(image, shift, direction, frameSize)
        angle = random.uniform(rotationBounds[0], rotationBounds[1])              
        newImage2 = rotate(newImage1, angle, reshape=False)
        Good = False
        fails = 0
        while not Good:
            zoomRate = random.uniform(zoomBounds[0], zoomBounds[1])  
            newImage3 = clipped_zoom(newImage2, zoomRate)
            newImages[k, 1] = newImage3.reshape(-1)
            if len(newImages[k, 1]) == vectorLength:
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
   



