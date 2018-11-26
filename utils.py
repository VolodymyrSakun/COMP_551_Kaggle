import pickle 
import numpy as np

def saveObject(fileName, obj):
    f = open(fileName, "wb")
    pickle.dump(obj, f)
    f.close()
    return

def loadObject(fileName):
    f = open(fileName, "rb")
    obj = pickle.load(f)
    f.close()
    return obj

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
