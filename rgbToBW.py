#!!! Image preprocessing; RGB to BW; keeps resolution
 
import numpy as np

def rgbToBW(fileIn, fileOut, encoding='latin1'):
    images = np.load(fileIn, encoding=encoding)
    imagesBW = np.zeros(shape=images.shape, dtype=object)
    imagesBW[:, 0] = images[:, 0]
    for i in range(0, len(images), 1):
        image = images[i, 1]
#        imageBW = np.where(image < 128, 0, 1).astype(np.uint8) # split colors
        imageBW = np.where(image < 1, 0, 1).astype(np.uint8) # all colors to 1   
        imagesBW[i, 1] = imageBW    
    np.save(fileOut, imagesBW, fix_imports=False)
    return

rgbToBW('train_images30x30Clean.npy', 'train_imagesBW30x30Clean.npy', encoding='latin1')
rgbToBW('test_images30x30Clean.npy', 'test_imagesBW30x30Clean.npy', encoding='latin1')
