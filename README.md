To create those 4 files 

test_images30x30Clean.npy, test_imagesBW30x30Clean.npy, train_images30x30Clean.npy, train_imagesBW30x30Clean.npy

follow the procedure:

Run first zoomImage.py

#!!! Image preprocessing; 100x100 to 30x30
#!!! Works with RGB or BW

then run removeOutliers.py

lastly run rgbToBW.py

Or just use generated 4 files for image classification.
They have similar format to test_images.npy and train_images.npy,
but image size is 30x30 with less noise

