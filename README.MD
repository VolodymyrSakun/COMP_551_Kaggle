preprocessImages.py

Initial dataset comtaines 10000 images 100x100 with lots of noise. 
Preprocessing part has two goals: reduce image size since real images are much smaller than 100x100 and remove all noise.
Do that we used two approaches: 
1. Contour approach. We used findContours from OpenCV library to split one image into many small images.
First, each image is transformed into 0 / 1 array of 100x100. Then findContours gives list of contours of that image.
Selection criteria: greatest product of area and perimeter performed by our function getBigContour.
When contour is selected, image is reprodued from original on blank 100x100 array. Cntour is provided as set of x, y coordinates.
For each point our function fillFromPoint reporduces image from original.
Next step is to reduce image size to lowest possible by cropImage.
Then centerImage puts newly obtained frame befined as 32x32. Our analysis showed that this area is sufficient to fit all images.
If findContours failed and its greatest contour does not satisfies minimun requirement (at least one dimension must be > than some value)
we used findImage that finds the densiest region in 100x100 space assigned by frame size = 32x32
After, findContours is applied in smaller region.
If image is still too small, it is assigned as empty.
Result of preprocessImages.py is two numpy objects with images 32x32 without noice and corrected labels.

utils.py 

Contains few usefull functions.

imageTransform.py

Used to augment number of images by shifting, flipping and rotation.
Parameters are:
flipProbability = 0.2
directions = ['left', 'right', 'up', 'down']
shiftBounds = [0, 5]
rotationBounds = [-20, 20]
zoomBounds = [0.8, 1.2]
flips = ['LR', 'UD']
nClones = 40

Result:
New augmented array for training set with updated classes. Each image contains nClones replicated copies.

svc.py

Fits preprocessed images and validates accuracy using few folds
Algorithms used:

SVC with linear kernel
SVC with squared exponential kernel
Random Forest Classifier
XGBoost classifier
Multinomial Naive Bayes

cnn11.py
Eleventh version of convolution neural network.
Fits preprocessed images and saves model.

cnnPredict.py

Loads saved model and predicts test set.
Used for submission.

cnn14.py

Loads saved model and continues fitting. Then saves new model.
 
