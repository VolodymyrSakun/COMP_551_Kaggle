import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utils
import random

images = np.load('test_images.npy', encoding='latin1')    
images32 = np.load('test_images32x32.npy', encoding='latin1')
df = utils.loadObject('df.dat')    

aNew = []
i = random.randint(0, len(images))
a1 = images[i, 1].reshape(100, 100)
b1 = images32[i, 1].reshape(32, 32)
aNew.append(a1)
aNew.append(b1)
i = random.randint(0, len(images))
a2 = images[i, 1].reshape(100, 100)
b2 = images32[i, 1].reshape(32, 32)
aNew.append(a2)
aNew.append(b2)
i = random.randint(0, len(images))
a3 = images[i, 1].reshape(100, 100)
b3 = images32[i, 1].reshape(32, 32)
aNew.append(a3)
aNew.append(b3)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)
ax6 = fig.add_subplot(326)
ax1.imshow(aNew[0])
ax2.imshow(aNew[1])
ax3.imshow(aNew[2])
ax4.imshow(aNew[3])
ax5.imshow(aNew[4])
ax6.imshow(aNew[5])
plt.show()


