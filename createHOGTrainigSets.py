import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from glob import glob
import os
import numpy as np


source = 'cropped/*.png'
x = []
y = []
for file in glob(source):
    image = cv2.imread(file)
    size = (320,320)
    image =  cv2.resize(image, size)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    '''
    #stuff for visualizing the hog features
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    ax1.axis('off')
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.set_title('Input image')

    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()
    '''
    x.append(fd)
    y.append(1)

source = 'croppedwrong/*.png'
for file in glob(source):
    image = cv2.imread(file)
    size = (320,320)
    image =  cv2.resize(image, size)

    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)
    x.append(fd)
    y.append(0)
x = np.asarray(x)
y = np.asarray(y)
np.save("images", x)
np.save("labels", y)
