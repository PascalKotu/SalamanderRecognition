from sklearn import svm
import numpy as np
import cv2
import matplotlib.pyplot as plt

from skimage.feature import hog

X = np.load("images.npy", allow_pickle=True)
y = np.load("labels.npy", allow_pickle=True)

clf = svm.SVC(gamma = 'auto')
clf.fit(X, y)
print(clf.score(X, y))

image = cv2.imread("salamander.jpg")
predictions = []
for x in range(image.shape[0]-320):

    print(x, "/", image.shape[0]-320)
    list = []
    for y in range(image.shape[1]-320):

        img = image[x:x+320,y:y+320]
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)

        list.append(clf.predict( [fd] )[0])
    predictions.append(list)

#by now in predictions should be filled with 1 and 0.
#1 indicates if a salamander was recognized by the svm
plt.imshow(predictions)
print(predictions)
plt.show()
