import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from glob import glob
import os

class GetPointClass():
    def __init__(self, image):
        self.img = image
        self.point = ()

    def getCoord(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.imshow(self.img)
        cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.point

    def __onclick__(self,click):
        self.point = (click.xdata,click.ydata)
        plt.close()
        return self.point





'''
This will iterate trough every image in the source directory.
It will open up every image multiple times.
The first time you are supposed to pick the top-left corner of the region you want to cutout.
The second time you are supposed to pick the bottom-right corner.
Afterwards it will show the image with a box drawn onto it. The box indicates what region will be cut out.
Just close the window.
If you are happy with the result type "y" in the console. If you do so the cropped image is saved.
Otherwise you have to pick the top-left and bottom-right corner again.
'''
source = 'images/*.jpg'
for file in glob(source):
    correctlyCropped = False
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    while correctlyCropped == False:

        pointGetter = GetPointClass(img)
        cord1 = pointGetter.getCoord()
        cord2 = pointGetter.getCoord()
        fig,ax = plt.subplots(1)
        ax.imshow(img)

        rect = patches.Rectangle(cord1,cord2[0]-cord1[0],cord2[1]-cord1[1],linewidth=1,edgecolor='r',facecolor='none')

        ax.add_patch(rect)
        plt.show()
        txt = input("Happy with the results?(y/n)")
        if txt == "y":
            correctlyCropped = True
    cropedImg = img[int(cord1[1]):int(cord2[1]), int(cord1[0]):int(cord2[0])]
    cropedImg = cv2.cvtColor(cropedImg, cv2.COLOR_RGB2BGR)
    destination =  'croppedwrong/' + os.path.splitext(os.path.basename(file))[0] + '_cropped.png'
    cv2.imwrite(destination, cropedImg)
