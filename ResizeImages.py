import cv2
from glob import glob
import os
from skimage import exposure

source = 'Feuersalamander_Frames/*.jpg'
size = (512,512)

for file in glob(source):
    img = cv2.imread(file)
    small_img = cv2.resize(img, size)
    #cv2.imshow('Resized', small_img)
    #cv2.waitKey()
    eq_img = exposure.equalize_adapthist(small_img, clip_limit=0.01)
    eq_img = cv2.convertScaleAbs(eq_img, alpha=(255.0))
    #cv2.imshow('Equalized Alpha', eq_img)
    #cv2.waitKey()
    destination =  'Small_Salamander/' + os.path.splitext(os.path.basename(file))[0] + '_small.png'
    #print(destination)
    cv2.imwrite(destination, eq_img)

