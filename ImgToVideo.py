import cv2
import numpy as np
from glob import glob

source = 'salamanderpics/*.png'
img_array = []
#size = ()

for file in glob(source):
    img = cv2.imread(file)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)

out = cv2.VideoWriter('Salamander_Video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
