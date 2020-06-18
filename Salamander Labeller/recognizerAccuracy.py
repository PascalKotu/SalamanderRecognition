"""
If you encounter an error saying: This algorithm is patented and is excluded in this configuration; Set OPENCV_ENABLE_NONFREE CMake
option and rebuild the library in function 'cv::xfeatures2d::SIFT::create'

Perform these installations:
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16

"""

import cv2
import os


def dsift(image, step_size=10):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()

    kp = [cv2.KeyPoint(x, y, step_size) for y in range(0, gray.shape[0], step_size)
          for x in range(0, gray.shape[1], step_size)]

    key_pt, des = sift.compute(gray, kp)

    # Code for visualization only
    # image = cv2.drawKeypoints(gray, kp, image, color=(0,0,255))
    # plt.figure()
    # plt.imshow(image)
    # plt.pause(0.05)
    # plt.show()

    return key_pt, des


def image_similarity(image1, image2, step_size=10):
    key_pt1, des1 = dsift(image1, step_size)
    key_pt2, des2 = dsift(image2, step_size)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    # Code for visualization only
    # M = max([image1.shape[0], image2.shape[0]])
    # N = image1.shape[1] + image2.shape[1]
    # img_res = np.zeros((M, N))
    # # imgr = cv2.drawMatchesKnn(img1,key_pt1,img2,key_pt2,good,imgr,flags=2, matchColor=[255,0,0])
    # imgr = cv2.drawMatchesKnn(image1, key_pt1, image2, key_pt2, good, img_res, flags=2)
    # cv2.imshow("MyMatch", imgr)
    # cv2.waitKey()

    return len(good)


num_correct = 0
database = []

for l in range(195):
    database.append(cv2.imread('Belly_Rectified_Noise/' + str(l) + '/10_belly.png'))

print("Database Created")
print("Starting Comparisons")
for i in range(195):
    for j in range(10):
        prediction = {}
        target_img_path = 'Belly_Rectified_Noise/' + str(i) + '/' + str(j) + '_belly.png'
        target_img = cv2.imread(target_img_path)
        true_class = int(os.path.basename(os.path.dirname(target_img_path)))

        for k in range(len(database)):
            prediction[k] = image_similarity(database[k], target_img)

        if true_class == max(prediction, key=prediction.get):
            num_correct += 1

    print("Folder done:", i)
    print("Number of correct predictions = ", num_correct)
    print("Accuracy = ", num_correct/((i+1)*10)*100)
print("Final Accuracy = ", num_correct/(195*10)*100)

