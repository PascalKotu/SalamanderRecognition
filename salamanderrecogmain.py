import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models.densenet import DenseNet, _DenseLayer, _DenseBlock, _Transition
import torch.nn as nn
from collections import OrderedDict
from glob import glob
import csv
import os
import PySimpleGUI as sg
from PIL import Image
from torch.utils.data import Dataset
import math
from scipy import interpolate
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
import cv2
import pandas as pd
import pdb; pdb.set_trace()


# A class to create Neural Networks (specifically Densenet)
class MyDenseNet(DenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                # memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


# A class to rescale images and labelled points
class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, belly_points = sample['image'], sample['belly_points']
        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))
        belly_points = belly_points * [new_w / w, new_h / h]

        return {'image': img, 'belly_points': belly_points}


# A class to convert images and labelled points to Tensor
class ToTensor(object):
    def __call__(self, sample):
        image, belly_points = sample['image'], sample['belly_points']  # numpy image: H x W x C
        image = image.transpose((2, 0, 1))  # torch image: C X H X W
        image = torch.from_numpy(image.copy())
        belly_points = torch.from_numpy(belly_points.copy())
        return {'image': image, 'belly_points': belly_points}


# A function to display the predicted belly_points on the image
def show_points(image, preds):
    plt.imshow(image.permute(1, 2, 0))
    plt.scatter(preds[:, 0].detach().numpy(), preds[:, 1].detach().numpy(), s=10, marker='x', c='r')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.axis('off')
    plt.ioff()
    plt.savefig('Results/' + os.path.basename(file))  # Save the figure
    #plt.show()
################################################################################
#rectifier def

# Class to read labelled data
class SalaDataset(Dataset):

    def __init__(self, csv_file, transform=None):
        self.csv_file = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.csv_file) - 2

    def __getitem__(self, idx):
        idx = idx + 2
        img_path = self.csv_file.iloc[idx, 0]
        image = io.imread(img_path)
        belly_points = self.csv_file.iloc[idx, 1:]
        belly_points = np.asarray(belly_points)
        belly_points = belly_points.astype('float').reshape(-1, 2)
        sample = {'image': image, 'belly_points': belly_points}

        if self.transform:
            sample = self.transform(sample)
        return sample


# Function to show image of salamander with belly labelled
def show_points_known(image, belly_points, preds=None):
    image = torch.from_numpy(image.copy())
    plt.figure()
    plt.imshow(image)
    plt.scatter(belly_points[:, 0], belly_points[:, 1], s=10, marker='+', c='r')
    if preds is not None:
        plt.scatter(preds[:, 0].detach().numpy(), preds[:, 1].detach().numpy(), s=10, marker='x', c='b')
    plt.pause(0.001)  # pause a bit so that plots are updated
    plt.axis('off')
    plt.ioff()
    # plt.savefig('Results/' + os.path.basename(file))  # Save the figure
    plt.show()


# Function to perform interpolation
def getEquidistantPoints(belly_points, parts):
    coords = np.expand_dims(belly_points[0], axis=0)
    for i in range(len(belly_points) - 1):
        x = np.linspace(belly_points[i][0], belly_points[i + 1][0], parts + 1)
        y = np.linspace(belly_points[i][1], belly_points[i + 1][1], parts + 1)
        s = np.column_stack((x[1:], y[1:]))
        coords = np.concatenate((coords, s), axis=0)
    return coords

def getEquidistantPointsSmooth(belly_points, parts):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(belly_points, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0)
    alpha = np.linspace(distance.min(), int(distance.max()), parts)
    interpolator = interpolate.interp1d(distance, belly_points, kind='cubic', axis=0)
    coords = interpolator(alpha)
    return coords
###################################################################################
#recognizer accuracy

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

#################################################################################

# Crate an instance of the Densenet
test_network = MyDenseNet()
# Load the model's parameters
test_network.load_state_dict(torch.load('final_512.pth', map_location=torch.device('cpu')))
# Put the network in evaluation mode
test_network.eval()



# Path where images to be labelled are stored
layout = [
		    [
       		 sg.Text("File Folder"),
       		 sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
       		 sg.FolderBrowse(),
       
   		    ],
]

window = sg.Window("Path where images to be labelled are stored", layout)
 
# Run the Event Loop
while True:
     event, values = window.read()
     if event == "Exit" or event == sg.WIN_CLOSED: 
         break
    # Folder name was filled in, make a list of files in the folder
     if event == "-FOLDER-":
         folder = values["-FOLDER-"]

window.close()

path = folder

# name of csv file

filename = "Results/labelled_belly_points.csv"
# writing to csv file
with open(filename, 'w') as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    # writing the fields
    csvwriter.writerow(
        ['Path', 'topjoint x', 'topjoint y', 'torso1 x', 'torso1 y', 'torso2 x', 'torso2 y', 'torso3 x', 'torso3 y',
         'bottomjoint x', 'bottomjoint y'])

    for i, file in enumerate(glob(path)):
        # Read the image
        image = io.imread(file)
        # Create a sample
        dsample = {'image': image, 'belly_points': np.zeros((5, 2))}
        # Create a Rescale object
        scale = Rescale((512, 512))
        # Rescale the sample
        dsample = scale(dsample)
        # Create a ToTensor object
        tensor = ToTensor()
        # Convert the sample to Tensor
        dsample = tensor(dsample)

        # Get the image from the sample
        image = dsample['image']
        # Pass the image through the network to obtain the predicted belly_points
        preds = test_network(image.unsqueeze(0).float())
        # Reshaping the predictions for convenience
        predicted_points = preds.reshape(-1, 2)

        # Display the predicted belly_points on the image
        plt.title(os.path.basename(file))
        show_points(image, predicted_points)



        # Write the data to the csv file
        pred = np.array(preds.detach())
        csvwriter.writerow([file, pred[0][0], pred[0][1], pred[0][2], pred[0][3], pred[0][4], pred[0][5],
                            pred[0][6], pred[0][7], pred[0][8], pred[0][9]])

##################################################################################
num_samples = 500
perc_width = 0.8

#select csv file
layout = [
		[
       		 sg.Text("CSV Folder"),
       		 sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
       		 sg.FolderBrowse(),
       
   		],
   		[
        	 sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-"), 
       	],
        [sg.ReadFormButton('OK', size=(8,2)),],
]

window = sg.Window("Select CSV file", layout)
 
# Run the Event Loop
while True:
     event, values = window.read()
     if event == "Exit" or event == sg.WIN_CLOSED: 
         break
    # Folder name was filled in, make a list of files in the folder
     if event == "-FOLDER-":
         folder = values["-FOLDER-"]
         try:
            # Get list of files in folder
            file_list = os.listdir(folder)
         except:
            file_list = []

         fnames = [
                f 
                for f in file_list 
                if os.path.isfile(os.path.join(folder, f)) 
                and f.lower().endswith((".csv"))
            ]
         window["-FILE LIST-"].update(fnames)
     elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
        except:
            pass
     elif event == 'OK':
         break

window.close()

csv_file=filename

saladataset = SalaDataset(csv_file)

for i, sample in enumerate(saladataset):
    # Get imput image and belly points
    image, belly_points = sample['image'], sample['belly_points']
    image = transform.resize(image, (512, 512))

    # Plot more points inbetween known belly points
    # coords = getEquidistantPoints(belly_points, 500) # For liner interpolation
    coords = getEquidistantPointsSmooth(belly_points, 500) # For Smooth interpolation

    # Display salamander with new points
    # show_points_known(image, coords)

    """ CODE TO PERFORM RECTIFICATION BY PROFESSOR """

    dists = [np.linalg.norm(np.array(curr) - np.array(prev)) for (prev, curr) in zip(coords[:-1], coords[1:])]
    total_dist = sum(dists)
    total_width = total_dist * perc_width
    acc_dists = np.cumsum(dists)

    # print(dists)
    sample_stepsize_in_pix = total_dist * num_samples

    num_orth_samples = math.floor(perc_width * num_samples)
    sample_coords = np.zeros([num_samples, num_orth_samples, 2], np.float32)

    for curr_sample in range(num_samples):
        curr_dist = curr_sample / num_samples * total_dist
        larger_idxs = np.argwhere(acc_dists > curr_dist)
        if len(larger_idxs) > 0:
            next_idx = int(larger_idxs[0])
            curr_idx = int(larger_idxs[0] - 1)
            if curr_idx >= 0:
                curr_coord = ((curr_dist - acc_dists[curr_idx]) / (acc_dists[next_idx] - acc_dists[curr_idx])) * coords[
                    next_idx] \
                             + ((acc_dists[next_idx] - curr_dist) / (acc_dists[next_idx] - acc_dists[curr_idx])) * \
                             coords[curr_idx]
                curr_tang_vec = coords[next_idx] - coords[curr_idx]
                curr_orth_tang_vec = np.array([-curr_tang_vec[1], curr_tang_vec[0]])
                curr_orth_tang_vec = curr_orth_tang_vec / np.linalg.norm(curr_orth_tang_vec)
                for curr_orth_sample in range(num_orth_samples):
                    curr_sample_pnt = curr_coord - total_width / 2 * curr_orth_tang_vec + curr_orth_sample / num_orth_samples * total_width * curr_orth_tang_vec
                    sample_coords[curr_sample, curr_orth_sample, :] = curr_sample_pnt

    # print(sample_coords)

    x, y = np.arange(0, image.shape[0]), np.arange(0, image.shape[1])

    img_func_r = interpolate.interp2d(y, x, np.squeeze(image[:, :, 0]), kind='cubic')
    img_func_g = interpolate.interp2d(y, x, np.squeeze(image[:, :, 1]), kind='cubic')
    img_func_b = interpolate.interp2d(y, x, np.squeeze(image[:, :, 2]), kind='cubic')

    sampled_img = np.zeros([num_samples, num_orth_samples, 3], np.float32)

    for sampled_img_r in range(num_samples):
        for sampled_img_c in range(num_orth_samples):
            sampled_img[sampled_img_r, sampled_img_c, :] = [img_func(*sample_coords[sampled_img_r, sampled_img_c, :])
                                                            for img_func in [img_func_r, img_func_g, img_func_b]]
    # Remove the top padding of zeros
    sampled_img = sampled_img[20:, :, :]
    plt.figure()
    plt.imshow(sampled_img)
    plt.axis('off')
    plt.savefig('Belly_Rectified_Smooth/belly' + str(i) + '.png')  # Save the figure
    plt.show()
    plt.pause(0.05)
    print("Images Done :", i + 1, "/", len(saladataset))

#############################################################################################
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











