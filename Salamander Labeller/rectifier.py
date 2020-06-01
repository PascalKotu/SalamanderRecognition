import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import math
from scipy import interpolate


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


num_samples = 500
perc_width = 0.8

saladataset = SalaDataset(csv_file='Data/CollectedData_Brijesh.csv')

for i, sample in enumerate(saladataset):
    # Get imput image and belly points
    image, belly_points = sample['image'], sample['belly_points']
    image = transform.resize(image, (512, 512))

    # Plot more points inbetween known belly points
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
    plt.show()
    plt.pause(0.05)
