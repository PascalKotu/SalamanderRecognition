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
    plt.show()


# Crate an instance of the Densenet
test_network = MyDenseNet()
# Load the model's parameters
test_network.load_state_dict(torch.load('final_512.pth', map_location=torch.device('cpu')))
# Put the network in evaluation mode
test_network.eval()

# Path where images to be labelled are stored
path = 'Test_Data/*.*'

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
        # Canvert the sample to Tensor
        dsample = tensor(dsample)

        # Get the image from the sample
        image = dsample['image']
        # Pass the image through the network to obtain the predicted belly_points
        preds = test_network(image.unsqueeze(0).float())
        # Reshapeing the predictions for convenience
        predicted_points = preds.reshape(-1, 2)

        # Display the predicted belly_points on the image
        plt.title(os.path.basename(file))
        show_points(image, predicted_points)



        # Write the data to the csv file
        pred = np.array(preds.detach())
        csvwriter.writerow([file, pred[0][0], pred[0][1], pred[0][2], pred[0][3], pred[0][4], pred[0][5],
                            pred[0][6], pred[0][7], pred[0][8], pred[0][9]])
