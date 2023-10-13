import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import models

import torchvision.transforms.functional as F
from transformation_lib import transform_batch

import copy


import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

import yaml

from tqdm import tqdm
from itertools import islice

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

parser = argparse.ArgumentParser()
parser.add_argument("jobID", type=str)
parser.add_argument('-c', '--config-file', required=True, type=str, help="xxx.yaml")
args = parser.parse_args()

# job_id
job_id = args.jobID

# Load config
with open(args.config_file) as file:
    config = yaml.safe_load(file)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Configuration
datapath = config["dataset"]["root_dir"]
trained_weights_path = config["model"]["trained_weights_path"]
output_dir = config["output"]["output_dir"] + job_id

backbone = config["model"]["backbone"]
batch_size = config["test"]["batch_size"]

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# SEED=12345
# _=np.random.seed(SEED)
# _=torch.manual_seed(SEED)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


test_data = ParticleImage2D(data_files = [datapath],
                             start = 0.0, # start of the dataset fraction to use. 0.0 = use from 1st entry
                             end   = 1.0, # end of the dataset fraction to use. 1.0 = use up the last entry
                            )

# We use a specifically designed "collate" function to create a batch data
from iftool.image_challenge import collate
from torch.utils.data import DataLoader

test_loader = DataLoader(test_data,
                          collate_fn  = collate,
                          shuffle     = False,
                          num_workers = 1,
                          batch_size  = batch_size
                         )

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from DNN import ResNet152, VGG16

class My_Model(nn.Module):
    def __init__(self, model_type, num_classes=4):
        super(My_Model, self).__init__()
        
        if model_type == 'vgg16':
            self.model = VGG16(num_classes)
            
        elif model_type == 'resnet152':
            # Load the pretrained ResNet-152 model
            self.model = models.resnet152(pretrained=True)
            # Modify the last fully connected layer to match the number of classes
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Linear(num_ftrs, num_classes)
            
        # elif model_type == 'vitb16':
        #     # Load the pretrained ResNet-152 model
        #     self.model = models.vit_b_16(pretrained=True)
        #     # Modify the last fully connected layer to match the number of classes
        #     num_ftrs = self.model.heads.head.in_features
        #     self.model.heads.head = nn.Linear(num_ftrs, num_classes)
        
        else:
            raise ValueError("Invalid model_type. Expected 'vgg16' or 'resnet152'")

    def forward(self, x):
        x = self.model(x)
        return x

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

model = torch.load(trained_weights_path) #, map_location=torch.device('cpu'))
if isinstance(model, torch.nn.DataParallel):
    model = model.module  # Access the original model
model.eval()


# For UMAP
model_reduced = copy.deepcopy(model)
model_reduced.model.fc2 = nn.Identity()
# Check if the model is a DataParallel model
model_reduced.eval()  # Set the model to evaluation mode

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
model = model.to(device)  # Move the model to CPU
model_reduced = model_reduced.to(device)  # Move the model to CPU

# Assuming you have a DataLoader `data_loader` for your dataset
batch_image_list = []
features_list = []
outputs_list = []
labels_list = []

with torch.no_grad():  # No need to track the gradients
    for batch in tqdm(test_loader):
        batch_images = transform_batch(batch['data'], backbone).to(device)
        batch_labels = batch['label'].to(device)
        
        # Perform a forward pass through the model
        outputs = model(batch_images)
        outputs_fc1 = model_reduced(batch_images)
        
        # Append the batch images, features and labels to the lists
        # batch_image_list.append(batch_images.cpu().numpy())
        outputs_list.append(outputs.cpu().numpy())
        labels_list.append(batch_labels.cpu().numpy())
        features_list.append(outputs_fc1.cpu().numpy())

# Concatenate all the batch images, features and labels from all batches
# batch_images = np.concatenate(batch_image_list, axis=0)
test_outputs = np.concatenate(outputs_list, axis=0)
test_pred = np.argmax(test_outputs, axis=1)
test_labels = np.concatenate(labels_list, axis=0)
test_features = np.concatenate(features_list, axis=0)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import umap

# Perform UMAP dimensionality reduction
reducer = umap.UMAP()
test_embedding = reducer.fit_transform(test_features)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# np.save(f'{output_dir}/batch_images.npy', batch_images)
np.save(f'{output_dir}/test_outputs.npy', test_outputs)
np.save(f'{output_dir}/test_pred.npy', test_pred)
np.save(f'{output_dir}/test_labels.npy', test_labels)
np.save(f'{output_dir}/test_features.npy', test_features)
np.save(f'{output_dir}/test_embedding.npy', test_embedding)
np.save(f'{output_dir}/test_embedding_eg.npy', test_embedding_eg)

