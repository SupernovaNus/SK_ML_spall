import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import models


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
root_dir = config["dataset"]["root_dir"]
output_dir = config["output"]["output_dir"] + job_id

ch_id = config["dataset"]["channel_id"]

backbone = config["model"]["backbone"]
batch_size = config["train"]["batch_size"]
epochs = config["train"]["num_epochs"]
learning_rate = config["train"]["lr"]

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Define a custom dataset
class MyDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.labels_frame.iloc[idx, 0])
        image_arr = np.load(img_name)
        
        ## If we only train on single channel:
        if ch_id != 'all':
            image_arr = image_arr[ch_id].reshape(1, image_arr.shape[1], image_arr.shape[2])
            
        # Convert the shape of arrays in .npy files from the original (ch, h, w) to (h, w, ch).
        # As '.ToTensor()' in our transform will make a (h, w, ch) array to a (ch, h, w) tensor
        # If the original .npy files have already contained array with shape (h, w, ch), we don't need this line.
        image_arr = np.transpose(image_arr, (1, 2, 0))  
        
        label = int(self.labels_frame.iloc[idx, 1])
        if self.transform:
            image_arr = self.transform(image_arr)
        return image_arr, label


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Deal with the images that only contain 0 pixel values
# def normalize_image(x):
#     min_val = x.min()
#     max_val = x.max()
#     if max_val - min_val == 0:  # Check if the image has the same pixel values everywhere
#         return x * 0  # Set the entire image to 0
#     else:
#         return (x - min_val) / (max_val - min_val)

# Transformations including converting the image to 5 channels and to Tensor
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Lambda(normalize_image),  # normalized [0, 1]
])


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Set the random seed
torch.manual_seed(42)

# Creating dataset
dataset = MyDataset(csv_file=root_dir + "/labels.csv", root_dir=root_dir, transform=transform)

# Splitting dataset into training and validation sets (80-20 split)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Creating DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from DNN import ResNet152, VGG16

class My_Model(nn.Module):
    def __init__(self, model_type, num_classes=2):
        super(My_Model, self).__init__()
        if model_type == 'resnet152':
            self.model = ResNet152(num_classes)
        elif model_type == 'vgg16':
            self.model = VGG16(num_classes, ch_id)
        else:
            raise ValueError("Invalid model_type. Expected 'resnet152' or 'vgg16'")

    def forward(self, x):
        x = self.model(x)
        return x

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Let's use", torch.cuda.device_count(), "GPUs!")


# model = ResNet152()
model = My_Model(backbone)
model = model.to(device)
model = nn.DataParallel(model)

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

criterion  = nn.CrossEntropyLoss()
# criterion  = nn.BCEWithLogitsLoss()
    
#------------------------------------------
    
global_progress = tqdm(range(0, epochs), desc='Training')

epoch_list = []
train_loss_list = []
test_loss_list = []

train_label = [] # Keep track of the label list of all epochs (each epoch is different because we shuffle the training set)
train_pred = []  # Keep track of the prediction list of all epochs

test_label = []  # Keep track of the label list (each epoch is the same because we didn't shuffle the test set)
test_pred = []   # Keep track of the prediction list of all epochs


for epoch in global_progress:

    model.train()

    train_local_progress = tqdm(
        train_dataloader,
        desc = f'Epoch {epoch}/{epochs}',
    )

    correct_predictions, total_predictions, train_loss = 0, 0, 0
    train_label_epoch = torch.empty((0, 2)).to(device)
    train_pred_epoch = torch.empty((0, 2)).to(device)

    for _, (images, labels) in enumerate(train_local_progress):  

        """
        images.shape = torch.Size([batch_size, channels, h, w])
        labels.shape = torch.Size([batch_size])   (long tensor)
        """
        outputs = model(
            images.float().to(device, non_blocking=True))
        """ outputs.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
        
        # One-hot encoding the long-tensor labels
        labels = nn.functional.one_hot(labels, num_classes=2).to(device, non_blocking=True)
        # 'CrossEntropyLoss' only accepts input tensors in float type
        labels = labels.float()
        """ labels.shape = torch.Size([batch_size, num_classes]) """

        # Compute loss for each batch
        """
        'CrossEntropyLoss' will apply a Softmax to the output (one-hot encoded),
        so the outputs here doesn't need an additional Softmax function
        """ 
        batch_loss = criterion(outputs, labels).mean()
        """ batch_loss.shape = torch.Size([]) """
        train_loss += batch_loss

        # Backward pass and optimize
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

        m = nn.Softmax(dim=1)
        outputs_prob = m(outputs)
        """ outputs_prob.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
        
        # Count the correct presictions and total predictions from each batch
        labels_long = labels.argmax(dim=1)
        outputs_prob_long = outputs_prob.argmax(dim=1)
        correct_predictions += (outputs_prob_long == labels_long).sum().item()
        total_predictions += labels_long.size(0)
        
        # Concatenate the labels and predictioins from each batch
        train_label_epoch = torch.cat((train_label_epoch, labels), dim=0)
        train_pred_epoch = torch.cat((train_pred_epoch, outputs_prob), dim=0)

    # Accuracy of each epoch (assuming threshold = 0.5) 
    accuracy = correct_predictions/total_predictions
    # Number of batch in training set
    train_num_batch = train_size/batch_size
    # Average the total loss of all batches
    train_loss /= train_num_batch
    print(f'Epoch {epoch}: Training loss: {train_loss:.6f} / Accuracy: {accuracy:.3f}')

    train_label.append(train_label_epoch.detach().cpu().numpy())
    train_pred.append(train_pred_epoch.detach().cpu().numpy()) 
    np.save(f'{output_dir}/train_label_epoch.npy', train_label)
    np.save(f'{output_dir}/train_pred_epoch.npy', train_pred)

# -------------------------------------------------------------------------------------------------

    # Test the model on test set
    test_local_progress = tqdm(
        test_dataloader,
        desc = f'Epoch {epoch}/{epochs}',
    )

    
    correct_predictions, total_predictions, test_loss = 0, 0, 0
    
    with torch.no_grad():
        model.eval()
        
        if epoch == 0:
            test_label_0 = torch.empty((0, 2)).to(device)
        test_pred_epoch = torch.empty((0, 2)).to(device)

        for _, (images, labels) in enumerate(test_local_progress):  
            
            """
            images.shape = torch.Size([batch_size, channels, h, w])
            labels.shape = torch.Size([batch_size])   (long tensor)
            """
            outputs = model(
                images.float().to(device, non_blocking=True))
            """ outputs.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
            
            # One-hot encoding the long-tensor labels
            labels = nn.functional.one_hot(labels, num_classes=2).to(device, non_blocking=True)
            # 'CrossEntropyLoss' only accepts input tensors in float type
            labels = labels.float()
            """ labels.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
        
            # Compute loss for each batch
            """
            'CrossEntropyLoss' will apply a Softmax to the output (one-hot encoded),
            so the outputs here doesn't need an additional sigmoid function
            """ 
            batch_loss = criterion(outputs, labels).mean()
            """ batch_loss.shape = torch.Size([]) """
            test_loss += batch_loss        

            m = nn.Softmax(dim=1)
            outputs_prob = m(outputs)
            """ outputs_prob.shape = torch.Size([batch_size, num_classes]); one-hot encoded """
            
            # Count the correct presictions and total predictions from each batch
            labels_long = labels.argmax(dim=1)
            outputs_prob_long = outputs_prob.argmax(dim=1)
            correct_predictions += (outputs_prob_long == labels_long).sum().item()
            total_predictions += labels_long.size(0)
            
            # Concatenate the labels and predictioins from each batch
            if epoch == 0:
                test_label_0 = torch.cat((test_label_0, labels), dim=0)
            test_pred_epoch = torch.cat((test_pred_epoch, outputs_prob), dim=0)

        # Accuracy of each epoch (assuming threshold = 0.5) 
        accuracy = correct_predictions/total_predictions
        # Number of batch in test set
        test_num_batch = test_size/batch_size
        # Average the total loss of all batches
        test_loss /= test_num_batch
        
        print(f'Epoch {epoch}: Test loss: {test_loss:.6f} / Accuracy: {accuracy:.3f}')

        if epoch == 0:
            test_label.append(test_label_0.detach().cpu().numpy())
            np.save(f'{output_dir}/test_label_epoch.npy', test_label)

        test_pred.append(test_pred_epoch.detach().cpu().numpy()) 
        np.save(f'{output_dir}/test_pred_epoch.npy', test_pred)
                

    # Record all losses, make plot, and save the outputs
    epoch_list.append(epoch)
    train_loss_list.append(train_loss.detach().cpu().numpy())
    test_loss_list.append(test_loss.detach().cpu().numpy())

    plt.plot(epoch_list, train_loss_list, color= 'navy')
    plt.plot(epoch_list, test_loss_list, color= 'coral')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.savefig(f'{output_dir}/plotter.pdf')

    
    output_dict = {
        'Epoch': epoch_list,
        'Train_loss': train_loss_list,
        'Test_loss': test_loss_list,
    }
    df = pd.DataFrame(output_dict)
    df.to_csv(f'{output_dir}/output_history.csv', index=False)

    
    if epoch == 0:
        lowest_test_loss = test_loss
        
    # Save the model that achieves the lowest loss on the test set
    if test_loss < lowest_test_loss:
        lowest_test_loss = test_loss
        model_save_path = f'{output_dir}/epoch_{epoch}_loss_{lowest_test_loss:.6f}.pt'
        torch.save(model, model_save_path)