import torch
from torch import nn, flatten
from torch.nn.functional import relu
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from IPython.display import display

from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid
import os
import re

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
print(f'Using device: {device}')

# we're going to crop all images to this resolution
image_resolution = 224

# how many colour channels?
# 3 for RGB, 1 for greyscale
colour_channels = 3

# make sure the path to your image folder is correct
folder_path = '../data/texture-data'

from torchvision.transforms.v2 import Compose, ToImage, ToDtype, Grayscale, Resize, RandomCrop, Normalize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

# this transformation function will help us pre-process images during the training (on-the-fly)
transformation = Compose([   
    # convert an image to tensor
    ToImage(),
    ToDtype(torch.float32, scale=True),
    
    # resize and crop
    Resize(image_resolution),
    RandomCrop(image_resolution),

    # apply random transformations
    RandomHorizontalFlip(),
    RandomVerticalFlip(),
    RandomRotation(30),

    # normalise pixel values to be between -1 and 1 
    Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    
    # if you're training on greyscale images, convert them to greyscale
    # otherwise just do nothing
    Grayscale() if colour_channels == 1 else torch.nn.Identity()
    
])


# how many images are going to be put into the testing set, 
# e.g. 0.2 means 20% percent of images
test_size = 0.2 

# how many images will be used in one epoch, 
# this usually depend on your model / types of data / CPU or GPU's capability
batch_size = 16

# Instatiate train and test datasets
train_dataset = ImageFolder(folder_path, transform=transformation)
test_dataset = ImageFolder(folder_path, transform=transformation)

# Get length of dataset and indicies
num_train = len(train_dataset)
indices = list(range(num_train))

# Get train / test split for data points
train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)

# Override dataset classes to only be samples for each split
train_sub_dataset = torch.utils.data.Subset(train_dataset, train_indices)
test_sub_dataset = torch.utils.data.Subset(test_dataset, test_indices)

# Create training and tresing data loaders
train_loader = DataLoader(train_sub_dataset, batch_size=batch_size, num_workers=0, shuffle=True)
test_loader = DataLoader(test_sub_dataset, batch_size=batch_size, num_workers=0, shuffle=True)

# sort out the names for each classes (according to the folder names)
class_names = train_dataset.classes

print(f'{len(train_indices)} training images loaded')
print(f'{len(test_indices)} testing images loaded')
print(f'classes: {class_names}')

num_classes=len(class_names)

data, labels = next(iter(train_loader))

print(f'data shape: {data.shape}')
print(f'labels shape: {labels.shape}')


conv2d_layer = nn.Conv2d(3, 32, kernel_size=3, padding=1)
x = conv2d_layer(data)
maxpooling_layer = nn.MaxPool2d(2, 2)
x = maxpooling_layer(x)

activation_layer = nn.ReLU()
x = activation_layer(x)

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        modules = []
        modules.append(nn.Conv2d(3, 8, kernel_size=3, padding=1))
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(8, 16, kernel_size=3, padding=1))
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU())
        modules.append(nn.Conv2d(16, 32, kernel_size=3, padding=1))
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU()) 
        modules.append(nn.Conv2d(32, 32, kernel_size=3, padding=1))
        modules.append(nn.MaxPool2d(2, 2))
        modules.append(nn.ReLU()) 
        self.convolutions = nn.ModuleList(modules)
        # CHANGE THIS LINE:
        self.fc1 = nn.Linear(32 * 14 * 14, 32)
        self.fc2 = nn.Linear(32, num_classes)
        

    # Definition of the forward pass
    # Here the classifier takes an image as input and predicts an vector of probabilites
    def forward(self, x):

        # Pass input through all layers we have added
        for layer in self.convolutions:
            x = layer(x)
            
        # Flatten the output of the last convolutional layer into a 1-dimensional vector
        x = flatten(x, 1) 
        
        # Pass through the first and the second fully connected layer with relu activation function
        x = self.fc1(x)
        x = relu(x)
        x = self.fc2(x)
        
        # Output a vector of class probabilities
        return x
    
model = ConvNeuralNetwork()
model.to(device)

# Cross entropy loss
loss_function = torch.nn.CrossEntropyLoss().to(device)
# Adam optimizer
optimizer = torch.optim.Adam(model.parameters())

# we can save the model regularly
save_every_n_epoch = 5
# total number of epochs we aim for
num_epochs = 150

# keep track of the losses, we can plot them in the end
train_losses = []
test_losses = []

# save state folder
save_dir = "saved_state"
os.makedirs(save_dir, exist_ok=True)

# Find latest checkpoint
def get_latest_checkpoint(save_dir):
    files = [f for f in os.listdir(save_dir) if f.startswith("model_epoch") and f.endswith(".pt")]
    if not files:
        return None, 0
    # Extract epoch numbers
    epochs = [int(re.findall(r"model_epoch(\d+).pt", f)[0]) for f in files]
    max_epoch = max(epochs)
    latest_file = f"model_epoch{max_epoch:04}.pt"
    return os.path.join(save_dir, latest_file), max_epoch + 1  # start from next epoch

checkpoint_path, start_epoch = get_latest_checkpoint(save_dir)

if checkpoint_path:
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
else:
    print("No checkpoint found, training from scratch.")
    start_epoch = 0

print(f'Starting at epoch {start_epoch}')

for epoch in range(start_epoch, num_epochs):

    #---- Training loop -----------------------------
    train_loss = 0.0
    model.train()
    
    for i, data in enumerate(train_loader, 0):
        # Load: The training data loader loads a batch of training data and their true class labels.
        inputs, true_labels = data
        inputs = inputs.to(device)
        true_labels = true_labels.to(device)
        
        # Pass: Forward pass the training data to our model, and get the predicted classes.
        pred_labels = model(inputs)
        
        # Loss: The loss function compares the predicted classes to the true classes, and calculates the error.
        loss = loss_function(pred_labels, true_labels)
        train_loss += loss.item()
        
        # Optimise: The optimizer slightly optimises our model based on the error.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if i % 50 == 0:
            print(f'  -> Step {i + 1:04}, train loss: {loss.item():.4f}')
    
    
    #---- Testing loop -----------------------------
    test_loss = 0.0
    model.eval()
    
    with torch.inference_mode():
        test_loss = 0.0
        for i, data in enumerate(test_loader, 0):
            # Load: The testing data loader loads a batch of testing data and their true class labels.
            inputs, true_labels = data
            inputs = inputs.to(device)
            true_labels = true_labels.to(device)
            
            # Pass: Forward pass the testing data to our model, and get the predicted classes.
            pred_labels = model(inputs)
            
            # Loss: The loss function compares the predicted classes to the true classes, and calculates the error.
            loss = loss_function(pred_labels, true_labels)
            test_loss += loss.item()
    
    
    #---- Report some numbers -----------------------------
    
    # Calculate the cumulative losses in this epoch
    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)
    
    # Added cumulative losses to lists for later display
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    print(f'Epoch {epoch + 1}, train loss: {train_loss:.3f}, test loss: {test_loss:.3f}')
    
    # save our model every n epoch
    if (epoch+1) % save_every_n_epoch==0:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
    }, f'{save_dir}/model_epoch{epoch:04}.pt')
        
# save the model at the end of the training process
torch.save(model.state_dict(), f'model_final.pt')



print("training finished, model saved to 'model_final.pt'")

# make sure the parameters are the same as when the model is created
eval_model = ConvNeuralNetwork()

# load the saved model, make sure the path is correct
eval_model.load_state_dict(torch.load('model_final.pt'))

eval_model.to(device)
eval_model.eval() 

num_samples = 0
num_correct = 0

with torch.inference_mode():
    for i, data in enumerate(test_loader, 0):
        # Load: The testing data loader loads a batch of testing data and their true class labels.
        inputs, true_labels = data
        inputs = inputs.to(device)
        true_labels = true_labels.to(device)

        # Pass: Forward pass the testing data to our model, and get the predicted classes.
        pred_labels = eval_model(inputs)
        pred_labels = torch.argmax(pred_labels, dim=1)
        
        num_correct += pred_labels.size(0) - torch.count_nonzero(pred_labels - true_labels)
        num_samples += pred_labels.size(0) 
        
accuracy = num_correct / num_samples
print(f'correct samples: {num_correct}  \ntotal samples: {num_samples}  \nmodel accuracy: {accuracy:.3f}')


import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,3))
plt.plot(train_losses, label='training loss')
plt.plot(test_losses, label = 'validation loss')
plt.xticks(np.arange(len(train_losses)))
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='upper right')
plt.show()

