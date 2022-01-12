# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 19:24:24 2021

@author: mrinal

"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

#%% Set GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Load dataset
train_dataset = dataset.MNIST(root='./data',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=False)
test_dataset = dataset.MNIST(root='./data',
                             train=False,
                             transform=transforms.ToTensor())

#%% Data description
pick = np.random.randint(0, 60000) # pick a number between 0 to 60000-1
sample = train_dataset[pick] 

# Sample is a tuple containing both the data and the label
sam_data = sample[0] # torch.Size([1, 28, 28])
sam_data = torch.squeeze(sam_data) # torch.Size([28, 28])
sam_data = sam_data.numpy() # converting to numpy array
sam_label = sample[1]
print('Number displaying: ', sam_label)

plt.figure()
plt.imshow(sam_data, cmap='gray')
plt.show()

#%% Prepare dataset
batch_size = 100
num_iters = 3000
num_batches = len(train_dataset)/batch_size
num_epochs = int(num_iters/num_batches)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

#%% Create model 
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        out = self.linear(x)
        return out

#%% Create an instance
input_dim = 28*28
output_dim = 10 # 0-9, so 10 classes
model = LogisticRegressionModel(input_dim, output_dim).to(device)

# Loss function  
criterion = nn.CrossEntropyLoss() # Note: it 1st calculate softmax, then does cross entropy

# Learning rate
learning_rate = 0.001

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

#%% Training
losses = []
iter = 0
for epoch in range(num_epochs):
    # Collect each batch
    for i, (images, labels) in enumerate(train_loader):
        
        # Create Variables
        images = Variable(images.view(-1, 28*28)) # creating a Variable (like a container) for images.           
        labels = Variable(labels) # Variable allows us accumulating gradients
        images = images.to(device) # move to gpu if available
        labels = labels.to(device) # move to gpu if available
            
        # Forward pass
        outputs = model(images) # generate output/logits
        
        # Calculate loss (softmax --> cross entropy)
        loss = criterion(outputs, labels)
        losses.append(loss)
        
        # Backward pass and optimizer
        optimizer.zero_grad() # Clear previous gradients
        loss.backward() # Calculate gradients w.r.t. parameters               
        optimizer.step() # Update parameters (or weights)
        
        iter += 1
        
        # Test/Validation step
        # Reference: Practical Deep Learning by Deep Learning Wizard
        if iter % 500 == 0:
            with torch.no_grad(): # meaning we don't want backward propagation
                # Calculate accuracy
                correct = 0
                total = 0
                
                # Iterate through test dataset
                for images, labels in test_loader:
                    
                    # Create Variables
                    images = Variable(images.view(-1, 28*28)) # creating a Variable (like a container) for images.           
                    labels = Variable(labels) # Variable allows us accumulating gradients
                    images = images.to(device) # move to gpu if available
                    labels = labels.to(device) # move to gpu if available
                    
                    # Forward pass
                    outputs = model(images) # torch.Size([100, 10])
                    
                    # Find the output node that has maximum probability
                    predictions = torch.argmax(outputs.data, dim=1)
                    
                    # Total number of labels
                    total += labels.size(0)
                    
                    # Total correct predictions
                    correct += (predictions.cpu() == labels.cpu()).sum()
                
            accuracy = 100 * correct / total
            
            # Print Loss
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))
