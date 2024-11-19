import quantus
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch.optim as optim
import captum
from captum.attr import IntegratedGradients, Lime

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForwardNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    def forward(self, x):
        return self.network(x)
    
    def fit(self, train_data, train_labels, epochs=10, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(train_data)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

MODEL_PATH = "feedforward.pth"

# Define the model
model = FeedForwardNetwork(input_size=51, hidden_size=32, output_size=2)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

#check what XAI models are available
available_methods = quantus.AVAILABLE_XAI_METHODS_CAPTUM
print("Available XAI Methods:", available_methods)


# Import the training and testing data .pt files
train_data = torch.load("X_train.pt").float()
train_labels = torch.load("y_train.pt").long()
test_data = torch.load("X_test.pt").float()
test_labels = torch.load("y_test.pt").long()

#print their shapes
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

model.fit(train_data, train_labels, epochs=10, learning_rate=0.001)

#initialize the XAI object
lime = Lime(forward_func=model.forward, interpretable_model=model)

print("LIME")
attr_coefs = lime.attribute(inputs=train_data, target= 1)


