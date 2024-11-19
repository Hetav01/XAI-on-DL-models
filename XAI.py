import quantus
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

MODEL_PATH = "feedforward.pth"

# Define the model
model = FeedForwardNetwork(input_size=51, hidden_size=32, output_size=2)

model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

#check what XAI models are available
available_methods = quantus.AVAILABLE_XAI_METHODS_CAPTUM
print("Available XAI Methods:", available_methods)