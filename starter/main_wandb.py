import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from densenet import DenseNet121
from utils import make, train, test

import wandb
wandb.login()

config = dict(
    epochs=2,
    batch_size=128,
    learning_rate=0.005,
    momentum=0.9,
)

def model_pipeline(hyperparameters):
    with wandb.init(project="", config=hyperparameters):
        config = wandb.config

        model, train_loader, test_loader, criterion, optimizer = make(config)
        
        train(model, train_loader, criterion, optimizer, config)
        test(model, test_loader)


model = model_pipeline(config)