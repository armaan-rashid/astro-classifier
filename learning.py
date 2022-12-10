import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from preprocess import ChartBioData, LitChartBioData
from torch.utils.tensorboard import SummaryWriter

MAX_EPOCHS = 10

class NeuralNetwork(nn.Module):
    """Barebones 2-layer neural network class."""
    def __init__(self, num_features, num_hidden) -> None:
        super().__init__()
        self.net =  nn.Sequential(
            nn.Linear(num_features, num_hidden),
            nn.ReLU(),
            nn.Linear(num_hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)


class DeepChartClassifier(pl.LightningModule):
    """
    This class performs chart classification with a deep neural network. 
    Very similar to the Logistic Regression model in baseline, but with 
    a two layer neural net: ultimately outputs a 0 or 1.
    """
    def __init__(self, neural_net: NeuralNetwork):
        super().__init__()
        self.neural = neural_net
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.to(torch.float32)
        y = y.to(torch.float32)    # convert dtypes as needed

        y_hat = self.neural(x)   # run through network
        loss = F.binary_cross_entropy(y_hat, y)
        self.log('training_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer


    


def train_NN_Lit(num_hidden, max_tokens = 1024, 
                 train_filepath = None, test_filepath = None):
    """
    Run the two layer neural net for classification with
    passed in number of hidden neurons. Plot the training
    loss and test accuracy.
    
    Args:
        (num_hidden) number of hidden nodes
        (int) max_tokens: if processing the data from scratch, number of tokens to process bios with
        (str) train_filepath, test_filepath: filepaths with stored data: if these are here, then max_tokens ignored!
    """
    data_module = LitChartBioData(train_filepath='train_features_len_1024_tensor.pt', test_filepath='test_features_len_1024_tensor.pt')
    trainer = pl.Trainer(max_epochs=10)
    model = DeepChartClassifier(NeuralNetwork(data_module.num_features, num_hidden))
    for param in model.parameters():
        param.requires_grad_()
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


def train_NN(num_hidden, max_tokens = 1024, 
            train_filepath = None, test_filepath = None):
    """
    Raw PyTorch implementation of the neural network.
    Args:
        (int) num_hidden: hidden neurons in NeuralNetwork
        (int) max_tokens: if processing the data from scratch, number of tokens to process bios with
        (str) train_filepath, test_filepath: filepaths with stored data: if these are here, then max_tokens ignored!
    """
    train_dataset = ChartBioData('train', tensor_filepath=train_filepath)
    model = NeuralNetwork(train_dataset.num_chart_features + train_dataset.num_bio_features, num_hidden)
    train_loader = DataLoader(train_dataset, 100)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.BCELoss()

    writer = SummaryWriter(f'runs/NN_trainer_testing')


    # training loop!
    for epoch in range(MAX_EPOCHS):
        running_loss = 0.
        last_loss = 0.
        for i, batch in enumerate(train_loader):
            features, labels = batch
            optimizer.zero_grad()

            # forward! then loss! then back!
            model.train(True)
            output = model(features)
            y_hat = (output > 0.5).float()
            loss = loss_fn(y_hat, labels.to(torch.float))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000 # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = i * len(train_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.



def main():
    # train_NN(12, train_filepath='train_features_len_1024_tensor.pt', test_filepath='test_features_len_1024_tensor.pt')
    train_NN_Lit(12, train_filepath='train_features_len_1024_tensor.pt', test_filepath='test_features_len_1024_tensor.pt')


if __name__ == '__main__':
    main()

