import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, Layers=[25, 16, 4], activation = nn.LeakyReLU(), X_mean = 0, X_std = 1, device = 'cpu'):
        super().__init__()
        self.Layers = Layers
        self.activation = activation
        self.device = device
        self.activation = activation
        self.device = device
        self.Model = self.build_model().to(device)
        self.X_mean = X_mean 
        self.X_std = X_std

    def build_model(self):
        net = nn.Sequential()
        for ii in range(len(self.Layers)-2):
            this_module = nn.Linear(self.Layers[ii], self.Layers[ii+1])
            nn.init.xavier_normal_(this_module.weight)
            net.add_module("Linear" + str(ii), this_module)
            net.add_module("Activation" + str(ii), self.activation)
        last_module = nn.Linear(self.Layers[-2], 1, True)
        net.add_module("Linear_last", last_module)
        last_activation = nn.LeakyReLU()
        net.add_module("Activation_last", last_activation)
        return net 
    
    def forward(self, X):
        X = ((X-self.X_mean)/self.X_std).t(self.device)
        return self.Model.forward(X)