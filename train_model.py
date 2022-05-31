import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import DNN
import data

class train_model:
    def Loss(y, y_pred):
        return torch.mean(((y - y_pred)/y)**2)

    def train(train_data, net, optimizer, epochs = 300):
        losses =[]
        for epoch in range(epochs):
            running_loss = 0.0
            for i, batch in enumerate(train_data.generate_data()):
                X, Y = batch
                inputs = torch.Tensor(X/np.mean(X))
                labels = torch.Tensor(Y)
                del X, Y
                outputs =net(inputs)
                optimizer.zero_grad(set_to_none=True)
                loss = train_model.Loss(labels, outputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            print('Epoch ' + str(epoch+1) +': ' + str(loss.item()))
        
            losses.append(loss.item())
        print('Finished Training')
        return losses