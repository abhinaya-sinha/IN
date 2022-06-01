import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import DNN
import data
from sklearn.model_selection import train_test_split

class train_model:
    def Loss(y, y_pred):
        return torch.mean(((y - y_pred)/y)**2)

    def train(train_data, net, optimizer, epochs = 300):
        losses =[]
        test_losses = []
        for epoch in range(epochs):
            epoch_loss = []            
            epoch_testloss = []
            running_loss = 0.0
            for i, batch in enumerate(train_data.generate_data()):
                X_0, Y_0 = batch
                X, X_test, Y, Y_test = train_test_split(X_0, Y_0, test_size=0.33)
                inputs = torch.Tensor(X/np.mean(X))
                labels = torch.Tensor(Y)
                test_inputs = torch.Tensor(X_test/np.mean(X_test))
                test_labels = torch.Tensor(Y_test)
                del X, Y, X_test, Y_test, X_0, Y_0
                outputs =net(inputs)
                optimizer.zero_grad(set_to_none=True)
                loss = train_model.Loss(labels, outputs)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                epoch_loss.append(loss.item())
                epoch_testloss.append(train_model.Loss(test_labels, net(test_inputs)).item())
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
            print('Epoch ' + str(epoch+1) +': ' + str(loss.item()))
            losses.append(np.mean(epoch_loss))
            test_losses.append(np.mean(epoch_testloss))
        print('Finished Training')
        return losses, test_losses