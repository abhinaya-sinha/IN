import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.optim as optim
from DNN import DNN
from data import CSVData
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
                inputs = F.normalize(torch.Tensor(X))
                labels = torch.Tensor(Y)
                test_inputs = F.normalize(torch.Tensor(X_test))
                test_labels = torch.Tensor(Y_test)
                del X, Y, X_test, Y_test, X_0, Y_0
                outputs =net(inputs)
                optimizer.zero_grad()
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

if __name__ == "__main__":
    features = ['pz_in1', 
    'pid_in1', 
    'pid_in2', 
    'px_out1',
    'py_out1',
    'pz_out1',
    'e_out1',
    'pid_out1',
    'px_out2',
    'py_out2',
    'pz_out2',
    'e_out2',
    'pid_out2',
    'px_out3',
    'py_out3',
    'pz_out3',
    'e_out3',
    'pid_out3',
    'px_out4',
    'py_out4',
    'pz_out4',
    'e_out4',
    'pid_out4',
    'Mtarget',
    'Gtarget',]
    label = 'f_rwt'

    VLQData = CSVData(batch_size=1024, features_name=features, labels_name=label, file_names=['./train_'+str(i)+'.csv' for i in range(0,10)])

    net = DNN(device = 'cuda').build_model()
    optimizer = optim.Adam(net.parameters(), lr=0.1)
    epochs =200

    losses, test_losses = train_model.train(train_data=VLQData, net = net, optimizer=optimizer, epochs=epochs)

    torch.save(net, 'first_model.pt')


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(np.linspace(0, epochs, epochs), losses, label = 'train loss')
    ax.plot(np.linspace(0,epochs, epochs), test_losses, label = 'test loss')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig('plot.png')
    plt.show(block=True)
