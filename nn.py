import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

from random import randint

class Net(nn.Module):
    '''
    Neural network class.
    Architecture:
        Convolution (16 layers, 14x14)
        Maxpool     (16 layers, 7x7)
        Convolution (36 layers, 7x7)
        Two nonlinear activation functions relu.
    '''

    def __init__(self):
        super(Net, self).__init__()
        
        # Layer 1
        #   Image   (?, 14, 14, 1) + padding (+2)
        #   Conv    (?, 14, 14, 12)
        #   Pool    (?, 7,  7,  12)
        self.conv1    = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.maxpool  = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2    = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(36*7*7,100)
        self.fc2 = nn.Linear(100,5)
    
    
    def init_data(self,data,cuda):
        if cuda:
            self.inputs_train  = torch.from_numpy(data.x_train).cuda()
            self.targets_train = torch.from_numpy(data.y_train).cuda().long()
            
            self.inputs_test  = torch.from_numpy(data.x_test).cuda()
            self.targets_test = torch.from_numpy(data.y_test).cuda().long()
        else:
            self.inputs_train  = torch.from_numpy(data.x_train)
            self.targets_train = torch.from_numpy(data.y_train).long()

            self.inputs_test  = torch.from_numpy(data.x_test)
            self.targets_test = torch.from_numpy(data.y_test).long()
            
    def forward(self,x):
        out = func.relu(self.conv1(x))
        out = self.maxpool(out)
        out = func.relu(self.conv2(out))
        out = out.view(out.size(0),-1)  #flatten for FC
        out = func.relu(self.fc1(out))
        out = self.fc2(out)
        return out
        
    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
    
    def backprop(self, loss, optimizer):
        self.train()
        
        n_train    = 2000
        args_batch = randint(0, len(self.inputs_train)-n_train)
        
        outputs= self(self.inputs_train[args_batch: args_batch+n_train])        
        obj_val= loss(outputs, self.targets_train[args_batch: args_batch+n_train])
        optimizer.zero_grad()
        obj_val.backward()
        optimizer.step()
        return obj_val.item()
    
    def test(self, loss):
        self.eval()
        with torch.no_grad():
            outputs= self(self.inputs_test)
            cross_val= loss(outputs, self.targets_test)  #self.forward(inputs)
        return cross_val.item()
    
    def predict_test(self):
        self.eval()
        with torch.no_grad():
            inputs  = self.inputs_test
            targets = self.targets_test
            outputs = self(inputs)
            
            predictions = np.argmax(np.array(outputs.cpu()),axis=1)
            actual      = np.array(targets.cpu(), dtype=int)
            
            list_accurate_predictions = np.zeros(5)
            list_all_predictions      = np.zeros(5)
            
            for i in range(0,len(targets)):
                n = predictions[i]
                if predictions[i] == actual[i]:
                    list_accurate_predictions[n] += 1
                    list_all_predictions[n]      += 1
            
                else:
                    list_all_predictions[n]      += 1
            
            
            print('Prediction Accuracies')
            for i in range(0,5):
                print('target = {}; acc = {:.4f}'.format(i*2, list_accurate_predictions[i]/list_all_predictions[i]))
            