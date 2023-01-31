import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM_Model(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_units):
        super().__init__()
        
        self.lstm = nn.LSTM(input_size=input_dimension, hidden_size=hidden_dimension, num_layers=1, 
#                             dropout=0.2, 
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dimension, 20)
        self.dropout = nn.Dropout(0.25)
        self.fc2 = nn.Linear(20, num_units)
        
    def forward(self, x):
        _, (x,_) = self.lstm(x)
        print('before linear' , x.shape)
#         x = torch.squeeze(x[1])
        x = self.fc2(self.dropout(self.fc1(x)))
        return torch.squeeze(x)


class MLP_Model(nn.Module):
    def __init__(self, input_dimension, hidden_dimension, num_units):
        super().__init__()

        self.fc1 = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, num_units)
    
    def forward(self, x):
        x = torch.squeeze(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
