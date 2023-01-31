import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
from torch.autograd import Variable

class Data_generator(Dataset):
    def __init__(self, X, train_window, horizon):
        super().__init__()
        self.X = X
        self.train_window = train_window
        self.horizon = horizon 
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        element = self.X[idx]
        return element[:-self.horizon, :], element[-self.horizon:, 0]

def sliding_window_transform(train_window, horizon, data):
    transform_np_data = np.lib.stride_tricks.sliding_window_view(data, train_window+horizon, axis=0, writeable=True)
    return torch.from_numpy(np.expand_dims(transform_np_data,2)).float()

def create_loaders(merged_data, train_window, horizon, batch_size, split_ratio = 0.2):

    data = sliding_window_transform(train_window, horizon, merged_data)

    train_limit = int(data.shape[0]*(1-split_ratio))

    train_data = data[:train_limit]
    train_generator = Data_generator(train_data, train_window, horizon)  
    train_loader = DataLoader(train_generator, batch_size=batch_size,shuffle=True, num_workers=1)


    if split_ratio:
        validation_data = data[train_limit:]
        validation_generator = Data_generator(validation_data, train_window, horizon)  
        validation_loader = DataLoader(train_generator, batch_size=validation_generator.__len__(),shuffle=True, num_workers=1)
        
        return train_loader, validation_loader
    return train_loader, None



### fisher    
def diag_fisher(model, data, batch_size_test):
    '''
    model is from base task, data is from target task.
    data: dataloader form
    '''
    precision_matrices = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = Variable(p.data)

    model.eval()

    # loss function during testing/computing the Fisher Information matrices
    loss_fn = nn.MSELoss()

    for _, item in enumerate(data):
    
        inputs, target = item
        inputs = inputs.type(torch.float32)
        target = target.type(torch.float32)    
        
        model.zero_grad()
        outputs = model(inputs)
        outputs = model(inputs)
        loss = loss_fn(outputs, target)
        loss.backward()

        # Compute the Fisher Information as (p.grad.data ** 2)
        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)  # only diagonal

    # Fisher Information
    precision_matrices = {n: p for n, p in precision_matrices.items()}

    return precision_matrices


def Fisher_distance(fisher_matrix_source, fisher_matrix_target, model):
    '''
    fisher_matrix_source: output of diag_fisher(model_base, data_base)
    fisher_matrix_target: output of diag_fisher(model_base, data_target)
    model: neural network trained on the source
    Fisher_distance: return the square of distance
    '''
    distance = 0
    for n, p in model.named_parameters():
        distance += 0.5 * np.sum(((fisher_matrix_source[n] ** 0.5 - fisher_matrix_target[n] ** 0.5) ** 2).cpu().numpy())
    return distance


def compute_distance(model, source_task, target_task, batch_size_test_source, batch_size_test_target):
    '''
    model: trained neural network
    source_task: the data on which the neural network was trained
    target_task: the data that we want to estimate how close is it to the original source task
    '''

    fisher_source = diag_fisher(model, source_task, batch_size_test_source)
    fisher_target = diag_fisher(model, target_task, batch_size_test_target)

    distance = Fisher_distance(fisher_source, fisher_target, model)
    return distance

####metrics
def MAPE(Y_actual,Y_Predicted):
    mape = (torch.abs(Y_actual-Y_Predicted)/Y_actual).mean(axis=1)
    return mape
def MAE(Y_actual,Y_Predicted):
    return torch.abs(Y_actual-Y_Predicted).mean(axis=1)
def RMSE(Y_actual,Y_Predicted):
    return torch.sqrt(torch.square(Y_actual-Y_Predicted).mean(axis=1))