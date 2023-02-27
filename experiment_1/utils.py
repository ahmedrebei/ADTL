import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.autograd import Variable


# fisher distance
def fisher_matrix(model, data, batch_size_test):

    fisher = {}
    params = {key: param for key, param in model.named_parameters()
              if param.requires_grad}

    for key, param in deepcopy(params).items():
        param.data.zero_()
        fisher[key] = Variable(param.data)

    model.eval()
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

        for key, param in model.named_parameters():
            fisher[key].data += (param.grad.data ** 2).mean(0)
    return {key: param for key, param in fisher.items()}



def fisher_distance(model, source_data, target_data, source_batch_size, target_batch_size):
    fisher_source = fisher_matrix(model, source_data, source_batch_size)
    fisher_target = fisher_matrix(model, target_data, target_batch_size)

    distance = 0
    for key, param in model.named_parameters():
        distance += 0.5 * \
            np.sum(
                ((fisher_source[key] ** 0.5 - fisher_target[key] ** 0.5) ** 2).cpu().numpy())
    return distance

# metrics
def MAPE(Y_actual, Y_Predicted):
    mape = (torch.abs(Y_actual-Y_Predicted)/Y_actual).mean(axis=1)
    return mape


def MAE(Y_actual, Y_Predicted):
    return torch.abs(Y_actual-Y_Predicted).mean(axis=1)


def RMSE(Y_actual, Y_Predicted):
    return torch.sqrt(torch.square(Y_actual-Y_Predicted).mean(axis=1))
