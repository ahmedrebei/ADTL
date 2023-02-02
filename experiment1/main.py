import numpy as np
import matplotlib.pyplot as plt

from models import LSTM_Model
from train import Train

from utils import fisher_distance
from data_utils import create_loaders, simulate_data



train_window = 36
horizon = 12

input_dimension = 1
hidden_dimension = 40
batch_size = 256

device = 'cpu'

merged_data = simulate_data(dataset_number = 10, weekend_gain = 1.2, number_of_weeks = 1)


train_loader, validation_loader = create_loaders(
    merged_data[0], train_window, horizon, batch_size)
model_0 = LSTM_Model(input_dimension, hidden_dimension, horizon)
train_object_0 = Train(model_0, train_loader, validation_loader)
train_object_0.train(num_epochs=50, verbose=True, verbose_every=1)

distances_list_0 = []
for i in range(0, len(merged_data)):
    print(f'computing distance {i}')
    t_loader, v_loader = create_loaders(merged_data[i], train_window, horizon, batch_size)
    distances_list_0.append(compute_distance(train_object_0.model, validation_loader, v_loader, batch_size, batch_size))


train_loader, validation_loader = create_loaders(
    merged_data[5], train_window, horizon, batch_size)
model_5 = LSTM_Model(input_dimension, hidden_dimension, horizon)
train_object_5 = Train(model_5, train_loader, validation_loader)
train_object_5.train(num_epochs=50, verbose=True, verbose_every=1)

distances_list_5 = []
for i in range(0, len(merged_data)):
    print(f'computing distance {i}')
    t_loader, v_loader = create_loaders(merged_data[i], train_window, horizon, batch_size)
    distances_list_5.append(compute_distance(train_object_5.model, train_loader, t_loader, batch_size, batch_size))








fig, axes = plt.subplots(1, 2,  figsize=(16, 4))

axes[0].bar(range(dataset_number), distances_list_0)
axes[0].set_xlabel('Dataset Number')
axes[0].set_ylabel('Task Affinity Score')
axes[0].set_title('Fig a: model trained on dataset 1')

axes[1].bar(range(dataset_number), distances_list_5)
axes[1].set_xlabel('Dataset Number')
axes[1].set_ylabel('Task Affinity Score')
axes[1].set_title('Fig b: model trained on dataset 5')

# plt.savefig('figure.png')
plt.show()