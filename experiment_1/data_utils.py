import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models import LSTM_Model
from train import Train
from utils import fisher_distance


# data functions
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
    transform_np_data = np.lib.stride_tricks.sliding_window_view(
        data, train_window+horizon, axis=0, writeable=True)
    return torch.from_numpy(np.expand_dims(transform_np_data, 2)).float()


def create_loaders(merged_data, train_window, horizon, batch_size, split_ratio=0.2):

    data = sliding_window_transform(train_window, horizon, merged_data)

    train_limit = int(data.shape[0]*(1-split_ratio))

    train_data = data[:train_limit]
    train_generator = Data_generator(train_data, train_window, horizon)
    train_loader = DataLoader(
        train_generator, batch_size=batch_size, shuffle=True, num_workers=1)

    if split_ratio:
        validation_data = data[train_limit:]
        validation_generator = Data_generator(
            validation_data, train_window, horizon)
        validation_loader = DataLoader(
            train_generator, batch_size=validation_generator.__len__(), shuffle=True, num_workers=1)

        return train_loader, validation_loader
    return train_loader, None

# data simulation


def simulate_data(dataset_number=10, weekend_gain=1.2, number_of_weeks=1):

    number_of_samples_per_day = 48
    number_of_samples_per_week = 48 * 7
    week_length = number_of_samples_per_day * 7
    weekend_period_length = number_of_samples_per_day * 2

    weekend_gain_list = [1]*(number_of_samples_per_week -
                             weekend_period_length)+[weekend_gain]*weekend_period_length
    week_simulation = (1+np.sin(np.linspace(-np.pi/2, 14*np.pi -
                                            np.pi/2, week_length)))*weekend_gain_list

    nbr_of_days = 7 * number_of_weeks
    timeseries_simulation = np.tile(week_simulation, number_of_weeks)

    trend = [np.linspace(0, i, num=number_of_samples_per_day*nbr_of_days)
             for i in np.linspace(0, 2, dataset_number)]
    noise = [np.random.normal(scale=scale, size=number_of_samples_per_day*nbr_of_days)
             for scale in np.linspace(0.01, 0.1, dataset_number)]
    merged_data = timeseries_simulation + trend + noise

    return merged_data

    ##


def train_and_get_distance(merged_data, dataset_index, train_window, horizon, batch_size, input_dimension, hidden_dimension, num_epochs=50, verbose=True, verbose_every=10):
    train_loader, validation_loader = create_loaders(
        merged_data[dataset_index], train_window, horizon, batch_size)
    model = LSTM_Model(input_dimension, hidden_dimension, horizon)
    train_object = Train(model, train_loader, validation_loader)
    train_object.train(num_epochs=num_epochs, verbose=verbose,
                       verbose_every=verbose_every)

    distances_list = []
    loss_list = []
    for i in range(0, len(merged_data)):
        t_loader, v_loader = create_loaders(
            merged_data[i], train_window, horizon, batch_size)
        distances_list.append(fisher_distance(
            train_object.model, validation_loader, v_loader, batch_size, batch_size))
        loss_list.append(train_object.validation_evaluation().item())
    return distances_list, loss_list
