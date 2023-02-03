import glob
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from models import MLP_Model
from utils import create_loaders
from train import Train


def get_australian_data(dataset_name, year):
    data_files = glob.glob('./AEMO/'+dataset_name+'*'+year+'/*')
    if not data_files:
        print('data for does not exist')
        return
    else:
        df_list = []
        for file in data_files:
            df_list.append(pd.read_csv(file, parse_dates=True, index_col='SETTLEMENTDATE', usecols=[
                           'SETTLEMENTDATE', 'TOTALDEMAND']))
        data = pd.concat(df_list).sort_index()

        scaler = MinMaxScaler()
        data = scaler.fit_transform(data).reshape(-1)

        return data


def create_model_loader_train_objects(data,
                                      train_loader_dictionary, validation_loader_dictionary, train_object_dictionary,
                                      train_window, hidden_dimension, horizon, batch_size,
                                      train_model, num_epochs=30):
    for i, dataset_ref in enumerate(data):

        print(
            f'{i+1}- creating objects from {dataset_ref[0]} on year {dataset_ref[1]}')

        train_loader_dictionary['train_loader'+''.join(dataset_ref)], validation_loader_dictionary['validation_loader' +
                                                                                                    ''.join(dataset_ref)] = create_loaders(get_australian_data(*dataset_ref), train_window, horizon, batch_size)

        train_object_dictionary['train_object'+''.join(dataset_ref)] = Train(MLP_Model(train_window, hidden_dimension, horizon),
                                                                              train_loader_dictionary['train_loader'+''.join(dataset_ref)], validation_loader_dictionary['validation_loader'+''.join(dataset_ref)])
        if(train_model):
            train_object_dictionary['train_object'+''.join(dataset_ref)].train(num_epochs=num_epochs)
