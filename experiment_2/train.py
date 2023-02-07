import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import json


class Train():
    def __init__(self, model, train_loader, validation_loader, learning_rate=0.01, eps=1e-05, device='cpu'):
        super().__init__()

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.model = model
        self.learning_parmeters = {
            'learning_rate': learning_rate,
            'eps': eps,
            'device': device
        }

        self.optimizer = Adam(self.model.parameters(),
                              lr=self.learning_parmeters['learning_rate'],
                              eps=self.learning_parmeters['eps'])
        self.loss_fn = nn.MSELoss()

        self.training_loss_items = []
        self.validation_loss_items = []

    def save(self, model_name):
        PATH = f'./models/{model_name}.pth'
        torch.save(self.model, PATH)
        with open(f"./training_loss_items/training_loss_items_{model_name}.json", 'w') as f:
            json.dump(self.training_loss_items, f)
        with open(f"./validation_loss_items/validation_loss_items_{model_name}.json", 'w') as f:
            json.dump(self.validation_loss_items, f)

    def load(self, model_name):
        PATH = f'./models/{model_name}.pth'
        self.model = torch.load(PATH)
        with open(f"./training_loss_items/training_loss_items_{model_name}.json", 'r') as f:
            self.training_loss_items = json.load(f)

        with open(f"./validation_loss_items/validation_loss_items_{model_name}.json", 'r') as f:
            self.validation_loss_items = json.load(f)

    def validation_evaluation(self):

        inputs, target = next(enumerate(self.validation_loader, 0))[1]
        inputs = inputs.type(torch.float32).to(self.learning_parmeters['device'])
        target = target.type(torch.float32).to(self.learning_parmeters['device'])
        outputs = self.model(inputs)
        with torch.no_grad():
            return nn.MSELoss()(target, outputs)

    def train(self, num_epochs=100, verbose=True, verbose_every=10):
        if verbose:
            print('-----starting training-----')
        self.model.train()
        for epoch in range(num_epochs):
            if verbose:
                progress_verbose = not ((epoch+1) % verbose_every)
            else:
                progress_verbose = False
            epoch_loss = []
            for i, data in enumerate(self.train_loader, 0):

                inputs, target = data
                inputs = inputs.type(torch.float32).to(self.learning_parmeters['device'])
                target = target.type(torch.float32).to(self.learning_parmeters['device'])
                outputs = self.model(inputs)

                loss = self.loss_fn(target, outputs)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss.append(loss.item())
                if progress_verbose:
                    print('\repoch {:0>2} train loss {:.5f}'.format(
                        epoch+1, loss.item()), end='')

            self.training_loss_items.append(np.mean(epoch_loss))
            val_loss = self.validation_evaluation().item()
            self.validation_loss_items.append(val_loss)
            if progress_verbose:
                print(' ------ validation loss {:.5f}'.format(val_loss))

        if progress_verbose:
            print('\repoch {:0>2} train loss {:.5f}'.format(
                epoch+1, loss.item()), end='')
            print(' ------ validation loss {:.5f}'.format(val_loss))
