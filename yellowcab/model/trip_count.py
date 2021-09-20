from .. import io

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer


def trip_count_nn(df, df_backup, scaled_features, epochs=15, l_rate=0.03,
                  vis=True, val_len=15, test_len=42, layer_nodes=20, save=False):
    # Full training workflow requires this commands:
    #   df, df_backup, scaled_features = yellowcab.model.full_preparing_workflow()
    #   yellowcab.model.trip_count_nn(df, df_backup, scaled_features, save=True, epochs=15)



    # Split of <val_len> days of data from the end of the df for validation
    validation_length = -val_len * 24
    validation_data = df[validation_length:]
    df = df[:validation_length]

    # Split of <test-len> days of data from the end of the df for testing
    test_length = -test_len * 24
    test_data = df[test_length:]
    df = df[:test_length]

    # The remaining (/earlier) data will be used for training
    train_data = df

    # shows lengths
    print(f'''Validation data length: {len(validation_data)}
    Test data length: {len(test_data)}
    Train data length: {len(train_data)}''')

    # Separate the data into feature and target fields
    target_fields = ['cnt']

    # Split into train, test and val sets. Each into features and targets
    train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
    validation_features, validation_targets = validation_data.drop(target_fields, axis=1), validation_data[
        target_fields]

    # Loss function
    mse_loss = nn.MSELoss(reduction='mean')

    class Regression(pl.LightningModule):

        # The Model

        def __init__(self):
            super(Regression, self).__init__()
            self.fc1 = nn.Linear(train_features.shape[1], layer_nodes)
            self.fc2 = nn.Linear(layer_nodes, 1)

        # Question: how should the forward pass be performed, and what will its ouputs be?
        # Perform the forward pass
        # We're using the sigmoid activation function on our hidden layer, but our output layer has no activation
        # function as we're predicting a continuous variable so we want the actual number predicted
        def forward(self, x):
            x = torch.sigmoid(self.fc1(x))
            x = self.fc2(x)
            return x

        # Dataloaders

        def train_dataloader(self):
            train_dataset = TensorDataset(torch.tensor(train_features.values).float(),
                                          torch.tensor(train_targets[['cnt']].values).float())
            train_loader = DataLoader(dataset=train_dataset, batch_size=64, num_workers=8)
            return train_loader

        def val_dataloader(self):
            validation_dataset = TensorDataset(torch.tensor(validation_features.values).float(),
                                               torch.tensor(validation_targets[['cnt']].values).float())
            validation_loader = DataLoader(dataset=validation_dataset, batch_size=64, num_workers=8)
            return validation_loader

        def test_dataloader(self):
            test_dataset = TensorDataset(torch.tensor(test_features.values).float(),
                                         torch.tensor(test_targets[['cnt']].values).float())
            test_loader = DataLoader(dataset=test_dataset, batch_size=64, num_workers=8)
            return test_loader

        # The Optimizer

        # Question: what optimizer will I use?
        # Define optimizer function: here we are using Stochastic Gradient Descent
        def configure_optimizers(self):
            return optim.SGD(self.parameters(), lr=l_rate)

        # Training

        # Question: what should a training step look like?
        # Define training step
        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = mse_loss(logits, y)
            # Add logging
            logs = {'loss': loss}
            return {'loss': loss, 'log': logs}

        ### Validation ###

        # Question: what should a validation step look like?
        # Define validation step
        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = mse_loss(logits, y)
            return {'val_loss': loss}

        # Define validation epoch end
        def validation_epoch_end(self, outputs):
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            print('\navg_val_loss', avg_loss)
            return {'avg_loss': avg_loss}

        ### Testing ###

        # Question: what should a test step look like?
        # Define test step
        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self.forward(x)
            loss = mse_loss(logits, y)
            correct = torch.sum(logits == y.data)

            # I want to visualize my predictions vs my actuals so here I'm going to
            # add these lines to extract the data for plotting later on
            predictions_pred.append(logits)
            predictions_actual.append(y.data)
            return {'test_loss': loss, 'test_correct': correct, 'logits': logits}

        # Define test end
        def test_epoch_end(self, outputs):
            avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
            logs = {'test_loss': avg_loss}
            return {'avg_test_loss': avg_loss, 'log': logs, 'progress_bar': logs}

    model = Regression()
    trainer = Trainer(max_epochs=epochs)
    trainer.fit(model)

    # Here I'm creating 2 empty lists into which I'll be appending my predictions and actuals as I go
    # - you don't have to do this, but if you want to examine them in detail or plot them, then it's convenient
    predictions_pred = []
    predictions_actual = []
    trainer.test()

    # Here I'm getting the mean and standard deviation values for 'cnt', so I can convert my predicted
    # values back to actual numbers of bike rides again, instead of the scaled values used to predict on
    mean = scaled_features['cnt'][0]
    std = scaled_features['cnt'][1]

    if vis:
        # Get dates for plotting
        datesx = list(df_backup[test_length:]['Date time'])

        # Get predicted points (scaled back to their original size)
        plot_pred = []
        for i in range(len(predictions_pred)):
            plot_pred.extend(predictions_pred[i].T.numpy()[0] * std + mean)

        # Get actual points (scaled back to their original size)
        plot_actual = []
        for i in range(len(predictions_actual)):
            plot_actual.extend(predictions_actual[i].T.numpy()[0] * std + mean)

        # And finally we can see that our network has done a decent job of estimating!
        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(plot_pred, label='Prediction')
        ax.plot(plot_actual, label='Data')
        ax.set_xticks(np.arange(len(datesx))[12::24])
        ax.set_xticklabels(datesx[12::24], rotation=45)
        ax.set_title('Number of Trips per Hour', size=24)
        ax.set_xlabel("Date time")
        ax.set_ylabel("Trips per Hour")
        ax.legend()
        io.save_fig(fig, 'trip_count_NN')

    if save:
        torch.save(model.state_dict(), '../data/output/models/trip_count_NN')
        print('model saved data/output/models/trip_count_NN')
    return model


def full_preparing_workflow():
    # And then import and view the data
    print('Reading in the data')
    df_orig = io.read_all_files('parquet')
    print('Add weather data')
    df_orig = io.utils.add_weather_data(df_orig)
    # df_orig = df_orig[df_orig['tpep_pickup_datetime'] < datetime(2020, 12, 31)]
    print('Sort values')
    df_orig.sort_values('tpep_pickup_datetime')

    df_edit = df_orig[['Date time', 'start_month', 'start_day', 'start_hour',
                       'weekend', 'weekday', 'Maximum Temperature',
                       'Minimum Temperature', 'Temperature',
                       'Precipitation', 'Snow', 'Wind Speed', 'Visibility',
                       'Cloud Cover', 'Relative Humidity', 'Conditions',
                       'trip_distance']]

    df_edit = df_edit.dropna(axis=0)

    onehot_fields = ['Conditions']
    for field in onehot_fields:
        dummies = pd.get_dummies(df_edit[field], prefix=field, drop_first=False)
        df_edit_1 = pd.concat([df_edit, dummies], axis=1)
    df_edit_1 = df_edit_1.drop(onehot_fields, axis=1)
    df_edit_1.head()

    df = df_edit_1.groupby(['Date time', 'start_hour']).median().reset_index()
    df['cnt'] = df_edit_1.groupby(['Date time', 'start_hour']).count().reset_index()['trip_distance']
    df.drop(columns='trip_distance', inplace=True)
    df.sample(5)

    # onehot encoding
    onehot_fields = ['start_hour', 'start_month', 'start_day', 'weekend',
                     'weekday']
    for field in onehot_fields:
        dummies = pd.get_dummies(df[field], prefix=field, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
    df = df.drop(onehot_fields, axis=1)
    df.head()

    # Being a timeseries problem, let's look at the date ranges we have available
    print(f"""Earliest date - {df['Date time'].min()} 
    Latest date - {df['Date time'].max()}
    Total number of days - {len(df) / 24}""")

    df_vis = df[df['Date time'] < datetime(2021, 1, 1)]
    df_vis[df_vis['Date time'].isin(df_vis['Date time'].unique()[:])].plot(x='Date time', y='cnt', figsize=(16, 4))

    # scaling
    continuous_fields = ['Maximum Temperature', 'Minimum Temperature',
                         'Temperature', 'Precipitation', 'Snow', 'Wind Speed', 'Visibility',
                         'Cloud Cover', 'Relative Humidity', 'cnt']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for field in continuous_fields:
        mean, std = df[field].mean(), df[field].std()
        scaled_features[field] = [mean, std]
        df.loc[:, field] = (df[field] - mean) / std
    scaled_features

    # Create a backup of df before we drop these fields - we'll need this later to plot our testing data
    df_backup = df.copy()

    fields_to_drop = ['Date time']
    df.drop(fields_to_drop, axis=1, inplace=True)

    return df, df_backup, scaled_features


def load_pytorch_model(name):
    class Regression(pl.LightningModule):

        # The Model

        def __init__(self):
            super(Regression, self).__init__()
            self.fc1 = nn.Linear(94, 20)
            self.fc2 = nn.Linear(20, 1)

        # Question: how should the forward pass be performed, and what will its ouputs be?
        # Perform the forward pass
        # We're using the sigmoid activation function on our hidden layer, but our output layer has no activation
        # function as we're predicting a continuous variable so we want the actual number predicted
        def forward(self, x):
            x = torch.sigmoid(self.fc1(x))
            x = self.fc2(x)
            return x

    save_path = ('../data/output/models/' + name)
    model = Regression()
    model.load_state_dict(torch.load(save_path))
    return model