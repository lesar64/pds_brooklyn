from .utils import get_data_path
import os
import pickle
import torch


def save_model(name, model, end='.pkl'):
    pickle.dump(model, open(os.path.join(get_data_path(), "output", "models", name + "_model" + end), 'wb'))


def save_fig(fig, name, file_type='png'):
    fig.savefig(os.path.join(get_data_path(), "output", name + "." + file_type), transparent=True, dpi=250)


def save_csv(name, df):
    df.to_csv(os.path.join(get_data_path(), "output", name + ".csv"))
