import numpy as np


def rescale(obs, pred, variable, train_means_path, train_stds_path):
    train_means = np.loadtxt(train_means_path, dtype="float32")
    train_stds = np.loadtxt(train_stds_path, dtype="float32")
    train_x_mean = train_means[:-1]
    train_y_mean = train_means[-1]
    train_x_std = train_stds[:-1]
    train_y_std = train_stds[-1]
    if variable == 'inputs':
        obs_rescaled = obs * train_x_std + train_x_mean
        pred_rescaled = pred * train_x_std + train_x_mean
    elif variable == 'output':
        obs_rescaled = obs * train_y_std + train_y_mean
        pred_rescaled = pred * train_y_std + train_y_mean
    else:
        raise RuntimeError(f"Unknown variable type {variable}")

    return obs_rescaled, pred_rescaled
