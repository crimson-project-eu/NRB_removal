import pandas as pd
import numpy as np
import pickle

import matplotlib
matplotlib.use('TKAgg')

import matplotlib.pyplot as plt

import torch
import scipy.io
from torch.utils.data import TensorDataset, DataLoader
from torch import Tensor
from torcheval.metrics import R2Score
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from torch import nn

from pytorch_models import SimilHuber_Generator as Generator

with open(r'data_predictions\raman.pkl', 'rb') as fin:
    raman = pickle.load(fin)
with open(r'data_predictions\cars.pkl', 'rb') as fin:
    cars = pickle.load(fin)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
generator = Generator(1000).to(device)
generator.load_state_dict(torch.load(rf'models/model_name.pt'))


test_tensor = TensorDataset(Tensor(cars.astype(np.float32))[:, None, :],
                            Tensor(raman.astype(np.float32))[:, None, :])
test_dataset = DataLoader(test_tensor, batch_size=1000, num_workers=0)

for _, (input_batch, target_batch) in enumerate(test_dataset):
    x_ = input_batch.to(device)
    y_ = target_batch.to(device)
    generator.eval()
    pred = generator(x_)

r2_gan = r2_score(y_.cpu().detach().numpy()[:, 0, :].reshape(-1, 1).ravel(),
                  pred.cpu().detach().numpy()[:, 0, :].reshape(-1, 1).ravel())

r2 = []
mse = []
mse_3 = []
for i in range(1000):
    r2.append(r2_score(y_.cpu().detach().numpy()[i, 0, :].reshape(-1, 1).ravel(),
                       pred.cpu().detach().numpy()[i, 0, :].reshape(-1, 1).ravel()))

ser_r2 = pd.Series(r2)
ser_mse_3 = pd.Series(mse_3)
ser_mse = pd.Series(mse)
ser_r2.describe()
ser_mse.describe()
ser_mse_3.describe()
