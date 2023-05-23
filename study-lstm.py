from fastai.tabular.all import *
import pandas as pd
import numpy as np

data = pd.read_csv('CAC40-^FCHI.csv')
# bla bla
# Preprocessing
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

cat_names = []
cont_names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
y_names = ['Close']

def TimeSplitter(data):
    train_size = int(len(data) * 0.8)
    train_idx = list(range(train_size))
    valid_idx = list(range(train_size, len(data)))
    return train_idx, valid_idx

splits = TimeSplitter(data)

# Making a TabularPandas object
to = TabularPandas(data, procs=[Categorify, FillMissing, Normalize],
                   cat_names = cat_names,
                   cont_names = cont_names,
                   y_names=y_names,
                   splits=splits)

dls = to.dataloaders(bs=64)

# Define LSTM model
class LSTMModel(Module):
    def __init__(self, input_sz: int, hidden_sz: int):
        self.lstm = nn.LSTM(input_sz, hidden_sz, batch_first=True)
        self.fc = nn.Linear(hidden_sz, 1)

    def forward(self, x_cat: Tensor, x_cont: Tensor):
        lstm_out, _ = self.lstm(x_cont.view(x_cont.shape[0], 1, -1))
        last_lstm_out = lstm_out[:, -1, :]
        return self.fc(last_lstm_out)

# Training the model
model = LSTMModel(len(cont_names), 50)
learn = Learner(dls, model, loss_func=nn.MSELoss(), metrics=accuracy)
learn.fit_one_cycle(10, 1e-3)
