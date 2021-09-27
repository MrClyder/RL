import pandas as pd
import time
import torch
import math
import torch.nn as nn
import tensorflow.keras.models as keras
import numpy as np
import datetime
from icecream import ic
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
# from transformer_singlestep import TransAm


input_window = 100 # number of input steps
output_window = 1 # number of prediction steps, in this model its fixed to one
batch_size = 64
device = torch.device("cuda")


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransAm(nn.Module):
    def __init__(self, feature_size=250, num_layers=1, dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)  # , self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + output_window:i + tw + output_window]
        inout_seq.append((train_seq, train_label))
    return torch.FloatTensor(inout_seq)


# Dataloader
def get_data(file):

    # loading data from a file
    series = read_csv(file, header=0, index_col=0, parse_dates=True, squeeze=True)
    print(len(series))
    # looks like normalizing input values curtial for the model
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)

    test_data = amplitude

    # convert our train data into a pytorch train tensor
    # train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment..
    # train_sequence = create_inout_sequences(train_data, input_window)
    # train_sequence = train_sequence[
    #                  :-output_window]  # todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack..

    # test_data = torch.FloatTensor(test_data).view(-1)
    test_data = create_inout_sequences(test_data, input_window)
    test_data = test_data[:-output_window]  # todo: fix hack?

    return test_data.to(device)


def get_batch(source, i, batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i + seq_len]
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window, 1))  # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window, 1))
    return input, target


# def plot_and_loss(model, data_source, epoch):
#     model.eval()
#     total_loss = 0.
#     test_result = torch.Tensor(0)
#     truth = torch.Tensor(0)
#     with torch.no_grad():
#         for i in range(0, len(data_source) - 1):
#             data, target = get_batch(data_source, i, 1)
#             output = model(data)
#             total_loss += criterion(output, target).item()
#             test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
#             truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)
#
#     # test_result = test_result.cpu().numpy() -> no need to detach stuff..
#     len(test_result)
#
#     pyplot.plot(test_result, color="red")
#     pyplot.plot(truth, color="blue")
#     pyplot.plot(test_result - truth, color="green")
#     pyplot.grid(True, which='both')
#     # pyplot.axhline(y=0, color='k')
#     pyplot.savefig('/home/dl/ren/transformer-prediction/transformer-epoch%d.png' % epoch)
#     pyplot.close()
#
#     return total_loss / i

# def evaluate(model, data_source):
#     model.eval() # Turn on the evaluation mode
#     total_loss = 0.
#     eval_batch_size = 64
#     with torch.no_grad():
#         for i in range(0, len(data_source) - 1, eval_batch_size):
#             data, targets = get_batch(data_source, i,eval_batch_size)
#             output = model(data)
#             #print("data",data)
#             #print("output", output)
#             total_loss += len(data[0])* criterion(output, targets).cpu().item()
#     return total_loss / len(data_source)


file = '/home/dl/ren/transformer-prediction/0712_15s.csv'
test_data = get_data(file)
model = TransAm().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()


def predict_(model, data):
    data = get_data(file)
    output = model(data)


