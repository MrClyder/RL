import numpy as np
import torch
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import causal_convolution_layer
import Dataloader_revise
import math
from Dataloader_revise import *
# from epoch import *
from torch.utils.data import DataLoader

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device( "cuda")

# model class
class TransformerTimeSeries(torch.nn.Module):
    """
    Time Series application of transformers based on paper

    causal_convolution_layer parameters:
        in_channels: the number of features per time point
        out_channels: the number of features outputted per time point
        kernel_size: k is the width of the 1-D sliding kernel

    nn.Transformer parameters:
        d_model: the size of the embedding vector (input)

    PositionalEncoding parameters:
        d_model: the size of the embedding vector (positional vector)
        dropout: the dropout to be used on the sum of positional+embedding vector

    """

    def __init__(self):
        super(TransformerTimeSeries, self).__init__()
        self.input_embedding = causal_convolution_layer.context_embedding(2, 256, 9)
        self.positional_embedding = torch.nn.Embedding(3000, 256)

        self.decode_layer = torch.nn.TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_decoder = torch.nn.TransformerEncoder(self.decode_layer, num_layers=3)

        self.fc1 = torch.nn.Linear(256, 1)

    def forward(self, x, y, attention_masks):
        # concatenate observed points and time covariate
        # (B*feature_size*n_time_points)
        z = torch.cat((y.unsqueeze(1), x.unsqueeze(1)), 1)

        # input_embedding returns shape (Batch size,embedding size,sequence len) -> need (sequence len,Batch size,embedding_size)
        z_embedding = self.input_embedding(z).permute(2, 0, 1)

        # get my positional embeddings (Batch size, sequence_len, embedding_size) -> need (sequence len,Batch size,embedding_size)
        positional_embeddings = self.positional_embedding(x.type(torch.long)).permute(1, 0, 2)

        input_embedding = z_embedding + positional_embeddings

        transformer_embedding = self.transformer_decoder(input_embedding, attention_masks)
        output = self.fc1(transformer_embedding.permute(1, 0, 2))

        return output


def load_data(train_data_path):
    print('---------[Data Loading]----------')
    # 数据加载
    train_data = pd.read_csv(train_data_path, index_col=[0])

    scaler = StandardScaler().fit(train_data)

    # 保存归一化模型
    pickle.dump(scaler, open('scaler_model.pth', 'wb'))
    print('Scaler model saved to {}'.format('scaler_model.pth'))

    train_data = pd.DataFrame(scaler.transform(train_data))
    train_data = time_series_decoder_paper(train_data)  #last output for train data

    return train_data

train_data_path = '/home/dl/ren/transformer-prediction/eth_7.24-7.27_data_150s.csv'
train_data = load_data(train_data_path)


# test_data and val_data
test_data_path = '/home/dl/ren/transformer-prediction/7.31_8.2_eth_swap_150s_data.csv'
df = pd.read_csv(test_data_path, index_col=[0])
val_data = df.values[:1000]
test_data = df.values[1000:]

# scaler test_data and val_data
fr = open('scaler_model.pth', 'rb')
scaler = pickle.load(fr)
val_data = pd.DataFrame(scaler.transform(val_data))
val_data = time_series_decoder_paper(val_data)
test_data = pd.DataFrame(scaler.transform(test_data))
test_data = time_series_decoder_paper(test_data)


# model part
def Dp(y_pred,y_true,q):
    return max([q*(y_pred-y_true),(q-1)*(y_pred-y_true)])

def Rp_num_den(y_preds,y_trues,q):
    numerator = np.sum([Dp(y_pred,y_true,q) for y_pred,y_true in zip(y_preds,y_trues)])
    denominator = np.sum([np.abs(y_true) for y_true in y_trues])
    return numerator,denominator


def train_epoch(model, train_dl, t0, future):
    model.train()
    train_loss = 0
    n = 0
    for step, (x, y, attention_masks) in enumerate(train_dl):
        optimizer.zero_grad()
        output = model(x.to(device), y.to(device), attention_masks[0].to(device))
        loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + future - 1)], y.to(device)[:, t0:])  # not missing data
        # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data
        loss.backward()
        optimizer.step()

        train_loss += (loss.detach().cpu().item() * x.shape[0])
        n += x.shape[0]
    return train_loss / n

def eval_epoch(model, val_dl, t0, future):
    model.eval()
    eval_loss = 0
    n = 0
    with torch.no_grad():
        for step, (x, y, attention_masks) in enumerate(val_dl):
            output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())
            loss = criterion(output.squeeze()[:, (t0 - 1):(t0 + 24 - 1)], y.cuda()[:, t0:])  # not missing data
            # loss = criterion(output.squeeze()[:,(t0-1-10):(t0+24-1-10)],y.cuda()[:,(t0-10):]) # missing data

            eval_loss += (loss.detach().cpu().item() * x.shape[0])
            n += x.shape[0]

    return eval_loss / n

def test_epoch(model, test_dl, t0, future):
    with torch.no_grad():
        predictions = []
        observations = []

        model.eval()
        for step, (x, y, attention_masks) in enumerate(test_dl):
            output = model(x.cuda(), y.cuda(), attention_masks[0].cuda())

            for p, o in zip(output.squeeze()[:, (t0 - 1):(t0 + future - 1)].cpu().numpy().tolist(),
                            y[:, t0:].cpu().numpy().tolist()):  # not missing data
                # for p,o in zip(output.squeeze()[:,(t0-1-10):(t0+24-1-10)].cpu().numpy().tolist(),y.cuda()[:,(t0-10):].cpu().numpy().tolist()): # missing data

                predictions.append(p)
                observations.append(o)

        num = 0
        den = 0
        for y_preds, y_trues in zip(predictions, observations):
            num_i, den_i = Rp_num_den(y_preds, y_trues, .5)
            num += num_i
            den += den_i
        Rp = (2 * num) / den

    return Rp


criterion = torch.nn.MSELoss()

train_dl = DataLoader(train_data,batch_size=32,shuffle=True)
val_dl = DataLoader(val_data,batch_size=32)
test_dl = DataLoader(test_data,batch_size=32)

model = TransformerTimeSeries().to(device)

lr = .0005 # learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
epochs = 100

train_epoch_loss = []
eval_epoch_loss = []
Rp_best = 1e5

torch.save(model.state_dict(), 'ConvTransformer_nologsparse_1.pth')

# model_save_path = 'ConvTransformer_nologsparse.pth'
for e, epoch in enumerate(range(epochs)):
    train_loss = []
    eval_loss = []

    if (epoch % 10 is 0):

        l_t = train_epoch(model, train_dl, t0=2073, future=231)
        train_loss.append(l_t)

        l_e = eval_epoch(model, val_dl, t0=1345, future=150)
        eval_loss.append(l_e)

        Rp = test_epoch(model, test_dl, t0=1345, future=150)

        # if Rp_best > Rp:
        #     Rp_best = Rp

        train_epoch_loss.append(np.mean(train_loss))
        eval_epoch_loss.append(np.mean(eval_loss))

        print('-' * 89)
        print("Epoch {}: Train loss: {} \t Validation loss: {} \t ".format(e,
                                                                                np.mean(train_loss),
                                                                                 np.mean(eval_loss)))
        print('-' * 89)
