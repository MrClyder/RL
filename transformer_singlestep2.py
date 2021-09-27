import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import math
from matplotlib import pyplot
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

torch.manual_seed(0)
np.random.seed(0)

# S is the source sequence length
# T is the target sequence length
# N is the batch size
# E is the feature number

# src = torch.rand((10, 32, 512)) # (S,N,E)
# tgt = torch.rand((20, 32, 512)) # (T,N,E)
# out = model(src, tgt)

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
        #pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
          

class TransAm(nn.Module):
    def __init__(self,feature_size=250,num_layers=1,dropout=0.1):
        super(TransAm, self).__init__()
        self.model_type = 'Transformer'
        
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(feature_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=10, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)        
        self.decoder = nn.Linear(feature_size,1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1    
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.pos_encoder(src)
        output = self.transformer_encoder(src,self.src_mask)#, self.src_mask)
        output = self.decoder(output)
        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask



# if window is 100 and prediction step is 1
# in -> [0..99]
# target -> [1..100]
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+output_window:i+tw+output_window]
        inout_seq.append((train_seq ,train_label))
    return torch.FloatTensor(inout_seq)

# Dataloader
def get_data(file):
    # loading data from a file
    series = read_csv(file, header=0, index_col=0, parse_dates=True, squeeze=True)
    print(len(series))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    amplitude = scaler.fit_transform(series.values.reshape(-1, 1)).reshape(-1)
    #print(amplitude)
    
    
    sampels = 2300
    train_data = amplitude[:sampels]
    test_data = amplitude[sampels:]

    # convert our train data into a pytorch train tensor
    #train_tensor = torch.FloatTensor(train_data).view(-1)
    # todo: add comment.. 
    train_sequence = create_inout_sequences(train_data,input_window)
    train_sequence = train_sequence[:-output_window] #todo: fix hack? -> din't think this through, looks like the last n sequences are to short, so I just remove them. Hackety Hack.. 

    #test_data = torch.FloatTensor(test_data).view(-1) 
    test_data = create_inout_sequences(test_data,input_window)
    test_data = test_data[:-output_window] #todo: fix hack?

    return train_sequence.to(device),test_data.to(device)

def get_batch(source, i,batch_size):
    seq_len = min(batch_size, len(source) - 1 - i)
    data = source[i:i+seq_len]    
    input = torch.stack(torch.stack([item[0] for item in data]).chunk(input_window,1)) # 1 is feature size
    target = torch.stack(torch.stack([item[1] for item in data]).chunk(input_window,1))
    return input, target


def train(train_data):
    model.train() # Turn on the train mode \o/
    total_loss = 0.
    start_time = time.time()

    for batch, i in enumerate(range(0, len(train_data) - 1, batch_size)):
        data, targets = get_batch(train_data, i,batch_size)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.7)
        optimizer.step()

        total_loss += loss.item()
        log_interval = int(len(train_data) / batch_size / 5)
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.6f} | {:5.2f} ms | '
                  'loss {:5.5f} | ppl {:8.2f}'.format(
                    epoch, batch, len(train_data) // batch_size, scheduler.get_lr()[0],
                    elapsed * 1000 / log_interval,
                    cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        #print(data.shape)
#%%
def plot_and_loss(eval_model, data_source,epoch):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    with torch.no_grad():
        for i in range(0, len(data_source) - 1):
            data, target = get_batch(data_source, i,1)
            output = eval_model(data)
            total_loss += criterion(output, target).item()
            test_result = torch.cat((test_result, output[-1].view(-1).cpu()), 0)
            truth = torch.cat((truth, target[-1].view(-1).cpu()), 0)

            
    #test_result = test_result.cpu().numpy() -> no need to detach stuff.. 
    len(test_result)


    pyplot.plot(test_result,color="red")
    pyplot.plot(truth,color="blue")
    pyplot.plot(test_result-truth,color="green")
    pyplot.grid(True, which='both')
    #pyplot.axhline(y=0, color='k')
    pyplot.savefig('/home/dl/ren/transformer-prediction/transformer-epoch%d.png'%epoch)
    pyplot.close()
    
    return total_loss / i

#%%
# predict the next n steps based on the input data 
def predict_future(eval_model, data_source,steps):
    eval_model.eval() 
    total_loss = 0.
    test_result = torch.Tensor(0)    
    truth = torch.Tensor(0)
    data, target = get_batch(data_source, 0,1)
    with torch.no_grad():
        for i in range(0, steps):            
            output = eval_model(data)#输出的预测值
            output = eval_model(data[-input_window:])
            # test_result = torch.cat(test_result, output[-1].view(-1).cpu(),0)
            data = torch.cat((data, output[-1:]))
            # truth = torch.cat(truth, target[-1].view(-1).cpu(),0)



    # data = data.cpu().view(-1)
    print('------------',target)
    print('____________',data)
    # I used this plot to visualize if the model pics up any long therm structure within the data.
    pyplot.plot(data,color="red")
    pyplot.plot(data,color="blue")
    pyplot.grid(True, which='both')
    #pyplot.axhline(y=0, color='k')
    pyplot.savefig('/home/dl/ren/transformer-prediction/transformer-future%d.png'%steps)
    pyplot.close()


def prediction(model, data_source, t0, steps):
    with torch.no_grad():
        for steps

def evaluate(eval_model, data_source):
    eval_model.eval() # Turn on the evaluation mode
    total_loss = 0.
    eval_batch_size = 64
    with torch.no_grad():
        for i in range(0, len(data_source) - 1, eval_batch_size):
            data, targets = get_batch(data_source, i,eval_batch_size)
            output = eval_model(data)
            #print("data",data)
            #print("output", output)
            total_loss += len(data[0])* criterion(output, targets).cpu().item()
    return total_loss / len(data_source)


#替换文件名 导入不同的文件
file_name = '/home/dl/ren/transformer-prediction/0712_15s.csv'
train_data, val_data = get_data(file_name)

model = TransAm().to(device)
model.load_state_dict(torch.load('best_model.pth'))
# model.torch.load('best_model.h5')
predict_future(model, val_data,100)


# criterion = nn.MSELoss()
# lr = 0.001
# #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

# best_val_loss = float("inf")
# epochs = 100 # The number of epochs
# best_model = None

# for epoch in range(1, epochs + 1):
#     epoch_start_time = time.time()
#     train(train_data)
    
#     if(epoch % 10 is 0):
#         val_loss = plot_and_loss(model, val_data,epoch)
#         predict_future(model, val_data,200)
#     else:
#         val_loss = evaluate(model, val_data)
   
#     print('-' * 89)
#     print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.5f} | valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
#                                      val_loss, math.exp(val_loss)))
#     print('-' * 89)

#     # if val_loss < best_val_loss:
#     #    best_val_loss = val_loss
#     #    best_model = model
#     #    torch.save(best_model, 'best_model.pth')

#     scheduler.step() 

# src = torch.rand(input_window, batch_size, 1) # (source sequence length,batch size,feature number)
# out = model(src)
#
#print(out)
#print(out.shape)
