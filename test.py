import pandas as pd
import time
import torch
import tensorflow.keras.models as keras
import numpy as np
import datetime
from icecream import ic
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot



file = '/home/dl/ren/transformer-prediction/0712_15s.csv'

model = TransAm().to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

def predict_(mdoel,data):
    data = get_data