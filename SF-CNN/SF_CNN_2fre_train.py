from keras.layers import Input, Dense, Dropout, Convolution2D, MaxPool2D, BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
from numpy import *
import numpy as np
import numpy.linalg as LA
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import copy
import math
import torch.nn.functional as FFF
# 检查 GPU 是否可用
if torch.cuda.is_available():
  print('GPU is available!')
# 选择要使用的 GPU 设备
device = torch.cuda.set_device(0)


# from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D, PositionalEncoding3D, Summer


epochs_num = 200
batch_size_num = 32
encoder_block_num = 9
learning_rate_num = 1e-4
key_dim_num = 256
print("TensorFlow 版本:", tf.__version__)
print("epochs_num = ", epochs_num)
print("batch_size_num = ", batch_size_num)
print("encoder_block_num = ", encoder_block_num)
print("learning_rate_num = ", learning_rate_num)
Nt=32
Nt_beam=32
Nr=16
Nr_beam=16
SNR_dB = 20
SNR=10.0**(SNR_dB/10.0) # transmit power
print("SNR = ", SNR)
# DFT matrix
def DFT_matrix(N):
    m, n = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    D = np.power( omega, m * n )
    return D

F_DFT=DFT_matrix(Nt)/np.sqrt(Nt)
F_RF=F_DFT[:,0:Nt_beam]
F=F_RF
F_conjtransp=np.transpose(np.conjugate(F))
FFH=np.dot(F,F_conjtransp)
invFFH=np.linalg.inv(FFH)
pinvF=np.dot(F_conjtransp,invFFH)

W_DFT=DFT_matrix(Nr)/np.sqrt(Nr)
W_RF=W_DFT[:,0:Nr_beam]
W=W_RF
W_conjtransp=np.transpose(np.conjugate(W))

scale=2
fre=2

############## training set generation ##################
data_num_train=1000
data_num_file=1000
H_train=zeros((data_num_train,Nr,Nt,2*fre), dtype=float)
H_train_noisy=zeros((data_num_train,Nr_beam,Nt_beam,2*fre), dtype=float)
current_directory = os.getcwd()
filedir = os.path.join(current_directory, '2fre_data')  # type the path of training data
n=0
SNRr=0
SNR_factor=5.9  # compensate channel power gain to approximate to 1
for filename in os.listdir(filedir):
    newname = os.path.join(filedir, filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        for j in range(fre):
            a=channel[:,:,j,i]
            H=np.transpose(a)
            H_re=np.real(H)
            H_im = np.imag(H)
            H_train[n*data_num_file+i,:,:,2*j]=H_re/scale
            H_train[n*data_num_file+i, :, :, 2*j+1] = H_im/scale
            N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
            NpinvF=np.dot(N,pinvF)
            Y = H + 1.0 / np.sqrt(SNR_factor*SNR) * NpinvF
            SNRr = SNRr + SNR_factor*SNR * (LA.norm(H)) ** 2 / (LA.norm(NpinvF)) ** 2
            Y_re = np.real(Y)
            Y_im = np.imag(Y)
            H_train_noisy[n*data_num_file+i, :, :, 2 * j] = Y_re / scale
            H_train_noisy[n*data_num_file+i, :, :, 2 * j + 1] = Y_im / scale
    n=n+1
print(n)
print(SNRr/(data_num_train*fre))
print(H_train.shape,H_train_noisy.shape)
index1=np.where(abs(H_train)>1)
row_num=np.unique(index1[0])
H_train=np.delete(H_train,row_num,axis=0)
H_train_noisy=np.delete(H_train_noisy,row_num,axis=0)
print(len(row_num))
print(H_train.shape,H_train_noisy.shape)

############## testing set generation ##################
data_num_test=1000
data_num_file=1000
H_test=zeros((data_num_test,Nr,Nt,2*fre), dtype=float)
H_test_noisy=zeros((data_num_test,Nr_beam,Nt_beam,2*fre), dtype=float)
filedir = os.path.join(current_directory, '2fre_data')  # type the path of testing data (Testing data should be different from training data. Here use the same data just for ease of demonstration.)
n=0
SNRr=0
SNR_factor=5.9
for filename in os.listdir(filedir):
    newname = os.path.join(filedir, filename)
    data = sio.loadmat(newname)
    channel = data['ChannelData_fre']
    for i in range(data_num_file):
        for j in range(fre):
            a=channel[:,:,j,i]
            H = np.transpose(a)
            H_re = np.real(H)
            H_im = np.imag(H)
            H_test[n*data_num_file+i, :, :, 2 * j] = H_re / scale
            H_test[n*data_num_file+i, :, :, 2 * j + 1] = H_im / scale
            N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
            NpinvF = np.dot(N, pinvF)
            Y = H + 1.0 / np.sqrt(SNR_factor*SNR) * NpinvF
            SNRr = SNRr + SNR_factor*SNR * (LA.norm(H)) ** 2 / (LA.norm(NpinvF)) ** 2
            Y_re = np.real(Y)
            Y_im = np.imag(Y)
            H_test_noisy[n*data_num_file+i, :, :, 2 * j] = Y_re / scale
            H_test_noisy[n*data_num_file+i, :, :, 2 * j + 1] = Y_im / scale
    n = n + 1
print(n)
print(SNRr/(data_num_test*fre))
print(H_test.shape,H_test_noisy.shape)
index3 = np.where(abs(H_test) > 1)
row_num = np.unique(index3[0])
H_test = np.delete(H_test, row_num, axis=0)
H_test_noisy = np.delete(H_test_noisy, row_num, axis=0)
print(len(row_num))
print(H_test.shape, H_test_noisy.shape)
print(((H_test)**2).mean())


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
    
# 模型
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # scores = torch.from_numpy(scores)
    p_attn = FFF.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(FFF.relu(self.w_1(x))))

class MyDataset(Dataset):
    def __init__(self, src, trg):
        self.src = src
        self.trg = trg

    def __len__(self):
        return len(self.src)

    def __getitem__(self, index):
        src_item = self.src[index]
        trg_item = self.trg[index]
        return src_item, trg_item

# 设置参数
epochs = 1
batch_size = 32

# 定义预处理步骤
def preprocess(data):
    # 将数据展平
    data = data.reshape(data.shape[0], -1)
    # 将数据转换为张量
    data = torch.from_numpy(data)
    # 将数据类型转换为 double
    data = data.float()
    # 檢查數據類型
    print("data type = ", data.dtype)
    return data
# 预处理数据
print("H_train_noisy type = ", type(H_train_noisy))
H_train_noisy = preprocess(H_train_noisy)
H_train = preprocess(H_train)
H_test_noisy = preprocess(H_test_noisy)
H_test = preprocess(H_test)

train_dataset = MyDataset(H_train_noisy, H_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 创建 EncoderLayer 实例
N=6
d_model=2048
d_ff=2048
h=8
dropout=0.1
c = copy.deepcopy
attn = MultiHeadedAttention(h, d_model)
ff = PositionwiseFeedForward(d_model, d_ff, dropout)
encoder_layer = EncoderLayer(d_model, c(attn), c(ff), dropout)

model = Encoder(encoder_layer, 6)  # 创建编码器对象
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_fn = nn.MSELoss()  # 示例损失函数，根据您的任务进行调整

for epoch in range(epochs):
    for batch in train_loader:
        x, y = batch
        x.cuda()
        y.cuda()
        print("x: ", x.device)
        print("y: ", y.device)
        outputs = model(x, mask=None)  # 假设不需要 mask
        loss = loss_fn(outputs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印训练进度
    print(f"Epoch {epoch}: loss = {loss}")

# 保存模型
torch.save(model, "model.pt")

# 加載模型
model = torch.load("model.pt")

# 预测输出
outputs = model(H_test_noisy, mask = None)  # 假设不需要 mask

# 計算 NMSE
#nmse2=zeros((data_num_test-len(row_num),1), dtype=float)
#for n in range(data_num_test-len(row_num)):
#    MSE=((H_test[n,:,:,:]-outputs[n,:,:,:])**2).sum()
#    norm_real=((H_test[n,:,:,:])**2).sum()
#    nmse2[n]=MSE/norm_real
#print(nmse2.sum()/(data_num_test-len(row_num)))  # calculate NMSE of current training stage
