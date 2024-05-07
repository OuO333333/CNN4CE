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
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tf_encodings import TFPositionalEncoding1D
from tensorflow.keras.layers import Add
from sparse_attention import SelfAttention, Multi_Head_Attention, reshape_input_output, FFT, IFFT, Inter_Modal_Multi_Head_Attention
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import AveragePooling1D
from tensorflow.keras.layers import BatchNormalization


epochs_num = 200
batch_size_num = 32
encoder_block_num = 2
decoder_block_num = 2
learning_rate_num = 1e-4
key_dim_num = 256
num_heads = 4  # Number of attention heads

print("TensorFlow 版本:", tf.__version__)
print("epochs_num = ", epochs_num)
print("batch_size_num = ", batch_size_num)
print("encoder_block_num = ", encoder_block_num)
print("decoder_block_num = ", decoder_block_num)
print("learning_rate_num = ", learning_rate_num)

Nt=32
Nr=16
SNR_dB = 20
# get command line argv
args = sys.argv
if len(args) == 2:
    try:
        SNR_dB = int(args[1])
    except ValueError:
        print("intput not valid")
SNR=10.0**(SNR_dB/10.0) # transmit power
print("SNR = ", SNR)# DFT matrix
# DFT matrix
def DFT_matrix(N):
    m, n = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    D = np.power( omega, m * n )
    return D

def sub_dftmtx(A, N):
    D=A[:,0:N]
    return D

F_DFT=DFT_matrix(Nt)/np.sqrt(Nt)
W_DFT=DFT_matrix(Nr)/np.sqrt(Nr)

Nt_beam=32
F_RF=F_DFT[:,0:Nt_beam]
F=F_RF
F_conj=np.conjugate(F)
F_conjtransp=np.transpose(F_conj)
FFH=np.dot(F,F_conjtransp)
Nr_beam=16
W_RF=W_DFT[:,0:Nr_beam]
W=W_RF
W_conj=np.conjugate(W)
W_conjtransp=np.transpose(W_conj)
WWH=np.dot(W,W_conjtransp)

Nt_beam1=16
F_RF1=F_DFT[:,0:Nt_beam1]
F1=F_RF1
F_conj1=np.conjugate(F1)
F_conjtransp1=np.transpose(F_conj1)
FFH1=np.dot(F1,F_conjtransp1)
Nr_beam1=4
W_RF1=W_DFT[:,0:Nr_beam1]
W1=W_RF1
W_conj1=np.conjugate(W1)
W_conjtransp1=np.transpose(W_conj1)
WWH1=np.dot(W1,W_conjtransp1)

Nt_beam2=16
F_RF2=F_DFT[:,0:Nt_beam2]
F2=F_RF2
F_conj2=np.conjugate(F2)
F_conjtransp2=np.transpose(F_conj2)
FFH2=np.dot(F1,F_conjtransp2)
Nr_beam2=4
W_RF2=W_DFT[:,0:Nr_beam2]
W2=W_RF2
W_conj2=np.conjugate(W2)
W_conjtransp2=np.transpose(W_conj2)
WWH2=np.dot(W2,W_conjtransp2)

Nt_beam3=16
F_RF3=F_DFT[:,0:Nt_beam3]
F3=F_RF3
F_conj3=np.conjugate(F3)
F_conjtransp3=np.transpose(F_conj3)
FFH3=np.dot(F3,F_conjtransp3)
Nr_beam3=4
W_RF3=W_DFT[:,0:Nr_beam3]
W3=W_RF3
W_conj3=np.conjugate(W3)
W_conjtransp3=np.transpose(W_conj3)
WWH3=np.dot(W3,W_conjtransp3)

scale=2
fre=2
time_steps=4

############## training set generation ##################
data_num_train=500
data_num_file=500
H_train=zeros((data_num_train,Nr,Nt,2*fre), dtype=float)
H_train_noisy=zeros((data_num_train,Nr,Nt,2*fre*time_steps), dtype=float)
current_directory = os.getcwd()
filedir = os.path.join(current_directory, '2fre4time_data')  # type the path of training datan=0
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
            for t in range(time_steps):
                a1=a[:,t*Nr:t*Nr+Nr]
                H=np.transpose(a1)
                H_re=np.real(H)
                H_im = np.imag(H)
                if t==3:
                    H_train[n * data_num_file + i, :, :, 2 * j] = H_re / scale
                    H_train[n * data_num_file + i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NFH = np.dot(N, F_conjtransp)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1))
                NFH1=np.dot(N,F_conjtransp1)
                N1=np.dot(WWH1,NFH1)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2))
                NFH2 = np.dot(N, F_conjtransp2)
                N2 = np.dot(WWH2, NFH2)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3))
                NFH3 = np.dot(N, F_conjtransp3)
                N3 = np.dot(WWH3, NFH3)
                if t == 0:
                    Y = H + 1.0 / np.sqrt(SNR_factor * SNR) * NFH
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H)) ** 2 / (LA.norm(NFH)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 1:
                    HF1=np.dot(H,FFH1)
                    H1=np.dot(WWH1,HF1)
                    Y = H1 + 1.0 / np.sqrt(SNR_factor * SNR) * N1
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H1)) ** 2 / (LA.norm(N1)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 2:
                    HF2 = np.dot(H, FFH2)
                    H2 = np.dot(WWH2, HF2)
                    Y = H2 + 1.0 / np.sqrt(SNR_factor * SNR) * N2
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H2)) ** 2 / (LA.norm(N2)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 3:
                    HF3 = np.dot(H, FFH3)
                    H3 = np.dot(WWH3, HF3)
                    Y = H3 + 1.0 / np.sqrt(SNR_factor * SNR) * N3
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H3)) ** 2 / (LA.norm(N3)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_train_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
    n=n+1
print(n)
print(SNRr/(data_num_train*fre*time_steps))
print(H_train.shape,H_train_noisy.shape)
index1=np.where(abs(H_train)>1)
row_num=np.unique(index1[0])
H_train=np.delete(H_train,row_num,axis=0)
H_train_noisy=np.delete(H_train_noisy,row_num,axis=0)
print(len(row_num))
print(H_train.shape,H_train_noisy.shape)

############## testing set generation ##################
data_num_test=500
data_num_file=500
H_test=zeros((data_num_test,Nr,Nt,2*fre), dtype=float)
H_test_noisy=zeros((data_num_test,Nr,Nt,2*fre*time_steps), dtype=float)
current_directory = os.getcwd()
filedir = os.path.join(current_directory, '2fre4time_data')  # type the path of training datan=0
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
            for t in range(time_steps):
                a1=a[:,t*Nr:t*Nr+Nr]
                H=np.transpose(a1)
                H_re=np.real(H)
                H_im = np.imag(H)
                if t==3:
                    H_test[n * data_num_file + i, :, :, 2 * j] = H_re / scale
                    H_test[n * data_num_file + i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NFH = np.dot(N, F_conjtransp)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam1))
                NFH1 = np.dot(N, F_conjtransp1)
                N1 = np.dot(WWH1, NFH1)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam2))
                NFH2 = np.dot(N, F_conjtransp2)
                N2 = np.dot(WWH2, NFH2)
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam3))
                NFH3 = np.dot(N, F_conjtransp3)
                N3 = np.dot(WWH3, NFH3)
                if t == 0:
                    Y = H + 1.0 / np.sqrt(SNR_factor * SNR) * NFH
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H)) ** 2 / (LA.norm(NFH)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 1:
                    HF1 = np.dot(H, FFH1)
                    H1 = np.dot(WWH1, HF1)
                    Y = H1 + 1.0 / np.sqrt(SNR_factor * SNR) * N1
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H1)) ** 2 / (LA.norm(N1)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 2:
                    HF2 = np.dot(H, FFH2)
                    H2 = np.dot(WWH2, HF2)
                    Y = H2 + 1.0 / np.sqrt(SNR_factor * SNR) * N2
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H2)) ** 2 / (LA.norm(N2)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
                if t == 3:
                    HF3 = np.dot(H, FFH3)
                    H3 = np.dot(WWH3, HF3)
                    Y = H3 + 1.0 / np.sqrt(SNR_factor * SNR) * N3
                    SNRr = SNRr + SNR_factor * SNR * (LA.norm(H3)) ** 2 / (LA.norm(N3)) ** 2
                    Y_re = np.real(Y)
                    Y_im = np.imag(Y)
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t] = Y_re / scale
                    H_test_noisy[n * data_num_file + i, :, :, j * 2 * time_steps + 2 * t + 1] = Y_im / scale
    n = n + 1
print(n)
print(SNRr/(data_num_test*fre*time_steps))
print(H_test.shape,H_test_noisy.shape)
index3 = np.where(abs(H_test) > 1)
row_num = np.unique(index3[0])
H_test = np.delete(H_test, row_num, axis=0)
H_test_noisy = np.delete(H_test_noisy, row_num, axis=0)
print(len(row_num))
print(H_test.shape, H_test_noisy.shape)
print(((H_test)**2).mean())

K=3
input_dim=(Nr,Nt,2*fre*time_steps)
reshape_input_dim = (int(8192 / key_dim_num), key_dim_num)
dropout_rate = 0.1

# Define the input layer
inputs = Input(shape=reshape_input_dim)
reshape_type = (0, 1, 2, 3)


# transpose
H_train_noisy = np.transpose(H_train_noisy, reshape_type)
H_train = np.transpose(H_train, reshape_type)
H_test_noisy = np.transpose(H_test_noisy, reshape_type)
H_test = np.transpose(H_test, reshape_type)

# 将 H_train_noisy, H_train, H_test_noisy, H_test 调整为形状为 (None, 1, int(2048 / key_dim_num), key_dim_num) 的数组
H_train_noisy = np.reshape(H_train_noisy, (-1, 1, int(8192 / key_dim_num), key_dim_num))
H_train = np.reshape(H_train, (-1, 1, int(2048 / key_dim_num), key_dim_num))
H_test_noisy = np.reshape(H_test_noisy, (-1, 1, int(8192 / key_dim_num), key_dim_num))
H_test = np.reshape(H_test, (-1, 1, int(2048 / key_dim_num), key_dim_num))

# start time
start_time = time.time()

# Add a rescaling layer to normalize inputs
x = Rescaling(scale=1.0 / scale)(inputs)

# position_encoding
positional_encoding_model = TFPositionalEncoding1D(key_dim_num)
position_encoding = positional_encoding_model(tf.zeros((1,int(8192 / key_dim_num),key_dim_num)))
position_encoding = tf.tile(position_encoding, multiples=[499, 1, 1])
position_encoding2 = positional_encoding_model(tf.zeros((1,int(2048 / key_dim_num),key_dim_num)))
position_encoding2 = tf.tile(position_encoding2, multiples=[499, 1, 1])

# 減少不必要維度
H_train_noisy = reshape_input_output(H_train_noisy)
H_train = reshape_input_output(H_train)
H_test_noisy = reshape_input_output(H_test_noisy)
H_test = reshape_input_output(H_test)

# 加上 position_encoding
H_train_noisy = Add()([H_train_noisy, position_encoding])
H_train = Add()([H_train, position_encoding2])
H_test_noisy = Add()([H_test_noisy, position_encoding])
H_test = Add()([H_test, position_encoding2])

enc_output = None
# FFT
FFT_layer = FFT()
x_time = x
x_fre = FFT_layer(x, key_dim_num)
for _ in range(encoder_block_num):  # Repeat the encoder encoder_block_num times
    
    # Transformer Encoder Layer
    
    # conv_output_encoder_time = Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu')(x_time)
    # conv_output_encoder_fre = Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu')(x_fre)

    # Intra_Modal_Multi_Head_Attention_Time
    Intra_Modal_Multi_Head_Attention_Time = Multi_Head_Attention(d_k=key_dim_num, d_v=key_dim_num, d_model=key_dim_num, num_heads = num_heads)
    Intra_Modal_Multi_Head_Attention_Time_Output = Intra_Modal_Multi_Head_Attention_Time(x_time)
    
    # Add & Norm
    x_time = Add()([x_time, Intra_Modal_Multi_Head_Attention_Time_Output])
    x_time = LayerNormalization(epsilon=1e-6)(x_time)

    # Intra_Modal_Multi_Head_Attention_Fre
    Intra_Modal_Multi_Head_Attention_Fre = Multi_Head_Attention(d_k=key_dim_num, d_v=key_dim_num, d_model=key_dim_num, num_heads = num_heads)
    Intra_Modal_Multi_Head_Attention_Fre_Output = Intra_Modal_Multi_Head_Attention_Fre(x_fre)
    
    # Add & Norm
    x_fre = Add()([x_fre, Intra_Modal_Multi_Head_Attention_Fre_Output])
    x_fre = LayerNormalization(epsilon=1e-6)(x_fre)

    # Inter_Modal_Multi_Head_Attention_Time
    Inter_Modal_Multi_Head_Attention_Time = Inter_Modal_Multi_Head_Attention(d_k=key_dim_num, d_v=key_dim_num, d_model=key_dim_num, num_heads = num_heads)
    Inter_Modal_Multi_Head_Attention_Time_Output = Inter_Modal_Multi_Head_Attention_Time(x_time, enc_output = x_time)

    # Add & Norm
    x_time = Add()([x_time, Inter_Modal_Multi_Head_Attention_Time_Output])
    x_time = LayerNormalization(epsilon=1e-6)(x_time)

    # Inter_Modal_Multi_Head_Attention_Fre
    Inter_Modal_Multi_Head_Attention_Fre = Inter_Modal_Multi_Head_Attention(d_k=key_dim_num, d_v=key_dim_num, d_model=key_dim_num, num_heads = num_heads)
    Inter_Modal_Multi_Head_Attention_Fre_Output = Inter_Modal_Multi_Head_Attention_Fre(x_fre, enc_output = x_fre)

    # Add & Norm
    x_fre = Add()([x_fre, Inter_Modal_Multi_Head_Attention_Fre_Output])
    x_fre = LayerNormalization(epsilon=1e-6)(x_fre)

    # Feed Forward Layer Time
    ff_output_encoder_1 = Dense(units=key_dim_num, activation='relu')(x_time)
    # ff_output_encoder_1 = Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu')(x_time)
    x_time = Add()([x_time, ff_output_encoder_1])
    x_time = LayerNormalization(epsilon=1e-6)(x_time)

    # Feed Forward Layer Time
    ff_output_encoder_2 = Dense(units=key_dim_num, activation='relu')(x_fre)
    # ff_output_encoder_2 = Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu')(x_fre)
    x_fre = Add()([x_fre, ff_output_encoder_2])
    x_fre = LayerNormalization(epsilon=1e-6)(x_fre)

    # x_time = Add()([x_time, conv_output_encoder_time])
    # x_time = LayerNormalization(epsilon=1e-6)(x_time)

    # x_fre = Add()([x_fre, conv_output_encoder_fre])
    # x_fre = LayerNormalization(epsilon=1e-6)(x_fre)

    # Transformer Decoder Layer

    # Multi-Head Attention Time(attention to encoder outputs)
    enc_output = x_time  # Assuming encoder output is available
    cross_attention_layer_time = Multi_Head_Attention(d_k=key_dim_num, d_v=key_dim_num, d_model=key_dim_num, num_heads = num_heads)
    cross_attention_layer_time_output = cross_attention_layer_time(x_time, enc_output = enc_output)
    
    # Add & Norm
    x_time = Add()([x_time, cross_attention_layer_time_output])
    x_time = LayerNormalization(epsilon=1e-6)(x_time)

    # Multi-Head Attention Fre(attention to encoder outputs)
    enc_output = x_fre  # Assuming encoder output is available
    cross_attention_layer_fre = Multi_Head_Attention(d_k=key_dim_num, d_v=key_dim_num, d_model=key_dim_num, num_heads = num_heads)
    cross_attention_layer_fre_output = cross_attention_layer_fre(x_fre, enc_output = enc_output)
    
    # Add & Norm
    x_fre = Add()([x_fre, cross_attention_layer_fre_output])
    x_fre = LayerNormalization(epsilon=1e-6)(x_fre)

    # Feed Forward Layer Time
    ff_output_decoder_1 = Dense(units=key_dim_num, activation='relu')(x_time)
    # ff_output_decoder_1 = Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu')(x_time)
    x_time = Add()([x_time, ff_output_decoder_1])
    x_time = LayerNormalization(epsilon=1e-6)(x_time)

    # Feed Forward Layer Time
    ff_output_decoder_2 = Dense(units=key_dim_num, activation='relu')(x_fre)
    # ff_output_decoder_2 = Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu')(x_fre)
    x_fre = Add()([x_fre, ff_output_decoder_2])
    x_fre = LayerNormalization(epsilon=1e-6)(x_fre)


# Concat x_time & x_fre
IFFT_layer = IFFT()
x_fre = IFFT_layer(x_fre,key_dim_num)
x = (x_time + x_fre) / 2
# x =  LayerNormalization(epsilon=1e-6)(x)

# Output layer
outputs = Conv1D(filters=key_dim_num, kernel_size=K, padding='Same', activation='tanh')(x_time)
outputs = AveragePooling1D(pool_size=4, padding='Same')(outputs)

# Create the model
model = Model(inputs=inputs, outputs=outputs)

# Compile the model
adam=Adam(learning_rate=learning_rate_num, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
model.compile(optimizer='adam', loss='mse')

# Print model summary
model.summary()


# checkpoint
filepath='CNN_UMi_3path_2fre_SNRminus10dB_200ep.tf'

checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print("H_train shape = ", H_train.shape, "H_train_noisy shape =", H_train_noisy.shape)
model.fit(H_train_noisy, H_train, epochs=epochs_num, batch_size=batch_size_num, callbacks=callbacks_list, verbose=2, shuffle=True, validation_split=0.1)

# end time
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print("执行时间：", execution_time, "秒")

# load model
CNN = tf.keras.models.load_model('CNN_UMi_3path_2fre_SNRminus10dB_200ep.tf')

decoded_channel = CNN.predict(H_test_noisy)

nmse2=zeros((data_num_test-len(row_num),1), dtype=float)
for n in range(data_num_test-len(row_num)):
    MSE = tf.reduce_sum(tf.square(H_test - decoded_channel))
    norm_real = tf.reduce_sum(tf.square(H_test))
    nmse2[n]=MSE/norm_real
print("NMSE = ", nmse2.sum()/(data_num_test-len(row_num)))  # calculate NMSE of current training stage

def Sumrate(h_test,h_est,bandwidth):
    numerator = np.sum((h_test-h_est)**2)
    denominator = np.sum((h_test-np.mean(h_test))**2)
    rate = bandwidth * np.log2(1+(2*denominator-numerator)/denominator)
    return rate
print("Sumrate(bandwidth = 10) = ", Sumrate(H_test, decoded_channel, 10))  # calculate NMSE of current training stage

def print_shape(reshape_type):
    shapes = {'Nr': 0, 'Nt': 1, 'channel': 2}
    shape_order = [None] * 3
    for shape, index in shapes.items():
        shape_order[index] = shape

    # 重新排列形状顺序
    reshaped_order = [shape_order[i] for i in reshape_type]

    # 打印结果
    print(f"({', '.join(reshaped_order)})")

# 打开文件以写入模式
with open('output.txt', 'a') as f:
    # 保存原始的标准输出
    original_stdout = sys.stdout
    
    # 将标准输出重定向到文件
    sys.stdout = f

    # 执行print语句，输出将被重定向到文件中
    # print("Execution_time: ", execution_time)
    print("Encoder * ", encoder_block_num, ", Decoder * ", decoder_block_num, ", reshape_type = ", end='')
    if reshape_type == (0, 1, 2, 3):
        print("(Nr, Nt, channel)")
    elif reshape_type == (0, 2, 1, 3):
        print("(Nt, Nr, channel)")
    elif reshape_type == (0, 3, 1, 2):
        print("(channel, Nr, Nt)")
    print("   ", '{:>3}'.format(SNR_dB), "        ", '{:>20}'.format(nmse2.sum()/(data_num_test-len(row_num))), "        ", '{:>20}'.format(Sumrate(H_test, decoded_channel, 10)))

    # 恢复原始的标准输出
    sys.stdout = original_stdout