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
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True   #allow growth
import scipy.io as sio

from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tf_encodings import TFPositionalEncoding1D
from tensorflow.keras.layers import Add
from sparse_attention import SelfAttention, Multi_Head_Attention
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D

epochs_num = 200
batch_size_num = 32
encoder_block_num = 3
decoder_block_num = 3
learning_rate_num = 1e-4
key_dim_num = 256
print("TensorFlow 版本:", tf.__version__)
print("epochs_num = ", epochs_num)
print("batch_size_num = ", batch_size_num)
print("encoder_block_num = ", encoder_block_num)
print("decoder_block_num = ", decoder_block_num)
print("learning_rate_num = ", learning_rate_num)

Nt=32
Nt_beam=32
Nr=16
Nr_beam=16
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
time_steps=2

############## training set generation ##################
data_num_train=1000
data_num_file=1000
H_train=zeros((data_num_train,Nr,Nt,2*fre), dtype=float)
H_train_noisy=zeros((data_num_train,Nr,Nt,2*fre*time_steps), dtype=float)
current_directory = os.getcwd()
filedir = os.path.join(current_directory, '2fre2time_data')  # type the path of training data
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
                if t==1:
                    H_train[n * data_num_file + i, :, :, 2 * j] = H_re / scale
                    H_train[n * data_num_file + i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NpinvF=np.dot(N,pinvF)
                Y = H + 1.0 / np.sqrt(SNR_factor*SNR) * NpinvF
                SNRr = SNRr + SNR_factor*SNR * (LA.norm(H)) ** 2 / (LA.norm(NpinvF)) ** 2
                Y_re = np.real(Y)
                Y_im = np.imag(Y)
                H_train_noisy[n*data_num_file+i, :, :, j*2*time_steps+2*t] = Y_re / scale
                H_train_noisy[n*data_num_file+i, :, :, j*2*time_steps+2*t + 1] = Y_im / scale
    n=n+1
print(n)
print(SNRr/(data_num_train*fre*time_steps))
print(H_train.shape,H_train_noisy.shape)
index1=np.where(abs(H_train)>1)
row_num=np.unique(index1[0])
H_train=np.delete(H_train,row_num,axis=0)
H_train_noisy=np.delete(H_train_noisy,row_num,axis=0)
print(len(row_num))
print("H_train shape = ", H_train.shape, "H_train_noisy shape =", H_train_noisy.shape)

############## testing set generation ##################
data_num_test=1000
data_num_file=1000
H_test=zeros((data_num_test,Nr,Nt,2*fre), dtype=float)
H_test_noisy=zeros((data_num_test,Nr,Nt,2*fre*time_steps), dtype=float)
current_directory = os.getcwd()
filedir = os.path.join(current_directory, '2fre2time_data') # type the path of testing data (Testing data should be different from training data. Here use the same data just for ease of demonstration.)
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
            for t in range(time_steps):
                a1=a[:,t*Nr:t*Nr+Nr]
                H=np.transpose(a1)
                H_re=np.real(H)
                H_im = np.imag(H)
                if t==1:
                    H_test[n * data_num_file + i, :, :, 2 * j] = H_re / scale
                    H_test[n * data_num_file + i, :, :, 2 * j + 1] = H_im / scale
                N = np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam)) + 1j * np.random.normal(0, 1 / np.sqrt(2), size=(Nr, Nt_beam))
                NpinvF=np.dot(N,pinvF)
                Y = H + 1.0 / np.sqrt(SNR_factor*SNR) * NpinvF
                SNRr = SNRr + SNR_factor*SNR * (LA.norm(H)) ** 2 / (LA.norm(NpinvF)) ** 2
                Y_re = np.real(Y)
                Y_im = np.imag(Y)
                H_test_noisy[n*data_num_file+i, :, :, j*2*time_steps+2*t] = Y_re / scale
                H_test_noisy[n*data_num_file+i, :, :, j*2*time_steps+2*t + 1] = Y_im / scale
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
reshape_input_dim = (1, int(4096 / key_dim_num), key_dim_num)
num_heads = 4  # Number of attention heads
dropout_rate = 0.1

# Define the input layer
# change here
#inputs = Input(shape=input_dim)
#key_dim_num = 4
inputs = Input(shape=reshape_input_dim)
reshape_type = (0, 1, 2, 3)

# H_train = np.tile(H_train, (1, 1, 2))
# H_test = np.tile(H_test, (1, 1, 2))
#print("after H_train shape = ", H_train.shape)
#print("after H_test shape = ", H_test.shape)

# transpose
H_train_noisy = np.transpose(H_train_noisy, reshape_type)
H_train = np.transpose(H_train, reshape_type)
H_test_noisy = np.transpose(H_test_noisy, reshape_type)
H_test = np.transpose(H_test, reshape_type)
# change here
# 将 H_train_noisy, H_train, H_test_noisy, H_test 调整为形状为 (None, 1, int(2048 / key_dim_num), key_dim_num) 的数组
H_train_noisy = np.reshape(H_train_noisy, (-1, 1, int(4096 / key_dim_num), key_dim_num))
H_train = np.reshape(H_train, (-1, 1, int(2048 / key_dim_num), key_dim_num))
H_test_noisy = np.reshape(H_test_noisy, (-1, 1, int(4096 / key_dim_num), key_dim_num))
H_test = np.reshape(H_test, (-1, 1, int(2048 / key_dim_num), key_dim_num))

# Add a rescaling layer to normalize inputs
x = Rescaling(scale=1.0 / scale)(inputs)

# learn positional encoding
# position_encoding = Conv2D(filters=key_dim_num, kernel_size=(K, K), padding='Same', activation='relu')(x)
# Returns the position encoding only
positional_encoding_model = TFPositionalEncoding1D(256)
# change here
position_encoding = positional_encoding_model(tf.zeros((1,int(4096 / key_dim_num),key_dim_num)))
position_encoding = tf.expand_dims(position_encoding, axis=0)
position_encoding = tf.tile(position_encoding, multiples=[999, 1, 1, 1])

position_encoding2 = positional_encoding_model(tf.zeros((1,int(2048 / key_dim_num),key_dim_num)))
position_encoding2 = tf.expand_dims(position_encoding2, axis=0)
position_encoding2 = tf.tile(position_encoding2, multiples=[999, 1, 1, 1])
H_train_noisy = Add()([H_train_noisy, position_encoding])
H_train = Add()([H_train, position_encoding2])
H_test_noisy = Add()([H_test_noisy, position_encoding])
H_test = Add()([H_test, position_encoding2])

enc_output = None
# Transformer Encoder Layer
for _ in range(encoder_block_num):  # Repeat the encoder encoder_block_num times
    # Multi-Head Attention
    # attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=32, dropout=dropout_rate)(x, x)
    #self_attention_layer = SelfAttention(d_k=256, d_v=256, d_model=256)
    self_attention_layer = Multi_Head_Attention(d_k=256, d_v=256, d_model=256, num_heads = 4)
    attn_output = self_attention_layer(x)
    # Add & Norm
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward Layer
    ff_output = Dense(units=key_dim_num, activation='relu')(x)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)

enc_output = x

# Transformer Decoder Layer
for _ in range(decoder_block_num):  # Repeat the decoder decoder_block_num times
    # change here
    # sequence mask
    mask = tf.sequence_mask([16], maxlen=16, dtype=tf.float32)
    #mask = tf.sequence_mask([32], maxlen=32, dtype=tf.float32)
    mask = tf.expand_dims(tf.expand_dims(mask, axis=0), axis=0)

    # Masked Multi-Head Attention (self-attention on decoder inputs)
    # attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=32, dropout=dropout_rate)(x, x, attention_mask=mask)
    # 创建 SelfAttention 层实例
    masked_self_attention_layer = Multi_Head_Attention(d_k=256, d_v=256, d_model=256, num_heads = 4)
    attn_output = masked_self_attention_layer(x, mask = mask)

    # Add & Norm
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Multi-Head Attention (attention to encoder outputs)
    enc_output = enc_output  # Assuming encoder output is available
    # attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=32, dropout=dropout_rate)(x, enc_output)
    cross_attention_layer = Multi_Head_Attention(d_k=256, d_v=256, d_model=256, num_heads = 4)
    attn_output = cross_attention_layer(x, enc_output = enc_output)
    # Add & Norm
    x = Add()([x, attn_output])
    x = LayerNormalization(epsilon=1e-6)(x)

    # Feed Forward Layer
    ff_output = Dense(units=key_dim_num, activation='relu')(x)
    x = Add()([x, ff_output])
    x = LayerNormalization(epsilon=1e-6)(x)


# Output layer
outputs = Conv2D(filters=key_dim_num, kernel_size=(K, K), padding='Same', activation='tanh')(x)
outputs = AveragePooling2D(pool_size=(1, 2))(outputs)

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