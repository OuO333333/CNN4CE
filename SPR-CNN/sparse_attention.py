import tensorflow as tf
import numpy as np

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, d_model, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.d_k = d_k  # Dimension of key vector
        self.d_v = d_v  # Dimension of value vector
        self.d_model = d_model  # Dimension of model (same as input feature dimension)

        # Create trainable weights with appropriate shapes
        self.Wq = self.add_weight(name='Wq', shape=(self.d_model, d_k), initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(name='Wk', shape=(self.d_model, d_k), initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(name='Wv', shape=(self.d_model, d_v), initializer='glorot_uniform', trainable=True)

    def call(self, inputs, mask=None):
        # Reshape input to facilitate efficient matrix multiplication
        batch_size, seq_len, feature_dim = inputs[0], 8, 256
        inputs_reshaped = tf.squeeze(inputs, axis=1)  # (B, L, F)

        # Project input features to query, key, and value vectors
        Q = tf.matmul(inputs_reshaped, self.Wq)  # (B, L, d_k)
        K = tf.matmul(inputs_reshaped, self.Wk)  # (B, L, d_k)
        V = tf.matmul(inputs_reshaped, self.Wv)  # (B, L, d_v)

        # Scaled dot-product attention
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))  # (B, L, L)

        # Apply masking if provided
        if mask is not None:
            scores = scores * mask - tf.constant(1e10, dtype=tf.float32) * (1 - mask)

        # Apply softmax for attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)  # (B, L, L)

        # Context vector (weighted sum of values)
        output = tf.matmul(attention_weights, V)  # (B, L, d_v)

        # Reshape output to match original input
        output = tf.reshape(output, (tf.shape(inputs)[0], 1, 8, 256))


        return output

class Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, d_model, num_heads, **kwargs):
        super(Multi_Head_Attention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k  # Dimension of key vector
        self.d_v = d_v  # Dimension of value vector

        # Create query, key, and value projections for each head
        self.Wq = self.add_weight(
            name='Wq',
            shape=(num_heads, d_model, d_k),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Wk = self.add_weight(
            name='Wk',
            shape=(num_heads, d_model, d_k),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Wv = self.add_weight(
            name='Wv',
            shape=(num_heads, d_model, d_k),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, mask = None, enc_output = None, FFT = False):
        
        shape = inputs.get_shape().as_list()
        seq_len = int(shape[1] * shape[2] / self.d_model)

        # Initialize the total_output
        total_output = tf.zeros_like(inputs)

        # Process each head iteratively
        for i in range(self.num_heads):

            Q = tf.matmul(inputs, self.Wq[i])  # (B, L, d_k)

            if enc_output is not None:
                K = tf.matmul(enc_output, self.Wk[i])  # (B, L, d_k)
                V = tf.matmul(enc_output, self.Wv[i])  # (B, L, d_v)
            else:
                K = tf.matmul(inputs, self.Wk[i])  # (B, L, d_k)
                V = tf.matmul(inputs, self.Wv[i])  # (B, L, d_v)

            # Scaled dot-product attention
            scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))  # (B, L, L)
            
            # Apply masking if provided
            if mask is not None:
                scores = scores * mask - tf.constant(1e10, dtype=tf.float32) * (1 - mask)
            
            # Apply sparse attention mask
            # mask = atrous_self_attention_mask(N = seq_len, dilation_rate = 2)
            # mask = local_self_attention_mask(N = seq_len, window_size = 2)
            # mask = stride_sparse_self_attention_mask(N = seq_len, local_range = 2, stride = 2)
            # scores = scores * mask - tf.constant(1e10, dtype=tf.float32) * (1 - mask)

            # Apply softmax for attention weights
            attention_weights = tf.nn.softmax(scores, axis=-1)  # (B, L, L)

            # Context vector (weighted sum of values)
            output = tf.matmul(attention_weights, V)  # (B, L, d_v)
            
            # Sum up the outputs of all heads
            total_output += output
        
        total_output = total_output / self.num_heads
        
        return total_output

class Inter_Modal_Multi_Head_Attention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, d_model, num_heads, **kwargs):
        super(Inter_Modal_Multi_Head_Attention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_k  # Dimension of key vector
        self.d_v = d_v  # Dimension of value vector

        # Create query, key, and value projections for each head
        self.Wq = self.add_weight(
            name='Wq',
            shape=(num_heads, d_model, d_k),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Wk = self.add_weight(
            name='Wk',
            shape=(num_heads, d_model, d_k),
            initializer='glorot_uniform',
            trainable=True
        )
        self.Wv = self.add_weight(
            name='Wv',
            shape=(num_heads, d_model, d_k),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs, mask = None, enc_output = None, FFT = False):
        
        shape = inputs.get_shape().as_list()
        seq_len = int(shape[1] * shape[2] / self.d_model)

        # Initialize the total_output
        total_output = tf.zeros_like(inputs)

        # Process each head iteratively
        for i in range(self.num_heads):

            K = tf.matmul(inputs, self.Wk[i])  # (B, L, d_k)
            V = tf.matmul(inputs, self.Wv[i])

            if enc_output is not None:
                Q = tf.matmul(enc_output, self.Wq[i])  # (B, L, d_k)
            else:
                Q = tf.matmul(inputs, self.Wq[i])  # (B, L, d_k)

            # Scaled dot-product attention
            scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))  # (B, L, L)
            
            # Apply masking if provided
            if mask is not None:
                scores = scores * mask - tf.constant(1e10, dtype=tf.float32) * (1 - mask)
            
            # Apply sparse attention mask
            # mask = atrous_self_attention_mask(N = seq_len, dilation_rate = 2)
            # mask = local_self_attention_mask(N = seq_len, window_size = 2)
            # mask = stride_sparse_self_attention_mask(N = seq_len, local_range = 2, stride = 2)
            # scores = scores * mask - tf.constant(1e10, dtype=tf.float32) * (1 - mask)

            # Apply softmax for attention weights
            attention_weights = tf.nn.softmax(scores, axis=-1)  # (B, L, L)

            # Context vector (weighted sum of values)
            output = tf.matmul(attention_weights, V)  # (B, L, d_v)
            
            # Sum up the outputs of all heads
            total_output += output
        
        total_output = total_output / self.num_heads
        
        return total_output

def atrous_self_attention_mask(N, dilation_rate):
    # [[1. 0. 1. 0. 1. 0.]
    #  [0. 1. 0. 1. 0. 1.]
    #  [1. 0. 1. 0. 1. 0.]
    #  [0. 1. 0. 1. 0. 1.]
    #  [1. 0. 1. 0. 1. 0.]
    #  [0. 1. 0. 1. 0. 1.]]
    mask = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if abs(i - j) % dilation_rate == 0:
                mask[i, j] = 1
    return mask

def local_self_attention_mask(N, window_size):
    # [[1. 1. 1. 0. 0. 0.]
    #  [1. 1. 1. 1. 0. 0.]
    #  [1. 1. 1. 1. 1. 0.]
    #  [0. 1. 1. 1. 1. 1.]
    #  [0. 0. 1. 1. 1. 1.]
    #  [0. 0. 0. 1. 1. 1.]]
    mask = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if abs(i - j) <= window_size:
                mask[i, j] = 1
    return mask

def stride_sparse_self_attention_mask(N, local_range, stride):
    mask = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if abs(j - i) <= local_range or abs(j - i) % stride == 0:
                mask[i, j] = 1
    return mask

class FFT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(FFT, self).__init__(**kwargs)

    @tf.function
    def call(self, x):
        # 假设 x 为(batchsize, 32, 256)的张量
        batch_size = tf.shape(x)[0]
        result = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=(32, 256))

        # 对每个张量执行FFT操作, q_fft 為 (32, 256)
        for i in tf.range(batch_size):
            q_batch = x[i]
            q_fft = fft_single(q_batch)  # 假设已定义了 ifft_single 函数
            result = result.write(i, q_fft)
        result = result.stack()
        return result

class IFFT(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(IFFT, self).__init__(**kwargs)

    @tf.function
    def call(self, x):
        # 假设 x 为(batchsize, 32, 256)的张量
        batch_size = tf.shape(x)[0]
        result = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=(32, 256))

        # 对每个张量执行FFT操作, q_fft 為 (32, 256)
        for i in tf.range(batch_size):
            q_batch = x[i]
            q_fft = ifft_single(q_batch)  # 假设已定义了 ifft_single 函数
            result = result.write(i, q_fft)
        result = result.stack()
        return result

@tf.function
def fft(x):
    # 假设 x 为(batchsize, 32, 256)的张量
    batch_size = tf.shape(x)[0]
    result = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=(32, 256))

    # 对每个张量执行FFT操作, q_fft 為 (32, 256)
    for i in tf.range(batch_size):
        q_batch = x[i]
        q_fft = fft_single(q_batch)
        result = result.write(i, q_fft)
    result = result.stack()
    return result

@tf.function
def ifft(x):
    # 假设 x 为(batchsize, 32, 256)的张量
    batch_size = tf.shape(x)[0]
    result = tf.TensorArray(dtype=tf.float32, size=batch_size, element_shape=(32, 256))

    # 对每个张量执行FFT操作, q_fft 為 (32, 256)
    for i in tf.range(batch_size):
        q_batch = x[i]
        q_fft = ifft_single(q_batch)
        result = result.write(i, q_fft)
    result = result.stack()
    return result

def fft_single(x):
    # print("x shape = ", x.shape)
    x = tf.reshape(x, (16, 32, 4, 2, 2))
    # x = tf.reshape(x, (16, 32, 8, 2))
    # 將實部和虛部拆分為兩個單獨的張量
    real_x = x[..., 0]
    imag_x = x[..., 1]

    # reshape 為 [16, 32, 2, 4]
    # real_x = tf.reshape(real_x, (16, 32, 2, 4))
    # imag_x = tf.reshape(imag_x, (16, 32, 2, 4))

    # 將實部和虛部合併為一個複數張量
    x_complex = tf.complex(real_x, imag_x)

    # 進行 2D FFT
    x = tf.signal.fft2d(x_complex)

    # 取出實部和虛部
    real_x = tf.math.real(x)
    imag_x = tf.math.imag(x)
    
    # 將實部和虛部合併為一個張量
    x = tf.concat([real_x, imag_x], axis=-1)
    x = tf.reshape(x, (32, 256))
    # print("x shape 2 = ", x.shape)
    return x

def ifft_single(x):
    # print("x shape = ", x.shape)
    x = tf.reshape(x, (16, 32, 4, 2, 2))
    # x = tf.reshape(x, (16, 32, 8, 2))
    # 將實部和虛部拆分為兩個單獨的張量
    real_x = x[..., 0]
    imag_x = x[..., 1]

    # reshape 為 [16, 32, 2, 4]
    # real_x = tf.reshape(real_x, (16, 32, 2, 4))
    # imag_x = tf.reshape(imag_x, (16, 32, 2, 4))

    # 將實部和虛部合併為一個複數張量
    x_complex = tf.complex(real_x, imag_x)

    # 進行 2D FFT
    x = tf.signal.ifft2d(x_complex)

    # 取出實部和虛部
    real_x = tf.math.real(x)
    imag_x = tf.math.imag(x)
    
    # 將實部和虛部合併為一個張量
    x = tf.concat([real_x, imag_x], axis=-1)
    x = tf.reshape(x, (32, 256))
    # print("x shape 2 = ", x.shape)
    return x

def reshape_input_output(x):
    x = tf.squeeze(x, axis=1)  # (B, L, F)
    return x

def one_gate_moe(x, key_dim_num):
  """
  使用 Keras 實現 One Gate MoE

  Args:
    x: 輸入資料
    key_dim_num: 鍵向量維度

  Returns:
    輸出資料
  """

  # 建立專家網路

  expert_network = keras.Sequential([
    keras.layers.Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu'),
    keras.layers.Conv1D(filters=key_dim_num, kernel_size=3, padding='same', activation='relu'),
  ])

  # 建立門控網路

  gating_network = keras.Sequential([
    keras.layers.Dense(units=1, activation='sigmoid'),
  ])

  # 計算專家網路輸出

  expert_outputs = expert_network(x)

  # 計算門控網路輸出

  gating_outputs = gating_network(x)

  # 計算 MoE 輸出

  moe_outputs = gating_outputs * expert_outputs

  return moe_outputs