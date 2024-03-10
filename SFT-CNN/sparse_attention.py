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

        assert d_model % num_heads == 0

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

    def call(self, inputs, mask = None, enc_output = None):
        #print("call!!!")
        batch_size =  tf.shape(inputs)[0]
        shape = inputs.get_shape().as_list()
        seq_len = int(shape[1] * shape[2] * shape[3] / self.d_model)
        feature_dim = self.d_model
        inputs_reshaped = tf.squeeze(inputs, axis=1)  # (B, L, F)
        if enc_output is not None:
            enc_output_reshaped = tf.squeeze(enc_output, axis=1)  # (B, L, F)

        total_output = tf.zeros_like(inputs)  # Initialize the total_output

        # Process each head iteratively
        for i in range(self.num_heads):
            # Compute the query, key, and value for the i-th head
            Q = tf.matmul(inputs_reshaped, self.Wq[i])  # (B, L, d_k)
            if enc_output is not None:
                K = tf.matmul(enc_output_reshaped, self.Wk[i])  # (B, L, d_k)
                V = tf.matmul(enc_output_reshaped, self.Wv[i])  # (B, L, d_v)
            else:
                K = tf.matmul(inputs_reshaped, self.Wk[i])  # (B, L, d_k)
                V = tf.matmul(inputs_reshaped, self.Wv[i])  # (B, L, d_v)
            
            # Scaled dot-product attention
            scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.d_k, tf.float32))  # (B, L, L)
            
            # Apply masking if provided
            if mask is not None:
                scores = scores * mask - tf.constant(1e10, dtype=tf.float32) * (1 - mask)
            
            # Apply sparse attention mask
            # mask = atrous_self_attention_mask(N = seq_len, dilation_rate = 2)
            # mask = local_self_attention_mask(N = seq_len, window_size = 2)
            mask = stride_sparse_self_attention_mask(N = seq_len, local_range = 2, stride = 2)
            scores = scores * mask - tf.constant(1e10, dtype=tf.float32) * (1 - mask)

            # Apply softmax for attention weights
            attention_weights = tf.nn.softmax(scores, axis=-1)  # (B, L, L)
            
            # Context vector (weighted sum of values)
            output = tf.matmul(attention_weights, V)  # (B, L, d_v)

            # Reshape output to match original input
            output = tf.reshape(output, (tf.shape(inputs)[0], 1, seq_len, feature_dim))

            # Sum up the outputs of all heads
            total_output += output
        
        # Calculate the average
        total_output = total_output / self.num_heads
        #print("inputs shape = ", inputs) # (None, 1, 8, 256)
        #print("inputs_reshaped shape = ", inputs_reshaped) # (None, 8, 256)
        #print("Q shape = ", Q) # (None, 8, 256)
        #print("scores shape = ", scores) # (None, 8, 8)
        #print("attention_weights shape = ", attention_weights) # (None, 8, 8)
        #print("total_output shape = ", total_output) # (None, 1, 8, 256)

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