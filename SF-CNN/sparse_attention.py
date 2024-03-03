import tensorflow as tf

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

        # Final projection layer to combine heads
        # self.dense = tf.keras.layers.Dense(d_model, activation='linear')

    def call(self, inputs, mask=None):
        batch_size, seq_len, feature_dim = tf.shape(inputs)[0], 8, 256
        inputs_reshaped = tf.squeeze(inputs, axis=1)  # (B, L, F)

        final_output = tf.zeros_like(inputs)  # 初始化 final_output

        # 循环处理每个头
        for i in range(self.num_heads):
            # 计算第 i 个头的查询、键和值
            Q = tf.matmul(inputs_reshaped, self.Wq[i])  # (B, L, d_k)
            K = tf.matmul(inputs_reshaped, self.Wk[i])  # (B, L, d_k)
            V = tf.matmul(inputs_reshaped, self.Wv[i])  # (B, L, d_v)
            
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

            # 将所有头的输出连接起来
            final_output += output
        
        # 算平均
        final_output = final_output / self.num_heads

        return final_output