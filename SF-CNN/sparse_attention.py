from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Activation
import tensorflow as tf

class SelfAttention(Layer):
    def __init__(self, units, **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.Wq = self.add_weight(name='Wq', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.Wk = self.add_weight(name='Wk', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.Wv = self.add_weight(name='Wv', shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        self.softmax = Activation('softmax')
        super(SelfAttention, self).build(input_shape)

    def call(self, inputs):
        an = inputs
        # (None, 1, 8, 256)
        print("an shape = ", an.shape)
        
        # Compute qn and kn
        qn = tf.matmul(an, self.Wq)
        # (None, 1, 8, units)
        kn = tf.matmul(an, self.Wk)
        # (None, 1, 8, units)
        print("qn shape = ", qn.shape)
        print("kn shape = ", kn.shape)
        
        # Compute alpha_n
        alpha_n = tf.matmul(qn, kn, transpose_b=True)
        print("alpha_n shape = ", alpha_n.shape)
        # Apply softmax
        alpha_n_softmax = self.softmax(alpha_n)
        print("self.Wv shape = ", self.Wv.shape)
        print("alpha_n_softmax shape = ", alpha_n_softmax.shape)
        # Compute output
        output = tf.matmul(alpha_n_softmax, self.Wv)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[1], input_shape[2])

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({'units': self.units})
        return config