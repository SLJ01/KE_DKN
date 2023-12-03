# -*- coding: utf-8 -*-
# @File   : PreTraining
# @Time   : 2023/8/2 21:58
# @Author : linjin😀

from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Layer, Activation
import tensorflow as tf
from tensorflow.python.keras.backend import relu
import matplotlib.pyplot as plt
import pywt
import pymysql
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models, optimizers, losses, metrics
# from Knowldge_phase import acquire

'''acquire the local excel testing data sample'''
def read_originals(url,column):
    """
    :param url:Test data sample local address
    :param column:target variable column
    :return:target variable of testing samples
    """
    data = pd.read_excel(url,parse_dates=True)
    data_ndarr=np.array(data[column])
    return data_ndarr


'''acquire the full data ranging 4 working days from mysql, user:user,passwaord:123456,db=SHUINI, 
    startID=15895459,endID=16667985'''
def read_originals_Database(startID,endID):
    """
    :param startID:Begining id in database of training samples
    :param endID:Ending id in database of training samples
    :return:ndarray  Returns an array containing the target and precondition variables with timestamp
    """
    conn = pymysql.connect(host='10.11.112.202', port=3306, user='user', password='123456', db='SHUINI')
    sql = "select ID,TIMESTAMP,L0024,L0033,L0167,L0191,L0093,L0022,L0029,L0060,L0069 from DB_JIANGYIN where id  between %s and %s" % (
    startID, endID)
    slj = pd.read_sql(sql, conn)
    slj = slj.sort_values(by='ID', axis=0, ascending=True)
    data_arr = np.array(slj)
    return data_arr

'''Add noise to an array to get a noisy array'''
def Noise_adding(data_arr,mean,std,Ratio):
    """

    :param data_arr: original sampels
    :param mean:noise mean
    :param std:Noise standard deviation
    :return:Contaminated data samples
    """
    noise = tf.random.normal(shape=tf.shape(data_arr), mean=mean, sstddev=std)
    # Calculate the noise ratio, which is the ratio of the root mean square of the noise to the root mean square of the data
    noise_ratio = tf.sqrt(tf.reduce_mean(tf.square(noise))) / tf.sqrt(tf.reduce_mean(tf.square(data_arr)))
    # Adjust the amplitude of the noise to meet the specified noise ratio
    noise = noise * (Ratio / noise_ratio)
    # Add noise to an array to get a noisy array
    noisy_data = data_arr + noise
    return noisy_data


'''Decomposing the original signals:Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1'''
def Signal_decomp_WD(data, w,k):
    """
    :param data: Signals to be decomposed
    :param w:Wavelet function
    :param k:Decomposition order
    :return:Return detail signal and approximation signal
    """
    w = pywt.Wavelet(w)#Select wavelet function
    a = data
    ca = []#Approximate components
    cd = []#detail component
    mode = pywt.Modes.smooth
    for i in range(k):
        (a, d) = pywt.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)
    rec_a = []
    rec_d = []
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))
    return rec_a,rec_d


'''Obtain knowledge items, this part of the work has not yet been made public '''

def knowldgeacquire(relationEquation, instanceSolution):
    """
    :param relationEquation: the relation equation between manupiated variables and traget variables
    :param instanceSolution: The insnaces from expert experience or experimental results
    :return:Returns a two-dimensional array consisting of relational equations and instance solutions
    """
    return [relationEquation,instanceSolution]


'''The following network construction and initialization process'''
def get_deep_order(feat_index, args):
    embedding = tf.nn.embedding_lookup(args, feat_index)[:, :, :]
    embedding_flatten = layers.Flatten()(embedding)
    return embedding_flatten
def get_cross_net(input, layer_num, kernelType, kernels, bias):
    # None x dim => None x dim x 1
    x_0 = tf.expand_dims(input, axis=2)
    x_l = x_0
    if kernelType == "Gaussian":
        for i in range(layer_num):
            # None x 32 x 1 => None x 1 x 32 x 32 x 1 => None x 1 x 1
            xl_w = tf.tensordot(x_l, kernels[i], axes=(1, 0))
            # x_0: None x 32 x 1 xl_w: None x 1 x 1
            dot_ = tf.matmul(x_0, xl_w)
            # dot: None x 32 x 1 + bias: 32 x 1
            x_l = dot_ + bias[i] + x_l
    elif kernelType == "matrix":
        for i in range(layer_num):
            # w: (dim, dim) x_l (None, dim, 1) => (None, dim, 1)
            xl_w = tf.einsum('ij,bjk->bik', kernels[i], x_l)  # W * xi (bs, dim, 1)
            dot_ = xl_w + bias[i]  # W * xi + b
            x_l = x_0 * dot_ + x_l  # x0 · (W * xi + b) +xl  Hadamard-product
    else:
        raise ValueError("kernelType should be 'Gaussian' or 'matrix'")
    re = tf.squeeze(x_l, axis=2)
    return re

class KE_DKN_pre (tf.keras.layers.Layer):
    def __init__(self, units, activation='sigmoid', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(KE_DKN_pre, self).__init__()
        self.mlp = models.Sequential([
        layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        layers.Dense(units, activation=activation, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
        ])
    def call(self, x1, x2):
        f1 = self.mlp(x1)
        f2 = self.mlp(x2)
        k = tf.matmul(f1, f2, transpose_b=True)
        return k
class DeepKernelLearning(tf.keras.Model):
    def __init__(self, units, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros'):
        super(DeepKernelLearning, self).__init__()
        self.kernel = KE_DKN_pre(units, activation, kernel_initializer, bias_initializer)
        self.gpr = tf.keras.layers.GaussianProcessRegressionModel(kernel=self.kernel, jitter=1e-6)

    def call(self, inputs):
        x_train, y_train, x_test = inputs
        y_mean, y_var = self.gpr([x_train, y_train, x_test])
        return y_mean, y_var
model = KE_DKN_pre(units=64, activation='relu')

'''Define knowledge embedding loss function'''
def knowledge_embedded_loss(y_true, y_pred,knowledge_term,pi,A):
    """
    :param y_true: trure lables
    :param y_pred: predicted value
    :param knowledge_term: knowledgeterm to be embedded into the loss function
    :param pi: weight
    :param A: weight
    :return: loss value
    """
    y_mean, y_var = y_pred
    loss_list=[]
    for i in range(len(knowledge_term[0])):
        loss = 0.5 * tf.math.log(2 * np.pi * y_var) + 0.5 * tf.square(y_true - y_mean) / y_var+knowledge_term[0][i]*pi+A*knowledge_term[1][i]
        loss_list.append(loss)
    return tf.reduce_mean(loss_list)
mse_metric = metrics.MeanSquaredError()
optimizer = optimizers.SGD(learning_rate=0.9, momentum=0.9)
@tf.function
def train_step(x_train, y_train, x_test, y_test):
    """
    :param x_train: training samples
    :param y_train: training lables
    :param x_test: tesing samples
    :param y_test: testing lables
    :return: loss value
    """
    '''Use tf.GradientTape to record gradients'''
    with tf.GradientTape() as tape:
        y_pred = model([x_train, y_train, x_test])
        loss = knowledge_embedded_loss(y_test, y_pred)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        mse_metric.update_state(y_test, y_pred[0])
        return loss

@tf.function
def test_step(x_train, y_train, x_new):
    """
    :param x_train:training samples
    :param y_train:training lables
    :param x_new:new samples
    :return: return predicted values
    """
    y_pred = model([x_train, y_train, x_new])
    return y_pred
x_train = tf.random.normal([100, 1])
y_train = tf.sin(x_train) + 0.1 * tf.random.normal([100, 1])
x_test = tf.random.normal([20, 1])
y_test = tf.sin(x_test) + 0.1 * tf.random.normal([20, 1])


'''Define one training epoch'''
def epochTraning(epochs):
    """
    :param epochs: Number of iteration rounds
    """
    for epoch in range(epochs):
        mse_metric.reset_states()
        loss = train_step(x_train, y_train, x_test, y_test)
        print(f'Epoch {epoch + 1}, Loss: {loss:.4f}, MSE: {mse_metric.result():.4f}')
x_new = tf.linspace(-5.0, 5.0, 50)[:, tf.newaxis]
y_pred = test_step(x_train, y_train, x_new)
# print(f'New data: {x_new.numpy().flatten()}')
# print(f'Predicted mean: {y_pred[0].numpy().flatten()}')
# print(f'Predicted variance: {y_pred[1].numpy().flatten()}')



# class KE_DKN(Layer):
#
#     def __init__(self, knowledge_embedded_loss,feature_num, embedding_dim, layer_num, dense1_dim, dense2_dim,
#                  kernelType, **kwargs):
#         self.feature_num = feature_num
#         self.embedding_dim = embedding_dim
#         self.dense1_dim = dense1_dim
#         self.dense2_dim = dense2_dim
#         self.activation = Activation('Sigmoid')
#         self.loss=knowledge_embedded_loss
#         # DKN parameter form and number of layers
#         self.kernelType = kernelType
#         self.layer_num = layer_num
#         super().__init__(**kwargs)
#
#
#
#     # model initialization
#     def build(self, input_shape):
#         # create a trainable weight variable for this layer
#         dim = int(input_shape[-1]) * self.embedding_dim
#         self.embedding = self.add_weight(name="embedding",shape=(self.feature_num, self.embedding_dim),initializer='he_normal',trainable=True)
#         self.dense1 = self.add_weight(name='dense1',shape=(input_shape[1] * self.embedding_dim, self.dense1_dim),initializer='he_normal',trainable=True)
#         self.bias1 = self.add_weight(name='bias1',shape=(self.dense1_dim,),initializer='he_normal',trainable=True)
#         self.dense2 = self.add_weight(name='dense2', shape=(self.dense1_dim, self.dense2_dim),initializer='he_normal', trainable=True)
#
#         # DNN Bias1
#         self.bias2 = self.add_weight(name='bias2',
#                                      shape=(self.dense2_dim,),
#                                      initializer='he_normal',
#                                      trainable=True)
#
#         if self.kernelType == 'vector':
#             self.kernels = [self.add_weight(name='kernel' + str(i),
#                                             shape=(dim, 1),
#                                             initializer="he_normal",
#                                             trainable=True) for i in range(self.layer_num)]
#         elif self.kernelType == 'matrix':
#             self.kernels = [self.add_weight(name='kernel' + str(i),
#                                             shape=(dim, dim),
#                                             initializer="he_normal",
#                                             trainable=True) for i in range(self.layer_num)]
#         else:  # error
#             raise ValueError("kernelType should be 'vector' or 'matrix'")
#         self.bias = [self.add_weight(name='bias4dcn' + str(i),
#                                      shape=(dim, 1),
#                                      initializer="he_normal",
#                                      trainable=True) for i in range(self.layer_num)]
#
#         # Be sure to call this at the end
#         super(KE_DKN, self).build(input_shape)


