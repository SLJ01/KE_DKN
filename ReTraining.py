# -*- coding: utf-8 -*-
# @File   : ReTraining
# @Time   : 2023/8/3 15:57
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
# class RE_KE_DKN(Layer):
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
#     # model initialization
#     def build(self, input_shape):
#         # create a trainable weight variable for this layer
#         dim = int(input_shape[-1]) * self.embedding_dim
#         self.embedding = self.add_weight(name="embedding",shape=(self.feature_num, self.embedding_dim),initializer='he_normal',trainable=True)
#         self.dense1 = self.add_weight(name='dense1',shape=(input_shape[1] * self.embedding_dim, self.dense1_dim),initializer='he_normal',trainable=True)
#         self.bias1 = self.add_weight(name='bias1',shape=(self.dense1_dim,),initializer='he_normal',trainable=True)
#         self.dense2 = self.add_weight(name='dense2', shape=(self.dense1_dim, self.dense2_dim),initializer='he_normal', trainable=True)
#