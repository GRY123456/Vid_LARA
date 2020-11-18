# -*- coding: utf-8 -*-
import pandas as pd
import tensorflow as tf
import numpy as np
import support

'''Hyper parameters'''
alpha = 0
attr_num = 18  # the number of attribute
attr_present_dim = 5  # the dimention of attribute present
batch_size = 1024
hidden_dim = 100  # G hidden layer dimention
user_emb_dim = attr_num

'''D variables'''
D_attri_matrix = tf.get_variable('D_attri_matrix', [2 * attr_num, attr_present_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
D_W1 = tf.get_variable('D_w1', [attr_num * attr_present_dim + user_emb_dim, hidden_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
D_b1 = tf.get_variable('D_b1', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
D_W2 = tf.get_variable('D_w2', [hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
D_b2 = tf.get_variable('D_b2', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
D_W3 = tf.get_variable('D_w3', [hidden_dim, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())
D_b3 = tf.get_variable('D_b3', [1, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())

D_params = [D_attri_matrix, D_W1, D_b1, D_W2, D_b2, D_W3, D_b3]

'''G variables'''
G_attri_matrix = tf.get_variable('G_attri_matrix', [2 * attr_num, attr_present_dim],
                                 initializer=tf.contrib.layers.xavier_initializer())
G_W1 = tf.get_variable('G_w1', [attr_num * attr_present_dim, hidden_dim],
                       initializer=tf.contrib.layers.xavier_initializer())
G_b1 = tf.get_variable('G_b1', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
G_W2 = tf.get_variable('G_w2', [hidden_dim, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
G_b2 = tf.get_variable('G_b2', [1, hidden_dim], initializer=tf.contrib.layers.xavier_initializer())
G_W3 = tf.get_variable('G_w3', [hidden_dim, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())
G_b3 = tf.get_variable('G_b3', [1, user_emb_dim], initializer=tf.contrib.layers.xavier_initializer())

G_params = [G_attri_matrix, G_W1, G_b1, G_W2, G_b2, G_W3, G_b3]

'''placeholder'''
attribute_id = tf.placeholder(shape=[None, attr_num], dtype=tf.int32)
real_user_emb = tf.placeholder(shape=[None, user_emb_dim], dtype=tf.float32)

neg_attribute_id = tf.placeholder(shape=[None, attr_num], dtype=tf.int32)
neg_user_emb = tf.placeholder(shape=[None, user_emb_dim], dtype=tf.float32)

'''G'''


def generator(attribute_id):
    attri_present = tf.nn.embedding_lookup(G_attri_matrix, attribute_id)  # batch_size x 18 x attr_present_dim

    attri_feature = tf.reshape(attri_present, shape=[-1, attr_num * attr_present_dim])

    l1_outputs = tf.nn.tanh(tf.matmul(attri_feature, G_W1) + G_b1)
    l2_outputs = tf.nn.tanh(tf.matmul(l1_outputs, G_W2) + G_b2)
    fake_user = tf.nn.tanh(tf.matmul(l2_outputs, G_W3) + G_b3)

    return fake_user


'''D'''


def discriminator(attribute_id, user_emb):
    attri_present = tf.nn.embedding_lookup(D_attri_matrix, attribute_id)  # batch_size x 18 x attr_present_dim
    attri_feature = tf.reshape(attri_present, shape=[-1, attr_num * attr_present_dim])
    emb = tf.concat([attri_feature, user_emb], 1)

    l1_outputs = tf.nn.tanh(tf.matmul(emb, D_W1) + D_b1)
    l2_outputs = tf.nn.tanh(tf.matmul(l1_outputs, D_W2) + D_b2)
    D_logit = tf.matmul(l2_outputs, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)

    return D_prob, D_logit


'''loss'''
fake_user_emb = generator(attribute_id)
D_real, D_logit_real = discriminator(attribute_id, real_user_emb)
D_fake, D_logit_fake = discriminator(attribute_id, fake_user_emb)

D_counter, D_logit_counter = discriminator(neg_attribute_id, neg_user_emb)

D_loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))

D_loss_fake = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
D_loss_counter = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_counter, labels=tf.zeros_like(D_logit_counter)))

D_regular = alpha * (tf.nn.l2_loss(D_attri_matrix) + tf.nn.l2_loss(D_W1) + tf.nn.l2_loss(D_b1) + tf.nn.l2_loss(
    D_W2) + tf.nn.l2_loss(D_b2) + tf.nn.l2_loss(D_W3) + tf.nn.l2_loss(D_b3))
G_regular = alpha * (tf.nn.l2_loss(G_attri_matrix) + tf.nn.l2_loss(G_W1) +
                     tf.nn.l2_loss(G_b1) + tf.nn.l2_loss(G_W2) + tf.nn.l2_loss(G_b2) + tf.nn.l2_loss(
            G_W2) + tf.nn.l2_loss(G_b2) + tf.nn.l2_loss(G_W3) + tf.nn.l2_loss(G_b3))

D_loss = (1 - alpha) * (D_loss_real + D_loss_fake + D_loss_counter) + D_regular
G_loss = (1 - alpha) * (tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))) + G_regular

'''optimizer'''
D_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(D_loss, var_list=D_params)
G_solver = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(G_loss, var_list=G_params)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

test_item_batch, test_attribute_vec = support.get_testdata()
print('111111111111111')
test_G_user = sess.run(fake_user_emb, feed_dict={attribute_id: test_attribute_vec})
print(test_G_user)
print('222222222222222')