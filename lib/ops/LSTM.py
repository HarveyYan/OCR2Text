import tensorflow as tf
import numpy as np
import lib.ops.Linear
import warnings
import functools
from lib.ops.Conv2D import conv2d


def bilstm(name, hidden_units, inputs, length):
    with tf.variable_scope(name):
        cell_forward = tf.nn.rnn_cell.LSTMCell(hidden_units, name='forward_cell')
        cell_backward = tf.nn.rnn_cell.LSTMCell(hidden_units, name='backward_cell')

        state_forward = cell_forward.zero_state(tf.shape(inputs)[0], tf.float32)
        state_backward = cell_backward.zero_state(tf.shape(inputs)[0], tf.float32)

        input_forward = inputs
        input_backward = tf.reverse(inputs, [1])

        output_forward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        output_backward = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)

        # unroll
        i = tf.constant(0)
        while_condition = lambda i, _1, _2, _3, _4: tf.less(i, length)

        def body(i, output_forward, output_backward, state_forward, state_backward):
            cell_output_forward, state_forward = cell_forward(input_forward[:, i, :], state_forward)
            output_forward = output_forward.write(i, cell_output_forward)
            cell_output_backward, state_backward = cell_backward(input_backward[:, i, :], state_backward)
            output_backward = output_backward.write(i, cell_output_backward)
            return [tf.add(i, 1), output_forward, output_backward, state_forward, state_backward]

        _, output_forward, output_backward, state_forward, state_backward = tf.while_loop(while_condition, body,
                                                                                          [i, output_forward,
                                                                                           output_backward,
                                                                                           state_forward,
                                                                                           state_backward])
        output_forward = tf.transpose(output_forward.stack(), [1, 0, 2])
        output_backward = tf.reverse(tf.transpose(output_backward.stack(), [1, 0, 2]), [1])
        output = tf.concat([output_forward, output_backward], axis=2)

        print(output.get_shape().as_list())
        return output


def attention(name, attention_size, inputs):
    batch_size, length, nb_features = inputs.shape.as_list()
    with tf.variable_scope(name):
        context_vec = tf.tanh(
            lib.ops.Linear.linear('Context_Vector', nb_features, attention_size, tf.reshape(inputs, [-1, nb_features])))
        pre_weights_exp = tf.exp(
            tf.reshape(lib.ops.Linear.linear('Attention_weights', attention_size, 1, context_vec),
                       [tf.shape(inputs)[0], -1]))
        weights = pre_weights_exp / tf.reduce_sum(pre_weights_exp, 1)[:, None]
        output = tf.reduce_sum(inputs * weights[:, :, None], 1)
        return output


def sn_non_local_block_sim(name, inputs):
    with tf.variable_scope(name):
        batch_size, h, w, num_channels = inputs.get_shape().as_list()
        location_num = h * w

        # theta path
        theta = conv2d('theta-1x1', num_channels, num_channels, 1, inputs)
        theta = tf.reshape( theta, [-1, location_num, num_channels])

        # phi path
        phi = conv2d('phi-1x1', num_channels, num_channels, 1, inputs)
        # phi = tf.layers.max_pooling2d(inputs=phi, pool_size=[2, 2], strides=2)
        phi = tf.reshape(phi, [-1, location_num, num_channels])


        attn = tf.matmul(theta, phi, transpose_b=True) # [batch_size, location_num, downsampled_num]
        attn = tf.nn.softmax(attn)

        # g path
        # g = conv2d('g-1x1', num_channels, num_channels, 1, inputs)
        # g = tf.layers.max_pooling2d(inputs=g, pool_size=[2, 2], strides=2)
        g = tf.reshape(inputs, [-1, location_num, num_channels])

        attn_g = tf.matmul(attn, g) # [batch_size, location_num, num_channels]
        attn_g = tf.reshape(attn_g, [-1, h, w, num_channels])
        # sigma = tf.get_variable(
        #     'sigma_ratio', [], initializer=tf.constant_initializer(0.0))
        # attn_g = conv2d('recover-channel', num_channels, num_channels, 1, attn_g)
        return attn_g
