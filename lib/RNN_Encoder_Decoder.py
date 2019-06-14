import tensorflow as tf
from lib.ops.Linear import linear

"""
What matters in the attention mechanism?

As hinted in the above equations, there are many different attention variants. 
These variants depend on the form of the scoring function and the attention function, 
and on whether the previous state $$h_{t-1}$$ is used instead of $$h_t$$ in the scoring function 
as originally suggested in (Bahdanau et al., 2015). 
Empirically, we found that only certain choices matter. 
First, the basic form of attention, i.e., direct connections between target and source, needs to be present. 
Second, it's important to feed the attention vector to the next timestep 
to inform the network about past attention decisions as demonstrated in (Luong et al., 2015). 
Lastly, choices of the scoring function can often result in different performance. 
See more in the benchmark results section.
"""


def BiLSTMEncoder(name, hidden_units, inputs, length):
    with tf.variable_scope(name):
        cell_forward = tf.nn.rnn_cell.LSTMCell(hidden_units, name='forward_lstm_cell')
        cell_backward = tf.nn.rnn_cell.LSTMCell(hidden_units, name='backward_lstm_cell')

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

        encoder_state = tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat([state_forward[0], state_backward[0]], axis=-1),
                                                      h=tf.concat([state_forward[1], state_backward[1]], axis=-1))

        return output, encoder_state


def AttentionDecoder(name, encoder_outputs, encoder_states, length):
    hidden_units = encoder_states[0].get_shape().as_list()[-1]
    print('hidden units in the decoder %d, same as the encoder' % (hidden_units))
    with tf.variable_scope(name):
        cell = tf.nn.rnn_cell.LSTMCell(hidden_units, name='decoder_lstm_cell')
        output = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        start_token = tf.zeros((tf.shape(encoder_states[0])[0], hidden_units))

        # unroll
        i = tf.constant(0)
        while_condition = lambda i, _1, _2, _3: tf.less(i, length)

        def body(i, output, state, input):
            cell_output, state = cell(input, state)
            # attention ( cell_output, encoder_outputs)
            attention_vector = Attention('DecoderATT', encoder_outputs, cell_output)
            output = output.write(i, attention_vector)
            return [tf.add(i, 1), output, state, attention_vector]

        _, output, state, att_vec = tf.while_loop(while_condition, body, [i, output, encoder_states, start_token])
        output = tf.transpose(output.stack(), [1, 0, 2])
        return output, state


def BeamAttDecoder(name, encoder_outputs, encoder_states, length, nb_emb,
                   teacher_forced_output=None, mode='training'):
    '''
    At training stage, teaching forcing can be enabled or disabled depending on
    if teacher_forced_output argument has been specified a tensor.

    At inference stage, teacher_forced_output must be disabled, and a beam searcher will be employed to
    extract top-k hypothesis.

    For most of the time, teacher_forced_output feature is disabled. Might be fun to try it someday...
    '''
    hidden_units = encoder_states[0].get_shape().as_list()[-1]
    if mode == 'training':
        print('hidden units in the decoder %d, same as the encoder' % (hidden_units))
        with tf.variable_scope(name):
            cell = tf.nn.rnn_cell.LSTMCell(hidden_units, name='decoder_lstm_cell')
            # output stores the probability logits given by the lstm projection
            output = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
            start_token = tf.zeros((tf.shape(encoder_states[0])[0], nb_emb))

            # unroll
            i = tf.constant(0)
            while_condition = lambda i, _1, _2, _3: tf.less(i, length)

            def body(i, output, state, input):
                cell_output, state = cell(input, state)
                # attention ( cell_output, encoder_outputs), output with dimension nb_emb
                attention_vector = Attention('DecoderATT', encoder_outputs, cell_output)
                logits_vector = linear('Logits', hidden_units, nb_emb, attention_vector)
                if teacher_forced_output is None:
                    token = tf.one_hot(tf.multinomial(logits_vector, 1)[:, 0], nb_emb)
                else:
                    token = teacher_forced_output[:, i, :]
                # token = tf.concat([attention_vector, token], axis=-1) # concatenate the token with attention_vector
                output = output.write(i, logits_vector)
                return [tf.add(i, 1), output, state, token]

            _, output, state, att_vec = tf.while_loop(while_condition, body, [i, output, encoder_states, start_token])
            output = tf.transpose(output.stack(), [1, 0, 2])
            return output, state
    elif mode == 'inference':
        '''add beam search for inference when it's ready'''
        raise Exception('Not there yet!')


def Attention(name, encoder_outputs, cell_output):
    input_dim = cell_output.get_shape().as_list()[-1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cell_output = linear('linear', input_dim, input_dim, cell_output)
        scores = tf.matmul(encoder_outputs, cell_output[:, None, :], transpose_b=True)[:, :, 0]
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.reduce_sum(encoder_outputs * attention_weights[:, :, None], axis=1)
        return tf.nn.tanh(linear('ATT_vector', input_dim * 2, input_dim,
                                 tf.concat([context_vector, cell_output], axis=-1)))


if __name__ == "__main__":
    encoder_outputs, encoder_states = BiLSTMEncoder('Encoder', 128, tf.random_normal((200, 32, 4)), 32)
    decoder_outputs, decoder_states = AttentionDecoder('Decoder', encoder_outputs, encoder_states, 32)

    print(decoder_outputs.get_shape().as_list())
