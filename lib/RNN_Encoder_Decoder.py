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
        att_weights = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True)
        start_token = tf.zeros((tf.shape(encoder_states[0])[0], hidden_units))

        # unroll
        i = tf.constant(0)
        while_condition = lambda i, _1, _2, _3, _4: tf.less(i, length)

        def body(i, output, att_weights, state, input):
            cell_output, state = cell(input, state)
            # attention ( cell_output, encoder_outputs)
            attention_vector, attention_weights = \
                Attention('DecoderATT', encoder_outputs, cell_output)
            output = output.write(i, attention_vector)
            att_weights = att_weights.write(i, attention_weights)
            return [tf.add(i, 1), output, att_weights, state, attention_vector]

        _, output, att_weights, state, att_vec = tf.while_loop(while_condition, body,
                                                               [i, output, att_weights, encoder_states, start_token])
        output = tf.transpose(output.stack(), [1, 0, 2])
        att_weights = tf.transpose(att_weights.stack(), [1, 0, 2], name='stack_att_weights')
        return output, state, att_weights


def BeamAttDecoder(name, encoder_outputs, encoder_states, length, nb_emb,
                   teacher_forced_output=None, mode='training', beam_size=5):
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
                attention_vector, _ = Attention('DecoderATT', encoder_outputs, cell_output)
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
        print('Inference uses beam search with width %d' % (beam_size))
        # Imperative to keep the names the same for the training stage
        with tf.variable_scope(name, reuse=True):
            cell = tf.nn.rnn_cell.LSTMCell(hidden_units, name='decoder_lstm_cell')
            batch_size = tf.shape(encoder_states[0])[0]
            # output stores the probability logits given by the lstm projection
            beam_tokens = tf.TensorArray(tf.float32, size=length, infer_shape=True, dynamic_size=True,
                                         clear_after_read=False)
            # tensorflow tensorarray forces to write once.
            start_token = tf.zeros((batch_size, nb_emb))  # [batch_size, nb_emb]
            index_base = tf.reshape(
                tf.tile(tf.expand_dims(tf.range(batch_size) * beam_size, axis=1), [1, beam_size]),
                [-1])  # [batch_size * beam_size, ]

            # nested while loop for updating old tokens ...
            def update_func(j, beam_tokens, real_path, i):
                # todo double check
                # j = tf.Print(j, [j])
                updates = tf.gather(beam_tokens.read(j+i*(i-1)//2), real_path)
                updates = tf.Print(updates, [tf.shape(updates)])
                beam_tokens = beam_tokens.write(j+i*(i+1)//2, updates)
                # beam_tokens = beam_tokens.write(j, tf.gather(updates, real_path))
                return [tf.add(j, 1), beam_tokens, real_path, i]

            # unroll
            i = tf.constant(0)
            while_condition = lambda i, _1, _2, _3, _4, _5: tf.less(i, length)

            def body(i, beam_tokens, encoder_outputs, state, input, marginal_logprob):
                # print(input.get_shape().as_list())
                # print(state[0].get_shape().as_list())
                cell_output, state = cell(input, state)

                # attention ( cell_output, encoder_outputs), output with dimension nb_emb
                attention_vector, _ = Attention('DecoderATT', encoder_outputs, cell_output)
                logits_vector = linear('Logits', hidden_units, nb_emb, attention_vector)

                # log p(x_i | x_{i-1}, .., x_1)
                conditional_logprob = tf.nn.log_softmax(logits_vector)  # [batch_size * beam_size, nb_emb]

                # log p(x_i, x_{i-1}, .., x_1) = log p(x_i | x_{i-1}, .., x_1) + log p(x_{i-1}, .., x_1)
                # dynamic programming
                marginal_logprob = tf.cond(tf.less(i, 1),  # [batch_size * beam_size, nb_emb]
                                           lambda: conditional_logprob,
                                           lambda: conditional_logprob + marginal_logprob[:, None])

                # marginal_logprob = tf.Print(marginal_logprob, [tf.shape(marginal_logprob)])
                # log p(x_{r1} | ...)
                # reshape to [batch_size, beam_size * nb_emb],
                # then select beam_size hypothesis from each batch, leading to [batch_size, beam_size]
                best_prob, best_idx = tf.nn.top_k(
                    tf.reshape(marginal_logprob, (batch_size, -1)), beam_size
                    # note, mixing all the beams here, [batch_size, beam_size * nb_emb]
                )
                # print(best_prob.get_shape().as_list())

                best_prob = tf.reshape(best_prob, (-1,))
                best_idx = tf.reshape(best_idx, (-1,))
                beam_index = best_idx // nb_emb  # beam index where the token comes from, [batch_size * beam_size]
                token_index = best_idx % nb_emb  # token index inside the beam
                real_path = index_base + beam_index

                # update marginal_logprob
                marginal_logprob = best_prob

                # update cell states
                state = tf.cond(
                    tf.less(i, 1),
                    lambda: tf.nn.rnn_cell.LSTMStateTuple(
                        c=tf.reshape(tf.stack([state[0]] * beam_size, axis=1),
                                     ((batch_size * beam_size, hidden_units))),
                        h=tf.reshape(tf.stack([state[1]] * beam_size, axis=1),
                                     ((batch_size * beam_size, hidden_units))),
                    ),
                    lambda: tf.nn.rnn_cell.LSTMStateTuple(
                        c=tf.gather(state[0], real_path),
                        h=tf.gather(state[1], real_path),
                    )
                )

                encoder_outputs = tf.cond(
                    tf.less(i, 1),
                    lambda: tf.reshape(tf.stack([encoder_outputs] * beam_size, axis=1),
                                       ((batch_size * beam_size, length, hidden_units))),
                    lambda: encoder_outputs
                )

                token = tf.reshape(tf.one_hot(token_index, depth=nb_emb), (-1, nb_emb))
                # [batch_size * beam_size, nb_emb]


                j = tf.constant(0)
                update_condtion = lambda j, *args: tf.less(j, i)
                _, beam_tokens, _, _ = tf.while_loop(update_condtion, update_func, [j, beam_tokens, real_path, i],
                                                  parallel_iterations=1)
                beam_tokens = beam_tokens.write(i*(i+3)//2, token)
                return [tf.add(i, 1), beam_tokens, encoder_outputs, state, token, marginal_logprob]

            _, beam_tokens, encoder_outputs, state, att_vec, marginal_logprob = \
                tf.while_loop(while_condition, body,
                              loop_vars=[i, beam_tokens, encoder_outputs, encoder_states, start_token,
                                         tf.zeros((batch_size * beam_size,))],
                              shape_invariants=[tf.TensorShape(()), tf.TensorShape(()),
                                                tf.TensorShape((None, None, None)),
                                                tf.nn.rnn_cell.LSTMStateTuple(c=tf.TensorShape((None, None)),
                                                                              h=tf.TensorShape((None, None))),
                                                tf.TensorShape((None, nb_emb)), tf.TensorShape((None,))])

            beam_tokens = tf.transpose(beam_tokens.stack(), [1, 0, 2])[:,-length:,:]

            return (beam_tokens, marginal_logprob), state


def Attention(name, encoder_outputs, cell_output):
    input_dim = cell_output.get_shape().as_list()[-1]

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        cell_output = linear('linear', input_dim, input_dim, cell_output)
        scores = tf.matmul(encoder_outputs, cell_output[:, None, :], transpose_b=True)[:, :, 0]
        attention_weights = tf.nn.softmax(scores, axis=-1)
        context_vector = tf.reduce_sum(encoder_outputs * attention_weights[:, :, None], axis=1)
        return tf.nn.tanh(linear('ATT_vector', input_dim * 2, input_dim,
                                 tf.concat([context_vector, cell_output], axis=-1))), attention_weights


if __name__ == "__main__":
    encoder_outputs, encoder_states = BiLSTMEncoder('Encoder', 128, tf.random_normal((200, 8, 4)), 8)
    BeamAttDecoder('Decoder', encoder_outputs, encoder_states, 8, 4)
    (beam_tokens, marginal_logprob), decoder_states = BeamAttDecoder('Decoder', encoder_outputs, encoder_states, 8, 4,
                                                                     mode='inference', beam_size=2)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    res = sess.run([beam_tokens, marginal_logprob])
    print(res[0].shape)
    print(res[1].shape)
