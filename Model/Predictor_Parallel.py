import numpy as np
import locale
import os
import sys
import tensorflow as tf

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.resutils import OptimizedResBlockDisc1, resblock
from lib.RNN_Encoder_Decoder import BiLSTMEncoder, AttentionDecoder, BeamAttDecoder
import lib.ops.LSTM, lib.ops.Linear
import lib.plot, lib.dataloader, lib.clr


class Predictor:

    def __init__(self, max_size, nb_emb, nb_class, nb_length_class, nb_max_digits, gpu_device_list=['/gpu:0'],
                 **kwargs):
        self.max_size = max_size
        self.nb_emb = nb_emb
        self.nb_class = nb_class
        self.nb_length_class = nb_length_class
        self.nb_max_digits = nb_max_digits
        self.gpu_device_list = gpu_device_list
        self.nb_mnist_class = 10

        # hyperparams
        self.arch = kwargs.get('arch', 0)
        self.use_bn = kwargs.get('use_bn', False)
        self.use_lstm = kwargs.get('use_lstm', True)
        self.nb_layers = kwargs.get('nb_layers', 4)
        self.resample = kwargs.get('resample', None)
        self.filter_size = kwargs.get('filter_size', 3)
        self.residual_connection = kwargs.get('residual_connection', 1.0)
        self.output_dim = kwargs.get('output_dim', 32)
        self.learning_rate = kwargs.get('learning_rate', 2e-4)
        self.use_clr = kwargs.get('use_clr', False)
        self.use_momentum = kwargs.get('use_momentum', False)
        self.length_obj_ratio = kwargs.get('length_obj_ratio', 0.1)
        self.beam_size = kwargs.get('beam_size', 5)

        self.g = tf.get_default_graph()
        with self.g.as_default():
            self._placeholders()
            self.mnist_pretrain_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

            if self.use_momentum:
                # everybody loves NAG — wait, scrape that one
                self.optimizer = tf.contrib.opt.MomentumWOptimizer(
                    1e-4, self.learning_rate * self.lr_multiplier,
                    0.9, use_nesterov=True
                )
            else:
                self.optimizer = tf.contrib.opt.AdamWOptimizer(
                    weight_decay=1e-4,
                    learning_rate=self.learning_rate * self.lr_multiplier
                )

            for i, device in enumerate(self.gpu_device_list):
                with tf.device(device), tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                    if self.arch == 0:
                        raise Exception('baseline disabled (constructing CTC)')
                        # self._build_residual_classifier(i)
                    elif self.arch == 1:
                        self._build_seq2seq(i, mode='training')
                    else:
                        self._build_beam_seq2seq(i, mode='training')
                    self._loss('CE' if self.nb_class > 1 else 'MSE', i)
                    self._train(i)

            with tf.device(self.gpu_device_list[0]), tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE):
                if self.arch == 0:
                    raise Exception('baseline disabled (constructing CTC)')
                    # self._build_residual_classifier(i)
                elif self.arch == 1:
                    self._build_seq2seq(None, mode='inference')
                else:
                    self._build_beam_seq2seq(None, mode='inference')

            self._merge()
            self.mnist_pretrain_op = self.mnist_pretrain_optimizer.apply_gradients(self.mnist_gv)
            self.train_op = self.optimizer.apply_gradients(self.gv)
            self._stats()
            self.saver = tf.train.Saver(max_to_keep=1000)
            self.init = tf.global_variables_initializer()
        self._init_session()

    def _placeholders(self):
        self.input_ph = tf.placeholder(tf.float32, shape=[None, *self.max_size, self.nb_emb])
        self.input_splits = tf.split(self.input_ph, len(self.gpu_device_list))

        self.inference_input_ph = tf.placeholder(tf.float32, shape=[None, *self.max_size, self.nb_emb])

        self.labels = tf.placeholder(tf.int32, shape=[None, self.nb_max_digits])
        self.labels_split = tf.split(
            tf.one_hot(self.labels, depth=self.nb_class),
            len(self.gpu_device_list)
        )

        self.mnist_labels = tf.placeholder(tf.int32, shape=[None, ])
        self.mnist_labels_splits = tf.split(
            tf.one_hot(self.mnist_labels, depth=10),
            len(self.gpu_device_list)
        )

        self.nb_digits_labels = tf.placeholder(tf.int32, shape=[None, ])
        self.nb_digits_labels_splits = tf.split(
            tf.one_hot(self.nb_digits_labels, depth=self.nb_length_class),
            len(self.gpu_device_list)
        )

        self.is_training_ph = tf.placeholder(tf.bool, ())
        self.global_step = tf.placeholder(tf.int32, ())
        self.hf_iters_per_epoch = tf.placeholder(tf.int32, ())
        if self.use_clr:
            self.lr_multiplier = lib.clr.cyclic_learning_rate(self.global_step, 0.5, 5.,
                                                              self.hf_iters_per_epoch, mode='exp_range')
        else:
            self.lr_multiplier = 1.

    def _build_seq2seq(self, split_idx, mode):
        '''
        A fairly basic seq2seq without beam search
        that passes attention vector to the next timestep at the decoding stage
        '''
        if mode == 'training':
            output = self.input_splits[split_idx]
        elif mode == 'inference':
            output = self.inference_input_ph
        else:
            raise ValueError('unknown mode')

        with tf.variable_scope('pretrain_effect_zone'):
            for i in range(self.nb_layers):
                if i == 0:
                    output = OptimizedResBlockDisc1(output, self.nb_emb, self.output_dim,
                                                    resample=None)
                else:
                    shape = output.get_shape().as_list()
                    output = resblock('ResBlock%d' % (i), shape[-1], shape[-1] * 2 if i % 2 == 1 else shape[-1],
                                      self.filter_size, output, self.resample if i % 2 == 1 else None,
                                      self.is_training_ph, use_bn=self.use_bn, r=self.residual_connection)

            output = tf.nn.relu(output)
            shape = output.get_shape().as_list()
            output = tf.reshape(
                tf.transpose(output, [0, 2, 1, 3]),
                [-1, np.prod(shape[1:3]), shape[-1]])

            mnist_output = lib.ops.Linear.linear('mnist_output', np.prod(shape[1:]), self.nb_mnist_class,
                                                 tf.reshape(output, [-1, np.prod(shape[1:])]))

        # auxiliary loss on length
        nb_digits_output = lib.ops.Linear.linear('NBDigitsLinear', shape[-1], self.nb_length_class,
                                                 tf.reduce_sum(output, axis=1))

        encoder_outputs, encoder_states = BiLSTMEncoder('Encoder', shape[-1], output, np.prod(shape[1:3]))
        # feature dim from BiLSTMEncoder is shape[-1] * 2
        decoder_outputs, decoder_states, att_weights = AttentionDecoder('Decoder', encoder_outputs, encoder_states,
                                                           self.nb_max_digits, )

        # translation output
        output = lib.ops.Linear.linear('MapToOutputEmb', shape[-1] * 2, self.nb_class, decoder_outputs)

        if mode == 'training':
            if not hasattr(self, 'output'):
                self.output = [output]
                self.mnist_output = [mnist_output]
                self.nb_digits_output = [nb_digits_output]
            else:
                self.output += [output]
                self.mnist_output += [mnist_output]
                self.nb_digits_output += [nb_digits_output]
        else:
            self.inference_output = output
            self.inference_att_weights = att_weights

    def _build_beam_seq2seq(self, split_idx, mode):
        '''
        a seq2seq that uses beam search at inference stage.
        Only the sampled token is passed down to the next step at the inference stage.
        '''
        if mode == 'training':
            output = self.input_splits[split_idx]
        elif mode == 'inference':
            output = self.inference_input_ph
        else:
            raise ValueError('unknown mode')

        with tf.variable_scope('pretrain_effect_zone'):
            for i in range(self.nb_layers):
                if i == 0:
                    output = OptimizedResBlockDisc1(output, self.nb_emb, self.output_dim,
                                                    resample=None)
                else:
                    shape = output.get_shape().as_list()  # no downsampling
                    output = resblock('ResBlock%d' % (i), shape[-1], shape[-1] * 2 if i % 2 == 1 else shape[-1],
                                      self.filter_size, output, self.resample if i % 2 == 1 else None,
                                      self.is_training_ph, use_bn=self.use_bn, r=self.residual_connection)

            output = tf.nn.relu(output)
            shape = output.get_shape().as_list()
            output = tf.reshape(
                tf.transpose(output, [0, 2, 1, 3]),
                [-1, np.prod(shape[1:3]), shape[-1]])

            mnist_output = lib.ops.Linear.linear('mnist_output', np.prod(shape[1:]), self.nb_mnist_class,
                                                 tf.reshape(output, [-1, np.prod(shape[1:])]))

        # auxiliary loss on length
        nb_digits_output = lib.ops.Linear.linear('NBDigitsLinear', shape[-1], self.nb_length_class,
                                                 tf.reduce_sum(output, axis=1))

        encoder_outputs, encoder_states = BiLSTMEncoder('Encoder', shape[-1], output, np.prod(shape[1:3]))
        # feature dim from BiLSTMEncoder is shape[-1] * 2
        decoder_outputs, decoder_states = BeamAttDecoder('Decoder', encoder_outputs, encoder_states, self.nb_max_digits,
                                                         self.nb_class, mode=mode, beam_size=self.beam_size)

        # translation output
        output = decoder_outputs

        if mode == 'training':
            if not hasattr(self, 'output'):
                self.output = [output]
                self.mnist_output = [mnist_output]
                self.nb_digits_output = [nb_digits_output]
            else:
                self.output += [output]
                self.mnist_output += [mnist_output]
                self.nb_digits_output += [nb_digits_output]
        else:
            self.inference_output = output # [0]: beam tokens, [1] marginal logprob, [2] attention_weights

    def _loss(self, type, split_idx):
        if type == 'CE':
            # compute a more efficient loss?
            cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.output[split_idx],
                    labels=self.labels_split[split_idx]
                ))
            mnist_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.mnist_output[split_idx],
                    labels=self.mnist_labels_splits[split_idx]
                ))
            length_cost = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.nb_digits_output[split_idx],
                    labels=self.nb_digits_labels_splits[split_idx]
                )
            )
            prediction = tf.nn.softmax(self.output[split_idx], axis=-1)
        elif type == 'MSE':
            raise ValueError('MSE is not appropriate in this problem!')
        else:
            raise ValueError('%s doesn\'t supported. Valid options are \'CE\' and \'MSE\'.' % (type))

        pretrain_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(tf.argmax(self.mnist_output[split_idx], axis=-1)),
                    tf.to_int32(tf.argmax(self.mnist_labels_splits[split_idx], axis=-1))
                ),
                tf.float32
            )
        )

        # accuracy by comparing the modes
        char_acc = tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.to_int32(tf.argmax(self.output[split_idx], axis=-1)),
                    tf.to_int32(tf.argmax(self.labels_split[split_idx], axis=-1))
                ),
                tf.float32
            )
        )

        sample_acc = tf.reduce_mean(
            tf.reduce_prod(
                tf.cast(
                    tf.equal(
                        tf.to_int32(tf.argmax(self.output[split_idx], axis=-1)),
                        tf.to_int32(tf.argmax(self.labels_split[split_idx], axis=-1))
                    ),
                    tf.float32
                )
                , axis=-1)
        )

        if not hasattr(self, 'cost'):
            self.cost, self.prediction, self.char_acc, self.sample_acc, \
            self.mnist_cost, self.pretrain_acc, self.length_cost = \
                [cost], [prediction], [char_acc], [sample_acc], \
                [mnist_cost], [pretrain_acc], [length_cost]
        else:
            self.cost += [cost]
            self.prediction += [prediction]
            self.char_acc += [char_acc]
            self.sample_acc += [sample_acc]
            self.mnist_cost += [mnist_cost]
            self.pretrain_acc += [pretrain_acc]
            self.length_cost += [length_cost]

    def _train(self, split_idx):
        gv = self.optimizer.compute_gradients(self.cost[split_idx] + self.length_obj_ratio * self.length_cost[split_idx]
                                              , var_list=[var for var in tf.trainable_variables() if
                                                          'mnist_output' not in var.name])  # and 'NBDigits' not in var.name])

        mnist_gv = self.mnist_pretrain_optimizer.compute_gradients(self.mnist_cost[split_idx]
                                                                   , var_list=[var for var in tf.trainable_variables()
                                                                               if 'pretrain' in var.name])
        if not hasattr(self, 'gv'):
            self.gv = [gv]
            self.mnist_gv = [mnist_gv]
        else:
            self.gv += [gv]
            self.mnist_gv += [mnist_gv]

    def _stats(self):
        # show all trainable weights
        for name, grads_and_vars in [('Predictor arch %d' % (self.arch), self.gv)]:
            print("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    print("\t{} ({})".format(v.name, shape_str))
            print("Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            ))

    def _merge(self):
        # output, prediction, cost, acc, pears, gv
        self.output = tf.concat(self.output, axis=0)
        self.prediction = tf.concat(self.prediction, axis=0)

        self.cost = tf.add_n(self.cost) / len(self.gpu_device_list)
        self.char_acc = tf.add_n(self.char_acc) / len(self.gpu_device_list)
        self.sample_acc = tf.add_n(self.sample_acc) / len(self.gpu_device_list)

        self.gv = self._average_gradients(self.gv)

        self.mnist_cost = tf.add_n(self.mnist_cost) / len(self.gpu_device_list)
        self.pretrain_acc = tf.add_n(self.pretrain_acc) / len(self.gpu_device_list)
        self.mnist_gv = self._average_gradients(self.mnist_gv)

        self.length_cost = tf.add_n(self.length_cost) / len(self.gpu_device_list)

    def _average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def _init_session(self):
        gpu_options = tf.GPUOptions()
        gpu_options.allow_growth = True
        self.sess = tf.Session(graph=self.g, config=tf.ConfigProto(gpu_options=gpu_options))
        self.sess.run(self.init)

    def reset_session(self):
        del self.saver
        with self.g.as_default():
            self.saver = tf.train.Saver(max_to_keep=100)
        self.sess.run(self.init)
        lib.plot.reset()

    def uniform_data(self, data):
        if data.shape[1] != self.max_size[0]:  # height
            top = abs(self.max_size[0] - data.shape[1]) // 2
            down = abs(self.max_size[0] - data.shape[1]) - top

            if data.shape[1] < self.max_size[0]:
                data = np.concatenate(
                    [np.zeros((data.shape[0], top, data.shape[2], self.nb_emb)), data,
                     np.zeros((data.shape[0], down, data.shape[2], self.nb_emb))], axis=1)
            else:
                data = data[:, top:data.shape[1] - down, :, :]

        if data.shape[2] != self.max_size[1]:
            left = abs(data.shape[2] - self.max_size[1]) // 2
            right = abs(data.shape[2] - self.max_size[1]) - left

            if data.shape[2] < self.max_size[1]:
                data = np.concatenate(
                    [np.zeros((data.shape[0], data.shape[1], left, self.nb_emb)), data,
                     np.zeros((data.shape[0], data.shape[1], right, self.nb_emb))], axis=2)
            else:
                data = data[:, :, left:data.shape[2] - right, :]

        return data

    def mnist_pretrain(self, batch_size, output_dir):
        ''' first stage of pretraining — CNN filters'''
        pretrain_dir = os.path.join(output_dir, 'pretrain_resnet/')
        os.makedirs(pretrain_dir)

        ((train_data, train_targets), (test_data, test_targets)) = tf.keras.datasets.mnist.load_data()
        train_data = np.concatenate([train_data[:, :, :, None]] * self.nb_emb, axis=-1)
        test_data = np.concatenate([test_data[:, :, :, None]] * self.nb_emb, axis=-1)
        train_data = self.uniform_data(train_data) / 255.
        test_data = self.uniform_data(test_data) / 255.

        size_train = len(train_data)
        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        lib.plot.set_output_dir(pretrain_dir)
        for epoch in range(50):
            permute = np.random.permutation(np.arange(size_train))
            train_data = train_data[permute]
            train_targets = train_targets[permute]

            # trim
            train_rmd = train_data.shape[0] % len(self.gpu_device_list)
            if train_rmd != 0:
                train_data = train_data[:-train_rmd]
                train_targets = train_targets[:-train_rmd]

            for i in range(iters_per_epoch):
                _data, _labels = train_data[i * batch_size: (i + 1) * batch_size], \
                                 train_targets[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.mnist_pretrain_op,
                              feed_dict={self.input_ph: _data,
                                         self.mnist_labels: _labels,
                                         self.is_training_ph: True}
                              )

            train_cost, train_acc = self.mnist_evaluate(train_data, train_targets, batch_size)
            lib.plot.plot('train_cost', train_cost)
            lib.plot.plot('train_acc', train_acc)

            dev_cost, dev_acc = self.mnist_evaluate(test_data, test_targets, batch_size)
            lib.plot.plot('dev_cost', dev_cost)
            lib.plot.plot('dev_acc', dev_acc)

            lib.plot.flush()
            lib.plot.tick()
        lib.plot.reset()

        ''' second stage of pretraining — seq2seq'''
        pretrain_dir = os.path.join(output_dir, 'pretrain_seq2seq/')
        os.makedirs(pretrain_dir)
        # arrange targets to translation format
        train_targets = np.array(list(map(
            lambda x: [x] + [lib.dataloader.all_allowed_characters.index('!')]
                      * (self.nb_max_digits - 1), train_targets)))
        test_targets = np.array(list(map(
            lambda x: [x] + [lib.dataloader.all_allowed_characters.index('!')]
                      * (self.nb_max_digits - 1), test_targets)))

        train_length_targets = np.ones((train_targets.shape[0],))
        test_length_targets = np.ones((test_targets.shape[0],))

        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        lib.plot.set_output_dir(pretrain_dir)
        for epoch in range(50):
            permute = np.random.permutation(np.arange(size_train))
            train_data = train_data[permute]
            train_targets = train_targets[permute]

            # trim
            train_rmd = train_data.shape[0] % len(self.gpu_device_list)
            if train_rmd != 0:
                train_data = train_data[:-train_rmd]
                train_targets = train_targets[:-train_rmd]

            for i in range(iters_per_epoch):
                _data, _labels, _length_labels = train_data[i * batch_size: (i + 1) * batch_size], \
                                                 train_targets[i * batch_size: (i + 1) * batch_size], \
                                                 train_length_targets[i * batch_size: (i + 1) * batch_size]

                self.sess.run(self.train_op,
                              feed_dict={self.input_ph: _data,
                                         self.labels: _labels,
                                         self.global_step: i,
                                         self.hf_iters_per_epoch: iters_per_epoch // 2,
                                         self.nb_digits_labels: _length_labels,
                                         self.is_training_ph: True}
                              )

            train_cost, train_char_acc, train_sample_acc, train_length_cost = \
                self.evaluate(train_data, train_targets, train_length_targets, batch_size)
            lib.plot.plot('train_cost', train_cost)
            lib.plot.plot('train_char_acc', train_char_acc)
            lib.plot.plot('train_sample_acc', train_sample_acc)
            lib.plot.plot('train_length_cost', train_length_cost)

            dev_cost, dev_char_acc, dev_sample_acc, dev_length_cost = \
                self.evaluate(test_data, test_targets, test_length_targets, batch_size)
            lib.plot.plot('dev_cost', dev_cost)
            lib.plot.plot('dev_char_acc', dev_char_acc)
            lib.plot.plot('dev_sample_acc', dev_sample_acc)
            lib.plot.plot('dev_length_cost', dev_length_cost)

            lib.plot.flush()
            lib.plot.tick()
        lib.plot.reset()

    def mnist_evaluate(self, X, y, batch_size):
        iters_per_epoch = len(X) // batch_size + (0 if len(X) % batch_size == 0 else 1)
        all_cost, all_acc = 0., 0.
        for i in range(iters_per_epoch):
            _data, _labels = X[i * batch_size: (i + 1) * batch_size], \
                             y[i * batch_size: (i + 1) * batch_size]
            _cost, _acc \
                = self.sess.run([self.mnist_cost, self.pretrain_acc],
                                feed_dict={self.input_ph: _data,
                                           self.mnist_labels: _labels,
                                           self.is_training_ph: False}
                                )
            all_cost += _cost * _data.shape[0]
            all_acc += _acc * _data.shape[0]
        return all_cost / len(X), all_acc / len(X)

    def fit(self, X, y, y_len, epochs, batch_size, output_dir,
            dev_data=None, dev_targets=None, dev_length_targets=None):
        checkpoints_dir = os.path.join(output_dir, 'checkpoints/')
        os.makedirs(checkpoints_dir)

        if dev_data is None or dev_targets is None or dev_length_targets is None:
            # split validation set
            dev_data = X[:int(len(X) * 0.1)]
            dev_targets = y[:int(len(X) * 0.1)]
            dev_length_targets = y_len[:int(len(X) * 0.1)]

            X = X[int(len(X) * 0.1):]
            y = y[int(len(y) * 0.1):]
            y_len = y_len[int(len(y) * 0.1):]

        # trim development set, batch size should be a multiple of len(self.gpu_device_list)
        dev_rmd = dev_data.shape[0] % len(self.gpu_device_list)
        if dev_rmd != 0:
            dev_data = dev_data[:-dev_rmd]
            dev_targets = dev_targets[:-dev_rmd]
            dev_length_targets = dev_length_targets[:-dev_rmd]

        size_train = len(X)
        iters_per_epoch = size_train // batch_size + (0 if size_train % batch_size == 0 else 1)
        best_dev_cost = np.inf
        lib.plot.set_output_dir(output_dir)
        for epoch in range(epochs):
            permute = np.random.permutation(np.arange(size_train))
            train_data = X[permute]
            train_targets = y[permute]
            train_length_targets = y_len[permute]

            # trim
            train_rmd = train_data.shape[0] % len(self.gpu_device_list)
            if train_rmd != 0:
                train_data = train_data[:-train_rmd]
                train_targets = train_targets[:-train_rmd]
                train_length_targets = train_length_targets[:-train_rmd]

            for i in range(iters_per_epoch):
                _data, _labels, _labels_len \
                    = train_data[i * batch_size: (i + 1) * batch_size], \
                      train_targets[i * batch_size: (i + 1) * batch_size], \
                      train_length_targets[i * batch_size: (i + 1) * batch_size]
                self.sess.run(self.train_op,
                              feed_dict={self.input_ph: _data,
                                         self.labels: _labels,
                                         self.global_step: i,
                                         self.hf_iters_per_epoch: iters_per_epoch // 2,
                                         self.nb_digits_labels: _labels_len,
                                         self.is_training_ph: True}
                              )

            train_cost, train_char_acc, train_sample_acc, train_length_cost = \
                self.evaluate(train_data, train_targets, train_length_targets, batch_size)
            lib.plot.plot('train_cost', train_cost)
            lib.plot.plot('train_char_acc', train_char_acc)
            lib.plot.plot('train_sample_acc', train_sample_acc)
            lib.plot.plot('train_length_cost', train_length_cost)

            dev_cost, dev_char_acc, dev_sample_acc, dev_length_cost = \
                self.evaluate(dev_data, dev_targets, dev_length_targets, batch_size)
            lib.plot.plot('dev_cost', dev_cost)
            lib.plot.plot('dev_char_acc', dev_char_acc)
            lib.plot.plot('dev_sample_acc', dev_sample_acc)
            lib.plot.plot('dev_length_cost', dev_length_cost)

            lib.plot.flush()
            lib.plot.tick()

            if dev_cost + dev_length_cost * self.length_obj_ratio < best_dev_cost:
                best_dev_cost = dev_cost + dev_length_cost * self.length_obj_ratio
                save_path = self.saver.save(self.sess, checkpoints_dir, global_step=epoch)
                print('Validation sample acc improved. Saved to path %s\n' % (save_path), flush=True)
            else:
                print('\n', flush=True)

        print('Loading best weights %s' % (save_path), flush=True)
        self.saver.restore(self.sess, save_path)

    def evaluate(self, X, y, y_len, batch_size):
        iters_per_epoch = len(X) // batch_size + (0 if len(X) % batch_size == 0 else 1)
        all_cost, all_char_acc, all_sample_acc, all_length_cost = 0., 0., 0., 0.
        for i in range(iters_per_epoch):
            _data, _labels, _labels_len = X[i * batch_size: (i + 1) * batch_size], \
                                          y[i * batch_size: (i + 1) * batch_size], \
                                          y_len[i * batch_size: (i + 1) * batch_size]
            _cost, _char_acc, _sample_acc, _length_cost \
                = self.sess.run([self.cost, self.char_acc, self.sample_acc, self.length_cost],
                                feed_dict={self.input_ph: _data,
                                           self.labels: _labels,
                                           self.nb_digits_labels: _labels_len,
                                           self.is_training_ph: False}
                                )
            all_cost += _cost * _data.shape[0]
            all_char_acc += _char_acc * _data.shape[0]
            all_sample_acc += _sample_acc * _data.shape[0]
            all_length_cost += _length_cost * _data.shape[0]
        return all_cost / len(X), all_char_acc / len(X), all_sample_acc / len(X), all_length_cost / len(X)

    def predict(self, X):
        return self.sess.run(self.inference_output,
                             {self.inference_input_ph: X,
                              self.is_training_ph: False})

    def delete(self):
        self.sess.close()

    def load(self, chkp_path):
        self.saver.restore(self.sess, chkp_path)
