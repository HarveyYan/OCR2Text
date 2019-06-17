import os
import sys
import datetime
import numpy as np
import tensorflow as tf

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lib.dataloader, lib.plot
from Model.Predictor_Parallel import Predictor

tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 200, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
tf.app.flags.DEFINE_integer('digits_limits', 8, '')
tf.app.flags.DEFINE_bool('use_cross_validation', False, '')
tf.app.flags.DEFINE_bool('use_clr', True, '')
tf.app.flags.DEFINE_bool('use_momentum', False, '')
tf.app.flags.DEFINE_float('length_obj_ratio', 0.1, '')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 200 * FLAGS.nb_gpus  if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
N_EMB = 3 # 3 channels for augmented images
DEVICES = ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']

lib.dataloader.digits_limit = FLAGS.digits_limits
lib.dataloader.nb_channels = N_EMB
dataset = lib.dataloader.load_ocr_dataset(use_cross_validation=FLAGS.use_cross_validation)
N_CLASS = len(lib.dataloader.all_allowed_characters)

arch = 1
use_bn = True
use_lstm = True
nb_layers = 8
filter_size = 3
output_dim = 16
learning_rate = 2e-4
use_clr = FLAGS.use_clr
use_momentum = FLAGS.use_momentum
length_obj_ratio = FLAGS.length_obj_ratio

HParams = ['arch', 'use_bn', 'use_lstm', 'nb_layers', 'filter_size', 'output_dim',
           'learning_rate', 'use_clr', 'use_momentum', 'length_obj_ratio']
metrics = ['cost', 'char_acc', 'sample_acc']
hp = {}
for param in HParams:
    hp[param] = eval(param)

print('Building model with hyper-parameters\n', hp)

cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if FLAGS.output_dir == '':
    output_dir = os.path.join(basedir, 'output', cur_time)
else:
    output_dir = os.path.join(basedir, 'output', cur_time + '-' + FLAGS.output_dir)
os.makedirs(output_dir)

if FLAGS.use_cross_validation:

    # build model
    model = Predictor(lib.dataloader.max_size, N_EMB, N_CLASS, DEVICES,
                      lib.dataloader.max_target_length, lib.dataloader.digits_limit, **hp)

    cost, char_acc, sample_acc = 0., 0., 0.
    splits = dataset['splits']

    for fold, (train_idx, test_idx) in enumerate(splits):
        fold_dir = os.path.join(output_dir, 'fold-%d' % (fold))
        os.makedirs(fold_dir)

        model.mnist_pretrain(BATCH_SIZE, output_dir)
        model.fit(dataset['all_images'][train_idx], dataset['all_targets'][train_idx], dataset['all_length_targets'][train_idx],
                  EPOCHS, BATCH_SIZE, fold_dir)

        test_rmd = dataset['all_images'][test_idx].shape[0] % len(DEVICES)
        if test_rmd != 0:
            test_data = dataset['all_images'][test_idx][:-test_rmd]
            test_targets = dataset['all_targets'][test_idx][:-test_rmd]
            test_length_targets = dataset['all_length_targets'][test_idx][:-test_rmd]
        else:
            test_data = dataset['all_images'][test_idx]
            test_targets = dataset['all_targets'][test_idx]
            test_length_targets = dataset['all_length_targets'][test_idx]

        test_cost, test_char_acc, test_sample_acc = \
            model.evaluate(test_data, test_targets, test_length_targets, BATCH_SIZE)

        cost += test_cost
        char_acc += test_char_acc
        sample_acc += test_sample_acc

        model.reset_session()
    model.delete()
    del model

    met = {}
    for metric in metrics:
        met[metric] = eval(metric) / 5
    print('Combined fold evaluations', met)

else:
    # build model
    model = Predictor(lib.dataloader.max_size, N_EMB, N_CLASS,
                      lib.dataloader.max_target_length, lib.dataloader.digits_limit, DEVICES, **hp)
    model.mnist_pretrain(BATCH_SIZE, output_dir)
    model.fit(dataset['train_images'], dataset['train_targets'], dataset['train_length_targets'],
              EPOCHS, BATCH_SIZE, output_dir, dataset['test_images'], dataset['test_targets'], dataset['test_length_targets'])


all_expr_images, all_expr_ids = lib.dataloader.load_expr_data()
predictions = model.predict(all_expr_images)

outfile = open(os.path.join(output_dir, 'validation_set_values.txt'), 'w')
outfile.write('filename;value\n')
# decode step
for pred, id in zip(predictions, all_expr_ids):
    decoded_digits = [lib.dataloader.all_allowed_characters[pos] for pos in np.argmax(pred, axis=-1)]
    cutoff = decoded_digits.index('!') if '!' in decoded_digits else len(decoded_digits)
    decoded = ''.join(decoded_digits[:cutoff])
    print('%s : %s' % (id, decoded))
    outfile.write('%s;%s\n' % (id, decoded))
    outfile.flush()
outfile.close()

print('Evaluation:', lib.dataloader.compute_score(os.path.join(output_dir, 'validation_set_values.txt')))