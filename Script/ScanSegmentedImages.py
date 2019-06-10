import os
import sys
import datetime
import numpy as np
import tensorflow as tf

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import lib.dataloader
from Model.Predictor_Parallel import Predictor

tf.app.flags.DEFINE_string('output_dir', '', '')
tf.app.flags.DEFINE_integer('epochs', 200, '')
tf.app.flags.DEFINE_integer('nb_gpus', 1, '')
FLAGS = tf.app.flags.FLAGS

BATCH_SIZE = 200 * FLAGS.nb_gpus  if FLAGS.nb_gpus > 0 else 200
EPOCHS = FLAGS.epochs  # How many iterations to train for
N_EMB = 1 # 3 channels for augmented images
DEVICES = ['/gpu:%d' % (i) for i in range(FLAGS.nb_gpus)] if FLAGS.nb_gpus > 0 else ['/cpu:0']

cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if FLAGS.output_dir == '':
    output_dir = os.path.join('output', cur_time)
else:
    output_dir = os.path.join('output', cur_time + '-' + FLAGS.output_dir)
os.makedirs(output_dir)

lib.dataloader.load_ocr_dataset()

arch = 1
use_bn = True
use_lstm = True
nb_layers = 2
filter_size = 3
output_dim = 32
learning_rate = 2e-4

HParams = ['arch', 'use_bn', 'use_lstm', 'nb_layers', 'filter_size', 'output_dim', 'learning_rate']
metrics = ['cost', 'char_acc', 'sample_acc']
hp = {}
for param in HParams:
    hp[param] = eval(param)

print('Building model with hyper-parameters\n', hp)

predictor = Predictor(lib.dataloader.max_size, 1, len(lib.dataloader.all_allowed_characters), DEVICES, **hp)
predictor.load(os.path.join(basedir, 'output', '20190609-003606-pretrain-seq2seq-include-punc', 'checkpoints', '-2060'))

# load segmented dataset
training_dataset_path = os.path.join(basedir, 'Data', 'cell_images', 'training_set', 'segments')
test_dataset_path = os.path.join(basedir, 'Data', 'cell_images', 'validation_set', 'segments')

all_img_path = []
all_img_id = []
# make predictions only
for test_sample_dir in os.listdir(test_dataset_path):
    for jpg in os.listdir(os.path.join(test_dataset_path, test_sample_dir)):
        all_img_path.append(os.path.join(test_dataset_path, test_sample_dir, jpg))
        all_img_id.append(test_sample_dir+'.jpg')

all_img = lib.dataloader.load_and_preprocess_image(all_img_path)
predictions = predictor.predict(all_img, BATCH_SIZE)

print(all_img_id)
print(predictions)

outfile = open(os.path.join(output_dir, 'validation_set_values.txt'), 'w')

prev_id = None
prev_decoded = 'filename;value'
# decode step
for pred, id in zip(predictions, all_img_id):
    decoded_digits = [lib.dataloader.all_allowed_characters[pos] for pos in np.argmax(pred, axis=-1)]
    cutoff = decoded_digits.index('!') if '!' in decoded_digits else len(decoded_digits)
    decoded = ''.join(decoded_digits[:cutoff])

    if prev_id is None or prev_id != id:
        if prev_id is None:
            outfile.write(prev_decoded+'\n')
        else:
            outfile.write('%s;%s\n' % (id, prev_decoded))
        prev_id = id
        prev_decoded = decoded
    else:
        prev_decoded += decoded
    outfile.flush()
outfile.close()