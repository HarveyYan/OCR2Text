import os
import cv2
import sys
import numpy as np
from sklearn.model_selection import KFold

basedir = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
sys.path.append(basedir)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

dataset_dir = os.path.join(basedir, 'Data', 'cell_images', 'training_set', 'BW')
dataset_target_file = os.path.join(basedir, 'Data', 'cell_images', 'training_set_values.txt')
expr_data_dir = os.path.join(basedir, 'Data', 'cell_images', 'validation_set', 'BW')
expr_target_file = os.path.join(basedir, 'Data', 'cell_images', 'validation_set_values.txt')
validation_annotations_file = os.path.join(basedir, 'Data', 'cell_images', 'validation_solutions.txt')

all_allowed_characters = list(map(lambda i: str(i), range(10))) + ['.' , '-', '!'] #+ ['-', '.', ',', '!']  # '!' is the eol signal
max_size = None
max_target_length = None
image_load_func = None
digits_limit = 10


def determine_largest_size(path_images):
    max_size = [0, 0]
    for path_image in path_images:
        im = cv2.imread(path_image)
        for i, s in enumerate(im.shape[:2]):
            if max_size[i] < s:
                max_size[i] = s
    return max_size


def load_and_preprocess_image(path_images):
    if max_size is None:
        raise ValueError('Need to determine a consensus size!')
    all_images = []
    for img_ph in path_images:
        img_data = cv2.imread(img_ph)[:, :, :1] // 255.
        size = list(img_data.shape)

        if size[0] != max_size[0]: # height
            top = abs(max_size[0] - size[0]) // 2
            down = abs(max_size[0] - size[0]) - top

            if size[0] < max_size[0]:
                img_data = np.concatenate(
                    [np.zeros((top, size[1], size[-1])), img_data, np.zeros((down, size[1], size[-1]))], axis=0)
            else:
                img_data = img_data[top:size[0] - down, :, :]
        size[0] = max_size[0]

        if size[1] != max_size[1]: # width
            left = abs(max_size[1] - size[1]) // 2
            right = abs(max_size[1] - size[1]) - left

            if size[1] < max_size[1]:
                img_data = np.concatenate(
                    [np.zeros((size[0], left, size[-1])), img_data, np.zeros((size[0], right, size[-1]))], axis=1)
            else:
                img_data = img_data[:, left:size[1] - right, :]
        size[1] = max_size[1]

        all_images.append(img_data)
    all_labeled_images = np.stack(all_images)
    return all_labeled_images


def load_ocr_dataset(**kwargs):
    # splits the dataset into 10 fold cross validation, or a held-out train-test split
    use_cross_validation = kwargs.get('use_cross_validation', False)

    if kwargs.get('seed') is not None:
        print('Setting seed to %d' % (kwargs.get('seed')))
        np.random.seed(kwargs.get('seed'))

    # load all training targets and ids
    all_targets = []
    all_length_targets = []
    all_ids = []
    with open(dataset_target_file, 'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            all_ids.append(line.rstrip().split(';')[0])
            target = line.rstrip().split(';')[-1]
            target = target.replace(',', '.')
            encode_target = []
            for i, c in enumerate(target):
                if c in all_allowed_characters:
                    encode_target.append(all_allowed_characters.index(c))
            available_length = len(encode_target)
            if len(encode_target) < digits_limit:
                encode_target += [all_allowed_characters.index('!')] * (digits_limit - len(encode_target))
            elif len(encode_target) > digits_limit:
                encode_target = encode_target[:digits_limit]
                available_length = digits_limit
            # this is to ensure all targets are of the same length
            all_targets.append(encode_target)
            all_length_targets.append(available_length)
    all_targets = np.array(all_targets)
    all_length_targets = np.array(all_length_targets)
    all_ids = np.array(all_ids)

    # maximum length target
    global max_target_length
    max_target_length = np.max(all_length_targets)

    # to estimate the largest picture width and height, for downstream padding
    # we need uniform length of the input to enable batch optimization
    global max_size
    max_size = determine_largest_size([os.path.join(dataset_dir, 'clean'+id) for id in all_ids])
    # load all images with the order of the targets
    all_labeled_images = load_and_preprocess_image(
        np.array([os.path.join(dataset_dir, 'clean'+id) for id in all_ids]))

    # initial shuffle
    total_size = len(all_labeled_images)
    permute = np.random.permutation(np.arange(total_size, dtype=np.int32))
    all_ids = all_ids[permute]
    all_labeled_images = all_labeled_images[permute]
    all_targets = all_targets[permute]
    all_length_targets = all_length_targets[permute]

    if use_cross_validation:
        kf = KFold(n_splits=10)
        splits = kf.split(all_labeled_images)
        return {
            'all_ids': all_ids,
            'all_images': all_labeled_images,
            'all_targets': all_targets,
            'all_length_targets': all_length_targets,
            'splits': splits
        }
    else:
        test_ids = all_ids[-int(total_size * 0.1):]
        test_images = all_labeled_images[-int(total_size * 0.1):]
        test_targets = all_targets[-int(total_size * 0.1):]
        test_length_targets = all_length_targets[-int(total_size * 0.1):]

        ids = all_ids[:-int(total_size * 0.1)]
        images = all_labeled_images[:-int(total_size * 0.1)]
        targets = all_targets[:-int(total_size * 0.1)]
        length_targets = all_length_targets[:-int(total_size * 0.1)]

        print('dataset size %d\ntraining set %d\ntest set %d' % (
            total_size, len(images), len(test_images)))

        return {
            'train_ids': ids,
            'train_images': images,
            'train_targets': targets,
            'train_length_targets': length_targets,
            'test_ids': test_ids,
            'test_images': test_images,
            'test_targets': test_targets,
            'test_length_targets': test_length_targets
        }


def load_expr_data():
    '''prepare the training targets'''
    all_ids = []
    with open(expr_target_file, 'r') as file:
        lines = file.readlines()[1:]
        for line in lines:
            all_ids.append(line.rstrip().split(';')[0])
    return load_and_preprocess_image([os.path.join(expr_data_dir, 'clean'+id) for id in all_ids]), \
           all_ids


def compute_score(path_to_prediction_file):
    print('Classification errors')
    with open(validation_annotations_file, 'r') as truth_file, open(path_to_prediction_file, 'r') as pred_file:
        truth = {}
        for line in truth_file.readlines()[1:]:
            truth[line.rstrip().split(';')[0]] = line.rstrip().split(';')[-1].replace(',', '.')
        correct = 0
        for line in pred_file.readlines()[1:]:
            if line.rstrip().split(';')[-1] == truth[line.rstrip().split(';')[0]]:
                correct += 1
            else:
                print(line.rstrip().split(';')[0], line.rstrip().split(';')[-1], truth[line.rstrip().split(';')[0]])
    return correct / len(truth)


if __name__ == "__main__":
    dataset = load_ocr_dataset(use_cross_validation=False)
    print(dataset['train_images'].shape)
    print(dataset['train_targets'])

    all_expr_images, all_expr_ids = load_expr_data()
    print(all_expr_images.shape)
    print(all_expr_ids)
