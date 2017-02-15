import numpy as np
from PIL import Image
import glob
import sys
import tflearn

'''
generate data set object
BoundingBoxes.txt>array
Folder SonarJpg>array
'''

PERCENTAGE_TRAIN = 0.85


def display_progression_epoch(j, id_max):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + '% of the folder completed' + chr(13))
    sys.stdout.flush


def read_jpg(image_dir: str) -> np.array:
    """
    decode jpeg images
    :param image_dir:  folder containing jpg images
    :return: numpy array of vectorized images (column vector)
    """
    image_paths = glob.glob(image_dir + '/*')
    # image_paths.sort()
    nb_frames = len(image_paths)
    images = np.zeros((nb_frames, 128 * 128), dtype=np.uint8)
    cpt = 0
    for image_path in image_paths:

        if cpt % 5 == 0:
            display_progression_epoch(cpt, nb_frames)

        image = Image.open(image_path).convert('L')
        image = image.resize((128, 128))
        image = np.array(image, dtype=np.uint8)
        images[cpt, :] = image.reshape(128 * 128)
        cpt += 1
    return images


def read_bb(fname: str) -> np.array:
    """
    create array of labels from text file
    :param fname: string - name of file text with bounding boxes coordinates
    :return: numpy array [4*L] of labels
    """
    with open(fname) as f:
        content = f.readlines()
        content = [x.strip().split() for x in content]
        content = [list(map(int, x)) for x in content]
        # !!!!! divide by 2 because images are resized 128*128
        content = np.asarray(content)
        return content / 2


def preprocess_images(images):
    """"
    preprocess the images (mean substraction and normalization)
    :param images: array of vectorized images
    :return: array of preprocessed images
    """
    # cast array to float
    images = images.astype(np.float32)
    images -= np.mean(images)
    images /= np.std(images)
    return images


def save_data(save_dir, fname, output_dir):
    images = read_jpg(save_dir)
    labels = read_bb(fname)
    np.savez(output_dir, images=images, labels=labels)
    return images, labels


def load_data(save_dir):
    """
    :return: DataSet object
    """
    data = np.load(save_dir)
    data_set = DataSet(data['images'], data['labels'])
    return data_set


class Train:
    def __init__(self, images, lbl):
        self.Images = images
        self.Labels = lbl
        self.num_examples = images.shape[0]

    def get_batch(self, batch_size, batch_number):
        """
        create a batch with images from the dataset
        id_max = int(data.Train.num_example/batch_size)
        :return: batch array of vector
        """
        beg = batch_number * batch_size
        end = (batch_number + 1) * batch_size
        if end > self.Images.shape[0]:
            print('carefull : batch index out of dataset !')
            batch = self.Images[-batch_size:, :]
            return batch
        batch_images = self.Images[beg:end, :]
        batch_labels = self.Labels[beg:end]
        return batch_images, batch_labels

    def shuffle_data(self):
        """
        shuffle data Train useful after each epoch
        :return: void
        """
        images = self.Images
        labels = self.Labels
        images, labels = tflearn.data_utils.shuffle(images, labels)
        self.Labels = labels
        self.Images = images


class Test:
    def __init__(self, images, lbl):
        self.Images = images
        self.Labels = lbl
        self.num_examples = images.shape[0]

    def get_batch(self, batch_size, batch_number):
        """
        create a batch with images from the dataset
        id_max = int(data.Train.num_example/batch_size)
        :return: batch array of vector
        """
        beg = batch_number * batch_size
        end = (batch_number + 1) * batch_size
        if end > self.Images.shape[0]:
            print('carefull : batch index out of dataset !')
            batch = self.Images[-batch_size:, :]
            return batch
        batch_images = self.Images[beg:end, :]
        batch_labels = self.Labels[beg:end]
        return batch_images, batch_labels


class DataSet:
    def __init__(self, images, labels):
        # shuffling data and labels
        # images = np.column_stack((np.arange(images.shape[0]), images))
        # np.random.shuffle(images)
        # labels = labels[images[:, 0]]
        # images = preprocess_images(images[:, 1:])

        # preprocess images
        images = preprocess_images(images)
        # shuffling datas before dividing dataset
        images, labels = tflearn.data_utils.shuffle(images, labels)

        # 85% of the images used for the dataset
        end = int(PERCENTAGE_TRAIN * images.shape[0])
        images_train = images[:end, :]
        images_test = images[end + 1:, :]
        label_train = labels[:end]
        label_test = labels[end + 1:]

        self.Train = Train(images_train, label_train)
        self.Test = Test(images_test, label_test)
        self.sizeImage = images.shape[1]
        self.sizeLabel = labels.shape[1]
        self.num_examples = labels.shape[0]
