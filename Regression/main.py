from data_management import *
from regression_network import *
import argparse
import tensorflow as tf
import sys
import numpy as np

FLAGS = None
KEEP_RATE = 0.8


def do_eval(sess, images_pl, labels_pl, images, labels, keep_prob):
    accuracy = evaluation(images_pl, labels_pl, keep_prob)
    result = sess.run(accuracy, feed_dict={images_pl: images, labels_pl: labels})
    print('evaluation (l2 loss) :', result)


def write_eval(sess,images_pl, labels_pl, Test, keep_prob, dir_out_bb):
    output = inference(images_pl, keep_prob)
    predicted_bb = sess.run(output, {images_pl: Test.Images,
                                  labels_pl: Test.Label, keep_prob: 1.0})

    np.savetxt(dir_out_bb, predicted_bb, delimiter=' ', fmt='%d')


def display_progression_epoch(j, id_max, epoch):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % of the epoch ' + str(epoch+1) + ' completed' + chr(13))
    sys.stdout.flush


def run_training():
    print('loading data set :', FLAGS.data_set_dir, '...')
    data_set = load_data(FLAGS.data_set_dir)
    with tf.device('/cpu:0'):
        print('constructing graph ...')

        labels_pl, images_pl, keep_prob = placeholder_training(data_set.sizeImage, data_set.sizeLabel)

        output = inference(images_pl, keep_prob)
        loss = regression_loss(output, labels_pl)
        train_op = training(loss)
        precision = evaluation(images_pl, labels_pl, keep_prob)

        init = tf.global_variables_initializer()

        #force on cpu
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)

        sess.run(init)

        id_max = int(data_set.Train.num_example / FLAGS.batch_size)

        for epoch in range(FLAGS.number_epoch):

            data_set.Train.shuffle_data()

            epoch_loss = 0
            for j in range(id_max):

                if j % 5 == 0:
                    display_progression_epoch(j, id_max, epoch)

                batch_x, batch_y = data_set.Train.get_batch(FLAGS.batch_size, j)

                _, loss_value = sess.run([train_op, loss], feed_dict={images_pl: batch_x,
                                                                      labels_pl: batch_y,
                                                                      keep_prob: KEEP_RATE})

                epoch_loss += loss_value

            print('Epoch ', epoch+1, '/', FLAGS.number_epoch, 'completed', '   loss:', epoch_loss)

        # print('Neural net sucessfully trained, starting evaluation on test data...')
        #
        # result = sess.run(precision, {images_pl: data_set.Test.Images,
        #                               labels_pl: data_set.Test.Label,
        #                               keep_prob: 1.0})
        # print('Evaluation DataTest :', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, default='/images',
                        help='Path to images Folder')
    parser.add_argument('--labels_file', type=str, default='labels.txt',
                        help='Path to label file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='How many images to train on at a time.')
    parser.add_argument('--number_epoch', type=int, default=1,
                        help='How many epoch.')
    parser.add_argument('--data_set_dir', type=str, default='BDD1.npz',
                        help='How many epoch.')
    FLAGS = parser.parse_args()
    run_training()
