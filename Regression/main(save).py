from Regression.data_management import *
from Regression.regression_network import *
import argparse
import tensorflow as tf
import numpy as np
import sys

FLAGS = None
KEEP_RATE = 0.8


def do_eval(sess, x_placeholder, y_placeholder, images, labels, keep_prob):
    accuracy = evaluation(x_placeholder, y_placeholder, keep_prob)
    result = sess.run(accuracy, feed_dict={x_placeholder: images, y_placeholder: labels})
    print('evaluation (l2 loss) :', result)


def display_progression_epoch(j, id_max, epoch):
    batch_progression = int((j / id_max) * 100)
    # display the progression of the current epoch on the terminal
    sys.stdout.write(str(batch_progression) + ' % of the epoch ' + str(epoch) + ' completed' + chr(13))
    sys.stdout.flush


def main():
    print('loading data set :', FLAGS.data_set_dir, '...')
    data_set = load_data(FLAGS.data_set_dir)

    print('number of test images : ', data_set.Train.num_example, '; number of test Images', data_set.Test.num_test,
          '\nshape of Images :',
          np.shape(data_set.Train.Images), '\nshape of Labels :', np.shape(data_set.Train.Label), )

    print('constructing graph ...')

    y_placeholder, x_placeholder, keep_prob = placeholder_training(data_set.sizeImage, data_set.sizeLabel)

    output = inference(x_placeholder, keep_prob)
    loss = regression_loss(output, y_placeholder)
    train_op = training(loss)
    precision = evaluation(x_placeholder, y_placeholder, keep_prob)

    # tensorboard
    writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())

    print('starting training...', '\nbatch size:', FLAGS.batch_size, '\nnumber of epoch :', FLAGS.number_epoch)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        id_max = int(data_set.Train.num_example / FLAGS.batch_size)

        # start the training loop
        for epoch in range(FLAGS.number_epoch):
            epoch_loss = 0
            for j in range(id_max):
                if j % 1 == 0:
                    display_progression_epoch(j, id_max, epoch)

                batch_x, batch_y = data_set.Train.get_batch(FLAGS.batch_size, j)

                _, loss_value = sess.run([train_op, loss], feed_dict={x_placeholder: batch_x,
                                                                      y_placeholder: batch_y, keep_prob: KEEP_RATE})

                epoch_loss += loss_value

            print('Epoch ', epoch, '/', FLAGS.number_epoch, 'completed', '   loss:', epoch_loss)

        print('Neural net sucessfully trained, starting evaluation on test data...')
        print('Evaluation DataTest :', precision.eval({x_placeholder: data_set.Test.Images,
                                                       y_placeholder: data_set.Test.Label, keep_prob: 1.0}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_folder', type=str, default='/images',
                        help='Path to images Folder')
    parser.add_argument('--labels_file', type=str, default='labels.txt',
                        help='Path to label file')
    parser.add_argument('--batch_size', type=int, default=3,
                        help='How many images to train on at a time.')
    parser.add_argument('--number_epoch', type=int, default=10,
                        help='How many epoch.')
    parser.add_argument('--data_set_dir', type=str, default='output2.npz',
                        help='How many epoch.')
    FLAGS = parser.parse_args()
    main()
