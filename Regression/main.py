from data_management import *
from regression_network import *
import argparse
import tensorflow as tf
import sys
import numpy as np


"""
to visualize summary on tensorboard : tensorboard --logdir=run1:../tmp/logs_regression --port 6006
"""


FLAGS = None
KEEP_RATE = 0.8

def fill_feed_dict(data_set, images_pl, labels_pl):

    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
    feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            loss,
            precision,
            output,
            images_pl,
            labels_pl,
            data_set,
            keep_prob):
    step_per_epoch = data_set.Test.num_examples // FLAGS.batch_size
    number_tested = step_per_epoch * FLAGS.batch_size
    error = 0
    pres = 0
    output_bb_array = np.zeros(shape=(number_tested, data_set.sizeLabel))
    for step in range(step_per_epoch):
        display_progression_epoch(step, step_per_epoch, 0)
        batch_images, batch_labels = data_set.Test.get_batch(FLAGS.batch_size, step)

        [e, p, o] = sess.run([loss, precision, output], feed_dict={images_pl: batch_images,
                                                                   labels_pl: batch_labels,
                                                                   keep_prob: 1.0})

        error += e
        pres += p
        beg = FLAGS.batch_size * step
        end = FLAGS.batch_size * (step + 1)
        output_bb_array[beg:end, :] = o

    precision_avg = pres / step_per_epoch

    print('\nEvaluation on data test :\nnumber of images', number_tested, '\nerror : (l2 loss)  ', error,
          '\nerror (avg of pixels) :    ', precision_avg)

    return output_bb_array


def write_eval(output_list, dir_out_bb):
    predicted_bb = np.asarray(output_list)
    np.savetxt(dir_out_bb, predicted_bb, delimiter=' ', fmt='%d')
    print('data test output sucessfully written in the file :   ', dir_out_bb)


def display_progression_epoch(j, id_max, epoch):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % of the epoch ' + str(epoch + 1) + ' completed' + chr(13))
    sys.stdout.flush


def run_training():
    print('loading data set :', FLAGS.data_set_dir, '...')
    data_set = load_data(FLAGS.data_set_dir)
    print('number of training images    ', data_set.Train.num_examples)

    print('constructing graph ...')

    labels_pl, images_pl, keep_prob_pl = placeholder_training(data_set.sizeImage,
                                                              data_set.sizeLabel,
                                                              FLAGS.batch_size)

    output = inference(images_pl, keep_prob_pl)
    loss = regression_loss(output, labels_pl)
    train_op = training(loss)
    precision = accuracy(output, labels_pl)

    summary = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    # force on cpu
    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    sess = tf.Session(config=config)

    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    sess.run(init)

    steps_per_epoch = data_set.Train.num_examples // FLAGS.batch_size
    train_step = 0

    for epoch in range(FLAGS.number_epoch):
        data_set.Train.shuffle_data()
        epoch_loss = 0
        for step in range(steps_per_epoch):
            if step % 1 == 0:  # to increase speed for testing...
                display_progression_epoch(step, steps_per_epoch, epoch)

                batch_images, batch_labels = data_set.Train.get_batch(FLAGS.batch_size, step)

                feed_dict = {images_pl: batch_images,
                             labels_pl: batch_labels,
                             keep_prob_pl: KEEP_RATE}

                _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
                epoch_loss += loss_value

                train_step += 1

                if step % 3 == 0:
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, train_step)
                    summary_writer.flush()

        print('Epoch ', epoch + 1, '/', FLAGS.number_epoch, 'completed', '   loss:', epoch_loss)
    output_list = do_eval(sess, loss, precision, output, images_pl, labels_pl, data_set, keep_prob_pl)
    write_eval(output_list, FLAGS.dir_out_bb)


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
    parser.add_argument('--data_set_dir', type=str, default='../BDD1.npz',
                        help='How many epoch.')
    parser.add_argument('--dir_out_bb', type=str, default='../out_bb.txt',
                        help='Directory for writing outputs')
    parser.add_argument('--log_dir', type=str, default='../tmp/logs_regression',
                        help='Directory for logs')
    FLAGS = parser.parse_args()

    # delete logs files
    if tf.gfile.Exists(FLAGS.log_dir):
        tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)

    run_training()
