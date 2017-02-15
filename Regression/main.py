from data_management import *
from regression_network import *
import argparse
import tensorflow as tf
import sys
import numpy as np

FLAGS = None
KEEP_RATE = 0.8


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
    output_list = []
    for step in range(step_per_epoch):
        display_progression_epoch(step, step_per_epoch,1)
        batch_images, batch_labels = data_set.Test.get_batch(FLAGS.batch_size, step)
        [e, p, o] = sess.run([loss, precision, output], feed_dict={images_pl: batch_images,
                                                                 labels_pl: batch_labels,
                                                                 keep_prob: 1.0})
        error += e
        pres += p
        output_list.append(o)

        precision_avg = precision / step_per_epoch
    print('evaluation on data test\n number of images', number_tested, '\nerror (l2 loss)', error,
          '\nerror (avg of pixels)', precision_avg)

    return output_list


def write_eval(output_list, dir_out_bb):
    predicted_bb = np.asarray(output_list)
    np.savetxt(dir_out_bb, predicted_bb, delimiter=' ', fmt='%d')


def display_progression_epoch(j, id_max, epoch):
    batch_progression = int((j / id_max) * 100)
    sys.stdout.write(str(batch_progression) + ' % of the epoch ' + str(epoch + 1) + ' completed' + chr(13))
    sys.stdout.flush


def run_training():
    print('loading data set :', FLAGS.data_set_dir, '...')
    data_set = load_data(FLAGS.data_set_dir)
    with tf.device('/cpu:0'):
        print('constructing graph ...')

        labels_pl, images_pl, keep_prob = placeholder_training(data_set.sizeImage,
                                                               data_set.sizeLabel,
                                                               FLAGS.batch_size)

        output = inference(images_pl, keep_prob)
        loss = regression_loss(output, labels_pl)
        train_op = training(loss)
        precision = accuracy(output, labels_pl)

        init = tf.global_variables_initializer()

        # force on cpu
        config = tf.ConfigProto(
            device_count={'GPU': 0}
        )
        sess = tf.Session(config=config)

        sess.run(init)

        id_max = int(data_set.Train.num_examples / FLAGS.batch_size)

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

            print('Epoch ', epoch + 1, '/', FLAGS.number_epoch, 'completed', '   loss:', epoch_loss)

        output_list = do_eval(sess, loss, precision, images_pl, output, labels_pl, data_set, keep_prob)
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
    FLAGS = parser.parse_args()
    run_training()
