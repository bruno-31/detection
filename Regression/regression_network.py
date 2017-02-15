"""Builds the Regression network.

Implements the tensorflow inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.
"""
import tensorflow as tf

OUTPUT_SIZE = 4


def placeholder_training(image_size, output_size):
    x_placeholder = tf.placeholder('float', [None, image_size])
    y_placeholder = tf.placeholder('float', [None, output_size])
    keep_prob = tf.placeholder(tf.float32)
    return y_placeholder, x_placeholder, keep_prob


def inference(x_placeholder, keep_prob):
    weights = {'W_conv1': tf.Variable(tf.random_normal([8, 8, 1, 32])),
               'W_conv2': tf.Variable(tf.random_normal([8, 8, 32, 64])),
               'W_fc': tf.Variable(tf.random_normal([32 * 32 * 64, 1024])),
               'out': tf.Variable(tf.random_normal([1024, OUTPUT_SIZE]))}

    biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
              'b_conv2': tf.Variable(tf.random_normal([64])),
              'b_fc': tf.Variable(tf.random_normal([1024])),
              'out': tf.Variable(tf.random_normal([OUTPUT_SIZE]))}

    x = tf.reshape(x_placeholder, shape=[-1, 128, 128, 1])
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)
    fc = tf.reshape(conv2, [-1, 32 * 32 * 64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_prob)

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output


def regression_loss(output, labels):
    loss = 2 * tf.nn.l2_loss(output - labels)
    return loss


def training(loss):
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return train_op


def evaluation(input_images, true_labels, keep_prob):
    prediction = inference(input_images, keep_prob)
    accuracy = tf.nn.l2_loss(prediction - true_labels)
    return accuracy


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
