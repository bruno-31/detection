"""Builds the Regression network.

Implements the tensorflow inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and apply gradients.
"""
import tensorflow as tf

OUTPUT_SIZE = 4


def placeholder_training(image_size, output_size, batch_size):
    x_placeholder = tf.placeholder('float32', [batch_size, image_size])
    y_placeholder = tf.placeholder('float32', [batch_size, output_size])
    keep_prob = tf.placeholder(tf.float32)
    return y_placeholder, x_placeholder, keep_prob


def inference(images, keep_prob):
    """
    :param images: Images placeholder
    :return: output tensor with coordinates BB
    """
    images = tf.reshape(images, shape=[-1, 128, 128, 1])

    with tf.name_scope('hidden_layer_1'):
        weights = tf.Variable(tf.random_normal([8, 8, 1, 32]),name='weights')
        biases = tf.Variable(tf.random_normal([32]), name='biases')
        hidden1 = tf.nn.relu(conv2d(images,weights)+biases)
        hidden1 = maxpool2d(hidden1)

    with tf.name_scope('hidden_layer_2'):
        weights = tf.Variable(tf.random_normal([8, 8, 32, 64]),name='weights')
        biases = tf.Variable(tf.random_normal([64]), name='biases')
        hidden2 = tf.nn.relu(conv2d(hidden1,weights)+biases)
        hidden2 = maxpool2d(hidden2)

    with tf.name_scope('fully_connected'):
        weights = tf.Variable(tf.random_normal([32 * 32 * 64, 1024]),name='weights')
        biases = tf.Variable(tf.random_normal([1024]),name='biases')
        fc = tf.reshape(hidden2, [-1, 32 * 32 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights) + biases)
        fc = tf.nn.dropout(fc, keep_prob)

    with tf.name_scope('out'):
        weights = tf.Variable(tf.random_normal([1024, OUTPUT_SIZE]))
        biases = tf.Variable(tf.random_normal([OUTPUT_SIZE]))
        output = tf.matmul(fc, weights) + biases

    return output


def regression_loss(output, true_labels):
    # loss = 2 * tf.nn.l2_loss(output - labels, name='l2_loss')
    loss = tf.reduce_mean (tf.reduce_sum((output - true_labels) ** 2, axis=1), name='l2_loss')
    return loss

def training(loss):
    train_op = tf.train.AdamOptimizer().minimize(loss)
    return train_op


def accuracy(output, true_labels):
    accuracy_bb = tf.reduce_mean(tf.sqrt(tf.reduce_mean((output-true_labels)**2,axis=1)), )
    return accuracy_bb


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
