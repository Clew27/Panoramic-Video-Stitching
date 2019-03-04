import cv2 as cv
import numpy as np
import tensorflow as tf
from os import listdir

class DataManager():
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.size   = len(images)
        self.i      = 0

    def next_batch(self, batch_size):
        imgs = self.images[self.i : self.i + batch_size]
        lbls = self.labels[self.i : self.i + batch_size]

        self.i = (self.i + batch_size) % len(self.images)
        return imgs, lbls

class BallDataManager():
    training_percent = 0.1

    def __init__(self, folder):
        self.data_folder = folder
        self.__get_data()

    def __get_data(self):
        nonball = list() # Isn't a ball cases
        numimgs_non = len(listdir(self.data_folder + '/Non Ball'))
        for i in range(1, numimgs_non + 1):
            img = cv.imread(self.data_folder + '/Non Ball/{}.jpg'.format(i))
            nonball.append(img)
        nonball = np.array(nonball)

        ball = list()    # Is a ball cases
        numimgs_ball = len(listdir(self.data_folder + '/Ball'))
        for i in range(1, numimgs_ball + 1):
            img = cv.imread(self.data_folder + '/Ball/{}.jpg'.format(i))
            ball.append(img)
        ball = np.array(ball)

        nonball_test, nonball_train = self.__get_random_subset(nonball, BallDataManager.training_percent)
        ball_test,    ball_train    = self.__get_random_subset(ball, BallDataManager.training_percent)

        self.test = DataManager([img for img in nonball_test] + [img for img in ball_test],
                                self.__process_labels([0] * int(numimgs_non * BallDataManager.training_percent) \
                                + [1] * int(numimgs_ball * BallDataManager.training_percent)))


        train_cases_non  = [[0, img] for img in nonball_train]
        train_cases_ball = [[1, img] for img in ball_train]

        train_data = train_cases_non + train_cases_ball
        np.random.shuffle(train_data)

        self.train = DataManager([img for [_, img] in train_data],
                                 self.__process_labels([lbl for [lbl, _] in train_data]))


    def __get_random_subset(self, input_arr, percent):
        indices        = list(range(len(input_arr)))
        subset_indices = np.random.choice(indices, int(len(input_arr) * percent))
        non_ss_indices = np.setdiff1d(indices, subset_indices)

        return input_arr[subset_indices], input_arr[non_ss_indices]

    def __process_labels(self, labels, vals=2):
        n   = len(labels)
        out = np.zeros((n, vals))
        out[range(n), labels] = 1
        return out

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b






# Main Code ##
STEPS      = 1000
BATCH_SIZE = 50


ball_data = BallDataManager('tmp/Training')

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 2])
keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5, 5, 3, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)
conv2_flat = tf.reshape(conv2_pool, [-1, 8 * 8 * 64])

full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

y_conv = full_layer(full1_drop, 2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv,
                                                               labels=y_))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
    X = ball_data.test.images.reshape(11, 6, 32, 32, 3)
    Y = ball_data.test.labels.reshape(11, 6, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i],
                                                 keep_prob: 1.0})
                   for i in range(11)])
    print("Accuracy: {:.4}%".format(acc * 100))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(STEPS):
        batch = ball_data.train.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],
                                        keep_prob: 0.5})

    test(sess)


