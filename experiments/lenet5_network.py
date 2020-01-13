import tensorflow as tf
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.contrib.layers import flatten
from sklearn_.utils import shuffle
import multiprocessing
import time
from tensorflow.contrib.opt import extend_with_decoupled_weight_decay
EPOCHS = 10
BATCH_SIZE = 128

image_size = 28
num_labels = 10
num_channels = 1 # grayscale

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def run_release_gpu(func):
    def parallel_wrapper(output_dict, *argv, **kwargs):
        ret = func(*argv, **kwargs)
        if ret is not None:
            output_dict['ret'] = ret

    def outer_wrapper(*argv, **kwargs):
        same_process = kwargs.pop('same_process', False)
        if same_process:
            return func(*argv, **kwargs)

        with multiprocessing.Manager() as manager:
            output = manager.dict()
            args = (output, ) + argv
            p = multiprocessing.Process(target=parallel_wrapper, args=args, kwargs=kwargs)
            p.start()
            p.join()
            ret_val = output.get('ret', None)

        return ret_val

    return outer_wrapper

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
    return dataset, labels



def LeNet_5(x, keep_prob_input1, keep_prob_input2):
    # Layer 1 : Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = 0.1))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b
    # TODO: Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling Layer. Input = 28x28x1. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    # TODO: Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = 0.1))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    # TODO: Activation.
    conv2 = tf.nn.relu(conv2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
    # TODO: Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)
    # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = 0, stddev = 0.1))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b
    # TODO: Activation.
    fc1 = tf.nn.relu(fc1)
    #dropout
    fc1_drop = tf.nn.dropout(fc1, keep_prob_input1)
    # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = 0, stddev = 0.1))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1_drop,fc2_w) + fc2_b
    # TODO: Activation.
    fc2 = tf.nn.relu(fc2)
    #dropout
    fc2_drop = tf.nn.dropout(fc2, keep_prob_input2)
    # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84,10), mean = 0 , stddev = 0.1))
    fc3_b = tf.Variable(tf.zeros(10))
    logits = tf.matmul(fc2_drop, fc3_w) + fc3_b
    return logits

def load_data():
    df_train = pd.read_csv('dataset/train.csv')
    df_test = pd.read_csv('dataset/test.csv')
    df_train.head()

    df_train = pd.get_dummies(df_train,columns=["label"])
    df_features = df_train.iloc[:, :-10].values
    df_label = df_train.iloc[:, -10:].values


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_features, df_label, test_size = 0.2, random_state = 1212)
    X_test, X_validation, y_test,y_validation = train_test_split(X_test, y_test, test_size=0.5, random_state=0)



    train_dataset, train_labels = reformat(X_train, y_train)
    valid_dataset, valid_labels = reformat(X_validation, y_validation)
    test_dataset , test_labels = reformat(X_test, y_test)
    df_test = df_test.values.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    # Pad images with 0s
    X_train      = np.pad(train_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_validation = np.pad(valid_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')
    X_test       = np.pad(test_dataset, ((0,0),(2,2),(2,2),(0,0)), 'constant')

    return X_train, X_validation, X_test, y_train, y_validation, y_test

@run_release_gpu
def run_mnist(lr, momentum, weight_decay, X_train, X_validation, X_test, y_train, y_validation, y_test, keep_prob1, keep_prob2):

    def evaluate(X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        sess = tf.get_default_session()
        for offset in range(0, num_examples, BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
            accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y_: batch_y, keep_prob_input1: keep_prob1, keep_prob_input2: keep_prob2})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples

    #X_train, X_validation, X_test, y_train, y_validation, y_test = load_data()


    x = tf.placeholder(tf.float32, shape=[None,32,32,1])
    y_ = tf.placeholder(tf.int32, (None))
    keep_prob_input1= tf.placeholder(tf.float32)
    keep_prob_input2 = tf.placeholder(tf.float32)

    #Invoke LeNet function by passing features
    logits = LeNet_5(x,keep_prob_input1, keep_prob_input2)
    #Softmax with cost function implementation
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels = y_, logits = logits)
    loss_operation = tf.reduce_mean(cross_entropy)
    mom_wd = extend_with_decoupled_weight_decay(tf.train.MomentumOptimizer)
    optimizer = mom_wd(weight_decay=weight_decay, learning_rate = lr, momentum=momentum)
    #optimizer = tf.train.MomentumOptimizer(learning_rate = lr, momentum=momentum)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y_,1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    # Evaluate function

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = len(X_train)

        #print("Training... with dataset - ", num_examples)
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y_: batch_y, keep_prob_input1: keep_prob1, keep_prob_input2: keep_prob2})

            # validation_accuracy = evaluate(X_validation, y_validation)
            # print("EPOCH {} ...".format(i+1), "Validation Accuracy = {:.3f}".format(validation_accuracy))

        test_accuracy = evaluate(X_test, y_test)
        # print("Test Accuracy = {:.3f}".format(test_accuracy))
    return test_accuracy


#run_mnist()