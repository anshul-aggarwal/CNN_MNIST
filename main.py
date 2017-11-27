import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import math

f = open('results.txt', 'w')

fsize1 = 5          # Convolution filters are 5 x 5 pixels.
nfilters1 = 16         # There are 16 of these filters.

fsize2 = 5          # Convolution filters are 5 x 5 pixels.
nfilters2 = 36         # There are 36 of these filters.

fc_size = 128             # Number of neurons in fully-connected layer.

data = input_data.read_data_sets('data/MNIST/', one_hot=True)

f.write("Training set:\t\t{}".format(len(data.train.labels)) + "\n")
f.write("Test set:\t\t{}".format(len(data.test.labels)) + "\n")
f.write("Validation set:\t\t{}".format(len(data.validation.labels)) + "\n")

data.test.cls = np.argmax(data.test.labels, axis=1)

iSize = 28   
iSize_flat = iSize * iSize
img_shape = (iSize, iSize)
nChannels = 1
nClasses = 10

def plot_imgs(imgs, classActual, classPredicted=None):
    assert len(imgs) == len(classActual) == 4
    fig, axes = plt.subplots(2, 2)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i].reshape(img_shape), cmap='binary')

        if classPredicted is None:
            xlabel = "Actual: {0}".format(classActual[i])
        else:
            xlabel = "Actual: {0}, Predicted: {1}".format(classActual[i], classPredicted[i])

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def CreateconvLayer(input, NinputChannels, filtSize, nfilters, use_pooling=True):
    shape = [filtSize, filtSize, NinputChannels, nfilters]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[nfilters]))
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

def flatten_layer(layer):
    layer_shape = layer.get_shape()
    Nfeatures = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, Nfeatures])
    return layer_flat, Nfeatures

def CreatefcLayer(input, Ninputs, Noutputs, use_relu=True):

    weights = tf.Variable(tf.truncated_normal([Ninputs, Noutputs], stddev=0.05))
    biases = tf.Variable(tf.constant(0.05, shape=[Noutputs]))
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

x = tf.placeholder(tf.float32, shape=[None, iSize_flat], name='x')
x_image = tf.reshape(x, [-1, iSize, iSize, nChannels])
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)

layer_conv1, weights_conv1 = CreateconvLayer(input=x_image, NinputChannels=nChannels, filtSize=fsize1, nfilters=nfilters1, use_pooling=True)
layer_conv2, weights_conv2 = CreateconvLayer(input=layer_conv1, NinputChannels=nfilters1, filtSize=fsize2, nfilters=nfilters2, use_pooling=True)
layer_flat, Nfeatures = flatten_layer(layer_conv2)
layer_fc1 = CreatefcLayer(input=layer_flat, Ninputs=Nfeatures, Noutputs=fc_size, use_relu=True)
layer_fc2 = CreatefcLayer(input=layer_fc1, Ninputs=fc_size, Noutputs=nClasses, use_relu=False)

y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, axis=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

session = tf.Session()
session.run(tf.global_variables_initializer())
train_batch_size = 64
Titerations = 0

def optimize(nIterations):
    global Titerations
    for i in range(Titerations, Titerations + nIterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 100 == 0 and i>0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimizing: {0:>6}th time, Accuracy on training set: {1:>6.2%}"
            f.write(msg.format(i, acc) + "\n")

    Titerations += nIterations

def example_errors(classPredicted, correct):
    incorrect = (correct == False)
    imgs = data.test.images[incorrect]
    classPredicted = classPredicted[incorrect]
    classActual = data.test.cls[incorrect]    
    plot_imgs(imgs=imgs[10:14], classActual=classActual[10:14], classPredicted=classPredicted[10:14])

test_batch_size = 256

def CalcAccuracy(show_example_errors=False):
    Ntest = len(data.test.images)
    classPredicted = np.zeros(shape=Ntest, dtype=np.int)
    i = 0
    while i < Ntest:
        j = min(i + test_batch_size, Ntest)
        imgs = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]
        feed_dict = {x: imgs, y_true: labels}
        classPredicted[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)
        i = j

    classActual = data.test.cls
    correct = (classActual == classPredicted)
    correct_sum = correct.sum()
    acc = float(correct_sum) / Ntest
    msg = "Test Set accuracy: {0:.2%}"
    f.write(msg.format(acc) + "\n")
    if show_example_errors:
        example_errors(classPredicted=classPredicted, correct=correct)

CalcAccuracy()
optimize(nIterations=1001)
CalcAccuracy(show_example_errors=True)
f.close()
