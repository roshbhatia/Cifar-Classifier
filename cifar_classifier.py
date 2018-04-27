import numpy as np
import tensorflow as tf
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #GIVE ME PATCHES TO MAKE IT CLEAR
import random
import os

def unpickle(file):
    """Adapted from the CIFAR page: http://www.cs.utoronto.ca/~kriz/cifar.html"""
    import pickle
    with open(file, 'rb') as fo:
        return pickle.load(fo, encoding='bytes')

def gather_data():
    # Gather data
    data_dir = os.getcwd() + "/cifar-10/"
    train = [unpickle(data_dir + 'data_batch_{}'.format(i)) for i in [1, 2, 3, 4]]
    X_train = np.concatenate([t[b'data'] for t in train], axis=0)
    y_train = np.array(list(itertools.chain(*[t[b'labels'] for t in train])))
    valid = unpickle(data_dir + 'data_batch_5')
    X_valid = valid[b'data']
    y_valid = np.array(valid[b'labels'])

    return X_train, y_train, X_valid, y_valid


def display_image_from_Xtrain(X_train, index):
    image = np.reshape(X_train[index], (3, 32, 32)).transpose(1, 2, 0)
    plt.imshow(image, interpolation="nearest")

def plot_trainingandvalid_acc(trainingacc, validacc):
    fig = plt.figure()
    fig.suptitle("Training and Validation Accuracy",fontsize=20)
    axis = fig.add_subplot(111)
    axis.set_ylim(0, 1)
    axis.set_xlim(0,50)
    TRAINING_LABEL = mpatches.Patch(color='red', label='Training Accuracy')
    VALID_LABEL = mpatches.Patch(color='blue', label='Validation Accuracy')

    axis.legend(handles = [TRAINING_LABEL, VALID_LABEL],loc=0)

    axis.set_xlabel('Epochs')
    axis.set_ylabel('Training and Validation Accuracy')


    plt.plot(trainingacc, color='r')
    plt.plot(validacc, color='b')
    plt.savefig("Training_Valid_Acc.png")




def main():
    X_train, y_train, X_valid, y_valid = gather_data()

    # Inputs and Outputs
    n_inputs = 32 * 32 * 3
    n_outputs = 10

    # Symbols
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

    ##BUILDS LAYERS
    with tf.name_scope("dnn"):
        sess = tf.Session()
        shaped = tf.transpose(tf.reshape(X, [-1, 3, 32, 32]), (0, 2, 3, 1))

        #First conv layer uses elu
        n_filters1 = 32
        conv1 = tf.layers.conv2d(shaped, n_filters1, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
        pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2, padding='valid')

        #Second conv layer uses elu
        n_filters2 = 64
        conv2 = tf.layers.conv2d(pool1, n_filters2, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2, padding='valid')

        #Third conv layer uses elu
        n_filters3 = 128
        conv3 = tf.layers.conv2d(pool2, n_filters3, kernel_size=3, strides=1, padding='same', activation=tf.nn.elu)


        flat = tf.reshape(conv3, [-1, 8 * 8 * n_filters3])

        #Deleted second hidden layer and the first one uses relu
        n_hidden1 = 1024
        hidden1 = tf.layers.dense(flat, n_hidden1, name="hidden1", activation=tf.nn.relu)

        logits = tf.layers.dense(hidden1, n_outputs, name="outputs")

    with tf.name_scope('loss'):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(xentropy, name='loss')
    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    #LEARNING RATE IS 0.001 AND USES ADAM OPTIMIZER
    #Attempted gradient descent but was no good
    learning_rate = 0.001
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate)
        trainoptimizer = optimizer.minimize(loss)

    ##RUNS
    init = tf.global_variables_initializer()
    training_size = len(X_train)
    epochs = 50
    size = 500
    valid_accuracy_list= []
    training_accuracy_list= []
    with tf.Session() as sess:
            init.run()
            for epoch in range(epochs):
                print("EPOCH #{} STARTED".format(epoch))
                for i in range(0, training_size, size):
                    count = range(i, i + size)
                    X_batch = X_train [count, :]
                    y_batch = y_train [count]
                    sess.run(trainoptimizer, feed_dict={X: X_batch, y: y_batch})
                training_loss = loss.eval(feed_dict={X: X_train, y: y_train})
                training_accuracy = accuracy.eval(feed_dict={X: X_train, y: y_train})
                valid_accuracy = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
                print("EPOCH {} | ACC TRAIN {:.4f} | ACC VALID {:.4f} | LOSS TRAIN {:.6f}".format(epoch, training_accuracy, valid_accuracy, training_loss))
                valid_accuracy_list.append(valid_accuracy)
                training_accuracy_list.append(training_accuracy)

    #Plotting data, should save as Training_Valid_Acc.png
    plot_trainingandvalid_acc(training_accuracy_list, valid_accuracy_list)

if __name__ == '__main__':
    main()
