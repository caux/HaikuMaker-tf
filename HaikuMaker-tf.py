import numpy as np
import tensorflow as tf
import random
import time
from os import listdir
from os.path import isfile, join
import sys

import Network


def main():
    inputSize = 512
    batch_size = 50

    data = readTrainingSet("/Users/caux/Documents/Development/Datasets/Haiku/", inputSize)

    topology = [inputSize, 32, 64, 16, 4]

    weights, biases = Network.createWeightsAndBiases(topology)

    x, y, y_, layers = Network.createTrainingTopology(topology, 2, weights, biases)
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for reps in range(10):
            for index in range(0, len(data), batch_size):
                batch = data[index:index+batch_size]
                train_step.run(feed_dict={x: batch, y_: batch})


        correct_prediction = tf.equal(y, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(accuracy.eval(feed_dict={x: data, y_: data}))

def readTrainingSet(path, inputSize):
    fileList = []

    for f in listdir(path):
        if ".txt" in f:
            filePath = join(path, f)
            if isfile(filePath):
                fileList.append(filePath)

    data = np.empty([len(fileList), inputSize])

    for indexFile in range(len(fileList)):
        with open(fileList[indexFile], 'r') as haikuFile:
            haikuData = haikuFile.read()

            for indexChar in range(len(haikuData)):
                data[indexFile][indexChar] = float(ord(haikuData[indexChar])) / 255.0

    return data



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))