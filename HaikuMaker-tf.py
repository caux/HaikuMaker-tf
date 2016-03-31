import numpy as np
import tensorflow as tf
import tensorflow.models.rnn as rnn
import time
from os import listdir
from os.path import isfile, join

import Network


def main():
    inputSize = 512
    numOfNeurons = 32

    data = readTrainingSet("/Users/caux/Documents/Development/Datasets/Haiku/", inputSize)

    # lstm = rnn.rnn_cell.BasicLSTMCell(numOfNeurons)
    # state = tf.zeros([inputSize, lstm.state_size])
    #
    # loss = 0.0
    #
    # for current_batch_of_words in data:
    #     # The value of state is updated after processing each batch of words.
    #     output, state = lstm(current_batch_of_words, state)
    #
    #     # The LSTM output can be used to make next word predictions
    #     logits = tf.matmul(output, softmax_w) + softmax_b
    #     probabilities = tf.nn.softmax(logits)
    #     loss += loss_function(probabilities, target_words)



    # topology = [inputSize, 256, 64, 16, 4]
    #
    # weights, biases = Network.createWeightsAndBiases(topology)
    #
    # x, y, y_, layers = Network.createTrainingTopology(topology, 2, weights, biases)
    # cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #
    # with tf.Session() as sess:
    #     sess.run(tf.initialize_all_variables())
    #     train_step.run(feed_dict={x: batch[0], y_: batch[1]})


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

            haikuData = haikuData.replace("\r\n", "\n")
            haikuData = haikuData.split("\n\n")[0]

            print haikuData
            print
            print

            for indexChar in range(len(haikuData)):
                data[indexFile][indexChar] = float(ord(haikuData[indexChar])) / 255.0

    return data



if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))