import numpy as np
import tensorflow as tf


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def createWeightsAndBiases(topology):
    weights = []
    biases = []

    for index in range(len(topology) - 1):
        layerWeights = weight_variable([topology[index], topology[index+1]])
        layerBias = bias_variable([topology[index+1]])

        weights.append(layerWeights)
        biases.append(layerBias)

    return weights, biases


def createTrainingTopology(topology, levels, weights, biases):
    layers = []

    numOutputs = numInputs = topology[0]

    x = tf.placeholder(tf.float32, shape=[None, numInputs])
    y_ = tf.placeholder(tf.float32, shape=[None, numOutputs])

    previousOutput = x

    for index in range(levels - 1):
        layerNode = tf.nn.softmax(tf.matmul(previousOutput, weights[index]) + biases[index])

        layers.append(layerNode)
        previousOutput = layerNode

    for index in reversed(range(levels - 1)):
        print previousOutput.get_shape()
        print tf.transpose(weights[index]).get_shape()
        layerNode = tf.nn.softmax(tf.matmul(previousOutput - biases[index], tf.transpose(weights[index])))

        layers.append(layerNode)
        previousOutput = layerNode


    y = previousOutput

    return x, y, y_, layers

def runNeuralNet(input, node, data):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        result = sess.run(node, feed_dict={input: data})

    return result
