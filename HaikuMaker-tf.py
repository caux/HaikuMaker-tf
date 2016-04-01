import numpy as np
import tensorflow as tf
import time
from os import listdir
from os.path import isfile, join

import NN_Model

class Config(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 20
    hidden_size = 200
    max_epoch = 4
    max_max_epoch = 13
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 20
    vocab_size = 10000


def run_epoch(session, m, data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = m.initial_state.eval()
    for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
        print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

def main():
    inputSize = 512
    numOfNeurons = 32

    data, dict, numWords = readTrainingSet("/Users/caux/Documents/Development/Datasets/Haiku/")

    config = Config()

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = NN_Model.PTBModel(is_training=True, config=config)

        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = NN_Model.PTBModel(is_training=False, config=config)
            mtest = NN_Model.PTBModel(is_training=False, config=eval_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


def readTrainingSet(path):
    fileList = []
    allHaikus = []
    wordDict = dict()
    numUniqueWords = 0

    wordDict[""] = numUniqueWords   # EOH (End Of Haiku)

    for f in listdir(path):
        if ".txt" in f:
            filePath = join(path, f)
            if isfile(filePath):
                fileList.append(filePath)

    for indexFile in range(len(fileList)):
        with open(fileList[indexFile], 'r') as haikuFile:
            haikuData = haikuFile.read()

            haikuData = haikuData.replace("\r\n", "\n")
            haikuData = haikuData.split("\n\n")[0]
            splitHaikuData = haikuData.split(" ")

            # ID all words using a map
            for word in splitHaikuData:
                if word != "":
                    if word not in wordDict:
                        wordDict[word] = numUniqueWords
                        numUniqueWords += 1

                    allHaikus.append(wordDict[word])

            allHaikus.append(wordDict[""])

    return allHaikus, wordDict, numUniqueWords

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))