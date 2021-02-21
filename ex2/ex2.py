from enum import Enum
import random
import sys
import numpy as np


# Class of enum for changing letters for correct value
class Letter(Enum):
    M = 0.9
    F = 1.6
    I = 1.2


def define_type(data):
    data_array = np.array(data)
    type_data = data_array.astype(np.float)
    return type_data


# Read files and make the replace od latters
def read_files(file):
    data = []
    read_file = open(file, "r+")
    # replace chars
    for line in read_file:
        data.append(
            line.replace('\n', '').replace('M', str(Letter.M.value)).replace('F', str(Letter.F.value)).replace('I', str(
                Letter.I.value)).split(','))

    type_d = define_type(data)
    read_file.close()
    return type_d


# min max normalization
def min_max(data):
    cols = len(data[0])
    normalize_data = np.zeros(data.shape)
    for j in range(cols):
        minimum = min(data[:, j])
        maximum = max(data[:, j])
        if minimum != maximum:
            normalize_data[:, j] = (data[:, j] - minimum) / (maximum - minimum)
    return normalize_data

# running the test on the algorithm
def test_algo(w, test):
    predictions = []
    counter = 0
    for i, example in enumerate(test):
        y_hat = np.argmax(np.dot(w, example))
        # this for checking the avg
        #if y_hat == dataY[i]:
        #    counter += 1
        predictions.append(y_hat)
   # print avg
    #print(str(counter / len(test) * 100) + '%')
    return predictions


def train_perceptron(features, trainX, trainY, epochs, eta):
    w = np.zeros((3, features), dtype=float)
    c = list(zip(trainX, trainY))
    random.shuffle(c)
    feat, labels = zip(*c)
    for i in range(epochs):
        for x, y in zip(feat, labels):
            y_hat = np.argmax(np.dot(w, x))
            if y_hat != y:
                w[int(y), :] = w[int(y), :] + (eta * x)
                w[y_hat, :] = w[y_hat, :] - (eta * x)
        eta /= 1.5

    return w


def train_svm(features, trainX, trainY, epochs, eta, lamda):
    w = np.zeros((3, features), dtype=float)
    c = list(zip(trainX, trainY))
    random.shuffle(c)
    training_inputs, labels = zip(*c)
    for set in range(epochs):
        for x, y in zip(training_inputs, labels):
            y_hat = np.argmax(np.dot(w, x))
            if (y_hat != y):
                if y[0] != y_hat:
                    w[int(y), :] = (1 - eta * lamda) * w[int(y), :] + eta * x
                    w[y_hat, :] = (1 - eta * lamda) * w[y_hat, :] - eta * x
                for i in range(w.shape[0]):
                    if i != int(y) and i != int(y_hat):
                        w[i:] = (1 - eta * lamda) * w[i:]
        eta /= 1.5
        lamda /= 1.5
    return w


def train_PassAgg(features, trainX, trainY, epochs):
    w = np.zeros((3, features), dtype=float)
    c = list(zip(trainX, trainY))
    random.shuffle(c)
    feat, labels = zip(*c)
    for i in range(epochs):
        for x, y in zip(feat, labels):
            y_hat = np.argmax(np.dot(w, x))
            if y != y_hat:
                yi = int(y[0])
                loss = max((1 - (np.dot(w[yi, :], x)) + (np.dot(w[y_hat, :], x))), 0)
                xx = 2 * (np.linalg.norm(x) ** 2)
                if xx != 0:
                    tau = loss / xx
                w[yi, :] += tau * x
                w[y_hat, :] -= tau * x
    return w


def main():
    dataX = read_files(sys.argv[1])
    dataY = read_files(sys.argv[2])
    testX = read_files(sys.argv[3])

    features = len(dataX[0])
    trainX = min_max(dataX)

    # array for the train
    #xTrain = xArr[:int(((totalAll / 5) * 4))]
    #xTest = xArr[int(((totalAll / 5) * 4)):]
    #yTrain = yArr[:int(((totalAll / 5) * 4))]
    #yTest = yArr[int(((totalAll / 5) * 4)):]


    test = min_max(testX)
    test_perceptron = test_algo(train_perceptron(features, trainX, dataY, 100, 0.10), test)
    test_svm = test_algo(train_svm(features, trainX, dataY, 50, 0.10, 0.2), test)
    test_passagg = test_algo(train_PassAgg(features, trainX, dataY, 100), test)
    for pred_perc, pred_svm, pred_pa in zip(test_perceptron, test_svm, test_passagg):
        print("perceptron: " + str(pred_perc) + ", svm: " + str(pred_svm) + ", pa: " + str(pred_pa))


if __name__ == '__main__':
    main()
