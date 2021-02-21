import sys
import numpy as np
import random

# Global
HIDE_SIZE = 110
ETA = 0.09
INPUT_SIZE = 28 * 28
CLUSTER = 10
EPOCHS = 24


def init_w_b():
    w1 = init_w(HIDE_SIZE, INPUT_SIZE)
    w2 = init_w(CLUSTER, HIDE_SIZE)
    b1 = init_b(HIDE_SIZE)
    b2 = init_b(CLUSTER)
    w_b = [w1, b1, w2, b2]
    return w_b


def shuffleData(x, y):
    zip_x_y = list(zip(x, y))
    random.shuffle(zip_x_y)
    new_x, new_y = zip(*zip_x_y)
    return new_x, new_y


def init_w(r, c):
    return np.random.randn(r, c)


def init_b(r):
    return np.random.randn(r, 1)


def normalization(value):
    value = np.divide(value, 255)
    return value


def softMax(x):
    list_sm = np.exp(x - np.max(x))
    sm_sum = list_sm.sum(axis=0)
    return list_sm / sm_sum


def relu(x):
    return max(0, x)


def dev_relu(x):
    if x > 0: return 1
    return 0


def update_weights(w_b, new_w_b):
    w1, b1, w2, b2 = w_b
    w1 -= ETA * new_w_b[0]
    b1 -= ETA * new_w_b[1]
    w2 -= ETA * new_w_b[2]
    b2 -= ETA * new_w_b[3]
    w_b = w1, b1, w2, b2
    return w_b


def loss_calc(y_hat, y):
    specific_y = np.zeros(y_hat.size)
    specific_y[int(y)] = 1
    temp = np.copy(y_hat)
    temp[temp == 0] = 1
    return -np.dot(specific_y, np.log2(temp))


def fprop(w_b, x):
    w1, b1, w2, b2 = w_b
    shape_x = np.reshape(x, (-1, 1))
    z1 = np.dot(w1, shape_x) + b1
    g = np.vectorize(relu)
    h = g(z1)
    h = h / (np.max(h) or 1)
    z2 = np.dot(w2, h) + b2
    y_hat = softMax(z2)
    values = y_hat, h, z1
    return values


def backprop(w_b, x, y, new_w_b):
    y_hat, h, z1 = new_w_b
    w1, b1, w2, b2 = w_b
    y_v = np.zeros((y_hat.size, 1))
    y_v[int(y)] = 1
    new_yhat = y_hat - y_v
    # w1

    dh1_z1 = np.vectorize(dev_relu)(z1)
    b1_loss = (np.dot(new_yhat.T, w2) * dh1_z1.T).T
    loss_w1 = np.dot(b1_loss, np.reshape(x, (-1, 1)).T)
    # w2
    calc_w2 = h
    loss_w2 = np.dot(new_yhat, calc_w2.T)
    b2_loss = np.copy(new_yhat)

    new_w_b = loss_w1, b1_loss, loss_w2, b2_loss
    return new_w_b


def train(w_b, train_x, train_y):
    for i in range(EPOCHS):
        sum_loss = 0.0
        shuffleData(train_x, train_y)
        for x, y in zip(train_x, train_y):
            values = fprop(w_b, x)
            loss = loss_calc(values[0], y)
            sum_loss += loss
            new_w_b = backprop(w_b, x, y, values)
            update_weights(w_b, new_w_b)
    return w_b


def main():
    # Open files and read content
    train_x_path = sys.argv[1]
    train_y_path = sys.argv[2]
    test_x_path = sys.argv[3]
    tx = np.loadtxt(train_x_path)
    ty = np.loadtxt(train_y_path)
    testx = np.loadtxt(test_x_path)
    train_x = normalization(tx)
    # Shuffle
    train_x, ty = shuffleData(train_x, ty)

    w_b = init_w_b()
    w_b = train(w_b, train_x, ty)
    file = open("test_y", 'w+')
    for i in testx:
        values = fprop(w_b, i)
        file.write(str(np.argmax(values[0])) + "\n")
    file.close()


if __name__ == '__main__':
    main()
