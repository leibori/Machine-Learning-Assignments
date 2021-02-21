import numpy as np

EPOCHS = 10
ETA = 0.05
INPUT_SIZE = 28 * 28
CLASSES = 10
HIDDEN_LAYER = 100


def sigmoid(x):
    return np.divide(1, (1 + np.exp(-x)))


def ReLU(x):
    return np.maximum(x, 0)



def softmax(w, b, h):
    sum = 0
    softmax_vec = np.zeros((CLASSES, 1))
    for j in range(CLASSES):
        sum += np.exp(np.dot(w[j], h) + b[j])
    for i in range(CLASSES):
        softmax_vec[i] = (np.exp(np.dot(w[i],h)+b[i])) / sum
    return softmax_vec


def load():
    train_x = np.loadtxt("train_x")
    train_y = np.loadtxt("train_y")
    test_x = np.loadtxt("test_x")

    train_x, train_y = shuffle(train_x, train_y)

    # split to val_set and train_set
    size_train = int(len(train_x) * 0.2)
    val_x = train_x[-size_train:, :]
    val_y = train_y[-size_train:]
    train_x = train_x[: -size_train, :]
    train_y = train_y[: -size_train]
    # normalization
    train_x = train_x / 255
    val_x = val_x / 255
    test_x = test_x / 255

    return train_x, train_y, val_x, val_y, test_x


def fprop(params, x):
    w1, b1, w2, b2 = params
    x = np.transpose(x)
    z1 = np.dot(w1, x) + b1
    h1 = sigmoid(z1)
    z2 = np.dot(w2, h1) + b2
    y_hat = softmax(w2, b2, h1)
    return y_hat, h1, z1, z2


def update_weights(params, gradient_mat):
    w1, b1, w2, b2 = params
    gb1, gw1, gb2, gw2 = gradient_mat
    w1 -= ETA * gw1
    w2 -= ETA * gw2
    b1 -= ETA * gb1
    b2 -= ETA * gb2
    return w1, b1, w2, b2


def train(params, train_x, train_y, val_x, val_y):
    for i in range(EPOCHS):
        sum_loss = 0.0
        train_x, train_y = shuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            x = np.reshape(x, (1, INPUT_SIZE))
            class_vec, h1, z1, z2 = fprop(params, x)
            sum_loss += loss(class_vec, y)
            gradient_mat = backprop(x, y, z1, h1, params, class_vec)
            params = update_weights(params, gradient_mat)
        val_loss, accurate = validation(params, val_x, val_y, class_vec, y)
        print (i, sum_loss / train_x.shape[0], val_loss, accurate * 100)
    return params


def validation(params, val_x,val_y, class_vec, y):
    success = 0
    sum_loss = 0
    for x, y in zip(val_x, val_y):
        x = np.reshape(x, (1, INPUT_SIZE))
        y_hat, h1, z1, z2 = fprop(params, x)
        loss_val = loss(y_hat, y)
        sum_loss += loss_val
        max_arr = y_hat.argmax(axis=0)
        if max_arr[0] == int(y):
            success += 1
    accuracy = success / float(np.shape(val_x)[0])
    avg_of_loss = sum_loss / np.shape(val_x)[0]
    return avg_of_loss, accuracy


def backprop(x, y, z1, h1, params, classes):
    y_classes = classes
    y_classes[int(y)] -= 1
    w1, b1, w2, b2 = params
    diff_w2 = np.dot(y_classes, np.transpose(h1))
    diff_b2 = y_classes
    diff_z1 = np.dot(np.transpose(w2), y_classes) * sigmoid(z1) * (1 - sigmoid(z1))
    diff_w1 = np.dot(diff_z1, x)
    diff_b1 = diff_z1
    return diff_b1, diff_w1, diff_b2, diff_w2


def loss(class_vec, y):
    return -np.log(class_vec[int(y)])


def shuffle(x, y):
    shape = np.arange(x.shape[0])
    np.random.shuffle(shape)
    y = y[shape]
    x = x[shape]
    return x, y

def main():
    w1 = np.random.uniform(-0.08, 0.08, [HIDDEN_LAYER, INPUT_SIZE])
    b1 = np.random.uniform(-0.08, 0.08, [HIDDEN_LAYER, 1])
    w2 = np.random.uniform(-0.08, 0.08, [CLASSES, HIDDEN_LAYER])
    b2 = np.random.uniform(-0.08, 0.08, [CLASSES, 1])

    train_x, train_y, val_x, val_y, test_x = load()
    params = [w1, b1, w2, b2]
    params = train(params, train_x, train_y, val_x, val_y)
    pred_file = open("test_y", 'w')
    for x in test_x:
        x = np.reshape(x, (1, INPUT_SIZE))
        y_hat, h1, z1, z2 = fprop(params, x)
        y_hat = y_hat.argmax(axis=0)
        pred_file.write(str(y_hat[0]) + "\n")
    pred_file.close()


if __name__ == '__main__':
    main()


