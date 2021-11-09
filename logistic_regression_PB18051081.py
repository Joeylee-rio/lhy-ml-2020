# logistic regression
# lab1 @2021-Intros to Machine Learning @ prof.defulian
# editted by Li Zhaoyi
# reference: prof.Lee Hungyi-2020-ML-courses-hw2-classification
# https://colab.research.google.com/drive/1JaMKJU7hvnDoUfZjvUKzm9u-JLeX6B2C#scrollTo=7NzAmkzU2MAS
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Eps = 1e-8
# part-I:processing data



# test_data = pd.read_csv(r"C:\Users\Strawberry\Desktop\ml_lab1\wdbc_test.data")
'''
test_data = pd.read_csv(sys.argv[1])
print(test_data)
print(type(test_data))
test_data = np.array(test_data)
print(test_data)
print(type(test_data))
#test_data = np.delete(test_data,0,1)
test_data = np.delete(test_data,[0,1],1)
print(test_data)
print(type(test_data))

'''


argv = sys.argv
training_set_path = argv[1]
testing_set_path = argv[2]
# print(argv, type(argv), argv[1], type(argv[1]))
training_set = pd.read_csv(argv[1])
testing_set = pd.read_csv(argv[2])
training_set = np.array(training_set)
testing_set = np.array(testing_set)
Y = training_set[:, 1]
Y_test = testing_set[:, 1]
X = np.delete(training_set, [0, 1], 1)
X_test = np.delete(testing_set, [0, 1], 1)
'''
print(training_set.dtype)
print(Y)
print(X.dtype)
'''
# testing_set = np.delete(testing_set, [0, 1], 1)


def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])


# split data into two parts: training data and development data
def _train_dev_split(X, Y, dev_ratio = 0.25):
    X, Y = _shuffle(X, Y)
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


# Normalize
def _normalize(X, istrain = True, mean_train = None, std_train = None):
    columns = np.arange(X.shape[1])
    if istrain:
        mean = np.mean(X[:, columns].astype(np.float), 0).reshape(1,-1)
        std = np.std(X[:, columns].astype(np.float), 0).reshape(1, -1)
    else :
        mean = mean_train
        std = std_train
    X[:, columns] = (X[:, columns] - mean) / (std + Eps)
    return X, mean, std

for i in range(len(Y)):
    if Y[i] == 'M':
        Y[i] = 1
    else:
        Y[i] = 0
# Y-element is numpy.str type
# change it to numpy.int32 type
Y = Y.astype(int)
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X, Y)
X_train, mean_train, std_train = _normalize(X_train)
X_dev, _, _ = _normalize(X_dev, istrain = False, mean_train = mean_train, std_train = std_train)
X_test, _, _ = _normalize(X_test, istrain = False, mean_train = mean_train, std_train = std_train)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
dim = X_train.shape[1]

# part-II:constructing logistic regression model functions

def _sigmoid(z):

    return np.clip(1 / (1.0 + np.exp(-z.astype(float))), Eps, 1-Eps)

def _f(X, w, b):
    return _sigmoid(np.matmul(X,w) + b)

def _predict(X, w, b):
    return np.round(_f(X,w,b).astype(np.int))

def _acc(Y_pred, Y_label):
    acc = 1 - np.mean(np.abs(Y_pred-Y_label))
    return acc


# part-III:gradient & loss
def _cross_entropy_loss(Y_pred, Y_label):
    cross_entropy = -(np.dot(Y_label, np.log(Y_pred)) + np.dot((1-Y_label), np.log(1-Y_pred)))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, axis = 1)
    # compress col
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

#part-IV training model

def _training(X_train, Y_train, X_dev, Y_dev, lr = 0.1, epochs = 100, batchsize = 10):
    w = np.zeros((dim,))
    b = np.zeros((1,))
    training_loss = []
    training_acc = []
    dev_loss = []
    dev_acc = []
    train_size = X_train.shape[0]
    dev_size = X_dev.shape[0]

    Y_train_pred_raw = _f(X_train, w, b)
    Y_train_pred_processed = np.round(Y_train_pred_raw)
    training_loss.append(_cross_entropy_loss(Y_train_pred_raw, Y_train) / train_size)
    training_acc.append(_acc(Y_train_pred_processed, Y_train))

    Y_dev_pred_raw = _f(X_dev, w, b)
    Y_dev_pred_processed = np.round(Y_dev_pred_raw)
    dev_loss.append(_cross_entropy_loss(Y_dev_pred_raw, Y_dev) / dev_size)
    dev_acc.append(_acc(Y_dev_pred_processed, Y_dev))
    for i in range(epochs):
        X_train, Y_train = _shuffle(X_train, Y_train)

       
        # mini-batch training
        for idx in range(int(np.floor(train_size / batchsize))):
            X_batch = X_train[idx*batchsize : (idx+1)*batchsize]
            Y_batch = Y_train[idx*batchsize : (idx+1)*batchsize]
            
            w_grad, b_grad = _gradient(X_batch, Y_batch, w, b)
            w = w - lr * w_grad
            b = b - lr * b_grad

            
            # gradient descend update w and b
            # compute loss
            Y_train_pred_raw = _f(X_train, w, b)
            Y_train_pred_processed = np.round(Y_train_pred_raw)
            training_loss.append(_cross_entropy_loss(Y_train_pred_raw, Y_train) / train_size)
            training_acc.append(_acc(Y_train_pred_processed, Y_train))

            Y_dev_pred_raw = _f(X_dev, w, b)
            Y_dev_pred_processed = np.round(Y_dev_pred_raw)
            dev_loss.append(_cross_entropy_loss(Y_dev_pred_raw, Y_dev) / dev_size)
            dev_acc.append(_acc(Y_dev_pred_processed, Y_dev))
            
            # debug
            

          
        '''
        # compute loss
        Y_train_pred_raw = _f(X_train, w, b)
        Y_train_pred_processed = np.round(Y_train_pred_raw)
        training_loss.append(_cross_entropy_loss(Y_train_pred_raw, Y_train) / train_size)
        training_acc.append(_acc(Y_train_pred_processed, Y_train))

        Y_dev_pred_raw = _f(X_dev, w, b)
        Y_dev_pred_processed = np.round(Y_dev_pred_raw)
        dev_loss.append(_cross_entropy_loss(Y_dev_pred_raw, Y_dev) / dev_size)
        dev_acc.append(_acc(Y_dev_pred_processed, Y_dev))
        '''      


    print('Training Loss = {}'.format(training_loss[-1]))
    print('Training Acc = {}'.format(training_acc[-1]))
    print('Development Loss: {}'.format(dev_loss[-1]))
    print('Development Acc: {}'.format(dev_acc[-1]))
    return training_loss, training_acc, dev_loss, dev_acc, w, b


# Plotting loss and acc curve
def _plotting(train_loss, train_acc, dev_loss, dev_acc):
    plt.plot(train_loss)
    plt.plot(dev_loss)
    plt.title('Loss')
    plt.legend(['train', 'dev'])
    plt.savefig('loss.png')
    plt.show()

    plt.plot(train_acc)
    plt.plot(dev_acc)
    plt.title('Accuracy')
    plt.legend(['train', 'dev'])
    plt.savefig('acc.png')
    plt.show()


def _predict(W, b, X_test, Y_test):
    test_size = X_test.shape[0]
    Y_test_predict_raw = _f(X_test, W, b)
    Y_test_predict_processed = np.round(Y_test_predict_raw)
    corr_count = 0
    f = open("predict_result.txt", 'w')
    for i in range(test_size):
        if Y_test_predict_processed[i] == 1:
            print('M', file = f)
            if Y_test[i] == 'M':
                corr_count += 1
        else:
            print('B', file = f)
            if Y_test[i] == 'B':
                corr_count += 1
    f.close()
    predict_acc = corr_count / test_size
    return predict_acc

def main():
    training_loss, training_acc, dev_loss, dev_acc, W, b = _training(X_train = X_train, Y_train = Y_train, X_dev = X_dev, Y_dev = Y_dev, lr = 0.01, epochs = 10)
    # _plotting(training_loss, training_acc, dev_loss, dev_acc)
    predict_acc = _predict(W, b, X_test, Y_test)
    print("Accuracy on testing set is ", predict_acc)

main()