import numpy as np
from copy import deepcopy
from data_processings import linearInd2Binary
import matplotlib.pyplot as plt


class Net(object):
    def __init__(self, input_size, nHidden, nLabels, std=1e-2):
        self.input_size = input_size
        self.nLabels = nLabels
        self.nHidden = nHidden
        self.layers = len(nHidden)
        # initialize the weights
        self.weights = {0: std * np.random.randn(input_size, nHidden[0])}
        self.bias = {0: np.zeros(nHidden[0])}
        for i in range(len(nHidden) - 1):
            self.weights[i + 1] = std * np.random.randn(nHidden[i], nHidden[i + 1])
            self.bias[i + 1] = np.zeros(nHidden[i + 1])
        self.weights[len(nHidden)] = std * np.random.randn(nHidden[-1], nLabels)
        self.bias[len(nHidden)] = np.zeros(nLabels)

    def trainFoward(self, X, y, reg=0.0, keep_prob=0.5):
        """
        compute the forward loss and backward gradient
        """
        y = linearInd2Binary(y, self.nLabels)

        # forward pass
        ip = {0: np.dot(X, self.weights[0]) + self.bias[0]}
        #fp = {0: np.tanh(ip[0])}
        fp = {0: np.maximum(0, ip[0])}

        H = {0: (np.random.rand(*fp[0].shape) < keep_prob) / keep_prob}  # dropout mask
        fp[0] *= H[0]  # drop
        for h in range(self.layers - 1):
            ip[h + 1] = np.dot(fp[h], self.weights[h + 1]) + self.bias[h + 1]
            fp[h + 1] = np.maximum(0, ip[h + 1])  # activation
            # fp[h + 1] = np.tanh(ip[h + 1])
            H[h+1] = (np.random.rand(*fp[h+1].shape) < keep_prob) / keep_prob  # dropout mask
            fp[h+1] *= H[h+1]  # drop
        output = np.dot(fp[self.layers - 1], self.weights[self.layers]) + self.bias[self.layers]
        # softmax
        output = output - np.max(output, axis=1, keepdims=True)
        y_hat = np.exp(output)
        y_hat = y_hat/np.sum(y_hat, axis=1).reshape((y_hat.shape[0], 1))

        # loss
        loss = np.sum(-np.log(np.sum(y_hat*y)))  # log-likelihood loss
        # l2 regularization
        reg_weights = 0
        for h in range(self.layers):
            reg_weights += np.sum(self.weights[h] ** 2)
        loss += reg * reg_weights

        # backward pass
        grad_w = {}
        grad_b = {}
        backprop = y_hat - y  # grad base on softmax
        grad_w[self.layers] = np.dot(fp[self.layers - 1].T, backprop) + 2*reg*self.weights[self.layers]
        grad_b[self.layers] = np.sum(backprop, axis=0)
        for h in range(self.layers, 1, -1):
            # backprop = np.dot(backprop, self.weights[h].T) * (1 - np.tanh(ip[h - 1]) ** 2)
            backprop = H[h-1] * np.dot(backprop, self.weights[h].T) * (ip[h - 1] > 0)  # after dropout
            grad_w[h - 1] = np.dot(fp[h - 2].T, backprop) + 2*reg*self.weights[h-1]
            grad_b[h - 1] = np.sum(backprop, axis=0)
        # backprop = np.dot(backprop, self.weights[1].T) * (1 - np.tanh(ip[0]) ** 2)
        backprop = H[0] * np.dot(backprop, self.weights[1].T) * (ip[0] > 0)  # after dropout
        grad_w[0] = np.dot(X.T, backprop) + 2*reg*self.weights[0]
        grad_b[0] = np.sum(backprop, axis=0)

        return loss, (grad_w, grad_b)

    def validForward(self, X, y, reg):
        y = linearInd2Binary(y, self.nLabels)
        output = np.maximum(0, np.dot(X, self.weights[0]) + self.bias[0])
        for h in range(1, self.layers):
            output = np.maximum(0, np.dot(output, self.weights[h]) + self.bias[h])
        output = np.dot(output, self.weights[self.layers]) + self.bias[self.layers]
        pred = np.argmax(output, axis=1) + 1
        # softmax
        output = output - np.max(output, axis=1, keepdims=True)
        y_hat = np.exp(output)
        y_hat = y_hat/np.sum(y_hat, axis=1).reshape((y_hat.shape[0], 1))
        # loss
        loss = np.sum(-np.log(np.sum(y_hat*y)))  # log-likelihood loss
        reg_weights = 0
        for h in range(self.layers):
            reg_weights += np.sum(self.weights[h] ** 2)
        loss += reg * reg_weights
        return pred, loss

    def train(self, X, y, X_valid, y_valid, reg, learning_rate, lr_decay_steps, lr_decay_rate, maxIters, batch_size, keep_prob=0.5):

        best_val_err = 1
        num_train = X.shape[0]

        old_weights = deepcopy(self.weights)
        old_bias = deepcopy(self.bias)

        train_loss_record = []
        valid_loss_record = []
        valid_acc_record = []
        for i in range(maxIters):
            # random choose
            batch_nid = np.random.choice(num_train, batch_size)
            batch_X = X[batch_nid]
            batch_y = y[batch_nid]

            # compute loss and grad
            loss, grads = self.trainFoward(batch_X, batch_y, reg, keep_prob)
            train_loss_record.append(loss)

            # to save current weights
            tmp_weights = deepcopy(self.weights)
            tmp_bias = deepcopy(self.bias)
            # update
            for h in range(self.layers + 1):
                self.weights[h] = self.weights[h] - learning_rate * grads[0][h] + 0.9 * (
                            self.weights[h] - old_weights[h])
                self.bias[h] = self.bias[h] - learning_rate * grads[1][h] + 0.9 * (self.bias[h] - old_bias[h])
            if lr_decay_steps > 0 and (i+1) % lr_decay_steps == 0:
                learning_rate *= lr_decay_rate
            old_weights = tmp_weights
            old_bias = tmp_bias

            # report the training info
            if (i + 1) % (maxIters // 100) == 0:
                yhat, valid_loss = self.validForward(X_valid, y_valid, reg)
                valid_loss_record.append(valid_loss)
                valid_err = np.mean(y_valid.reshape(yhat.shape) != yhat)
                valid_acc_record.append(1-valid_err)
                print("iterations: {:d} / {:d}, validation error: {:f}".format(i + 1, maxIters, valid_err))
                if valid_err < best_val_err:
                    best_val_err = valid_err
                    best_weights = {}
                    best_bias = {}
                    for k, v in self.weights.items():
                        best_weights[k] = v.copy()
                    for k, v in self.bias.items():
                        best_bias[k] = v.copy()

        axis = list(range(maxIters))
        valid_axis = list(filter(lambda x: x % (maxIters // 100) == 0, axis))
        plt.subplot(1, 2, 1)
        plt.title("loss curve")
        plt.plot(axis, train_loss_record, color='red', label='train loss')
        plt.plot(valid_axis, valid_loss_record, color='blue', label='valid loss')
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.subplot(1, 2, 2)
        plt.title('valid acc')
        plt.plot(valid_axis, valid_acc_record, color='green', label='valid acc')
        plt.legend()
        plt.xlabel('iterations')
        plt.ylabel('valid acc')
        plt.show()

        self.weights = best_weights
        self.bias = best_bias
        return best_val_err

    def test(self, X_test, y_test):
        """
        predict on test and report the acc
        """
        test_pred = self.predict(X_test)
        acc = np.mean(y_test.reshape(test_pred.shape) == test_pred)
        print("Test set accuracy:", acc)
        return acc

    def predict(self, X):
        """
        predict based on given X
        """
        output = np.maximum(0, np.dot(X, self.weights[0]) + self.bias[0])
        for h in range(1, self.layers):
            output = np.maximum(0, np.dot(output, self.weights[h]) + self.bias[h])
        pred = np.dot(output, self.weights[self.layers]) + self.bias[self.layers]
        return np.argmax(pred, axis=1) + 1

    def saveModel(self):
        args_dict = {'input_size': self.input_size,
                     'nHidden': self.nHidden,
                     'nLabels': self.nLabels}
        import pickle as pkl
        with open('args_dict.pkl', 'wb') as f:
            pkl.dump(args_dict, f)
        with open('model_weights.pkl', 'wb') as f:
            pkl.dump({'weight': self.weights, 'bias': self.bias}, f)

    def loadModel(self, w_dir='model_weights.pkl'):
        import pickle as pkl
        with open(w_dir, 'rb') as f:
            weights = pkl.load(f)
        self.weights, self.bias = weights['weight'], weights['bias']






