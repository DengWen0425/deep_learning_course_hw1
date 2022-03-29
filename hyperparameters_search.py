from data_processings import *
from full_connected_net import Net


if __name__ == "__main__":
    train_data, train_labels = load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
    test_data, test_labels = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    train_data = train_data.reshape(train_data.shape[0], -1)
    test_data = test_data.reshape(test_data.shape[0], -1)
    valid_data, valid_labels = train_data[50000:], train_labels[50000:]
    train_data, train_labels = train_data[:50000], train_labels[:50000]

    # standardize the data
    train_data, mu, sigma = standardize_cols(train_data)
    valid_data, _, _ = standardize_cols(valid_data, mu, sigma)
    test_data, _, _ = standardize_cols(test_data, mu, sigma)

    # search hyperparameters
    best_valid_error = 1
    best_lr = 1e-4
    best_hidden1 = 512
    best_hidden2 = 256
    best_reg = 1e-4
    for lr in [1e-4, 5e-4, 1e-3]:
        net = Net(28 * 28, [best_hidden1, best_hidden2], 10)
        valid_error = net.train(train_data, train_labels, valid_data, valid_labels, reg=best_reg,
                                learning_rate=lr, lr_decay_steps=100, lr_decay_rate=0.8, maxIters=5000,
                                batch_size=32)
        if best_valid_error > valid_error:
            best_lr = lr
            best_valid_error = valid_error
            net.saveModel()

    for hidden1 in [1024, 512, 256]:
        net = Net(28 * 28, [hidden1, best_hidden2], 10)
        valid_error = net.train(train_data, train_labels, valid_data, valid_labels, reg=best_reg,
                                learning_rate=best_lr, lr_decay_steps=100, lr_decay_rate=0.8, maxIters=5000,
                                batch_size=32)
        if best_valid_error > valid_error:
            best_hidden1 = hidden1
            best_valid_error = valid_error
            net.saveModel()

    for hidden2 in [512, 256, 128]:
        net = Net(28 * 28, [best_hidden1, hidden2], 10)
        valid_error = net.train(train_data, train_labels, valid_data, valid_labels, reg=best_reg,
                                learning_rate=best_lr, lr_decay_steps=100, lr_decay_rate=0.8, maxIters=5000,
                                batch_size=32)
        if best_valid_error > valid_error:
            best_hidden2 = hidden2
            best_valid_error = valid_error
            net.saveModel()

    for reg in [1e-4, 1e-3, 1e-2]:
        net = Net(28 * 28, [best_hidden1, best_hidden2], 10)
        valid_error = net.train(train_data, train_labels, valid_data, valid_labels, reg=reg,
                                learning_rate=best_lr, lr_decay_steps=100, lr_decay_rate=0.8, maxIters=5000,
                                batch_size=32)
        if best_valid_error > valid_error:
            best_reg = reg
            best_valid_error = valid_error
            net.saveModel()


