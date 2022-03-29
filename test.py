import pickle as pkl
import matplotlib.pyplot as plt
from full_connected_net import Net
from data_processings import load_mnist, standardize_cols

train_data, train_labels = load_mnist('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz')
test_data, test_labels = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
train_data = train_data.reshape(train_data.shape[0], -1)
train_data, train_labels = train_data[:50000], train_labels[:50000]
test_data = test_data.reshape(test_data.shape[0], -1)

train_data, mu, sigma = standardize_cols(train_data)
test_data, _, _ = standardize_cols(test_data, mu, sigma)

with open('args_dict.pkl', 'rb') as f:
    args = pkl.load(f)

model = Net(args['input_size'], args['nHidden'], args['nLabels'])
model.loadModel('model_weights.pkl')
acc = model.test(test_data, test_labels)

plt.imshow(model.weights[0], cmap='gray')
plt.title('hidden 1')
plt.show()
plt.imshow(model.weights[1], cmap='gray')
plt.title('hidden 2')
plt.show()
plt.imshow(model.weights[2], cmap='gray')
plt.title('output layer')
plt.show()

