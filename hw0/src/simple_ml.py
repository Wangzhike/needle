import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    X = None
    Y = None
    with gzip.open(image_filename) as img_f:
      img_buf = img_f.read()
      offset = 0
      magic_num = struct.unpack_from('>i', img_buf, offset)[0]
      assert(magic_num == 2051)
      offset += 4
      m = struct.unpack_from('>i', img_buf, offset)[0]
      offset += 4
      n_h = struct.unpack_from('>i', img_buf, offset)[0]
      offset += 4
      n_w = struct.unpack_from('>i', img_buf, offset)[0]
      offset += 4
      X = np.zeros((m, n_h * n_w), dtype='float32')
      for i in range(m):
        for r in range(n_h):
          for j in range(n_w):
            X[i, r* n_w + j] = struct.unpack_from('>B', img_buf, offset)[0]
            offset += 1
    X = X / 255.

    with gzip.open(label_filename) as label_f:
      label_buf = label_f.read()
      offset = 0
      magic_num = struct.unpack_from('>i', label_buf, offset)[0]
      assert(magic_num == 2049)
      offset += 4
      m = struct.unpack_from('>i', label_buf, offset)[0]
      assert(m == X.shape[0])
      offset += 4
      Y = np.zeros((X.shape[0]), dtype='uint8')
      for i in range(m):
        Y[i] = struct.unpack_from('>B', label_buf, offset)[0]
        offset += 1
    
    return X, Y


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.int8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    return np.mean(np.log(np.sum(np.exp(Z), axis=1)) - Z[np.arange(y.shape[0]), y])
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = X.shape[0]
    k = theta.shape[1]
    Y = np.zeros((m, k), dtype='uint8')
    Y[np.arange(m), y] = 1
    # loop over minibatch
    nbatch = m // batch + 1
    for i in range(nbatch):
      # minibatch
      start = i * batch
      end = min((i+1) * batch, m)
      minibatch = end - start
      if start == end:
        break
      X_batch = X[start:end, :]
      Y_batch = Y[start:end, :]
      Z_batch = np.exp(np.matmul(X_batch, theta))
      Z_batch = Z_batch / np.sum(Z_batch, axis=1, keepdims=True)
      dTheta = 1/minibatch * np.matmul(X_batch.T, Z_batch - Y_batch)
      theta -= lr * dTheta  # (inplace) vs theta = theta - lr * dTheta (non inplace)
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    m = y.shape[0]
    k = W2.shape[1]
    Y = np.zeros((m, k), dtype='uint8')
    Y[np.arange(m), y] = 1
    nbatch = m // batch + 1
    # loop over batch
    for i in range(nbatch):
      start = i * batch
      end = min(start+batch, m)
      if start == end:
        break
      minibatch = end - start

      # forward propagation
      X_batch = X[start:end, :]
      Y_batch = Y[start:end, :]
      Z1_batch = np.matmul(X_batch, W1)
      Z1_batch = np.maximum(0, Z1_batch)
      Z2_batch = np.exp(np.matmul(Z1_batch, W2))
      Z2_batch = Z2_batch / np.sum(Z2_batch, axis=1, keepdims=True)

      # backward propagation
      G2_batch = Z2_batch - Y_batch
      dW2 = 1/minibatch * np.matmul(Z1_batch.T, G2_batch)
      G1_batch = np.multiply(Z1_batch > 0, np.matmul(G2_batch, W2.T))
      dW1 = 1/minibatch * np.matmul(X_batch.T, G1_batch)

      # SGD
      W2 -= lr * dW2
      W1 -= lr * dW1
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
