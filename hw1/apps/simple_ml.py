import struct
import gzip
import numpy as np

import sys
sys.path.append('python/')
import needle as ndl


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
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
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
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    ### construct the computation graph of softmax_loss
    '''
    v1 = ndl.ops.Exp()(Z)
    v2 = v1.sum(axes=(1,))
    v3 = ndl.ops.Log()(v2)

    v4 = Z * y_one_hot
    v5 = v4.sum(axes=(1,))

    v6 = v3 - v5
    return v6.sum(axes=(0,)) / Z.shape[0]
    '''
    m = Z.shape[0]
    lhs = ndl.ops.Log()(ndl.ops.Exp()(Z).sum(axes=(1,)))
    rhs = (Z * y_one_hot).sum(axes=(1,))
    return (lhs - rhs).sum(axes=(0,)) / m
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    m = y.shape[0]
    k = W2.shape[1]
    Y = np.zeros((m, k), dtype='uint8')
    Y[np.arange(m), y] = 1
    nbatch = m // batch + 1
    ### loop over batch
    for i in range(nbatch):
      start = i * batch
      end = min(start+batch, m)
      if start == end:
        break
      minibatch = end - start

      ### forward propagation
      X_batch = ndl.Tensor(X[start:end, :])
      Y_batch = ndl.Tensor(Y[start:end, :])
      Z1_batch = ndl.ops.ReLU()(X_batch @ W1)
      Z2_batch = Z1_batch @ W2
      L = softmax_loss(Z2_batch, Y_batch)

      ### backward propagation
      L.backward()
      dW1 = W1.grad.realize_cached_data()
      dW2 = W2.grad.realize_cached_data()
  
      ### SGD
      W1_cached_data = W1.realize_cached_data()
      W2_cached_data = W2.realize_cached_data()
      W1_cached_data -= lr * dW1
      W2_cached_data -= lr * dW2

      ### Reset W1, W2
      W1 = ndl.Tensor(W1_cached_data)
      W2 = ndl.Tensor(W2_cached_data)
    
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h,y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
