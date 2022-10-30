import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

import struct
import gzip
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

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
          img = np.flip(img, axis=1).copy()
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        # print('shift_x = {}, shift_y = {}'.format(shift_x, shift_y))
        # print('img.shape = {}'.format(img.shape))
        nH, nW, nC = img.shape
        img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0,0)), mode = 'constant', constant_values = (0,0))
        # print('[padding] img.shape = {}'.format(img.shape))
        h_st, h_end = shift_x + self.padding, shift_x + self.padding + nH
        w_st, w_end = shift_y + self.padding, shift_y + self.padding + nW
        # print('h_st = {}, h_end = {}'.format(h_st, h_end))
        # print('w_st = {}, w_end = {}'.format(w_st, w_end))
        return img[h_st:h_end, w_st:w_end, :]
        ### END YOUR SOLUTION

class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        self.X, self.y = parse_mnist(image_filename, label_filename)
        self.m = self.X.shape[0]
        self.nH = int(self.X.shape[1] ** (1/2))
        self.nW = self.nH
        # print('m = {}, nH = {}, nW = {}'.format(self.m, self.nH, self.nH))
        self.X = self.X.reshape(self.m, self.nH, self.nW)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        img = self.X[index,:].reshape(self.nH, self.nW, 1)
        label = self.y[index]
        if self.transforms:
          for tform in self.transforms:
            img = tform(img)
        return (img, label)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.m
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
