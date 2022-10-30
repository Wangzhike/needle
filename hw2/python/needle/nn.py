"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype, requires_grad=True))
        if bias:
          self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype, requires_grad=True).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N = X.shape[0]
        if self.bias:
          return X @ self.weight + self.bias.broadcast_to((N, self.out_features))
        else:
          return X @ self.weight
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        B = X.shape[0]
        return X.reshape((B,-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
          x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        m, k = logits.shape
        y_one_hot = init.one_hot(k, y, dtype='float32', requires_grad=True)
        Z_y = (logits * y_one_hot).sum(axes=(1,))
        return (ops.logsumexp(logits, axes=(1,)) - Z_y).sum() / m
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_var = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_shape = list(x.shape)
        x_shape[0] = 1
        if self.training:
          B = x.shape[0]
          mean = x.sum(axes=(0,)) / B
          var = ((x - mean.reshape(x_shape).broadcast_to(x.shape)) ** 2).sum(axes=(0,)) / B
          self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mean.data
          self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * var.data
          mean = mean.reshape(x_shape)
          var = var.reshape(x_shape)
          norm_x = (x - mean.broadcast_to(x.shape)) / ((var.broadcast_to(x.shape) + self.eps) ** (1/2))
          return self.weight.broadcast_to(x.shape) * norm_x + self.bias.broadcast_to(x.shape)
        else:
          self.running_mean = self.running_mean.reshape(x_shape)
          self.running_var = self.running_var.reshape(x_shape)
          norm_x = (x - self.running_mean.broadcast_to(x.shape)) / ((self.running_var.broadcast_to(x.shape) + self.eps) ** (1/2))
          return self.weight.broadcast_to(x.shape) * norm_x + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(1, self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x_shape = list(x.shape)
        x_shape[1] = 1
        mean = x.sum(axes=(1,)) / self.dim
        var = ((x - mean.reshape(x_shape).broadcast_to(x.shape)) ** 2).sum(axes=(1,)) / self.dim
        # v = (x * x).sum(axes=(1,)) / self.dim - e ** 2 # Note: this formula will break numerical stability  
        mean = mean.reshape(x_shape)
        var = var.reshape(x_shape)
        norm_x = (x - mean.broadcast_to(x.shape)) / ((var.broadcast_to(x.shape) + self.eps) ** (1/2))
        return self.weight.broadcast_to(x.shape) * norm_x + self.bias.broadcast_to(x.shape)
        '''
        ### Note: can't cal mean and variance by numpy
        cached_data = x.realize_cached_data()
        e = Tensor(np.mean(cached_data, axis=1, keepdims=True))
        v = Tensor(np.var(cached_data, axis=1, keepdims=True))
        norm_x = (x - e.broadcast_to(x.shape)) / ((v.broadcast_to(x.shape) + self.eps) ** (1/2))
        print('norm_x.shape = {}'.format(norm_x.shape))
        return self.weight.broadcast_to(x.shape) * norm_x + self.bias.broadcast_to(x.shape)
        '''
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          prob = Parameter(init.randb(x.shape[0], x.shape[1], p=1-self.p))
          x = x * prob / (1 - self.p)
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



