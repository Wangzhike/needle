"""Operator implementations."""

from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, out_grad * (-lhs) / (rhs**2),)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        i, j = -2, -1
        if self.axes:
          i, j = self.axes
        # axes = array_api.arange(a.ndim)
        # axes[i], axes[j] = axes[j], axes[i]
        # return array_api.transpose(a, axes)
        return array_api.swapaxes(a, i, j)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad.transpose(self.axes),)
        # i, j = -2, -1
        # if self.axes:
        #   i, j = self.axes
        # cached_data = out_grad.cached_data
        # cached_data = array_api.swapaxes(cached_data, i, j)
        # return Tensor(cached_data) 
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad.reshape(x.shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        x_shape = x.shape[::-1]
        y_shape = out_grad.shape[::-1]
        x_ndim = len(x_shape)
        y_ndim = len(y_shape)

        for i in range(y_ndim):
          y_axis = y_ndim - 1 - i
          y_s = y_shape[i]
          if i < x_ndim:
            x_s = x_shape[i]
            if x_s < y_s:
              y_shape_lst = list(out_grad.shape)
              y_shape_lst[y_axis] = 1
              out_grad = out_grad.sum(y_axis).reshape(y_shape_lst)
          else:
            out_grad = out_grad.sum(y_axis)

        return (out_grad,)
        '''
        x = node.inputs[0]
        x_shape = x.shape[::-1]
        # print('x_shape = {}'.format(x_shape))
        y_shape = out_grad.shape[::-1]
        # print('y_shape = {}'.format(y_shape))
        x_ndim = len(x_shape)
        # print('x_ndim = {}'.format(x_ndim))
        y_ndim = len(y_shape)
        # print('y_ndim = {}'.format(y_ndim))
        cached_data = out_grad.cached_data
        # print('[begin] cached_data: {}'.format(cached_data))
        for i in range(y_ndim):
          y_axis = y_ndim - 1 - i
          # print('i = {}, y_axis = {}'.format(i, y_axis))
          y_s = y_shape[i]
          # print('y_s = {}'.format(y_s))
          if i >= x_ndim:
            # if y_s > 1:
            #   cached_data = array_api.delete(cached_data, [l for l in range(1, y_s)], axis=i)
            # cached_data = array_api.squeeze(cached_data, axis=i)
            cached_data = array_api.sum(cached_data, axis=y_axis)
            # print('[wzk] cached_data.shape = {}'.format(cached_data.shape))
            # cached_data = array_api.squeeze(cached_data, axis=y_axis)
          else:
            x_s = x_shape[i]
            # print('y_s = {}, x_s = {}'.format(y_s, x_s))
            if x_s < y_s:
              # print('do sum, axis = {}'.format(y_axis))
              # cached_data = array_api.delete(cached_data, [l for l in range(x_s, y_s)], axis=i)
              cached_data = array_api.sum(cached_data, axis=y_axis, keepdims=True)
              # print('cached_data: {}'.format(cached_data))
        return Tensor(cached_data)
        '''
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        x_shape = list(x.shape)
        for axis in self.axes if self.axes else range(len(x_shape)):
          x_shape[axis] = 1
        out_grad = out_grad.reshape(x_shape)
        return (out_grad.broadcast_to(x.shape),)
        '''
        x = node.inputs[0]
        cached_data = out_grad.cached_data
        for axis in self.axes:
          sz = x.shape[axis]
          cached_data = array_api.expand_dims(cached_data, axis=axis).repeat(sz, axis=axis)
        return Tensor(cached_data)
        '''
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        ld = out_grad @ rhs.transpose()
        rd = lhs.transpose() @ out_grad
        ### deal with broadingcast matmul
        ld_ndim = len(ld.shape)
        lhs_ndim = len(lhs.shape)
        rd_ndim = len(rd.shape)
        rhs_ndim = len(rhs.shape)
        if lhs_ndim < ld_ndim:
          axes = tuple([axis for axis in range(ld_ndim - lhs_ndim)])
          ld = ld.sum(axes)
        if rhs_ndim < rd_ndim:
          axes = tuple([axis for axis in range(rd_ndim - rhs_ndim)])
          rd = rd.sum(axes)
        return ld, rd
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return (out_grad / x,)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * node,)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


# TODO
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        cached_data = x.realize_cached_data()
        cached_data = array_api.where(cached_data > 0, 1, 0)
        return (out_grad * Tensor(cached_data),)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

