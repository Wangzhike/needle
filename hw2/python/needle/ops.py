"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


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
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return (out_grad * rhs, out_grad * lhs,)


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
        x = node.inputs[0]
        return (out_grad * self.scalar * (x**(self.scalar-1)),)
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
        return array_api.divide(a, self.scalar, dtype='float32')
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
        return array_api.swapaxes(a, i, j)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad.transpose(self.axes),)
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


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        cached_data = x.realize_cached_data()
        cached_data = array_api.where(cached_data > 0, 1, 0).astype('float32')
        return (out_grad * Tensor(cached_data),)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # print('axes = {}'.format(self.axes))
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        # print('max_Z.shape = {}'.format(max_Z.shape))
        return array_api.log(array_api.sum(array_api.exp(Z - max_Z), axis=self.axes))\
                + array_api.max(max_Z, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        x_shape = list(x.shape)
        for axis in self.axes if self.axes else range(len(x_shape)):
          x_shape[axis] = 1
        ### cal max_x by array_api
        cached_data = x.realize_cached_data()
        max_cached_data = array_api.max(cached_data, axis=self.axes, keepdims=True)
        max_x = Tensor(max_cached_data).broadcast_to(x.shape)
        ### cal exp after subtract
        exp_diff = exp(x - max_x)
        ### cal sum of exp_diff, and broacast to the input shape
        sum_exp_diff = exp_diff.sum(self.axes)
        sum_exp_diff = sum_exp_diff.reshape(x_shape)
        sum_exp_diff = sum_exp_diff.broadcast_to(x.shape)
        ### broadcast out_grad to the input shape
        out_grad = out_grad.reshape(x_shape)
        out_grad = out_grad.broadcast_to(x.shape)
        return (out_grad * (exp_diff / sum_exp_diff),)
        '''
        print('axes = {}'.format(self.axes))
        v1 = node.inputs[0]
        print('v1.shape = {}'.format(v1.shape))
        v2 = exp(v1)
        print('v2.shape = {}'.format(v2.shape))
        v3 = v2.sum(self.axes)
        print('v3.shape = {}'.format(v3.shape))
        print('out_grad.shape = {}'.format(out_grad.shape))
        g = out_grad / v3
        print('g.shape = {}'.format(g.shape))
        v1_shape = list(v1.shape)
        print('v1_shape = {}'.format(v1_shape))
        for axis in self.axes if self.axes else range(len(v1_shape)):
          v1_shape[axis] = 1
        g = g.reshape(v1_shape)
        print('[after reshape] g.shape = {}'.format(g.shape))
        g = g.broadcast_to(v1.shape)
        return (g * v2,)
        '''
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
