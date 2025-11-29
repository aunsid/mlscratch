"""
author: Aun

Contains the building blocks of a NN.
Most of the code is the implementation from cs231n assignments.
"""
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    temp = x.reshape(x.shape[0], -1) # (N, d_1, ..., d_k) -> (N, D)
    out = temp @ x + b
    cache = (x, w, b)
   
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    # dout is the same shape as out
    # (N, M)

    # dx -> (N, D)
    dx = dout @ w.T # (N, M) @ (M, D) -> (N, D)
    dx = dx.reshape(x.shape)

    # dw -> (D, M)
    dw = x.reshape(x.shape[0], -1).T @ dout # (D, N) @ (N, M) -> (D, M)

    # db
    db = dout.sum(0) # (M,)

    return dx, dw, db