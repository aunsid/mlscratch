"""
author: Aun

Contains loss functions
"""
import numpy as np

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Li = -log(e^f_yi / Σ_j e^f_yj)

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
        class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
        0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    n = x.shape[0]

    logits = x

    # numerical stability
    max_logits = np.max(logits, axis=1, keepdims=True)
    norm_logits = logits - max_logits

    counts = np.exp(norm_logits)
    probs = counts / counts.sum(1, keepdims=True)
    logprobs = np.log(probs)

    loss = -logprobs[range(n), y].mean()

    dx = probs.copy()
    dx[range(n), y] -= 1
    dx /= n

    return loss, dx


def svm_loss(x, y):
    pass
