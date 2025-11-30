"""
author: Aun

Implementation of conv and pooling layers
"""
import numpy as np

def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None

    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)

    pad_width = ((0, 0), (0, 0), (pad, pad))
    x_padded = np.pad(x, pad_width)

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    outH = int(1+(H + 2 * pad - HH) // stride)
    outW = int(1+(W + 2 * pad - WW) // stride)
    
    out = np.zeros((N, F, outH, outW))

    for n in range(N):
        for f in range(F):
            for h in range(outH):
                for w in range(outW):
                    h_start = h * stride
                    h_end = h_start + HH
                    w_start = w * stride
                    w_end = w_start + WW
                    out[n, f, h, w] = np.sum(x_padded[n, :, h_start:h_end, w_start:w_end] * w[f]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache

def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """

    dx, dw, db = None, None, None
    x, w, b, conv_param = cache
    
  
    dw = np.zeros_like(w)
    dx = np.zeros_like(x)
    db = np.zeros_like(b)


    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    pad_width = ((0, 0),        # n dimension
                 (0, 0),        # b dimension
                 (pad, pad),    # h dimension (2 rows on top, 2 on bottom)
                 (pad, pad))    
    x_padded = np.pad(x, pad_width,  mode='constant', constant_values=0)

    db = dout.sum(axis=(0, 2, 3))
    dx_padded = np.zeros_like(x_padded)

    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, outH, outW = dout.shape

    for n in range(N):
        for f in range(F):
            for h in range(outH):
                for w in range(outW):
                    h_start = h*stride
                    h_end = h_start + HH
                    w_start = w*stride
                    w_end = w_start + WW

                    dx_padded[n,:,h_start:h_end,w_start:w_end] += w[f] * dout[n,f,h,w]
                    dw[f] += x_padded[n,:,h_start:h_end,w_start:w_end] * dout[n,f,h,w]

    dx = dx_padded[:, :, pad:-pad, pad:-pad]

    return dx, dw, db



