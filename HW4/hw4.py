"""
HW4
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import numpy as np
import math

"""

function view_as_windows

"""


def view_as_windows(arr_in, window_shape, step=1):
    # -- basic checks on arguments
    if not torch.is_tensor(arr_in):
        raise TypeError("`arr_in` must be a pytorch tensor")

    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim
    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = torch.tensor(arr_in.shape)
    window_shape = torch.tensor(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    # window_strides = torch.tensor(arr_in.stride())
    window_strides = arr_in.stride()

    indexing_strides = arr_in[slices].stride()

    win_indices_shape = torch.div(arr_shape - window_shape
                                  , torch.tensor(step), rounding_mode='floor') + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = torch.as_strided(arr_in, size=new_shape, stride=strides)
    return arr_out


debug_flag = False


class nn_convolutional_layer:

    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):

        # Xavier init
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                              size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()

    #######
    # Q1. Complete this method
    #######
    def forward(self, x):
        batch_s, input_channel, in_width, in_height = x.shape
        num_filter, filter_w, filter_h = self.W.shape[0], self.W.shape[2], self.W.shape[3]
        window_w, window_h = in_width - filter_w + 1, in_height - filter_h + 1

        windows = torch.empty((batch_s, input_channel,
                               window_w, window_h,
                               filter_w, filter_h))

        if debug_flag:
            print("x shape", x.shape)
            print("Filter shape", self.W.shape)
            print("Window shape", windows.shape)

        for batch in range(batch_s):
            for channel in range(input_channel):
                windows[batch][channel] = view_as_windows(x[batch][channel], (filter_w, filter_h))

        w_reshaped = self.W.reshape(num_filter, -1)
        windows_reshaped = windows.permute(0, 2, 3, 1, 4, 5).reshape(batch_s, window_w, window_h, -1)

        if debug_flag:
            print(w_reshaped.shape, windows_reshaped.shape)

        out = torch.matmul(windows_reshaped, w_reshaped.T).permute(0, 3, 1, 2)
        return out


class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        self.stride = stride
        self.pool_size = pool_size

    #######
    # Q2. Complete this method
    #######
    def forward(self, x):
        batch_s, input_channel, in_width, in_height = x.shape

        windows = torch.empty((batch_s, input_channel,
                               int(in_width / 2), int(in_height / 2),
                               self.pool_size, self.pool_size))

        for batch in range(batch_s):
            for channel in range(input_channel):
                windows[batch][channel] = view_as_windows(x[batch][channel],
                                                          (self.pool_size, self.pool_size),
                                                          self.stride)
        out = windows.max(-1)[0].max(-1)[0]
        return out


"""
TESTING 
"""

if __name__ == "__main__":

    # data sizes
    batch_size = 8
    input_size = 32
    filter_width = 3
    filter_height = filter_width
    in_ch_size = 3
    num_filters = 8

    std = 1e0
    dt = 1e-3

    # number of test loops
    num_test = 50

    # error parameters
    err_fwd = 0
    err_pool = 0

    # for reproducibility
    # torch.manual_seed(0)

    # set default type to float64
    torch.set_default_dtype(torch.float64)

    print('conv test')
    for i in range(num_test):
        # create convolutional layer object
        cnv = nn_convolutional_layer(filter_height, filter_width, input_size,
                                     in_ch_size, num_filters)

        # test conv layer from torch.nn for reference
        test_conv_layer = nn.Conv2d(in_channels=in_ch_size, out_channels=num_filters,
                                    kernel_size=(filter_height, filter_width))

        # test input
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))

        with torch.no_grad():
            out = cnv.forward(x)
            W, b = cnv.get_weights()
            test_conv_layer.weight = nn.Parameter(W)
            test_conv_layer.bias = nn.Parameter(torch.squeeze(b))
            test_out = test_conv_layer(x)

            err = torch.norm(test_out - out) / torch.norm(test_out)
            err_fwd += err

    stride = 2
    pool_size = 2

    print('pooling test')
    for i in range(num_test):
        # create pooling layer object
        mpl = nn_max_pooling_layer(pool_size=pool_size, stride=stride)

        # test pooling layer from torch.nn for reference
        test_pooling_layer = nn.MaxPool2d(kernel_size=(pool_size, pool_size), stride=stride)

        # test input
        x = torch.normal(0, 1, (batch_size, in_ch_size, input_size, input_size))

        with torch.no_grad():
            out = mpl.forward(x)
            test_out = test_pooling_layer(x)

            err = torch.norm(test_out - out) / torch.norm(test_out)
            err_pool += err

    # reporting accuracy results.
    print('accuracy results')
    print('forward accuracy', 100 - err_fwd / num_test * 100, '%')
    print('pooling accuracy', 100 - err_pool / num_test * 100, '%')
