"""

nn_layers_pt.py

PyTorch version of nn_layers

"""

import torch
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
                          , torch.tensor(step), rounding_mode = 'floor') + 1

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))

    arr_out = torch.as_strided(arr_in, size=new_shape, stride=strides)
    return arr_out

#######
# if necessary, you can define additional functions which help your implementation,
# and import proper libraries for those functions.
#######

class nn_convolutional_layer:

    def __init__(self, f_height, f_width, input_size, in_ch_size, out_ch_size):
        
        # Xavier init
        self.W = torch.normal(0, 1 / math.sqrt((in_ch_size * f_height * f_width / 2)),
                                  size=(out_ch_size, in_ch_size, f_height, f_width))
        self.b = 0.01 + torch.zeros(size=(1, out_ch_size, 1, 1))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)
        
        self.input_size = input_size

    def update_weights(self, dW, db):
        self.W += dW
        self.b += db

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()

    def forward(self, x):
        ###################################
        # Q4. Implement your layer here
        ###################################
        batch_s, input_channel, in_width, in_height = x.shape
        num_filter, filter_w, filter_h = self.W.shape[0], self.W.shape[2], self.W.shape[3]
        window_w, window_h = in_width - filter_w + 1, in_height - filter_h + 1

        windows = torch.empty((batch_s, input_channel,
                               window_w, window_h,
                               filter_w, filter_h))

        for batch in range(batch_s):
            for channel in range(input_channel):
                windows[batch][channel] = view_as_windows(x[batch][channel], (filter_w, filter_h))

        w_reshaped = self.W.reshape(num_filter, -1)
        windows_reshaped = windows.permute(0, 2, 3, 1, 4, 5).reshape(batch_s, window_w, window_h, -1)

        out = torch.matmul(windows_reshaped, w_reshaped.T).permute(0, 3, 1, 2) + self.b
        return out
        
    
    def step(self, lr, friction):
        with torch.no_grad():
            self.v_W = friction*self.v_W + (1-friction)*self.W.grad
            self.v_b = friction*self.v_b + (1-friction)*self.b.grad
            self.W -= lr*self.v_W
            self.b -= lr*self.v_b
            
            self.W.grad.zero_()
            self.b.grad.zero_()

# max pooling
class nn_max_pooling_layer:
    def __init__(self, pool_size, stride):
        self.stride = stride
        self.pool_size = pool_size

    def forward(self, x):
        ###################################
        # Q5. Implement your layer here
        ###################################
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

# relu activation
class nn_activation_layer:

    # linear layer. creates matrix W and bias b
    # W is in by out, and b is out by 1
    def __init__(self):
        pass

    def forward(self, x):
        return x.clamp(min=0)

# fully connected (linear) layer
class nn_fc_layer:

    def __init__(self, input_size, output_size, std=1):
        
        # Xavier/He init
        self.W = torch.normal(0, std/math.sqrt(input_size/2), (output_size, input_size))
        self.b=0.01+torch.zeros((output_size))

        self.W.requires_grad = True
        self.b.requires_grad = True

        self.v_W = torch.zeros_like(self.W)
        self.v_b = torch.zeros_like(self.b)

    ## Q1
    def forward(self,x):
        # compute forward pass of given parameter
        # output size is batch x output_size x 1 x 1
        # input size is batch x input_size x filt_size x filt_size
        output_size = self.W.shape[0]
        batch_size = x.shape[0]
        Wx = torch.mm(x.reshape((batch_size, -1)),(self.W.reshape(output_size, -1)).T)
        out = Wx+self.b
        return out

    def update_weights(self,dLdW,dLdb):

        # parameter update
        self.W=self.W+dLdW
        self.b=self.b+dLdb

    def get_weights(self):
        return self.W.clone().detach(), self.b.clone().detach()

    def set_weights(self, W, b):
        self.W = W.clone().detach()
        self.b = b.clone().detach()
    
    def step(self, lr, friction):
        with torch.no_grad():
            self.v_W = friction*self.v_W + (1-friction)*self.W.grad
            self.v_b = friction*self.v_b + (1-friction)*self.b.grad
            self.W -= lr*self.v_W
            self.b -= lr*self.v_b
            self.W.grad.zero_()
            self.b.grad.zero_()


# softmax layer
class nn_softmax_layer:
    def __init__(self):
        pass

    def forward(self, x):
        s = x - torch.unsqueeze(torch.amax(x, axis=1), -1)
        return (torch.exp(s) / torch.unsqueeze(torch.sum(torch.exp(s), axis=1), -1)).reshape((x.shape[0],x.shape[1]))


# cross entropy layer
class nn_cross_entropy_layer:
    def __init__(self):
        self.eps=1e-15

    def forward(self, x, y):
        # first get softmax
        batch_size = x.shape[0]
        num_class = x.shape[1]
        
        onehot = np.zeros((batch_size, num_class))
        onehot[range(batch_size), (np.array(y)).reshape(-1, )] = 1
        onehot = torch.as_tensor(onehot)

        # avoid numerial instability
        x[x<self.eps]=self.eps
        x=x/torch.unsqueeze(torch.sum(x,axis=1), -1)

        return sum(-torch.sum(torch.log(x.reshape(batch_size, -1)) * onehot, axis=0)) / batch_size
