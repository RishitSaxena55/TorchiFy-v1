# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from mytorch.nn.resampling import *


class Conv1d_stride1():
    def __init__(self, in_channels, out_channels, kernel_size,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        (batch_size, in_channels, input_size) = A.shape
        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))  # TODO

        for i in range(input_size - self.kernel_size + 1):
            sliced_A = A[:, :, i:i + self.kernel_size]

            Z[:, :, i] = np.tensordot(sliced_A, self.W, axes=((1, 2), (1, 2))) + self.b[None, :]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        (batch_size, out_channels, output_size) = dLdZ.shape

        input_size = self.A.shape[2]
        # self.dLdW = np.zeros((batch_size, self.in_channels, output_size-input_size+1)) # TODO
        self.dLdW = np.zeros((out_channels, self.in_channels, self.kernel_size))
        for i in range(self.kernel_size):
            sliced_A = self.A[:, :, i:i+output_size]
            self.dLdW[:, :, i] = np.tensordot(dLdZ, sliced_A, axes=([0, 2], [0, 2]))

        self.dLdb = np.sum(dLdZ, axis=(0, 2))  # TODO

        dLdA = np.zeros_like(self.A)  # TODO
        dLdZ_padded = np.pad(dLdZ, pad_width=((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1)))
        flipped_W = np.flip(self.W, axis=2)
        padded_out_size = dLdZ_padded.shape[2]
        for i in range(input_size):
            dLdZ_padded_slice = dLdZ_padded[:, :, i:i + self.kernel_size]
            dLdA[:, :, i] = np.tensordot(dLdZ_padded_slice, flipped_W, axes=([1, 2], [0, 2]))

        # np.ones()
        return dLdA


class Conv1d():
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0,
                 weight_init_fn=None, bias_init_fn=None):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(in_channels, out_channels, kernel_size,
                                             weight_init_fn, bias_init_fn)  # TODO
        self.downsample1d = Downsample1d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        # Pad the input appropriately using np.pad() function
        # TODO
        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)), mode="constant")

        # Call Conv1d_stride1
        # TODO
        Z = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(Z)
        # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)  # TODO

        # Unpad the gradient
        # TODO
        if self.pad:
            dLdA = dLdA[:, :, self.pad:-self.pad]

        return dLdA
