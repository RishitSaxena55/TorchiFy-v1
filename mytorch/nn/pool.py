import numpy as np
from resampling import *


class MaxPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """

        self.A = A
        (batch_size, in_channels, input_width, input_height)=A.shape
        output_width=input_width-self.kernel+1
        output_height=input_height-self.kernel+1
        Z=np.zeros((batch_size, in_channels, output_width, output_height))
        Z_ind = np.zeros((batch_size, in_channels, output_width, output_height, 2))

        for i in range(output_width):
            for j in range(output_height):
                A_sliced = A[:, :, i:i + self.kernel, j:j + self.kernel]  # (B, C, k, k)
                A_flat = A_sliced.reshape(batch_size, in_channels, -1)  # (B, C, k*k)
                ind = np.argmax(A_flat, axis=2)  # (B, C)

                h_ind, w_ind = np.unravel_index(ind, (self.kernel, self.kernel))  # each (B, C)

                Z_ind[:, :, i, j, 0], Z_ind[:, :, i, j, 1] = h_ind, w_ind
                Z[:, :, i, j] = A_sliced[:, :, h_ind, w_ind]

        self.Z_ind = Z_ind

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        (batch_size, out_channels, output_width, output_height) = dLdZ.shape
        dLdA = np.zeros_like(self.A)

        for w in range(output_width):
            for h in range(output_height):
                for b in range(batch_size):
                    for c in range(out_channels):
                        h_max = int(self.Z_ind[b, c, w, h, 0])
                        w_max = int(self.Z_ind[b, c, w, h, 1])
                        dLdA[b, c, w + w_max, h + h_max] += dLdZ[b, c, w, h]

        return dLdA




class MeanPool2d_stride1():

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        (batch_size, in_channels, input_width, input_height)=A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1
        Z = np.zeros((batch_size, in_channels, output_width, output_height))
        for i in range(input_width-self.kernel+1):
            for j in range(input_height-self.kernel+1):
                A_sliced=A[:, :, i:i+self.kernel, j:j+self.kernel]
                Z[:, :, i, j]=np.mean(A_sliced, axis=(2, 3))

        return Z


    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """

        (batch_size, out_channels, output_width, output_height) = dLdZ.shape
        dLdA = np.zeros_like(self.A)

        for w in range(output_width):
            for h in range(output_height):
                for b in range(batch_size):
                    for c in range(out_channels):
                        dLdA[b, c, w:w+self.kernel, h:h+self.kernel] += dLdZ[b, c, w, h]/(self.kernel*self.kernel)

        return dLdA




class MaxPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride) # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z=self.maxpool2d_stride1.forward(A)
        Z=self.downsample2d.forward(A)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA


class MeanPool2d():

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)  # TODO
        self.downsample2d = Downsample2d(stride)  # TODO

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z=self.meanpool2d_stride1.forward(A)
        Z=self.downsample2d.forward(A)
        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA


