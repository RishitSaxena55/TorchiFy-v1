import numpy as np


class Flatten():


    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, in_width)
        Return:
            Z (np.array): (batch_size, in_channels * in width)
        """
        self.in_channels = A.shape[1]
        self.in_width = A.shape[2]
        Z = np.reshape(A, shape=(A.shape[0], -1))  # TODO

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch size, in channels * in width)
        Return:
            dLdA (np.array): (batch size, in channels, in width)
        """

        dLdA = np.reshape(dLdZ, (dLdZ.shape[0], self.in_channels, self.in_width))  # TODO

        return dLdA
