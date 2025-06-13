import numpy as np


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self, dLdA):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ*dLdA


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """

    def forward(self, Z):
        self.A = 1/(1+np.exp(-Z))
        return self.A

    def backward(self, dLdA):
        return (self.A-self.A*self.A)*dLdA



class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):
        self.A=(np.exp(Z)-np.exp(-Z))/(np.exp(Z)+np.exp(-Z))

        return self.A

    def backward(self, dLdA):
        return (1-self.A**2)*dLdA


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self, dLdA):
        return (self.A > 0).astype('float32')*dLdA




