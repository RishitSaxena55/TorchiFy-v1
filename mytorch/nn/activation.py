# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)
import numpy as np

class Identity:
    def forward(self, Z):
        self.A = Z
        return self.A

    def backward(self):
        dAdZ = np.ones(self.A.shape, dtype="f")
        return dAdZ


class Sigmoid:
    """
    Sigmoid activation function
    """
    def forward(self, Z):
        self.A = Z
        self.npVal = np.exp(-self.A)
        return 1 / (1 + self.npVal)

    def backward(self, dLdA):
        dAdZ = self.npVal / (1 + self.npVal) ** 2
        return dAdZ * dLdA


class Tanh:
    """
    Modified Tanh to work with BPTT.
    """
    def forward(self, Z):
        self.A = Z
        self.tanhVal = np.tanh(self.A)
        return self.tanhVal

    def backward(self, dLdA, state=None):
        if state is not None:
            dAdZ = 1 - state * state
        else:
            dAdZ = 1 - self.tanhVal * self.tanhVal
        return dAdZ * dLdA


class ReLU:
    def forward(self, Z):
        self.A = Z
        return np.maximum(0, Z)

    def backward(self, dLdA):
        dAdZ = (self.A > 0).astype(float)
        return dAdZ * dLdA
