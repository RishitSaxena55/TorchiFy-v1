# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
import os

# ========== 1. Base Class ==========
class Criterion(object):
    """
    Interface for loss functions.
    """

    def __init__(self):
        self.logits = None
        self.labels = None
        self.loss = None

    def __call__(self, x, y):
        return self.forward(x, y)

    def forward(self, x, y):
        raise NotImplemented

    def derivative(self):
        raise NotImplemented

# ========== 2. Softmax + CrossEntropy ==========
class SoftmaxCrossEntropy(Criterion):
    """
    Softmax + Cross Entropy Loss
    """

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()

    def forward(self, x, y):
        """
        Args:
            x (np.array): logits, shape (batch_size, num_classes)
            y (np.array): one-hot labels, shape (batch_size, num_classes)

        Returns:
            loss (np.array): shape (batch_size,)
        """
        self.logits = x
        self.labels = y
        self.batch_size = self.labels.shape[0]

        exps = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))  # stability fix
        self.softmax = exps / exps.sum(axis=1, keepdims=True)

        self.loss = np.sum(-self.labels * np.log(self.softmax + 1e-9), axis=1)
        return self.loss

    def backward(self):
        """
        Returns:
            gradient of loss w.r.t input logits, shape (batch_size, num_classes)
        """
        return self.softmax - self.labels

# ========== 3. MSE Loss (for regression) ==========
class MSELoss:
    """
    Mean Squared Error Loss
    """

    def forward(self, A, Y):
        """
        Args:
            A (np.array): predictions, shape (N, C)
            Y (np.array): ground truths, shape (N, C)

        Returns:
            mse (float): mean squared error
        """
        self.A = A
        self.Y = Y
        N, C = A.shape
        se = (A - Y) ** 2
        sse = np.sum(se)
        mse = sse / (N * C)
        return mse

    def backward(self):
        """
        Returns:
            dLdA (np.array): gradient of loss w.r.t. predictions A, shape (N, C)
        """
        N, C = self.A.shape
        dLdA = 2 * (self.A - self.Y) / (N * C)
        return dLdA
