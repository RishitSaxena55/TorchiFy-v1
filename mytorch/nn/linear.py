import numpy as np

# Final merged version of Linear layer
class Linear:

    def __init__(self, in_features, out_features, debug=False):
        self.W = np.zeros((out_features, in_features), dtype="f")
        self.b = np.zeros((out_features, 1), dtype="f")
        self.dLdW = np.zeros((out_features, in_features), dtype="f")
        self.dLdb = np.zeros((out_features, 1), dtype="f")
        self.debug = debug

    def __call__(self, A):
        return self.forward(A)

    def forward(self, A):
        self.A = A                      # Shape: (N, in_features)
        self.N = A.shape[0]
        self.Ones = np.ones((self.N, 1), dtype="f")
        Z = self.A @ self.W.T + self.Ones @ self.b.T  # Shape: (N, out_features)
        return Z

    def backward(self, dLdZ):
        dZdA = self.W                  # Shape: (out_features, in_features)
        dZdW = self.A                  # Shape: (N, in_features)
        dZdb = self.Ones              # Shape: (N, 1)

        dLdA = dLdZ @ dZdA            # Shape: (N, in_features)
        dLdW = dLdZ.T @ dZdW          # Shape: (out_features, in_features)
        dLdb = dLdZ.T @ dZdb          # Shape: (out_features, 1)

        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:
            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdi = None
            self.dZdb = dZdb
            self.dLdA = dLdA
            self.dLdi = None

        return dLdA
