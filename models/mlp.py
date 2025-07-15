import numpy as np


class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features).astype(np.float32)
        self.b = np.random.randn(out_features, 1).astype(np.float32)
        self.dLdW = np.zeros_like(self.W)
        self.dLdb = np.zeros_like(self.b)

    def forward(self, A):
        self.A = A.T  # Store input for backward pass (transposed for matmul)
        Z = self.W @ self.A + self.b  # (out, in) @ (in, batch) = (out, batch)
        return Z.T  # Return (batch, out)

    def backward(self, dLdZ):
        dLdZ = dLdZ.T  # (batch, out) -> (out, batch)

        # Gradient w.r.t. weights
        self.dLdW = dLdZ @ self.A.T  # (out, batch) @ (batch, in) = (out, in)

        # Gradient w.r.t. bias
        self.dLdb = np.sum(dLdZ, axis=1, keepdims=True)  # Sum over batch

        # Gradient w.r.t. input
        dLdA = self.W.T @ dLdZ  # (in, out) @ (out, batch) = (in, batch)

        return dLdA.T  # Return (batch, in)


class ReLU:
    def __init__(self):
        self.forward_input = None

    def forward(self, Z):
        self.forward_input = Z
        return np.maximum(0, Z)

    def backward(self, dLdA):
        return dLdA * (self.forward_input > 0).astype(np.float32)


class MLP0:
    def __init__(self, debug=False):
        self.layers = [Linear(2, 3), ReLU()]
        self.debug = debug

    def forward(self, A0):
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):
        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP1:
    def __init__(self, debug=False):
        self.layers = [
            Linear(2, 3),
            ReLU(),
            Linear(3, 2),
            ReLU()
        ]
        self.debug = debug

    def forward(self, A0):
        Z0 = self.layers[0].forward(A0)
        A1 = self.layers[1].forward(Z0)
        Z1 = self.layers[2].forward(A1)
        A2 = self.layers[3].forward(Z1)

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):
        dLdZ1 = self.layers[3].backward(dLdA2)
        dLdA1 = self.layers[2].backward(dLdZ1)
        dLdZ0 = self.layers[1].backward(dLdA1)
        dLdA0 = self.layers[0].backward(dLdZ0)

        if self.debug:
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    def __init__(self, debug=False):
        self.layers = [
            Linear(2, 4), ReLU(),
            Linear(4, 8), ReLU(),
            Linear(8, 8), ReLU(),
            Linear(8, 4), ReLU(),
            Linear(4, 2), ReLU()
        ]
        self.debug = debug

    def forward(self, A):
        if self.debug:
            self.A = [A]

        for layer in self.layers:
            A = layer.forward(A)
            if self.debug:
                self.A.append(A)

        return A

    def backward(self, dLdA):
        if self.debug:
            self.dLdA = [dLdA]

        for layer in reversed(self.layers):
            dLdA = layer.backward(dLdA)
            if self.debug:
                self.dLdA.insert(0, dLdA)

        return dLdA