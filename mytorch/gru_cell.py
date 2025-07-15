import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        # Add your code here.
        # Define your variables based on the writeup using the corresponding
        # names below.
        self.r = self.r_act(self.Wrx @ x + self.brh + self.Wrh @ h_prev_t + self.brh)
        self.z = self.z_act(self.Wzx @ x + self.bzx + self.Wzh @ h_prev_t + self.bzh)
        self.n = self.h_act(self.Wnx @ x + self.bnx + self.r * (self.Wnh @ h_prev_t + self.bnh))
        h_t = (1 - self.z) * self.n + self.z * h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)
        assert h_t.shape == (self.h,)  # h_t is the final output of you GRU cell.

        # return h_t
        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        # 1) Reshape self.x and self.hidden to (input_dim, 1) and (hidden_dim, 1) respectively
        #    when computing self.dWs...
        # 2) Transpose all calculated dWs...
        # 3) Compute all of the derivatives
        # 4) Know that the autograder grades the gradients in a certain order, and the
        #    local autograder will tell you which gradient you are currently failing.

        # ADDITIONAL TIP:
        # Make sure the shapes of the calculated dWs and dbs  match the
        # initalized shapes accordingly
        self.x = self.x.reshape((self.d, 1))
        self.hidden = self.hidden.reshape((self.h, 1))
        dh_t = delta

        dz_t = dh_t * self.n
        dn_t = dh_t * (1 - self.z)

        dn_act = (dn_t * (1 - self.n**2))
        dWnh = dn_act @ self.x.T
        dbnx = dn_act
        dr_t = dn_act * (self.Wnh @ self.hidden + self.bnh)
        dWnh = self.hidden @ dn_act.T
        dbnh = dn_act * self.r

        dz_act = (dz_t * self.z * (1 - self.z))
        dWzx = dz_act @ self.x.T
        dbzx = dz_act
        dWzh = self.hidden @ dz_act.T
        dbzh = dz_act

        dr_act = (dr_t * self.r * (1 - self.r))
        dWrx = dr_act @ self.x.T
        dbrx = dr_act
        dWrh = self.hidden @ dr_act.T
        dbrh = dr_act

        dxn = dn_act.T @ self.Wnx
        dxz = dz_act.T @ self.Wzx
        dxr = dr_act.T @ self.Wrx

        dx = dxn + dxz + dxr

        t1 = (dh_t * self.z).T
        t2 = (dn_act * self.r).T @ self.Wnh
        t3 = dz_act.T @ self.Wzh
        t4 = dr_act.T @ self.Wrh

        dh_prev_t = t1 + t2 + t3 + t4


        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t

