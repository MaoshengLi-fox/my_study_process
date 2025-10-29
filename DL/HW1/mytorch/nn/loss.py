import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N = A.shape[0]  # TODO
        self.C = A.shape[1]  # TODO
        se = (A - Y) ** 2 # TODO
        sse = np.sum(se)  # TODO
        mse = sse / (2 * self.N * self.C)  # TODO

        return np.array(mse, dtype='f')

    def backward(self):

        dLdA = (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N = A.shape[0]  # TODO
        C = A.shape[1]  # TODO

        Ones_C = np.ones((C, 1), dtype="f")  # TODO
        Ones_N = np.ones((N, 1), dtype="f")  # TODO

        A_shift = A - np.max(A, axis=1, keepdims=True)
        expA = np.exp(A_shift)
        sum_expA = np.sum(expA, axis=1, keepdims=True)
        self.softmax = expA / sum_expA  # TODO
        crossentropy = (-Y * np.log(self.softmax + 1e-12)) @ Ones_C  # TODO
        sum_crossentropy = (Ones_N.T @ crossentropy).item()  # TODO
        L = sum_crossentropy / N

        return np.array(L, dtype='f')

    def backward(self):

        dLdA = self.softmax - self.Y  # TODO

        return dLdA
