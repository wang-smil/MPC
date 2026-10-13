"""Estimate-only LQR fallback request. Shared command guards are deliberately external."""

import numpy as np

from ._validation import finite_scalar
from .mpc_controller import finite_vector


class LqrFallbackController:
    def __init__(self, K):
        raw = np.asarray(K, dtype=object)
        if raw.shape != (1, 2):
            raise ValueError('Fallback K must have shape (1,2).')
        self.K = finite_vector(raw[0], 2, 'K')[None, :]

    def request(self, x_hat, x_ref, torque_ff=0.) -> float:
        state = finite_vector(x_hat, 2, 'x_hat')
        reference = finite_vector(x_ref, 2, 'x_ref')
        ff = finite_scalar(torque_ff, 'torque_ff')
        with np.errstate(over='ignore', invalid='ignore'):
            request = ff - (self.K @ (state - reference)).item()
        return finite_scalar(request, 'fallback request')
