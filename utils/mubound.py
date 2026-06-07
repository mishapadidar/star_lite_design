import numpy as np
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec

from .singularperiodicfieldline import _mu_names


class MuBound(Optimizable):
    """CurrentBound analogue for one named (independent / free) mu dof of a
    SingularPeriodicFieldline:  J = max(|mu_k| - threshold, 0)^2."""

    def __init__(self, fl, name, threshold):
        Optimizable.__init__(self, depends_on=[fl])
        self.fl = fl
        self.name = name
        self._idx = _mu_names(fl.local_full_dof_size).index(name)
        self.threshold = threshold

    def J(self):
        v = float(self.fl.mu[self._idx])
        return max(abs(v) - self.threshold, 0.) ** 2

    @derivative_dec
    def dJ(self):
        v = float(self.fl.mu[self._idx])
        ex = max(abs(v) - self.threshold, 0.)
        g = np.zeros(self.fl.local_full_dof_size)
        g[self._idx] = 2.0 * ex * np.sign(v)
        return Derivative({self.fl: g})
