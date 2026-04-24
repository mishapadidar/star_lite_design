import simsoptpp as sopp
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt.objectives import forward_backward
import numpy as np
import jax.numpy as jnp
from jax import jit, grad

def penalty_pure(current, threshold):
    return jnp.max(jnp.array([jnp.abs(current)-threshold, 0.]))**2

class CurrentBound(Optimizable):
    def __init__(self, current, CURRENT_THRESHOLD):
        self.current = current
        self.threshold = CURRENT_THRESHOLD
        self.J_jax = jit(lambda current: penalty_pure(current, self.threshold))
        self.thisgrad0 = jit(lambda current: grad(self.J_jax, argnums=0)(current))
        super().__init__(depends_on=[current])  
    
    def J(self):
        """
        """
        return self.J_jax(self.current.full_x[0])

    @derivative_dec
    def dJ(self):
        grad = self.thisgrad0(self.current.full_x[0])
        return Derivative({self.current: np.array([grad])})

