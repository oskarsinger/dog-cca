import numpy as np

from linal.qr import get_q
from linal.utils import multi_dot, quadratic as quad
from linal.svd_funcs import get_svd_power
from optimization.utils import is_converged
from optimization.optimizers.ftprl import MatrixAdaGrad as MAG

class OnlineGenELinKSubroutine:

    def __init__(self,
        k, epsilon=10**(-5)):

        self.k
        self.epsilon = epsilon
        self.W = None

    def get_update(self, A, B): 
        
        print "Stuff"
