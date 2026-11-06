"""Exact reused box KDE and Numba seeding helper."""
import numpy as np
from scipy.spatial import cKDTree
from numba import njit

@njit
def seed_numba(s): np.random.seed(s)

class BoxKDE:
    def __init__(self, original, mode):
        N,d=original.shape
        self.h=N**(-1/(d+4))
        S=np.eye(d) if mode=='box_isotropic' else np.cov(original,rowvar=False,ddof=1)
        self.H=self.h**2*S
        eig,V=np.linalg.eigh(self.H)
        if eig.min()<=0: raise ValueError('Non-positive bandwidth matrix')
        self.invroot=(V*(1/np.sqrt(eig)))@V.T
        self.volume=2**d*np.sqrt(np.linalg.det(self.H))
        self.original=self.density(original)
    def density(self,coords):
        white=coords@self.invroot
        counts=cKDTree(white).query_ball_point(white,1.,p=np.inf,return_length=True)
        return counts/(len(coords)*self.volume)
