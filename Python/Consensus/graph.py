import numpy as np
from numpy import sin, cos
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class graph:
    
    
    
    def __init__(self,_mat,_directed = False):
        self.adjacency_mat = np.array(_mat)
        self.directed = _directed
        self.n = self.adjacency_mat.shape[1]
        self.deg = np.zeros(self.n)
        for i in range(self.n):
            self.deg[i] = self.adjacency_mat[i,:].sum()
    
    def laplacian(self):
        return np.diag(self.deg) - self.adjacency_mat
