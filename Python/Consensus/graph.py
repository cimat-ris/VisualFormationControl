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
        
        #   Grados
        self.deg = np.zeros(self.n)
        for i in range(self.n):
            self.deg[i] = self.adjacency_mat[i,:].sum()
        
        #   Lista de adjacencia
        self.list_adjacency = []
        for i in range(self.n):
            tmp = np.where(self.adjacency_mat == 1)
            self.list_adjacency.append(tmp)
        
    
    def laplacian(self):
        return np.diag(self.deg) - self.adjacency_mat
