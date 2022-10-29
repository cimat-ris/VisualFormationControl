import numpy as np
from numpy import sin, cos, pi
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
            tmp = np.where(self.adjacency_mat[i] == 1)
            self.list_adjacency.append(tmp)
        #print(self.list_adjacency)
    
    def laplacian(self):
        return np.diag(self.deg) - self.adjacency_mat
    
    def plot(self,name="Grafo"):
        
        fig, ax = plt.subplots()
        fig.suptitle(name)
        
        x = np.arange(0,2*pi,2*pi/self.n)
        x = cos(x)
        y = np.arange(0,2*pi,2*pi/self.n)
        y = sin(y)
        
        symbols = []
        for i in range(self.n):
            
            for j in self.list_adjacency[i][0]:
                ax.plot([x[i],x[int(j)]],[y[i],y[int(j)]] , 
                        '-k', linewidth=0.1)
            ax.plot(x[i],y[i],'ob')
            ax.text(1.1*x[i],1.1*y[i],str(i))
        
        plt.axis('off')  
        plt.tight_layout()
        plt.savefig(name+'.png',bbox_inches='tight')
        #plt.show()
        
