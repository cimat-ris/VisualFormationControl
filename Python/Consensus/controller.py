import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv

def Interaction_Matrix(in_points,Z):
    
    n = points.shape[1]
    L = np.zeros((n,12))
    L[:,0]  =   L[:,7] = -1/Z
    L[:,2]  =   points[0,:]/Z
    L[:,3]  =   points[0,:]*points[1,:]
    L[:,4]  =   -(1+points[0,:]**2)
    L[:,5]  =   points[1,:]
    L[:,8]  =   points[1,:]/Z
    L[:,9]  =   1+points[1,:]**2
    L[:,10] =   -points[0,:]*points[1,:]
    L[:,11] =   -points[0,:]
    
    return L.reshape((2*n,6))  

def Inv_Moore_Penrose(L):
    A = L.T@L
    if np.linalg.det(A) < 1.0e-9:
        return None
    return inv(A) @ L.T
