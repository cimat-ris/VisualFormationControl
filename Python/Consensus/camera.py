import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def rot(ang,ax):
    
    ca = cos(ang)
    sa = sin(ang)
    
    if ax == 'x':
        return np.array([[1.0, 0.0, 0.0],
                        [0.0,  ca, -sa],
                        [0.0,  sa,  ca]])
    elif ax == 'y':
        return np.array([[ ca, 0.0,  sa],
                        [0.0, 1.0, 0.0],
                        [-sa, 0.0,  ca]])
    elif ax == 'z':
        return np.array([[ ca, -sa, 0.0],
                        [ sa,  ca, 0.0],
                        [0.0, 0.0, 1.0]])
    
    return None
    
class camera:
    
    
    foco=0.002; #Focal de la camara
    rho = np.array([1.e-5,1.e-5])
    iMsize=[1024, 1024]; #Not working change this
    pPrinc=[iMsize[0]/2.0, iMsize[1]/2.0]; #Not working change this

    xymin=-0.9
    xymax=0.9
    zmin=0.8
    zmax=1.8
    angsmin=-30
    angsmax=30
    Ldown=180*pi/180
    p = np.zeros((3,1))
    roll = 0.0
    pitch = 0.0
    yaw = 0.0
    T = np.eye(4)
    
    def pose(self,p):
       
       print(p)
       self.p = p[0:3]
       self.roll = p[3]
       self.pitch = p[4]
       self.yaw = p[5]
       self.R = rot(self.yaw,'z') 
       self.R = self.R @ rot(self.pitch,'y')
       self.R = self.R @ rot(self.roll,'x')
       tmp = np.c_[ self.R, self.p ]
       print(tmp)
       self.T = np.r_[ tmp, [[0.0,0.0,0.0,1.0]] ]
       
    def project(self,p):
        K = np.array([[foco,  0.0, pPrinc[0]],
                      [ 0.0, foco, pPrinc[1]],
                      [ 0.0,  0.0,       1.0]])
        self.P = np.c_[ self.R.T, -self.R.T @ self.p ]
        P = K@P
    
    def normalize(self, in_points):
        
        f = self.foco/self.rho
        
        points = in_points.copy()
        points[0,:] -= self.pPrinc[0]#cu
        points[1,:] -= self.pPrinc[1]#cv
        points[0,:] /= f[0]
        points[1,:] /= f[1]
        
        return points

       
