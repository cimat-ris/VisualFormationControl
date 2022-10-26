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
    
    def __init__(self):
        self.foco=0.002; #Focal de la camara
        self.rho = np.array([1.e-5,1.e-5])
        self.iMsize=[1024, 1024]; #Not working change this
        self.pPrinc=[self.iMsize[0]/2.0, self.iMsize[1]/2.0]; #Not working change this
        
        self.xymin=-0.9
        self.xymax=0.9
        self.zmin=0.8
        self.zmax=1.8
        self.angsmin=-30
        self.angsmax=30
        self.Ldown=180*pi/180
        self.p = np.zeros((3,1))
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.T = np.eye(4)
        self.K = np.array( [[self.foco/self.rho[0],       0.0, self.pPrinc[0]],
                            [      0.0, self.foco/self.rho[1], self.pPrinc[1]],
                            [      0.0,       0.0,            1.0]])
    
    def pose(self,p):
       #print("--- begin pose")
       #print(p)
       self.p = p[0:3]
       self.roll = p[3]
       self.pitch = p[4]
       self.yaw = p[5]
       self.R = rot(self.yaw,'z') 
       self.R = self.R @ rot(self.pitch,'y')
       self.R = self.R @ rot(self.roll,'x')
       #print(self.R)
       tmp = np.c_[ self.R, self.p ]
       #print(tmp)
       self.T = np.r_[ tmp, [[0.0,0.0,0.0,1.0]] ]
       self.P = np.c_[ self.R.T, -self.R.T @ self.p ]
       self.P = self.K @ self.P
       #print(self.P)
       
       #print("--- begin pose")
       
    def project(self,p):
        #print("--- begin project")
        #print(p)
        n = p.shape[1]
        res = np.zeros((4,n))
        if p.shape[0] ==3:
            res = np.r_[p,np.ones((1,n))]
        else:
            res = p.copy()
        #print(res)
        res = self.P @ res
        #print(res)
        #res = res[0:3,:]
        res = res/res[2,:]
        res = res[0:2,:]
        #print(res)
        #print("--- begin project")
        return res
    
    def normalize(self, in_points):
        #print("--- begin normalize")
        f = self.foco/self.rho
        #print(in_points)
        points = in_points.copy()
        points[0,:] -= self.pPrinc[0]#cu
        points[1,:] -= self.pPrinc[1]#cv
        points[0,:] /= f[0]
        points[1,:] /= f[1]
        #print(points)
        #print("--- end normalize")
        
        return points


