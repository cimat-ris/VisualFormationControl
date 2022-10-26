import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv
import camera as cm
 
def Interaction_Matrix(points,Z):
    
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

class agent:
    
    def __init__(self,camera,p_obj,p_current,points ):
        
        self.n_points = points.shape[1]
        #self.deg = deg
        
        self.camera = camera
        
        self.camera.pose(p_obj)
        self.s_ref = self.camera.project(points)
        self.s_ref_n = self.camera.normalize(self.s_ref)
        
        self.camera.pose(p_current)
        self.s_current = self.camera.project(points)
        self.s_current_n = self.camera.normalize(self.s_current)
        
        self.error = self.s_current_n - self.s_ref_n
        self.error = self.error.reshape((1,2*self.n_points))
        
        
    def get_control(self, error, deg ):
        
        Ls = Interaction_Matrix(self.s_current_n,1.0)
        print(Ls)
        Ls = Inv_Moore_Penrose(Ls) 
        if Ls is None:
            return None
        print(error.T)
        print(Ls)
        U = (Ls @ error.T) / deg
        
        return U
    
    def update(self,U,dt, points):
        
        p = np.r_[self.camera.p.T , self.camera.roll, self.camera.pitch, self.camera.yaw]
        p += dt*U
        self.camera.pose(p) 
        
        self.s_ref = self.camera.project(points)
        self.s_ref_n = self.camera.normalize(self.s_ref)
        
        self.error = self.s_current_n - self.s_ref_n
        self.error = self.error.reshape((1,2*self.n_points))
