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

def IBVC(control_sel, error, s_current_n,Z,deg,inv_Ls_set):
    if control_sel ==1:
        Ls = Interaction_Matrix(s_current_n,Z)
        #print(Ls)
        Ls = Inv_Moore_Penrose(Ls) 
    elif control_sel ==2:
        Ls = inv_Ls_set
    elif control_sel ==3:
        Ls = Interaction_Matrix(s_current_n,Z)
        #print(Ls)
        Ls = Inv_Moore_Penrose(Ls) 
        Ls = 0.5*( Ls +inv_Ls_set)
    if Ls is None:
            print("Invalid Ls matrix")
            return None
    #print(error.T)
    #print(Ls)
    U = (Ls @ error) / deg
    U[0] = -U[0]
    return  -U

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
        
        self.error =  self.s_current_n - self.s_ref_n
        self.error = self.error.T.reshape((1,2*self.n_points))
        
        self.Ls_set = None
        self.inv_Ls_set = None
        
        
    def get_control(self, sel,error,lamb,Z, args ):
        
        if self.count_points_in_FOV(Z) < 4:
            return np.zeros(6)
        
        if sel == 1:
            return  lamb* IBVC(args["control_sel"],
                               error,
                               self.s_current_n,
                               Z,
                               args["deg"],
                               self.inv_Ls_set)
        
    
    def set_interactionMat(self,Z):
        self.Ls_set = Interaction_Matrix(self.s_ref_n,Z)
        #print(Ls)
        self.inv_Ls_set = Inv_Moore_Penrose(self.Ls_set) 
        
    def update(self,U,dt, points):
        
        p = np.r_[self.camera.p.T , self.camera.roll, self.camera.pitch, self.camera.yaw]
        
        p += dt*U
        self.camera.pose(p) 
        
        self.s_current = self.camera.project(points)
        self.s_current_n = self.camera.normalize(self.s_current)
        
        self.error =  self.s_current_n  - self.s_ref_n
        #print(self.error)
        self.error = self.error.T.reshape((1,2*self.n_points))
        #print(self.error)
    def count_points_in_FOV(self,Z):
        xlim = self.camera.rho[0]* self.camera.iMsize[0]/(2*self.camera.foco)
        ylim = self.camera.rho[1]*self.camera.iMsize[1]/(2*self.camera.foco)
        
        a = abs(self.s_current_n[0,:]) < xlim 
        b = abs(self.s_current_n[1,:]) < ylim
        
        test = []
        for i in range(self.s_current.shape[1]):
            test.append(a[i] and b[i] and Z[0,i] > 0.0)
        return test.count(True)
