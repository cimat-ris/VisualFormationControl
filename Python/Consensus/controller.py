import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv
import camera as cm
 
import cv2 


def Interaction_Matrix(points,Z,gdl):
    
    n = points.shape[1]
    if gdl == 1:
        m = 6
    if gdl == 2:
        m = 4
    if gdl == 3:
        m = 3
    
    L = np.zeros((n,2*m))
    if gdl == 1:
        L[:,0]  =   L[:,7] = -1/Z
        L[:,2]  =   points[0,:]/Z
        L[:,3]  =   points[0,:]*points[1,:]
        L[:,4]  =   -(1+points[0,:]**2)
        L[:,5]  =   points[1,:]
        L[:,8]  =   points[1,:]/Z
        L[:,9]  =   1+points[1,:]**2
        L[:,10] =   -points[0,:]*points[1,:]
        L[:,11] =   -points[0,:]
    if gdl == 2:
        L[:,0]  =   L[:,5] = -1/Z
        L[:,2]  =   points[0,:]/Z
        L[:,3]  =   points[1,:]
        L[:,6]  =   points[1,:]/Z
        L[:,7] =   -points[0,:]
    if gdl == 3:
        L[:,0]  =   L[:,4] = -1/Z
        L[:,2]  =   points[0,:]/Z
        L[:,5]  =   points[1,:]/Z
    
    
    return L.reshape((2*n,m)) 

def get_angles(R):
    if (R[2,0] < 1.0):
        if R[2,0] > -1.0:
            pitch = np.arcsin(-R[2,0])
            yaw = np.arctan2(R[1,0],R[0,0])
            roll = np.arctan2(R[2,1],R[2,2])
        else:
            pitch = np.pi/2.
            yaw = -np.arctan2(-R[1,2],R[1,1])
            roll = 0.
    else:
        pitch = -np.pi/2.
        yaw = -np.arctan2(R[1,2],R[1,1])
        roll = 0.
        
    return [roll, pitch, yaw]

def Inv_Moore_Penrose(L):
    #if L.shape[1] ==6:
        #return inv(L)
    #return inv(L)
    A = L.T@L
    #if np.linalg.det(A) < 1.0e-18:
        #return None
    if np.linalg.det(A) == 0:
        return None
    return inv(A) @ L.T

def IBVC(control_sel, error, s_current_n,Z,deg,inv_Ls_set,gdl):
    #return np.array([0.,0.,0.,0.,0.,np.pi/10,]), np.array([0.,0.,0.,0.,0.,0.])
    #if any (Z < 0.):
        #print("Negative depths")
        #print("")
        #return np.array([0.,0.,0.,0.,0.,0.]), np.array([0.,0.,0.,0.,0.,0.])
    
    if control_sel ==1:
        Ls = Interaction_Matrix(s_current_n,Z,gdl)
        #print(Ls)
        #Ls = Inv_Moore_Penrose(Ls) 
        Ls = np.linalg.pinv(Ls) 
    elif control_sel ==2:
        Ls = inv_Ls_set
    elif control_sel ==3:
        Ls = Interaction_Matrix(s_current_n,Z,gdl)
        #print(Ls)
        #Ls = Inv_Moore_Penrose(Ls) 
        Ls = np.linalg.pinv(Ls) 
        Ls = 0.5*( Ls +inv_Ls_set)
    if Ls is None:
            print("Invalid Ls matrix")
            return np.array([0.,0.,0.,0.,0.,0.]), np.array([0.,0.,0.,0.,0.,0.])
    #   BEGIN L range test
    u, s, vh  = np.linalg.svd(Ls)
    #if (s[0] < 1000):
        #print(Interaction_Matrix(s_current_n,Z,gdl))
    #if(s[0] > 400):
        ##print("PVAL > 1000")
        #print(Z)
        #print(Interaction_Matrix(s_current_n,Z,gdl))
        #print(s)
        #print(s_current_n)
        #return np.array([0.,0.,0.,0.,0.,0.]), np.array([0.,0.,0.,0.,0.,0.])
    #print(s)
    #   END L range test
    if gdl == 2:
        _comp = np.zeros((2,Ls.shape[1]))
        Ls = np.r_[Ls[:3],_comp,Ls[3].reshape((1,Ls.shape[1]))]
    if gdl == 3:
        _comp = np.zeros((3,Ls.shape[1]))
        Ls = np.r_[Ls[:3],_comp]
    #print(Ls)
    #print(error)
    U = (Ls @ error) / deg
    
    return  U.reshape(6), u, s, vh

def get_Homographies(agents):
    n = len(agents)
    H = np.zeros((n,n,3,3))
    
    for i in range(n):
        for j in range(n):
            [H_tmp, mask] = cv2.findHomography(
                                    agents[j].s_current_n.T,
                                    agents[i].s_current_n.T)
            #   Docondition
            H_tmp = H_tmp/np.linalg.norm(H_tmp[:,1])
            H[i,j,:,:] = H_tmp
    return H



def Homography(H, delta_pref,adj_list,gamma):
    
    n = H.shape[0]
    U = np.zeros(6)
    
    hxy = np.array([0.0,0.0])
    hz  = 0.0
    han = 0.0
    
    for i in adj_list:
        hxy += gamma* H[i,0:2,2] - delta_pref[i,0:2,0]
        hz  += 1.0 - H[i,2,2]
        han += np.arctan2(H[i,1,0],H[i,0,0]) - delta_pref[i,5,0]
    U[0:2] = hxy
    U[0]  *= -1.0
    U[2]   = -hz
    U[5]   = han
    return U

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
        
        
    def get_control(self, sel,lamb, Z,args  ):
        
        #if self.count_points_in_FOV(Z) < 4:
            #return np.zeros(6)
        
            #   IBVC
        if sel == 1:
            #return  -lamb* IBVC(args["control_sel"],
            U, u, s, vh =  IBVC(args["control_sel"],
                               args["error"],
                               self.s_current_n,
                               Z,
                               args["deg"],
                               self.inv_Ls_set,
                               args["gdl"])
            #print(s)
            return -lamb * U,u, s, vh
        elif sel == 2:
            return  -lamb* Homography(args["H"],
                                     args["delta_pref"],
                                     args["Adj_list"],
                                     args["gamma"])
        
    
    def set_interactionMat(self,Z,gdl):
        self.Ls_set = Interaction_Matrix(self.s_ref_n,Z,gdl)
        #print(Ls)
        self.inv_Ls_set = Inv_Moore_Penrose(self.Ls_set) 
        
    def update(self,U,dt, points,Z):
        
        #   TODO: reconfigurar con momtijano
        #p = np.r_[self.camera.p.T , self.camera.roll, self.camera.pitch, self.camera.yaw]
        #p += -dt*np.array([-1.,1.,1.,1.,1.,1.])*U
        #print(U)
        _U = U.copy()
        p = np.zeros(6)
        
        ##   Traslation
        ##   ## TODO transpuesta?
        _U[:3] =  self.camera.R @ U[:3]
        #_U[:3] =  -np.diag([-1.,1.,1.]) @ U[:3]
        p[:3] = self.camera.p + dt* _U[:3]
        
        ##if (np.linalg.norm(dt* _U[:3])> 0.5):
            ##print(self.s_current_n)
            ##L=Interaction_Matrix(self.s_current_n,Z,1)
            ##print(L)
            ##L=Inv_Moore_Penrose(L)
            ##print(L)
            ##print(np.linalg.svd(L))
            ##print(self.error)
            ##print()
        ##   Rotation
        kw = 1
        ##kw = 1.
        new_R = self.camera.R @ cm.rot(kw*dt*U[5],'z') @ cm.rot(kw*dt*U[4],'y') @ cm.rot(kw*dt*U[3],'x') #@ self.camera.R
        [p[3] , p[4], p[5] ] = get_angles(new_R)
        #p[3] = self.camera.roll #+ kw * dt * U[3]
        #p[4] = self.camera.pitch - kw * dt * U[4]
        #p[5] = self.camera.yaw# - kw * dt * U[5]
        #print(U)
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
