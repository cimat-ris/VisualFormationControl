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
    if (abs(R[2,1] ) > 1.e-18 and abs(R[2,2] ) > 1.e-18):
        roll = np.arctan2(R[2,1],R[2,2])
        yaw = np.arctan2(R[1,0],R[0,0])
    else:
        roll = np.pi
        yaw = np.arctan2(R[0,1],-R[1,1])
    
    pitch = np.arctan2(-R[2,0],np.sqrt(R[2,1]**2+R[2,2]**2))
    return [roll, pitch, yaw]

def Inv_Moore_Penrose(L):
    A = L.T@L
    #if np.linalg.det(A) < 1.0e-18:
        #return None
    return inv(A) @ L.T

def IBVC(control_sel, error, s_current_n,Z,deg,inv_Ls_set,gdl):
    if control_sel ==1:
        Ls = Interaction_Matrix(s_current_n,Z,gdl)
        #print(Ls)
        Ls = Inv_Moore_Penrose(Ls) 
    elif control_sel ==2:
        Ls = inv_Ls_set
    elif control_sel ==3:
        Ls = Interaction_Matrix(s_current_n,Z,gdl)
        #print(Ls)
        Ls = Inv_Moore_Penrose(Ls) 
        Ls = 0.5*( Ls +inv_Ls_set)
    if Ls is None:
            print("Invalid Ls matrix")
            return np.array([0.,0.,0.,0.,0.,0.])
    #print(error.T)
    #print(Ls)
    if gdl == 2:
        _comp = np.zeros((2,Ls.shape[1]))
        Ls = np.r_[Ls[:3],_comp,Ls[3]]
    if gdl == 3:
        _comp = np.zeros((3,Ls.shape[1]))
        Ls = np.r_[Ls[:3],_comp]
    U = (Ls @ error) / deg
    U[0] = -U[0]
    U[1] = -U[1]
    U[2] = -U[2]
    U[3] = -U[3]
    U[4] = -U[4]
    U[5] = -U[5]
    return  U

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
        
        
    def get_control(self, sel,lamb, Z,args ):
        
        #if self.count_points_in_FOV(Z) < 4:
            #return np.zeros(6)
        
            #   IBVC
        if sel == 1:
            return  -lamb* IBVC(args["control_sel"],
                               args["error"],
                               self.s_current_n,
                               Z,
                               args["deg"],
                               self.inv_Ls_set,
                               args["gdl"])
        elif sel == 2:
            return  -lamb* Homography(args["H"],
                                     args["delta_pref"],
                                     args["Adj_list"],
                                     args["gamma"])
        
    
    def set_interactionMat(self,Z,gdl):
        self.Ls_set = Interaction_Matrix(self.s_ref_n,Z,gdl)
        #print(Ls)
        self.inv_Ls_set = Inv_Moore_Penrose(self.Ls_set) 
        
    def update(self,U,dt, points):
        
        #   DEPRECATED
        #p = np.r_[self.camera.p.T , self.camera.roll, self.camera.pitch, self.camera.yaw]
        #p += dt*U
        
        #   BEGIN Testing
        _U = U.copy()
        _U[:3] =  self.camera.R.T @ U[:3]
        
        
        p = np.zeros(6)
        p[:3] = self.camera.p+ dt* _U[:3]
        
        #print(U)
        kw = 1.
        #   Rotation calc Naive
        
        #new_R = cm.rot(kw*dt*U[5],'z') @ cm.rot(kw*dt*U[4],'y') @ cm.rot(kw*dt*U[3],'x') @ self.camera.R
        #new_R = cm.rot(kw*dt*U[3],'x') @ cm.rot(kw*dt*U[4],'y') @ cm.rot(kw*dt*U[5],'z') @ self.camera.R
        
        #   yaw only
        #new_R = cm.rot(kw*dt*U[5],'z') @  self.camera.R
        
        #new_R =   self.camera.R @ cm.rot(kw*dt*U[5],'z') @ cm.rot(kw*dt*U[4],'y') @ cm.rot(kw*dt*U[3],'x') 
        new_R =   self.camera.R @ cm.rot(kw*dt*U[3],'x') @ cm.rot(kw*dt*U[4],'y') @ cm.rot(kw*dt*U[5],'z') 
        
        #   No rotation 
        #new_R =  self.camera.R
        
        #   Rotation : UPdate
        #[p[3] , p[4], p[5] ] = get_angles(new_R)
        
        #   Rotation: previoues
        p[3] = self.camera.roll + dt*U[3]
        p[4] = self.camera.pitch + dt*U[4]
        p[5] = self.camera.yaw + dt*U[5]
        
        print(_U)
        
        #   END Testing
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
