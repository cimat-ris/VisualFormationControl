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
    
    if control_sel ==1:
        Ls = Interaction_Matrix(s_current_n,Z,gdl)
        #Ls = Inv_Moore_Penrose(Ls) 
        Ls = np.linalg.pinv(Ls) 
        #Ls = np.linalg.inv(Ls) 
    elif control_sel ==2:
        Ls = inv_Ls_set
    elif control_sel ==3:
        Ls = Interaction_Matrix(s_current_n,Z,gdl)
        #Ls = Inv_Moore_Penrose(Ls) 
        Ls = np.linalg.pinv(Ls) 
        Ls = 0.5*( Ls +inv_Ls_set)
    if Ls is None:
            print("Invalid Ls matrix")
            return np.array([0.,0.,0.,0.,0.,0.]), np.array([0.,0.,0.,0.,0.,0.])
    u, s, vh  = np.linalg.svd(Ls)
    if gdl == 2:
        Ls = np.insert(Ls, 3, 0., axis = 0)
        Ls = np.insert(Ls, 4, 0., axis = 0)
        #_comp = np.zeros((2,Ls.shape[1]))
        #Ls = np.r_[Ls[:3],_comp,Ls[3].reshape((1,Ls.shape[1]))]
    if gdl == 3:
        np.insert(Ls, 3, [[0.],[0.],[0.]], axis = 0)
    U = (Ls @ error) / deg
    
    return  U.reshape(6), u, s, vh

#   Calcula el control basado en posición
#   Supone:
#       camara 1 = actual, 2 = referencia
#       Los puntos están normalizados
#       los vectores de puntos tienen dimención (n,2)
def PBVC(p1,p2,K, realR = np.eye(3), realT = np.ones(3) ):
    
    #   Calcular matriz de homografía a partir de p1 y p2
    #   src, dst
    #   dst = H src
    #   H tiene R,t en el sistema dst
    [H, mask] = cv2.findHomography(p2 ,p1)
    [ret, Rots, Tras, Nn] = cv2.decomposeHomographyMat(H,K)
    
    #   Revisa que las solución satisfaga
    a = []
    b = []
    
    if(ret ==1):
        return np.zeros(6)
    
    for i in range(ret):
        m = np.ones((p1.shape[0],1))
        m = np.c_[p1,m]
        p = m @ Rots[i] @ Nn[i] 
        #print(p)
        if all(p>0.):
            _R =  Rots[i].T @ realR.T
            errang = np.arccos((_R.trace()-1.)/2.)
            a.append(errang)
            b.append([Rots[i],Tras[i]])
    if len(a) == 0:
        print("a empty")
    if len(b) == 0:
        print("b empty")
    idx = a.index(min(a))
    R = b[idx][0].T
    T = b[idx][1]
    
    #   Ground Truth
    #R = realR
    #T = realT.reshape((3,1))

    #   U = et, ew
    theta = np.arccos(0.5*(R.trace()-1.))
    rot = 0.5*np.array([[R[2,1]-R[1,2]],
                        [-R[2,0]+R[0,2]],
                        [R[1,0]-R[0,1]]])/np.sinc(theta)
    
    U = np.r_[-T,rot]
    
    return  U.reshape(6)

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
    
    def __init__(self,
                 camera,
                 p_obj,
                 p_current,
                 points,
                 k = 1,
                 k_int = 0,
                 set_derivative = True,
                 set_consensoRef = True ):
        
        self.n_points = points.shape[1]
        #self.deg = deg
        
        self.k = k
        self.k_int = k_int
        
        self.camera = camera
        self.FOVxlim = self.camera.rho[0]* self.camera.iMsize[0]/(2*self.camera.foco)
        self.FOVylim = self.camera.rho[1]*self.camera.iMsize[1]/(2*self.camera.foco)
        
        self.set_consensoRef = set_consensoRef 
        self.set_derivative = set_derivative
        
        self.camera.pose(p_obj)
        self.s_ref = self.camera.project(points)
        self.s_ref_n = self.camera.normalize(self.s_ref)
        
        self.camera.pose(p_current)
        self.s_current = self.camera.project(points)
        self.s_current_n = self.camera.normalize(self.s_current)
        self.dot_s_current_n = np.zeros(self.s_current_n.size)
        
        if self.set_consensoRef:
            self.error_p =  self.s_current_n - self.s_ref_n
        else:
            self.error_p =  self.s_current_n
        self.error_p = self.error_p.T.reshape(2*self.n_points)
        self.error = self.error_p.copy()
        self.error_int = np.zeros(self.error.shape)
        
        self.Ls_set = None
        self.inv_Ls_set = None
        
        
    def get_control(self, sel,lamb, Z,args  ):
        
        #TODO: adapt Montijano
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
        if sel == 3:
            U = PBVC(args["p1"],
                    args["p2"],
                    args["K"],
                    args["realR"],
                    args["realT"])
            #print(s)
            return -lamb * U,None, None, None
    
    def set_interactionMat(self,Z,gdl):
        self.Ls_set = Interaction_Matrix(self.s_ref_n,Z,gdl)
        #print(Ls)
        self.inv_Ls_set = Inv_Moore_Penrose(self.Ls_set) 
        
    def update(self,U,dt, points,Z):
        
        #   TODO: reconfigurar con momtijano
        #p = np.r_[self.camera.p.T ,
                  #self.camera.roll,
                  #self.camera.pitch,
                  #self.camera.yaw]
        p = self.camera.p.T.copy()
        
        #   BEGIN local
        #kw = 1.
        #p += dt*np.array([1.,-1.,-1.,kw,-kw,-kw])*U
        #print(U)
        
        #   END GLOBAL
        #   BEGIN With global
        #_U = U.copy()
        #_U[:3] =  self.camera.R @ U[:3]
        #_U[3:] =  self.camera.R @ U[3:]
        #print("Rv = ",_U[:3])
        #print("t x R = " , np.cross(p[:3],_U[3:]))
        #print("vg = ",_U[:3]+ np.cross(p[:3],_U[3:]))
        #print("wg = ",_U[3:])
        
        #p[:3] += dt* (_U[:3]- np.cross(p[:3],_U[3:]))
        #p[3:] += dt* _U[3:]
        ##p[:3] += dt* _U[:3]
        
        #   END GLOBAL
        #   BEGIN With global + dR R
        _U = U.copy()
        _U[:3] =  self.camera.R @ U[:3]
        _U[3:] =  self.camera.R @ U[3:]
        #print("Rv = ",_U[:3])
        #print("t x R = " , np.cross(p[:3],_U[3:]))
        #print("vg = ",_U[:3]+ np.cross(p[:3],_U[3:]))
        #print("wg = ",_U[3:])
        
        #p[:3] += dt* (_U[:3]- np.cross(p[:3],_U[3:]))
        #p[:3] += dt* (_U[:3]+ np.cross(p[:3],_U[3:]))
        p[:3] += dt* _U[:3]
        
        _R = cm.rot(dt*U[5],'z') 
        _R = _R @ cm.rot(dt*U[4],'y')
        _R = _R @ cm.rot(dt*U[3],'x')
        
        _R = self.camera.R @ _R
        [p[3], p[4], p[5]] = get_angles(_R)
        
        #   END GLOBAL
        tmp = self.s_current_n.copy()
        self.camera.pose(p) 
        
        self.s_current = self.camera.project(points)
        self.s_current_n = self.camera.normalize(self.s_current)
        
        if self.set_derivative:
            self.dot_s_current_n = (self.s_current_n - tmp)/dt
            self.dot_s_current_n = self.dot_s_current_n.T.reshape(2*self.n_points)
        self.error_int += self.error_p*dt
        
        if self.set_consensoRef:
            self.error_p =  self.s_current_n - self.s_ref_n
        else:
            self.error_p =  self.s_current_n
        
        #print(self.error)
        self.error_p = self.error_p.T.reshape(2*self.n_points)
        #print(self.error)
        self.error = self.k * self.error_p -  self.dot_s_current_n
        #if any (self.error_p < .5):
        #if True:
        self.error += self.k_int * self.error_int  
        
    def count_points_in_FOV(self,P):
        
        Z = self.camera.Preal @ P
        Z = Z[2,:]
        
        a = abs(self.s_current_n[0,:]) < self.FOVxlim 
        b = abs(self.s_current_n[1,:]) < self.FOVylim
        
        test = []
        for i in range(self.s_current.shape[1]):
            test.append(a[i] and b[i] and Z[i] > 0.0)
        return test.count(True)
