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
    A = L.T@L
    if np.linalg.det(A) == 0:
        return None
    return inv(A) @ L.T

#   TODO: Adaptar Montijano
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


def rectify(camera, s_norm, Z):
    
    n_points = s_norm.shape[1]
    
    points_r = s_norm * Z
    points_r = np.r_[points_r, Z.reshape((1,n_points))]
    
    points_r = cm.rot(camera.p[3],'x') @ points_r
    points_r = cm.rot(camera.p[4],'y') @ points_r
    points_r = cm.rot(pi,'x').T @ points_r
    points_r = camera.K @ points_r
    points_r = points_r[0:2,:]/points_r[2,:]
    
    return points_r.copy()

class agent:
    
    def __init__(self,
                 camera,
                 p_obj,
                 p_current,
                 points,
                 k = 1,
                 k_int = 0,
                 intGamma0 = None,
                 intGammaInf = None,
                 intGammaSteep = 5.,
                 gamma0 = None,
                 gammaInf = None,
                 gammaSteep = 5.,
                 setleader = False,
                 setRectification = False,
                 set_derivative = True,
                 set_consensoRef = True ):
        
        self.n_points = points.shape[1]
        
        self.k = k
        self.k_int = k_int
        
        self.camera = camera
        self.FOVxlim = self.camera.rho[0]* self.camera.iMsize[0]/(2*self.camera.foco)
        self.FOVylim = self.camera.rho[1]*self.camera.iMsize[1]/(2*self.camera.foco)
        
        self.set_consensoRef = set_consensoRef 
        self.set_derivative = set_derivative
        self.setRectification = setRectification
        
        self.camera.pose(p_obj)
        self.s_ref = self.camera.project(points)
        self.s_ref_n = self.camera.normalize(self.s_ref)
        
        if self.setRectification:
            #points_r = self.camera.Preal @ points
            #points_r[:2,:] =  self.s_ref_n* points_r[2,:]
            #points_r = cm.rot(self.camera.p[3]-pi,'x') @ points_r
            #points_r = cm.rot(self.camera.p[4],'y') @ points_r
            ##print(points_r)
            #points_r = self.camera.K @ points_r
            #points_r = points_r[:2,:]/points_r[2,:]
            #self.s_ref = points_r.copy()
            Z = self.camera.Preal @ points
            Z = Z[2,:]
            self.s_ref = rectify(self.camera, self.s_ref_n, Z)
            self.s_ref_n = self.camera.normalize(self.s_ref)
            #print(self.s_ref)
        
        
        self.camera.pose(p_current)
        self.s_current = self.camera.project(points)
        self.s_current_n = self.camera.normalize(self.s_current)
        #print("---")
        #print(self.s_current)
        if self.setRectification:
            #points_r = self.camera.Preal @ points
            #points_r[:2,:] =  self.s_current_n* points_r[2,:]
            ##print(cm.rot(self.camera.p[3]-pi,'x'))
            ##print(cm.rot(self.camera.p[4],'y'))
            #points_r = cm.rot(self.camera.p[3]-pi,'x') @ points_r
            #points_r = cm.rot(self.camera.p[4],'y') @ points_r
            #points_r = self.camera.K @ points_r
            #points_r = points_r[0:2,:]/points_r[2,:]
            #self.s_current = points_r.copy()
            Z = self.camera.Preal @ points
            Z = Z[2,:]
            self.s_current = rectify(self.camera, self.s_current_n, Z)
            self.s_current_n = self.camera.normalize(self.s_current)
            #print(self.s_current)
        
        self.dot_s_current_n = np.zeros(self.s_current_n.size)
        
        if self.set_consensoRef:
            self.error_p =  self.s_current_n - self.s_ref_n
        else:
            self.error_p =  self.s_current_n
        self.error_p = self.error_p.T.reshape(2*self.n_points)
        self.error = self.error_p.copy()
        self.error_int = np.zeros(self.error.shape)
        
        self.gammaAdapt = False
        self.gammaSteep = gammaSteep
        if (not gamma0 is None) or (not gammaInf is None):
            self.gammaAdapt = True
            
            if gamma0 is None:
                self.gamma0 = 2.
            else:
                self.gamma0 = gamma0
            if gammaInf is None:
                self.gammaInf = 1.
            else:
                self.gammaInf = gammaInf
        
        self.intGammaAdapt = False
        self.intGammaSteep = intGammaSteep
        if (not intGamma0 is None) or (not intGammaInf is None):
            self.intGammaAdapt = True
            self.k_int = 1
            if intGamma0 is None:
                self.intGamma0 = 2.
            else:
                self.intGamma0 = intGamma0
            if intGammaInf is None:
                self.intGammaInf = 1.
            else:
                self.intGammaInf = intGammaInf
        
        self.Ls_set = None
        self.inv_Ls_set = None
        
        #   Plot data
        self.u_inv = np.eye(6)
        self.s_inv = np.zeros(6)
        self.vh_inv = np.eye(self.n_points*2)
        self.vh = np.eye(6)
        self.s = np.zeros(6)
        self.u = np.eye(self.n_points*2)
        
        
    def get_control(self, sel,lamb, Z,args  ):
        
        #TODO: adapt Montijano
        #if self.count_points_in_FOV(Z) < 4:
            #return np.zeros(6)
        
            #   IBVC
        if sel == 1:
            U =  self.IBVC(args["control_sel"],
                               args["error"],
                               #self.s_current_n,
                               Z,
                               args["deg"],
                               args["gdl"],
                               args["dt"])
            return -lamb *  U
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
            return -lamb * U
    
    def set_interactionMat(self,Z,gdl):
        self.Ls_set = Interaction_Matrix(self.s_ref_n,Z,gdl)
        #print(Ls)
        self.inv_Ls_set = Inv_Moore_Penrose(self.Ls_set) 
        
    def update(self,U,dt, points,Z):
        
        #   TODO: reconfigurar con momtijano
        p = self.camera.p.T.copy()
        
        #   BEGIN local
        #kw = 1.
        #p += dt*np.array([1.,-1.,-1.,kw,-kw,-kw])*U
        #print(U)
        
        #   END GLOBAL
        #   BEGIN With global
        #_U = U.copy()
        #_U[:3] =  self.camera.R @ U[:3]
        #_U[3:] =  self.camera.R @ U[3:] #   está de más
        
        #p[:3] += dt* _U[:3]
        
        #_R = cm.rot(dt*U[5],'z') 
        #_R = _R @ cm.rot(dt*U[4],'y')
        #_R = _R @ cm.rot(dt*U[3],'x')
        
        #_R = self.camera.R @ _R
        #[p[3], p[4], p[5]] = get_angles(_R)
        
        #   END GLOBAL
        #   BEGIN GLOBAL skew
        _U = U.copy()
        _U[:3] =  self.camera.R @ U[:3]
        _U[3:] =  self.camera.R @ U[3:]
        
        p[:3] += dt* _U[:3]
        
        S = np.array([[0,-_U[5],_U[4]],
                      [_U[5],0,-_U[3]],
                      [-_U[4],_U[3],0]])
        _R = dt * S @ self.camera.R + self.camera.R
        
        #print(np.linalg.det(_R))
        [p[3], p[4], p[5]] = get_angles(_R)
        
        #   END GLOBAL skew
        tmp = self.s_current_n.copy()
        self.camera.pose(p) 
        
        self.s_current = self.camera.project(points)
        self.s_current_n = self.camera.normalize(self.s_current)
        #print("-------")
        #print(self.s_current)
        if self.setRectification:
            #points_r = self.s_current_n * Z
            #points_r = np.r_[points_r, Z.reshape((1,self.n_points))]
            ##print(cm.rot(self.camera.p[3]-pi,'x'))
            ##print(cm.rot(self.camera.p[4],'y'))
            #points_r = cm.rot(self.camera.p[3]-pi,'x') @ points_r
            #points_r = cm.rot(self.camera.p[4],'y') @ points_r
            #points_r = self.camera.K @ points_r
            #points_r = points_r[0:2,:]/points_r[2,:]
            
            #self.s_current = points_r.copy()
            Z_local = self.camera.Preal @ points
            Z_local = Z_local[2,:]
            self.s_current = rectify(self.camera, self.s_current_n, Z_local)
            self.s_current_n = self.camera.normalize(self.s_current)
            #print(self.s_current)
        
        
        if self.set_derivative:
            self.dot_s_current_n = (self.s_current_n - tmp)/dt
            self.dot_s_current_n = self.dot_s_current_n.T.reshape(2*self.n_points)
        
        if self.set_consensoRef:
            self.error =  self.s_current_n - self.s_ref_n
        else:
            self.error =  self.s_current_n
        
        self.error = self.error.T.reshape(2*self.n_points)
        
    def IBVC(self,control_sel, error,Z,deg,gdl,dt):
        
        
        s_current_n = self.s_current_n.copy()
        Z_current = Z.copy()
        #if self.setRectification:
            #s_current_n = self.s_current.copy()
            #Z_current = Z_current.reshape((1,self.n_points))
            #s_current_n = np.r_[s_current_n, Z_current]
            ##print(cm.rot(self.camera.p[3]-pi,'x'))
            ##print(cm.rot(self.camera.p[4],'y'))
            ##s_current_n = cm.rot(self.camera.p[3]-pi,'x')@s_current_n
            ##s_current_n = cm.rot(self.camera.p[4],'y')@s_current_n
            #Z_current = s_current_n[2,:].copy()
            #s_current_n = s_current_n[0:2,:]/Z_current
            #s_current_n = self.camera.normalize(s_current_n)
            
        if control_sel ==1:
            Ls = Interaction_Matrix(s_current_n,Z_current,gdl)
            self.u, self.s, self.vh  = np.linalg.svd(Ls)
            #Ls = Inv_Moore_Penrose(Ls) 
            Ls = np.linalg.pinv(Ls) 
        elif control_sel ==2:
            Ls = self.inv_Ls_set
        elif control_sel ==3:
            Ls = Interaction_Matrix(s_current_n,Z_current,gdl)
            self.u, self.s, self.vh  = np.linalg.svd(Ls)
            #Ls = Inv_Moore_Penrose(Ls) 
            Ls = np.linalg.pinv(Ls) 
            Ls = 0.5*( Ls + self.inv_Ls_set)
        if Ls is None:
                print("Invalid Ls matrix")
                return np.array([0.,0.,0.,0.,0.,0.]), np.array([0.,0.,0.,0.,0.,0.])
        
        self.u_inv, self.s_inv, self.vh_inv  = np.linalg.svd(Ls)
        
        if gdl == 2:
            Ls = np.insert(Ls, 3, 0., axis = 0)
            Ls = np.insert(Ls, 4, 0., axis = 0)
        if gdl == 3:
            np.insert(Ls, 3, [[0.],[0.],[0.]], axis = 0)
        
        gamma = 1.
        if self.gammaAdapt:
            gamma =  np.linalg.norm(error )
            gamma =  -self.gammaSteep  * gamma
            gamma =  gamma / (self.gamma0 - self.gammaInf)
            gamma =  np.exp(gamma)
            gamma =  gamma * (self.gamma0 - self.gammaInf) 
            gamma += self.gammaInf
        
        _error = self.k * gamma * error
        
        if self.k_int != 0.:
            
            intGamma = 1.
            if self.intGammaAdapt:
                intGamma =  np.linalg.norm(self.error_int )
                intGamma =  -self.intGammaSteep * intGamma
                intGamma =  intGamma / (self.intGamma0 - self.intGammaInf)
                intGamma =  np.exp(intGamma)
                intGamma =  intGamma * (self.intGamma0 - self.intGammaInf) 
                intGamma += self.intGammaInf
            
            _error += self.k_int * intGamma * self.error_int
            
            self.error_int += dt * error
        
        U = (Ls @ _error) / deg
        #print('---')
        #print(U)
        if self.setRectification:
            
            #   Rectificado -> Camara
            _R =  cm.rot(pi,'x')
            #_R =  np.eye(3)
            _R =  cm.rot(self.camera.p[4],'y').T @ _R
            _R =  cm.rot(self.camera.p[3],'x').T @ _R
            
            #   Traslación
            U[:3] = _R @ U[:3]
            #U[:3] = .0
            #print(U)
            
            #   BEGIN fuerza bruta
            #U[3:] = _R @ U[3:]
            #U[3:] = _R @ (U[3:]*np.array([0.,0.,1.]))
            #U[3:] = 0.
            #   END 
            #   BEGIN extract angles and apply euler equation 
            #[phi, theta, psi] = get_angles(_R)
            ##[phi, theta, psi] = [self.camera.p[3]-pi,self.camera.p[4],0 ]
            
            ##_U = np.zeros(3)
            ##_U[0] = U[3] - U[5] * sin(theta)
            ##_U[1] = U[5] * cos(theta) * sin(phi) + U[4] * cos(phi)
            ##_U[2] = U[5] * cos(theta) * cos(phi) - U[4] * sin(phi)
            ##U[3:] = _U.copy()
            
            ##A = np.array([[1,         0,           sin(theta)],
                          ##[0,  cos(phi), -cos(theta) * sin(phi)],
                          ##[0, sin(phi), cos(theta) * cos(phi)]])
            #A = np.array([[0, -sin(psi), cos(psi) * sin(theta)],
                          #[0,  cos(psi), sin(psi) * sin(theta)],
                          #[1,         0,           cos(theta)]])
            
            #U[3:] = A @ (U[3:]*np.array([1.,1.,1.]))
            #U[3:] = _R @ U[3:]
            
            #A = np.eye(3)
            #A = np.flip(A,1)
            
            #U[3:] = inv(A) @ U[3:]
            
            #   END
            #   BEGIN rotación incremental
            #delta = 0.001
            ##[U[3], U[4], U[5]] = [0.,0.,0.]
            ##[U[3], U[4]] = [0.,0.]
            
            #R_U = cm.rot(delta*U[3],'x') 
            #R_U = R_U @ cm.rot(delta*U[4],'y')
            #R_U = R_U @ cm.rot(delta*U[5],'z')
            
            #R_U = _R @ R_U
            
            #[U[3], U[4], U[5]] = get_angles(R_U)
            ##[U[3], U[4], U[5]] = [0.,0.,0.]
            ##[U[3], U[4]] = [0.,0.]
            #U[3:] = U[3:] / delta
            
            #   END
            #   BEGIN Skew matrix trasnformation
            #R_U = np.array([[    0,-U[5], U[4]],
                            #[ U[5],    0,-U[3]],
                            #[-U[4], U[3],    0]])
            
            #R_U = _R @ R_U
            
            #U[3] = R_U[2,1]
            #U[4] = R_U[0,2]
            #U[5] = R_U[1,0]
            #   END
            
            #print(R_U)
            #print(U)
        
        return  U.reshape(6)
    
    def reset_int(self):
        self.error_int[:] =  0.0
    
    #Revisa si los puntos ingresados están en FOV
    #   TODO: pasarlo al modelo de cámara
    def count_points_in_FOV(self,P, enableMargin = True):
        
        PN = self.camera.Preal @ P
        Z = PN[2,:]
        
        #   BEGIN Only front ckeck
        if not enableMargin:
            return np.count_nonzero(Z > 0.)
        #   END Only front check
        
        PN = PN[[0,1],:]/Z
        #a = abs(self.s_current_n[0,:]) < self.FOVxlim 
        #b = abs(self.s_current_n[1,:]) < self.FOVylim
        a = abs(PN[0,:]) < self.FOVxlim 
        b = abs(PN[1,:]) < self.FOVylim
        
        test = []
        for i in range(PN.shape[1]):
            test.append(a[i] and b[i] and Z[i] > 0.0)
        return test.count(True)
