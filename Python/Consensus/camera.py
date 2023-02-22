import numpy as np
from numpy import sin, cos, pi
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt

import Arrow3D


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
    
    def draw_camera(self, ax,
                    color='cyan', 
                    scale=1.0,
                    linestyle='solid',
                    alpha=0.):
        #CAmera points: to be expressed in the camera frame;
        CAMup=np.array([[-1,-1,  1, 1, 1.5,-1.5,-1, 1 ],
                        [ 1, 1,  1, 1, 1.5, 1.5, 1, 1 ],
                        [ 2,-2, -2, 2,   3,   3, 2, 2 ],
                        [ 1, 1,  1, 1, 1  , 1 , 1, 1  ]])
        CAMup[0:3,:] = scale * CAMup[0:3,:] 
        #Ri2w    = np.dot(Rotations.rotox(pi), self.R)
        #trasl   = self.t.reshape(3, -1)
        CAMupTRASF = self.T @ CAMup # Ri2w.dot(CAMup) + trasl;
        CAMdwn=np.array([[-1,-1,  1, 1, 1.5,-1.5,-1, 1  ],
                        [ -1,-1, -1,-1,-1.5,-1.5,-1,-1 ],
                        [  2,-2, -2, 2,   3,   3, 2, 2 ],
                        [ 1, 1,  1, 1, 1  , 1 , 1, 1  ]])
        CAMdwn[0:3,:] = scale * CAMdwn[0:3,:] 
        CAMdwnTRASF     = self.T @ CAMdwn #Ri2w.dot( CAMdwn ) + trasl
        CAMupTRASFm     = CAMupTRASF.copy()
        CAMdwnTRASFm    = CAMdwnTRASF.copy()
        ax.plot(CAMupTRASFm[0,:],
                CAMupTRASFm[1,:],
                CAMupTRASFm[2,:],
                c=color,ls=linestyle)
        ax.plot(CAMdwnTRASFm[0,:],
                CAMdwnTRASFm[1,:],
                CAMdwnTRASFm[2,:],
                c=color,ls=linestyle)
        ax.plot([CAMupTRASFm[0,0],CAMdwnTRASFm[0,0]],
                [CAMupTRASFm[1,0],CAMdwnTRASFm[1,0]],
                [CAMupTRASFm[2,0],CAMdwnTRASFm[2,0]],
                c=color,ls=linestyle)
        ax.plot([CAMupTRASFm[0,1],CAMdwnTRASFm[0,1]],
                [CAMupTRASFm[1,1],CAMdwnTRASFm[1,1]],
                [CAMupTRASFm[2,1],CAMdwnTRASFm[2,1]],
                c=color,ls=linestyle)
        ax.plot([CAMupTRASFm[0,2],CAMdwnTRASFm[0,2]],
                [CAMupTRASFm[1,2],CAMdwnTRASFm[1,2]],
                [CAMupTRASFm[2,2],CAMdwnTRASFm[2,2]],
                c=color,ls=linestyle)
        ax.plot([CAMupTRASFm[0,3],CAMdwnTRASFm[0,3]],
                [CAMupTRASFm[1,3],CAMdwnTRASFm[1,3]],
                [CAMupTRASFm[2,3],CAMdwnTRASFm[2,3]],
                c=color,ls=linestyle)
        ax.plot([CAMupTRASFm[0,4],CAMdwnTRASFm[0,4]],
                [CAMupTRASFm[1,4],CAMdwnTRASFm[1,4]],
                [CAMupTRASFm[2,4],CAMdwnTRASFm[2,4]],
                c=color,ls=linestyle)
        ax.plot([CAMupTRASFm[0,5],CAMdwnTRASFm[0,5]],
                [CAMupTRASFm[1,5],CAMdwnTRASFm[1,5]],
                [CAMupTRASFm[2,5],CAMdwnTRASFm[2,5]],
                c=color,ls=linestyle)
        
        scale *= 10.0
        
        Oc = np.array([[0.,0,0,1]]).T
        Xc = np.array([[scale,0,0,1]]).T
        Yc = np.array([[0.,scale,0,1]]).T
        Zc = np.array([[0.,0,scale,1]]).T
        
        Oc1     = self.T @ Oc
        Xc1     = self.T @ Xc
        Yc1     = self.T @ Yc
        Zc1     = self.T @ Zc
        a1 = Arrow3D.Arrow3D([Oc1[0,0],Xc1[0,0]],
                             [Oc1[1,0],Xc1[1,0]],
                             [Oc1[2,0],Xc1[2,0]],
                             mutation_scale=5,
                             lw=1, arrowstyle="-|>",
                             color='k')
        a2 = Arrow3D.Arrow3D([Oc1[0,0],Yc1[0,0]],
                             [Oc1[1,0],Yc1[1,0]],
                             [Oc1[2,0],Yc1[2,0]],
                             mutation_scale=5, 
                             lw=1, arrowstyle="-|>", 
                             color='k')
        a3 = Arrow3D.Arrow3D([Oc1[0,0],Zc1[0,0]],
                             [Oc1[1,0],Zc1[1,0]],
                             [Oc1[2,0],Zc1[2,0]],
                             mutation_scale=5, 
                             lw=1, arrowstyle="-|>",
                             color='k')
        ax.add_artist(a1)
        ax.add_artist(a2)
        ax.add_artist(a3)
        #ax.text(Xc1[0,0], Xc1[1,0], Xc1[2,0], (r'$X_{cam}$'))
        #ax.text(Yc1[0,0], Yc1[1,0], Yc1[2,0], (r'$Y_{cam}$'))
        #ax.text(Zc1[0,0], Zc1[1,0], Zc1[2,0], (r'$Z_{cam}$'))
