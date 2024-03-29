# -*- coding: utf-8 -*-
"""
    2018
    @author: robotics lab (Patricia Tavares)
    @email: patricia.tavares@cimat.mx
    version: 1.0
    This code cointains a position based control
    of a camera using the Essential Matrix.
"""
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import shutil, os
from Functions.PlanarCamera import PlanarCamera
from Functions.Geometric import Rodrigues

#================================================================Functions

def tr(Camera, in_points):
    #return in_points
    
    f = Camera.focal/Camera.rho
    
    points = in_points.copy()
    points[0,:] -= Camera.pp[0]#cu
    points[1,:] -= Camera.pp[1]#cv
    points[0,:] /= f[0]
    points[1,:] /= f[1]
    
    return points

def Interation_Matrix(Camera,in_points,Z):
    points = tr(Camera,in_points)
    #points = in_points
    
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
    #L[:,[0,1,2,5,6,7,8,11]] *= 200
    #L[:,[3,4,9,10]] *= 200#40000
    return L.reshape((2*n,6))  

def Inv_Moore_Penrose(L):
    A = L.T@L
    if np.linalg.det(A) < 1.0e-9:
        return None
    #print(np.linalg.det(A))
    #print(A)
    return inv(A) @ L.T

#================================================================point cloud
#xx       = np.loadtxt('cloud/x.data')
#yy       = np.loadtxt('cloud/y.data')
#zz       = np.loadtxt('cloud/z.data')
#n_points = len(xx)
#w_points = np.vstack([xx, yy, zz])
xx = np.array([0, 2, 2 , 0],dtype=float)
yy = np.array([0,0,2,2],dtype=float)
zz = np.array([0,0,0,0],dtype=float)
n_points = 4
w_points = np.vstack([xx, yy, zz])

#==============================================================target camera
target_x        = 1.0
target_y        = 1.0
target_z        = 1.0
target_roll     = np.deg2rad(0.0) # Degrees to radians 'x'
target_pitch    = np.deg2rad(0.0) # Degrees to radians 'y'
target_yaw      = np.deg2rad(0.0) # Degrees to radians 'z'
target_camera = PlanarCamera() # Set the target camera
target_camera.set_position(target_x, target_y, target_z,target_roll, target_pitch, target_yaw)
p_target = target_camera.projection(w_points, n_points) # Project the points for camera 1

#=============================================================current camera
init_x     = 1.0
init_y     = 1.0
init_z     = 2.0
init_pitch   = np.deg2rad(0.0)
init_roll    = np.deg2rad(0.0)
init_yaw     = np.deg2rad(0.0)
moving_camera = PlanarCamera() # Set the init camera
moving_camera.set_position(init_x, init_y, init_z,init_roll, init_pitch, init_yaw)
p_moving = moving_camera.projection(w_points,n_points)

#============================================================defining params
dt = 0.01   # Time Delta, seconds.
t0 = 0      # Start time of the simulation
ite = 0 #iterations
steps = 10000 #max iterations
t1 = steps*dt

#==================================================variables to use and plot
#v       = np.array([[0],[0],[0]]) # speed in m/s
#omega   = np.array([[0],[0],[0]]) # angular velocity in rad/s
#U       = np.vstack((v,omega))
U = np.zeros((6,1))
UArray              = np.zeros((6,steps))           # Matrix to save controls history
tArray              = np.zeros(steps)               # Matrix to save the time steps
pixelCoordsArray    = np.zeros((2*n_points,steps))   # List to save points positions on the image
ErrorArray   = np.zeros((2,steps))               # Matrix to save error points positions
positionArray       = np.zeros((3,steps))           # Matrix to save  camera positions
I              = np.eye(3, 3)
lamb = 0.25
Z_estimada = 2.0
t       = t0

#=========================================================auxiliar variables
x_pos = init_x
y_pos = init_y
z_pos = init_z
roll    = init_roll
pitch   = init_pitch
yaw     = init_yaw
p20 = []
j = 0
err_pix = 10 #error in pixels
#to use a filter on the decomposition

#   Caso 1
#L = Interation_Matrix(p_target, Z_estimada)
#L_e = Inv_Moore_Penrose(L)

#   CASO 3
#L_in = L_e

print()
#=============================================================init algorithm
while( j<steps and err_pix > 1e-2):

    # ===================== Calculate new translation and rotation values using Euler's method====================
    x_pos     +=  dt * U[0, 0]
    y_pos     +=  dt * U[1, 0] # Note the velocities change due the camera framework
    z_pos     +=  dt * U[2, 0]
    roll      +=  dt * U[3, 0]
    pitch     +=  dt * U[4, 0]
    yaw       +=  dt * U[5, 0]
    

    moving_camera.set_position(x_pos, y_pos, z_pos,roll, pitch, yaw)
    p_moving = moving_camera.projection(w_points,n_points)
    
    # ==================================== CONTROL COMPUTATION =======================================
    # Chaumette Part I versión for v = -\lambda \ḩat \L^+_e (s^* -s )
    
    err = tr(moving_camera,p_target)-tr(moving_camera,p_moving)
    #err = p_target-p_moving
    
    #   CASO 2 y 3
    L =  Interation_Matrix(moving_camera,p_moving, Z_estimada)
    Z_estimada +=  dt * U[2, 0]
    #print(L)
    L_e = Inv_Moore_Penrose(L)
    
    #   CASO 3
    #L_e = (L_e+L_in)/2
    
    #print(L_e)
    if L_e is None:
        print("Err: state: "+str(x_pos)+" "+str(y_pos)+" "+str(z_pos))
        quit() 
    #print(L.shape)
    #print(L_e.shape)
    #print(err.shape)
    U = -lamb*L_e @ err.T.reshape((2*n_points,1))
    #Avoiding numerical error
    U[np.abs(U) < 1.0e-9] = 0.0
    
    # Copy data for plot
    UArray[:, j]    = U.reshape((6,))
    tArray[j]       = t
    pixelCoordsArray[:,j] = p_moving.reshape((2* n_points,1), order='F')[:,0]
    positionArray[0, j] = x_pos
    positionArray[1, j] = y_pos
    positionArray[2, j] = z_pos

    # =================================== Average feature error ======================================
    pixel_error             = np.max(abs(err),axis = 1)
    ErrorArray[:,j]         = pixel_error
    err_pix                 = np.mean(np.linalg.norm(err.reshape((2* n_points,1), order='F')))
    #ErrorArray[j]    = err_pix
    # ==================================== verifying error =======================================
    #if j == 1 and averageErrorArray[j] > averageErrorArray[j-1] and j == 1:
        #tr_filter = -tr_filter

    t += dt
    j += 1
    #print(pixel_error,U[:,0])
    print(j-1,pixel_error,end='\r')
print()
print("Finished at: "+str(j))
# ======================================  Draw cameras ========================================

fig = plt.figure(figsize=(15,10))
fig.suptitle('World setting')
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax = fig.gca()
ax.plot(xx, yy, zz, 'o')
ax.plot(positionArray[0,0:j],positionArray[1,0:j],positionArray[2,0:j]) # Plot camera trajectory
axis_scale      = 0.5
camera_scale    = 0.02
target_camera.draw_camera(ax, scale=camera_scale, color='red')
target_camera.draw_frame(ax, scale=axis_scale, c='black')
moving_camera.set_position(x_pos, y_pos, z_pos,roll,pitch,yaw)
moving_camera.draw_camera(ax, scale=camera_scale, color='brown')
moving_camera.draw_frame(ax, scale=axis_scale, c='black')
moving_camera.set_position(init_x, init_y, init_z,init_roll, init_pitch, init_yaw)
moving_camera.draw_camera(ax, scale=camera_scale, color='blue')
moving_camera.draw_frame(ax, scale=axis_scale, c='black')
limit_x = 1.0
limit_y = 1.0
limit_z = 1.0
ax.set_xlabel("$w_x$")
ax.set_ylabel("$w_y$")
ax.set_zlabel("$w_z$")
ax.grid(True)
ax.set_title('World setting')
# ======================================  Plot the pixels ==========================================
ax = fig.add_subplot(2, 2, 2)
p20 = pixelCoordsArray[:,0].reshape((2, n_points), order='F')
ax.plot(p_target[0, :],  p_target[1, :], 'o', color='red')
ax.plot(p20[0, :], p20[1, :], 'o', color='blue')

ax.set_ylim(0, target_camera.height)
ax.set_xlim(0, target_camera.width)
ax.grid(True)

ax.legend([mpatches.Patch(color='red'),mpatches.Patch(color='blue')],['Desired', 'Init'], loc=2)

for l in range( n_points):
    ax.plot(pixelCoordsArray[l*2,0:j], pixelCoordsArray[l*2+1,0:j])
# ======================================  Plot the controls ========================================
ax = fig.add_subplot(2, 2, 3)
ax.plot(tArray[0:j], UArray[0, 0:j], label='$v_x$')
ax.plot(tArray[0:j], UArray[1, 0:j], label='$v_y$')
ax.plot(tArray[0:j], UArray[2, 0:j], label='$v_z$')
ax.plot(tArray[0:j], UArray[3, 0:j], label='$\omega_x$')
ax.plot(tArray[0:j], UArray[4, 0:j], label='$\omega_y$')
ax.plot(tArray[0:j], UArray[5, 0:j], label='$\omega_z$')
ax.grid(True)
ax.legend(loc=0)

# ======================================  Plot the pixels position ===================================
#ax = fig.add_subplot(2, 2, 4)
#ax.plot(tArray[0:j], averageErrorArray[0:j], label='Average error')
#ax.grid(True)
#ax.legend(loc=0)
ax = fig.add_subplot(2, 2, 4)
ax.plot(tArray[0:j], ErrorArray[0, 0:j], label='$err_x$')
ax.plot(tArray[0:j], ErrorArray[1, 0:j], label='$err_y$')

ax.grid(True)
ax.legend(loc=0)


#deleting previous data
#dirs = os.listdir('.')
#if 'graphs' in dirs:
    #shutil.rmtree("graphs")
#os.mkdir('graphs')
#plt.savefig('graphs/complete.png',bbox_inches='tight')
plt.show()
