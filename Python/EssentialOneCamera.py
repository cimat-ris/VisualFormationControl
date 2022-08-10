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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import shutil, os
from Functions.PlanarCamera import PlanarCamera
from Functions.Geometric import Rodrigues

#================================================================point cloud
xx       = np.loadtxt('cloud/x.data')
yy       = np.loadtxt('cloud/y.data')
zz       = np.loadtxt('cloud/z.data')
n_points = len(xx)
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
init_x     = 1.5
init_y     = 1.5
init_z     = 1.5
init_pitch   = np.deg2rad(0.0)
init_roll    = np.deg2rad(0.0)
init_yaw     = np.deg2rad(18.0)
moving_camera = PlanarCamera() # Set the init camera
moving_camera.set_position(init_x, init_y, init_z,init_roll, init_pitch, init_yaw)
p_moving = moving_camera.projection(w_points,n_points)

#============================================================defining params
dt = 0.01   # Time Delta, seconds.
t0 = 0      # Start time of the simulation
ite = 0 #iterations
steps = 1000 #max iterations
t1 = steps*dt

#==================================================variables to use and plot
v       = np.array([[0],[0],[0]]) # speed in m/s
omega   = np.array([[0],[0],[0]]) # angular velocity in rad/s
U       = np.vstack((v,omega))
UArray              = np.zeros((6,steps))           # Matrix to save controls history
tArray              = np.zeros(steps)               # Matrix to save the time steps
pixelCoordsArray    = np.zeros((2*n_points,steps))   # List to save points positions on the image
averageErrorArray   = np.zeros(steps)               # Matrix to save error points positions
positionArray       = np.zeros((3,steps))           # Matrix to save  camera positions
I              = np.eye(3, 3)
lambdav         = 2.
lambdaw         = 6.
Gain            = np.zeros((6,1))
Gain[0]         = lambdav
Gain[1]         = lambdav
Gain[2]         = lambdav
Gain[3]         = lambdaw
Gain[4]         = lambdaw
Gain[5]         = lambdaw
t       = t0
K1      = target_camera.K
K2      = moving_camera.K
K1_inv  = np.linalg.inv(K1)
K2_inv  = np.linalg.inv(K2)

#=========================================================auxiliar variables
x_pos = init_x
y_pos = init_y
z_pos = init_z
roll    = init_roll
pitch   = init_pitch
yaw     = init_yaw
p20 = []
j = 0
error_e = 1
err_pix = 10 #error in pixels
#to use a filter on the decomposition
R_filter = []
tr_filter = []

#=============================================================init algorithm
while( j<steps and err_pix > 1e-2):

    # ===================== Calculate new translation and rotation values using Euler's method====================
    x_pos     +=  dt * U[0, 0]
    y_pos     +=  dt * U[1, 0] # Note the velocities change due the camera framework
    z_pos     +=  dt * U[2, 0]
    roll    +=  dt * U[3, 0]
    pitch       +=  dt * U[4, 0]
    yaw      +=  dt * U[5, 0]

    moving_camera.set_position(x_pos, y_pos, z_pos,roll, pitch, yaw)
    p_moving = moving_camera.projection(w_points,n_points)

    # =================================== Fundamental and Essential =======================================
    F, mask = cv2.findFundamentalMat(p_moving.T,p_target.T,cv2.FM_LMEDS)
    E = np.dot(target_camera.K.T,np.dot(F,moving_camera.K))
    # =================================== Decomposing and chosing decomposition =======================================
    if j == 0:
        R1, R2, tr = cv2.decomposeEssentialMat(E)
        R_filter = moving_camera.R.T.dot(target_camera.R)
        t_filter = moving_camera.R.T.dot(target_camera.t-moving_camera.t)
        t_filter = t_filter / np.linalg.norm(t_filter)
        if np.linalg.norm(R1-R_filter) < np.linalg.norm(R2-R_filter):
            R = R1
        else:
            R = R2
        if np.linalg.norm(-tr-tr_filter) < np.linalg.norm(tr-tr_filter):
            tr =  -tr

        tr_filter = tr.flatten()
        R_filter = R
    else:
        R1, R2, tr = cv2.decomposeEssentialMat(E)
        if np.linalg.norm(R1-R_filter) < np.linalg.norm(R2-R_filter):
            R = R1
        else:
            R = R2
        if np.linalg.norm(-tr-tr_filter) < np.linalg.norm(tr-tr_filter):
            tr =  -tr
        R_filter = R
        tr_filter = tr.flatten()
    u = Rodrigues(R).T

    # ==================================== CONTROL COMPUTATION =======================================
    #i used toscale the error, making it to be less than 10, but this fails when there are rotations
    #because of the unity vector tr. So i decided to use the pose, in the end this is a position based method
    err  = np.linalg.norm(moving_camera.t[0]-target_camera.t[0])
    ev       = R.T.dot(tr*err)
    ew       = u.reshape((3,1))
    e        = np.vstack((ev,ew))
    U        = -Gain*e.reshape((6,1))
    error_e  = np.linalg.norm(e)
    #Avoiding numerical error
    U[np.abs(U) < 1.0e-9] = 0.0

    # Copy data for plot
    UArray[:, j]    = U[:, 0]
    tArray[j]       = t
    pixelCoordsArray[:,j] = p_moving.reshape((2* n_points,1), order='F')[:,0]
    positionArray[0, j] = x_pos
    positionArray[1, j] = y_pos
    positionArray[2, j] = z_pos

    # =================================== Average feature error ======================================
    pixel_error             = p_moving-p_target
    err_pix                 = np.mean(np.linalg.norm(pixel_error.reshape((2* n_points,1), order='F')))
    averageErrorArray[j]    = err_pix
    # ==================================== verifying error =======================================
    if j == 1 and averageErrorArray[j] > averageErrorArray[j-1] and j == 1:
        tr_filter = -tr_filter

    t += dt
    j += 1
    print(j-1,averageErrorArray[j-1])

# ======================================  Draw cameras ========================================

fig = plt.figure(figsize=(15,10))
fig.suptitle('World setting')
ax = fig.add_subplot(2, 2, 1, projection='3d')
ax = fig.gca()
ax.plot(xx, yy, zz, 'o')
ax.plot(positionArray[0,0:j],positionArray[1,0:j],positionArray[2,0:j]) # Plot camera trajectory
axis_scale      = 0.5
camera_scale    = 0.09
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
ax = fig.add_subplot(2, 2, 4)
ax.plot(tArray[0:j], averageErrorArray[0:j], label='Average error')
ax.grid(True)
ax.legend(loc=0)
#deleting previous data
dirs = os.listdir('.')
if 'graphs' in dirs:
    shutil.rmtree("graphs")
os.mkdir('graphs')
plt.savefig('graphs/complete.png',bbox_inches='tight')
plt.show()
