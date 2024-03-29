"""
	Saturday September 22, 22:50:58 2018
	@author: robotics lab (Patricia Tavares)
	@email: patricia.tavares@cimat.mx
	version: 3.0
	This code uses ground truth to achieve a formation.
"""
"""
	Import section: from python and robotics.
"""
from Functions.Formation import line_formation, circle_formation, get_L_radius, get_relative
from Functions.Initialize import verified_random_restricted, copy_cameras, order_cameras
from Functions.Control import GTC
from Functions.Geometric import move_wf
#######################################ALWAYS NEEDED
import numpy as np
from math import pi
import shutil, os
from random import  seed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from Functions.Plot import *
from Functions.PlanarCamera import PlanarCamera
from Functions.Error import get_error, transform_to_frame, distances

"""
	***************************************************************************************************Example
"""
"""
	function: main
	description: example of how to use the functions and the consensus
	algorithm for control with relative positions and rotations.
"""

#===================================================================init cameras
#seed(40200) #init seed for comparision purposes
n_cameras = 5
init_cameras, init_poses = verified_random_restricted(n_cameras,0.5,[0.,2.,0.5,1.5,1.,3.,0.,0.,0.,0.,-pi/2.,pi/2.])
desired_cameras, desired_poses = circle_formation(n_cameras,1.,[1.,1.])
init_cameras, init_poses = order_cameras(n_cameras, init_cameras,init_poses, desired_poses)
copy = copy_cameras(n_cameras,init_poses)

#===================================getting Laplacian and desired relative poses
L = get_L_radius(n_cameras,init_poses,1.5)
p_aster,R_aster = get_relative(n_cameras,desired_cameras,L)
#transform translations to cameras 1 frame
p_n_a = transform_to_frame(desired_cameras[0].R,p_aster,desired_cameras)
#obtaining distances between drones in desired formation
dist_aster = distances(n_cameras, desired_cameras)

#==================================================================define params
# Define the gain
lambdav         = 2.
lambdaw         = 2.
# Timing parameters
dt = 0.01   # Time Delta, seconds.
ite = 0 #iterations
max_ite = 10000 #max iterations
# Error parameters
threshold = 1e-4 #threshold for errors
e_t = 10000 #actual error for translation
e_psi = 10000 #actual error for rotation

#=============================================================plotting variables
t_arr = [] #time array
err_t = [] #average error in translation
err_psi = [] #average error in rotation
err_s = [] #error for scale
w = [] #angular velocity
v = [] #velocity
x = [] #pos x for every camera
y = [] #pos y for every camera
z = [] #pos z for every camera
for i in range(n_cameras):
	x.append([])
	y.append([])
	z.append([])

#=================================================================init algorithm
while (e_t > threshold or e_psi > threshold) and ite < max_ite:
	#compute relative poses
	p,R = get_relative(n_cameras, init_cameras, L)
	#Compute velocities
	vi, wi = GTC(n_cameras,p,p_aster,R,R_aster,lambdaw,lambdav)
	#compute error
	p_n = transform_to_frame(init_cameras[0].R,p,init_cameras)#transform to camera 1 frame
	dist = distances(n_cameras,init_cameras)#computing real distances for simulation purposes
	e_t, e_psi, e_s = get_error(p_n,p_n_a,R,R_aster,init_cameras,desired_cameras, dist, dist_aster)
	err_t.append(e_t)
	err_psi.append(e_psi)
	err_s.append(e_s)

	#getting back to world frame with those velocities
	init_poses = move_wf(n_cameras,init_cameras,vi,wi,dt)
	#assing the new pose
	for i in range(n_cameras):
		x[i].append(init_poses[i][0])
		y[i].append(init_poses[i][1])
		z[i].append(init_poses[i][2])
		init_cameras[i].set_position(init_poses[i][0],init_poses[i][1],init_poses[i][2],init_poses[i][3],init_poses[i][4],init_poses[i][5])

	#save info for plots
	v.append(sum(vi)/n_cameras)
	w.append(sum(np.abs(wi))/n_cameras)
	t_arr.append(ite*dt)
	#Next
	ite+=1
	print('{} Translation error: {} Rotation error: {}'.format(ite,e_t,e_psi))

print("*********************\nLaplacian is {}\n".format(L))
#=======================================================================plotting
#converting to numpy
v = np.array(v)
w = np.array(w)
t_arr = np.array(t_arr)
err_t = np.array(err_t)
err_psi = np.array(err_psi)
err_s = np.array(err_s)

#clearing
dirs = os.listdir('.')
if 'graphs' in dirs:
	shutil.rmtree("graphs")
os.mkdir('graphs')

#plotting
rows = 3
cols = 2
ax,fig = configure_plot(3,2,15,20,'Position-based control using Ground Truth')
bounds = [min(-1.0,np.min(x)-0.2,np.min(y)-0.2,np.min(z)-0.2),max(2.5,np.max(x)+0.2,np.max(y)+0.2,np.max(z)+0.2)]
ax,fig = plot_all_cameras(ax,fig,n_cameras,init_cameras,desired_cameras,copy,init_poses,desired_poses,x,y,z,0.5,0.09,bounds)
ax,fig = plot_quarter(ax,fig,[rows,cols,2],t_arr,err_s,'Actual formation average scale',ite,None,['Time (s)','Actual formation average scale'],4)
ax,fig = plot_quarter(ax,fig,[rows,cols,3],t_arr,v,['$v_x$','$v_y$','$v_z$'],ite,None,['Time (s)','Average linear velocity (m/s)'])
ax,fig = plot_quarter(ax,fig,[rows,cols,4],t_arr,w,['$\omega_x$','$\omega_y$','$\omega_z$'],ite,None,['Time (s)','Average absolute angular velocity (rad/s)'])
ax,fig = plot_quarter(ax,fig,[rows,cols,5],t_arr,err_t,'Evaluation error in translation ($e_t$)',ite,None,['Time (s)','Evaluation error in translation (m)'])
ax,fig = plot_quarter(ax,fig,[rows,cols,6],t_arr,err_psi,'Evaluation error in translation ($e_\psi$)',ite,None,['Time (s)','Evaluation error in rotation (rad)'])

#save and show
plt.savefig('graphs/complete.pdf',bbox_inches='tight')
plt.show()
