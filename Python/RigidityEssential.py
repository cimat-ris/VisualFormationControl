"""
	Saturday September 28, 01:12:50 2018
	@author: robotics lab (Patricia Tavares)
	@email: patricia.tavares@cimat.mx
	version: 2.0
	This code uses essential matrix decomposition and incorporates the rigidity
	matrix as in Fabrizio Schiano' Thesis
"""
"""
	Import section: from python and robotics.
"""
from Functions.Formation import line_formation, circle_formation, get_L_radius, get_relative
from Functions.Initialize import verified_random_restricted, copy_cameras, order_cameras
from Functions.Control import RMC
from Functions.Geometric import move_wf, rigidity_function, RMF
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
#================================================================points in scene
xx       = np.loadtxt('cloud/x.data')
yy       = np.loadtxt('cloud/y.data')
zz       = np.loadtxt('cloud/z.data')
n_points = len(xx)
w_points = np.vstack([xx, yy, zz])

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
for (i,j) in p_aster:
	p_aster[(i,j)] = p_aster[(i,j)]/np.linalg.norm(p_aster[(i,j)])
#transform translations to cameras 1 frame
p_n_a = transform_to_frame(desired_cameras[0].R,p_aster,desired_cameras)	
#obtaining distances between drones in desired formation
dist_aster = distances(n_cameras, desired_cameras)

#==================================================================define params
# Define the gain
lambdav         = 2.
lambdaw         = 6.	
# Timing parameters
dt = 0.01   # Time Delta, seconds.
ite = 0 #iterations
max_ite = 10000 #max iterations
# Error parameters
threshold = 1e-4 #threshold for errors
e_t = 10000 #actual error for translation
e_psi = 10000 #actual error for rotation
p = {} #for filter purposes
R = {} #for filter purposes
#=============================================================plotting variables
t_arr = [] #time array
err_t = [] #average error in translation
err_psi = [] #average error in rotation
err_s = [] #error for scale
rk = [] #for J rank
eig = []#for sixth eigenvalue
w = [] #angular velocity
v = [] #velocity
x = [] #pos x for every camera
y = [] #pos y for every camera
z = [] #pos z for every camera
for i in range(n_cameras):	
	x.append([])
	y.append([])
	z.append([])	

#Verify the formation
rf_aster = rigidity_function(p_aster)
J = RMF(rf_aster,n_cameras,L,desired_cameras,p_aster,R_aster)
rank = np.linalg.matrix_rank(J)
u,s,vt = np.linalg.svd(J)
print 'Desired Formation'
print 'Jacobian dimension ', J.shape
print 'Jacobian rank ',rank
print 'Sixth eigenvalue of J.TJ', s[s.shape[0]-6]

#=================================================================init algorithm 
while (e_t > threshold or e_psi > threshold) and ite < max_ite and rank >= 4*n_cameras-5:
	#Compute velocities 
	p, R, vi, wi = RMC(n_cameras,init_cameras,w_points,n_points,R_aster,p_aster,lambdaw,lambdav,L,p,R,False)
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

	#rigidity matrix properties
	J = RMF(rf_aster,n_cameras,L,init_cameras,p,R)
	rank = np.linalg.matrix_rank(J)
	u,s,vt = np.linalg.svd(J)
	eig.append(s[s.shape[0]-6])
	rk.append(rank)

	#Next		
	ite+=1
	print ite,'Translation error: ',e_t, 'Rotation error: ', e_psi
	
print "*********************\nLaplacian is\n", L
#=======================================================================plotting 
#converting to numpy
v = np.array(v)
w = np.array(w)
t_arr = np.array(t_arr)
err_t = np.array(err_t)
err_psi = np.array(err_psi)
err_s = np.array(err_s)
rk = np.array(rk)
eig = np.array(eig)

#clearing
dirs = os.listdir('.')
if 'graphs' in dirs:
	shutil.rmtree("graphs")
os.mkdir('graphs')

#plotting
rows = 4
cols = 2
ax,fig = configure_plot(rows,cols,15,20,'Position-based control using Essential Decomposition and Rigidity Matrix')
bounds = [min(-1.0,np.min(x)-0.2,np.min(y)-0.2,np.min(z)-0.2),max(2.5,np.max(x)+0.2,np.max(y)+0.2,np.max(z)+0.2)]
ax,fig = plot_all_cameras(ax,fig,n_cameras,init_cameras,desired_cameras,copy,init_poses,desired_poses,x,y,z,0.5,0.09,bounds)
ax.plot(xx,yy,zz,'o', alpha=0.2) #plot points
ax,fig = plot_quarter(ax,fig,[rows,cols,2],t_arr,err_s,'Actual formation average scale',ite,None,['Time (s)','Actual formation average scale'],4)
ax,fig = plot_quarter(ax,fig,[rows,cols,3],t_arr,v,['$v_x$','$v_y$','$v_z$'],ite,None,['Time (s)','Average linear velocity (m/s)'])
ax,fig = plot_quarter(ax,fig,[rows,cols,4],t_arr,w,['$\omega_x$','$\omega_y$','$\omega_z$'],ite,None,['Time (s)','Average absolute angular velocity (rad/s)'])
ax,fig = plot_quarter(ax,fig,[rows,cols,5],t_arr,err_t,'Evaluation error in translation ($e_t$)',ite,None,['Time (s)','Evaluation error in translation (m)'])
ax,fig = plot_quarter(ax,fig,[rows,cols,6],t_arr,err_psi,'Evaluation error in rotation ($e_\psi$)',ite,None,['Time (s)','Evaluation error in rotation (rad)'])
ax,fig = plot_quarter(ax,fig,[rows,cols,7],t_arr,eig,'Sixth smaller eigenvalue of Rigidity Matrix',ite,None,['Time (s)','$\lambda_6$'])
ax,fig = plot_quarter(ax,fig,[rows,cols,8],t_arr,rk,'Rank of Rigidity Matrix',ite,None,['Time (s)','Rank of Rigidity Matrix'])

#save and show
plt.savefig('graphs/complete.pdf',bbox_inches='tight')
plt.show()
