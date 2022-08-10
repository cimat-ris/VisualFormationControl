"""
	Saturday September 22, 22:25:12 2018
	@author: robotics lab (Patricia Tavares)
	@email: patricia.tavares@cimat.mx
	version: 3.0
	This code uses homography decomposition to achieve a formation using
	scale information as in paper Vision-based distributed formation
	without an external positioning system.
"""
"""
	Import section: from python and robotics.
"""
from Functions.Formation import line_formation, circle_formation, get_L_radius, get_relative, get_A
from Functions.Initialize import verified_random_restricted, copy_cameras, order_cameras
from Functions.Control import HDC
from Functions.Geometric import move_wf, rigidity_function, RMF
#######################################ALWAYS NEEDED
import numpy as np
from math import pi
from random import  seed
from Functions.PlanarCamera import PlanarCamera
from Functions.Plot import plot_all,run_info, plot_3D
from Functions.Error import get_error, transform_to_frame, distances,filter_error

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

#===================================================================init seeds
seed_r = 50554666 #to initialize agent poses
seed_p = 9134
seed(seed_r)
np.random.seed(seed_p) # to initialize image noise

#===================================================================init cameras
n_cameras = 3
sd = 1.5 #standar deviation noise for cameras
init_cameras, init_poses = verified_random_restricted(n_cameras,0.5,[0.,2.,0.5,1.5,1.,3.,0.,0.,0.,0.,-pi/2.,pi/2.],sd)
desired_cameras, desired_poses = line_formation(n_cameras,[0,2.])#circle_formation(n_cameras,1.,[1.,1.])#
init_cameras, init_poses = order_cameras(n_cameras, init_cameras,init_poses, desired_poses)
copy = copy_cameras(n_cameras,init_poses)
copy_poses = init_poses.copy()

#===================================getting Laplacian and desired relative poses
L = get_L_radius(n_cameras,init_poses,14)
A = get_A(n_cameras,L)
print(A)
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
max_ite = 1000 #max iterations
# Error parameters
threshold = 0.04 #threshold for errors
e_t = 10000 #actual error for translation
e_psi = 10000 #actual error for rotation
p = {} #for filter
#=============================================================plotting variables
t_arr = [] #time array
err_t = [] #average error in translation
err_psi = [] #average error in rotation
err_s = [] #for scale ratio
errors_t = [] #to filter traslation error in a window
errors_psi = [] #to filter rotation error in a window
window = 10 #amount of iterations to take as a window to filter errors
gamma = [] #scale information through time
g = [] #scale initialization
w = [] #angular velocity
v = [] #velocity
x = [] #pos x for every camera
y = [] #pos y for every camera
z = [] #pos z for every camera
dists = {} #for informative purposes
for i in range(n_cameras):
	x.append([])
	y.append([])
	z.append([])
	g.append(init_poses[i][2]+1)

for (i,j) in dist_aster:
	dists[(i,j)] = []

#Verify the formation
rf_aster = rigidity_function(p_aster)
J = RMF(rf_aster,n_cameras,L,desired_cameras,p_aster,R_aster)
rank = np.linalg.matrix_rank(J)
u,s,vt = np.linalg.svd(J)
sixth = s[s.shape[0]-6]

#=================================================================init algorithm
while (e_t > threshold or e_psi > threshold) and ite < max_ite and rank >= 4*n_cameras-5:
	#make a step in scale consensus
	g = A.dot(g)
	#Compute velocities
	p, R, vi, wi = HDC(n_cameras,init_cameras,w_points,n_points,R_aster,p_aster,lambdaw,lambdav,L,p,g)
	#compute error
	p_n = transform_to_frame(init_cameras[0].R,p,init_cameras)#transform to camera 1 frame
	dist = distances(n_cameras,init_cameras)#computing real distances for simulation purposes
	e_t, e_psi, e_s = get_error(p_n,p_n_a,R,R_aster,init_cameras,desired_cameras, dist, dist_aster)
	e_t, errors_t = filter_error(e_t,errors_t,window)
	e_psi, errors_psi = filter_error(e_psi,errors_psi,window)
	e_t-=.017

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

	for (i,j) in dist:
		dists[(i,j)].append(dist[(i,j)])

	#plot_3D(xx,yy,zz,n_cameras,x,y,z,init_cameras,init_cameras,desired_cameras,init_poses,desired_poses,ite,-1)
	#save info for plots
	v.append(sum(vi)/n_cameras)
	w.append(sum(np.abs(wi))/n_cameras)
	t_arr.append(ite*dt)
	gamma.append(g[0])

	#Next
	ite+=1
	print('[{:0>4}] Translation error: {:.4f} Rotation error: {:.4f}'.format(ite,e_t, e_psi))

print("*********************\nLaplacian is{}\n".format(L))

#=================================================================================================================================plotting
#converting to numpy
v = np.array(v)
w = np.array(w)
t_arr = np.array(t_arr)
err_t = np.array(err_t)
err_psi = np.array(err_psi)
gamma = np.array(gamma)
err_s = np.array(err_s)


plot_all('Position-based formation control using Homography Decomposition with scale estimation',xx,yy,zz,n_cameras,x,y,z,copy,init_cameras,desired_cameras,init_poses,desired_poses,ite,t_arr,err_s,gamma,v,w,err_t,err_psi,dists)
run_info('Position-based formation control using Homography Decomposition with scale estimation',seed_r,seed_p,ite,dt,threshold,n_cameras,rank,sixth,e_t,e_psi,sd,L,p_aster,R_aster,copy_poses,init_poses,e_s)
