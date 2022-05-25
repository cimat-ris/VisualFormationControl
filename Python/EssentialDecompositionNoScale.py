"""
	Saturday September 22, 23:41:10 2018
	@author: robotics lab (Patricia Tavares)
	@email: patricia.tavares@cimat.mx
	version: 3.0
	This code uses essential decomposition to achieve a formation.
"""
"""
	Import section: from python and robotics.
"""
from Functions.Formation import line_formation, circle_formation, get_L_radius, get_relative
from Functions.Initialize import verified_random_restricted, copy_cameras, order_cameras
from Functions.Control import EDC
from Functions.Geometric import move_wf, RMF, rigidity_function
#######################################ALWAYS NEEDED
import numpy as np
from math import pi
import shutil, os, argparse, logging
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
# Parser arguments
parser = argparse.ArgumentParser(description='example of how to use the functions and the consensus algorithm for control with relative positions and rotations')
parser.add_argument('--log_level',type=int, default=20,help='Log level (default: 20)')
parser.add_argument('--log_file',default='',help='Log file (default: standard output)')
args = parser.parse_args()

# Loggin format
logging.basicConfig(format='%(levelname)s: %(message)s',level=args.log_level)

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
#normalize, its mandatory
for (i,j) in p_aster:
	p_aster[(i,j)] = p_aster[(i,j)] / np.linalg.norm(p_aster[(i,j)])
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
p = {} #relative positions
R = {} #relative rotations
#=============================================================plotting variables
t_arr = [] #time array
err_t = [] #average error in translation
err_psi = [] #average error in rotation
err_s = [] #error for scale
eig = [] #for eigenvalue
rk = [] #for matrix rank
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
logging.info('Desired Formation')
logging.info('Jacobian dimension {}'.format(J.shape))
logging.info('Jacobian rank {}'.format(rank))
logging.info('Sixth eigenvalue of J.TJ {}'.format(s[s.shape[0]-6]))

#=================================================================init algorithm
while (e_t > threshold or e_psi > threshold) and ite < max_ite and rank >= 4*n_cameras-5:
	#Compute velocities
	p, R, vi, wi = EDC(n_cameras,init_cameras,w_points,n_points,R_aster,p_aster,lambdaw,lambdav,L,p,R)
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
	logging.info('{} Translation error: {:.4f} Rotation error: {:.4f} '.format(ite,e_t ,e_psi))

logging.info("*********************\nLaplacian is {}\n".format(L))
#=======================================================================plotting
#converting to numpy
v = np.array(v)
w = np.array(w)
t_arr = np.array(t_arr)
err_t = np.array(err_t)
err_psi = np.array(err_psi)
err_s = np.array(err_s)
eig = np.array(eig)
rk = np.array(rk)

#clearing
dirs = os.listdir('.')
if 'graphs' in dirs:
	shutil.rmtree("graphs")
os.mkdir('graphs')

# Plot and save results
plot_consensus_results(n_cameras,init_cameras,desired_cameras,copy,init_poses,desired_poses,x,y,z,v,w,xx,yy,zz,t_arr,err_t,err_psi,None,legend='Position-based control using Essential Decomposition without scale estimation')
