"""
	Monday September 24th, 21:24:30 2018
	@author: robotics lab (Patricia Tavares)
	@email: patricia.tavares@cimat.mx
	version: 3.0

	This code contains functions to get the velocities depending of the desired 
	controller.
		- GTC (Ground Truth Control)
		- HD (Homography Decomposition Control)
		- RMC (Rigidity Matrix Control as Schiano)
"""

import cv2
import numpy as np
from Functions.Aux import closer_element
from Functions.Geometric import Rodrigues, H_from_points, homog_to_rt, decompose_E, scale_estimation_essential


"""
	function: GTC
	description: control for the velocities using ground truth.
	params: 
		n: how many cameras do you have? (int)
		p: relative positions of the cameras (dict)
		p_aster: relative desired positions (dict)
		R: relative rotations of the cameras (dict)
		R_aster: relative desired rotations (dict)
		lambdav: gain for linear velocity
		lambdaw: gain for angular velocity
	returns:
		v: velocities of every camera, its shape is (n,3) where every 
			row indicates de velocities for the i-th camera. Every element of the 
			row indicates the velocity on the x,y,z axis.
		w: angular velocities of every camera, its shape is (n,3) where every 
			row indicates de velocities for the i-th camera. Every element of the 
			row indicates the angular velocity on the x,y,z axis.
"""
def GTC(n,p,p_aster,R,R_aster,lambdav,lambdaw):
	#init velocities
	v = np.zeros((n,3))
	w = np.zeros((n,3))
	#calculate
	for (i,j) in p:						
			v[i]+=lambdav*(p[(i,j)]-p_aster[(i,j)])	
			w_r = Rodrigues(R[(i,j)].dot(R_aster[(i,j)].T))
			w[i]+=lambdaw*w_r			
	#return velocities
	return v,w

"""
	function: HDC
	description: returns the estimated pose (by homography means) and the 
	velocities using homography decomposition.
	params: 
		n: how many cameras do you want?
		cameras: a list of cameras (PlanarCamera)			
		w_points: the world points to project in the cameras
		n_points: how many w_points do you have?
		R_aster: the desired relative rotations (dict)
		p_aster: the desired relative positions (dict)
		gain_w: the gain for angular velocities
		gain_v: gain for velocities		
		L: Laplacian
		p_filter: the relative positions (dict) of the previous iteration
		gamma: optional value, it is a vector containing the approx distance
			from every drone to the plane. default []
	returns: 
		R: the computed relative orientations (dict)
		p: the computed relative positions (dict)
		v: the computed linear velocities (n x 3 numpy array)
		w: the computed angular velocities (n x 3 numpy array)
"""
def HDC(n,cameras,w_points,n_points,R_aster,p_aster,gain_w,gain_v,L,p_filter,gamma=[]):	
	#project the points to the camera
	points = []
	for i in range(n):			
		p_aux = cameras[i].projection(w_points,n_points)
		points.append(p_aux)

	#init velocities = 0
	w = np.zeros((n,3))
	v = np.zeros((n,3))
	
	#init dict
	p = {}
	R = {}

	#for the graph conections
	for i in range(n):
		for j in range(n):
			if L[i][j]==1 and i!=j:			
				#get homography
				Hij = H_from_points(points[i],points[j])
				#Euclidean Homography
				H_ijr = np.dot(np.linalg.inv(cameras[i].K),np.dot(Hij,cameras[j].K))
				#decompose euclidean homography
				Rij,tij,normal = homog_to_rt(H_ijr) #it we use the homography decomposition	
				#get the val of the desired yaw			
				yaw_aster = Rodrigues(R_aster[(i,j)])[2]				
				#get the value of yaw
				yaw = Rodrigues(Rij)[2]				

				if not (i,j) in p_filter:
					p_filter[(i,j)] = cameras[i].R.T.dot(cameras[j].t-cameras[i].t)
				
				tij = closer_element(p_filter[(i,j)],[tij, -tij])

				#if we have the scale
				if len(gamma) == n:							
					tij[0] = gamma[i]*tij[0]
					tij[1] =gamma[i]*tij[1]					
				
				#compute velocities	
				w[i][2]+=gain_w*(yaw-yaw_aster)#yaw
				v[i][0]+=gain_v*(tij[0]-p_aster[(i,j)][0])							
				v[i][1]+=gain_v*(tij[1]-p_aster[(i,j)][1])
				v[i][2]+=gain_v*(tij[2]-p_aster[(i,j)][2]) 				

				#save relative pose
				p[(i,j)] = tij
				R[(i,j)] = Rij

	#return velocities and relative poses
	return p,R,v,w

"""
	function: EDC
	description: returns the estimated posed (by essential means) and the velocities 
			using essential matrix decomposition.
	params: 
		n: how many cameras do you want?
		cameras: a list of cameras (PlanarCamera)			
		w_points: the world points to project in the cameras
		n_points: how many w_points do you have?
		R_aster: the desired relative rotations (dict)
		p_aster: the desired relative positions (dict)
		gain_w: the gain for angular velocities
		gain_v: gain for velocities		
		L: Laplacian
		p_filter: the relative positions (dict) of the previous iteration
		R_filter: the relative rotations (dict) of the previous iteration
		gamma: aproximated distances from every agent to the floor, it is only
			necessary for scale estimation, defaults to []
		spindex: index of the point we will reconstruct for scale estimation
	returns: 
		R: the computed relative orientations (dict)
		p: the computed relative positions (dict)		
		v: the computed linear velocities (n x 3 numpy array)
		w: the computed angular velocities (n x 3 numpy array)
"""
def EDC(n,cameras,w_points,n_points,R_aster,p_aster,gain_w,gain_v,L,p_filter,R_filter,gamma=[],spindex=-1):	
	#project the points to the camera
	points = []
	for i in range(n):			
		p_aux = cameras[i].projection(w_points,n_points)
		points.append(p_aux)

	#init velocities = 0
	w = np.zeros((n,3))
	v = np.zeros((n,3))
	
	#init dict
	p = {}
	R = {}

	#for the graph conections
	for i in range(n):
		for j in range(n):
			if L[i][j]==1 and i!=j:			
				#get fundamental	
				F, mask = cv2.findFundamentalMat(points[j].T,points[i].T,cv2.FM_LMEDS,1.0,0.99)				
				#get the essential matrix
				E = np.dot(cameras[i].K.T,np.dot(F,cameras[j].K))
				
				#obtaining desired orientation	
				yaw_aster = Rodrigues(R_aster[(i,j)])[2]

				#if its the first time, init filter
				if not (i,j) in p_filter:					
					R_filter[(i,j)] = cameras[i].R.T.dot(cameras[j].R)
					p_filter[(i,j)] = cameras[i].R.T.dot(cameras[j].t-cameras[i].t)
					p_filter[(i,j)] =  p_filter[(i,j)] / np.linalg.norm(p_filter[(i,j)])					
					
				#get decomposition
				tij, yaw = decompose_E(E,p_filter[(i,j)],R_filter[(i,j)])	
				#saving filters and other data				
				p[(i,j)] = tij
				R[(i,j)] = cameras[i].euler_to_rotmat(0,0,yaw)						
						
				#if we want to use scale
				if spindex != -1:
					qi = np.array([points[i][0][spindex],points[i][1][spindex]])#central point seen from camera i
					qj = np.array([points[j][0][spindex],points[j][1][spindex]])#central point seen from camera j
					s = scale_estimation_essential(cameras[i].K,cameras[j].K,qi,qj,yaw,tij)
					p[(i,j)][0] = tij[0]*(gamma[i]/s)#the scaled position for computing the error
					p[(i,j)][1] = tij[1]*(gamma[i]/s)#the scaled position for computing the error					

				##############updating velocities						
				w[i][2]+=gain_w*(yaw-yaw_aster)
				v[i][0]+=gain_v*(p[(i,j)][0]-p_aster[(i,j)][0])							
				v[i][1]+=gain_v*(p[(i,j)][1]-p_aster[(i,j)][1])
				v[i][2]+=gain_v*(p[(i,j)][2]-p_aster[(i,j)][2])				

	#return velocities and relative poses
	return p,R,v,w

"""
	function: RFC
	description: returns the estimated pose (by homography means) and the 
	velocities using homography decomposition.
	params: 
		n: how many cameras do you want?
		cameras: a list of cameras (PlanarCamera)			
		w_points: the world points to project in the cameras
		n_points: how many w_points do you have?
		R_aster: the desired relative rotations (dict)
		p_aster: the desired relative positions (dict)
		gain_w: the gain for angular velocities
		gain_v: gain for velocities		
		L: Laplacian
		p_filter: the relative positions (dict) of the previous iteration
		R_filter: the relative rotations (dict) of the previous iteration
		homography: optional boolean value to determine if we will use homography or
			essential matrix. defaults to True for using homography
	returns: 
		R: the computed relative orientations (dict)
		p: the computed relative positions (dict)
		v: the computed linear velocities (n x 3 numpy array)
		w: the computed angular velocities (n x 3 numpy array)
"""
def RMC(n,cameras,w_points,n_points,R_aster,p_aster,gain_w,gain_v,L,p_filter,R_filter,homography=True):	
	#project the points to the camera
	points = []
	for i in range(n):			
		p_aux = cameras[i].projection(w_points,n_points)
		points.append(p_aux)

	#init velocities = 0
	w = np.zeros((n,3))
	v = np.zeros((n,3))
	
	#init dict
	p = {}
	R = {}

	#for the graph conections
	for i in range(n):
		for j in range(n):
			if L[i][j]==1 and i!=j:			
				if homography: #we process it with the homography
					#get homography
					Hij = H_from_points(points[i],points[j])
					#Euclidean Homography
					H_ijr = np.dot(np.linalg.inv(cameras[i].K),np.dot(Hij,cameras[j].K))
					#decompose euclidean homography
					Rij,tij,normal = homog_to_rt(H_ijr) #it we use the homography decomposition
					tij = tij / np.linalg.norm(tij) #normalize
					if not (i,j) in p_filter:
						p_filter[(i,j)] = cameras[i].R.T.dot(cameras[j].t-cameras[i].t)
						p_filter[(i,j)] = p_filter[(i,j)] / np.linalg.norm(p_filter[(i,j)])
										
					tij = closer_element(p_filter[(i,j)],[tij,-tij])				
								
				else: #process the essential
					#get fundamental	
					F, mask = cv2.findFundamentalMat(points[j].T,points[i].T,cv2.FM_LMEDS,1.0,0.99)				
					#get the essential matrix
					E = np.dot(cameras[i].K.T,np.dot(F,cameras[j].K))
					if not (i,j) in p_filter:					
						R_filter[(i,j)] = cameras[i].R.T.dot(cameras[j].R)
						p_filter[(i,j)] = cameras[i].R.T.dot(cameras[j].t-cameras[i].t)
						p_filter[(i,j)] =  p_filter[(i,j)] / np.linalg.norm(p_filter[(i,j)])
					#get decomposition
					tij, yaw = decompose_E(E,p_filter[(i,j)],R_filter[(i,j)])	
					#saving filters and other data				
					Rij = cameras[i].euler_to_rotmat(0,0,yaw)

				#save relative pose
				p[(i,j)] = tij
				R[(i,j)] = Rij

	#compute velocities as in schiano thesis
	S = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]])
	for i in range(n):
		for j in range(n):
			if L[i][j]==1 and i!=j:				
				#compute velocities
				Bij = p[(i,j)]
				Bijd = p_aster[(i,j)] 
				Pij = np.eye(3)-np.outer(Bij,Bij)
				iRj = R[(i,j)]
				Bji = p[(j,i)]
				Bjid = p_aster[(j,i)] 
				Pji = np.eye(3)-np.outer(Bji,Bji)
				u_i = -gain_v*(Pij.dot(Bijd))+gain_v*(iRj.dot(Pji.dot(Bjid)))				
				w[i][2] += gain_w*np.dot(Bij,S.dot(Bijd))				
				for l in range(3):
					v[i][l]+=u_i[l]
					

	#return velocities and relative poses
	return p,R,v,w
