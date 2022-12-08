"""
	Python script to plot all the given information from the simulator
	email: patriciatt.tavares@cimat.mx
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import re
import os

n = int(sys.argv[1]) #number of quadrotors
gamma_used = int(sys.argv[2]) #plot additional information
dt = 0.02 #sampling time

#plot the information from every folder
for i in range(n):
	print 'Processing '+str(i)+'...'
	dir = str(i)

	velocities = np.loadtxt(dir+'/velocities.txt')
	errors = np.loadtxt(dir+'/errors.txt')
	if gamma_used == 1:
		gamma = np.loadtxt(dir+'/gamma.txt')
	ite = len(velocities[:,0])
	time = np.linspace(0,ite*dt,ite)

	#plot translation error
	plt.plot(time,errors[:,0])
	plt.ylabel('Average position error $(m)$')
	plt.xlabel('Time $(s)$')
	plt.grid(True)
	plt.savefig(dir+'/e_t.pdf',bbox_inches='tight')
	plt.clf()

	#plot rotation error
	plt.plot(time,errors[:,1])
	plt.ylabel('Average rotation error $(rad)$')
	plt.xlabel('Time $(s)$')
	plt.grid(True)
	plt.savefig(dir+'/e_psi.pdf',bbox_inches='tight')
	plt.clf()

	#plot velocities
	plt.suptitle("Linear Velocities for quadrotor "+dir)
	labels = ['$V_x$','$V_y$','$V_z$']
	for j in range (3):
		plt.plot(time, velocities[:,j], label=labels[j])
	plt.ylabel('Velocities $(m/s)$')
	plt.xlabel('Time $(s)$')
	plt.grid(True)
	plt.legend(loc=0)
	plt.savefig(dir+'/v.pdf',bbox_inches='tight')
	plt.clf()

	#plot angular velocities
	plt.suptitle("Angular Velocity for quadrotor "+dir)
	plt.plot(time, velocities[:,3], label='$W_z$')
	plt.ylabel('Velocity $(rad/s)$')
	plt.xlabel('Time $(s)$')
	plt.grid(True)
	plt.legend(loc=0)
	plt.savefig(dir+'/w.pdf',bbox_inches='tight')
	plt.clf();
	
	#plot altitude consensus
	if gamma_used ==1 :
		plt.suptitle("Altitude consensus "+dir)
		plt.plot(time, gamma, label='$\gamma(t)$')
		plt.ylabel('Altitude $(m)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(dir+'/gamma.pdf',bbox_inches='tight')
		plt.clf();
	
	for sub in os.walk(dir):

		if len(sub[0]) != 3: continue
		
		name = re.findall('\d', sub[0])

		errors = np.loadtxt(sub[0]+'/errors.txt')
		coordinates = np.loadtxt(sub[0]+'/coordinates.txt')
		matching = np.loadtxt(sub[0]+'/matches.txt')
		ite = len(matching)
		time = np.linspace(0,ite*dt,ite)
		
		#difference between estimation and desired
		plt.plot(time, errors[:,0], label='$||\hat{\mathbf{p}}_{'+name[0]+name[1]+'} - \mathbf{p}_{'+name[0]+name[1]+'}^{*}||$')
		plt.ylabel('Distance $(m)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/pij_estimation_desired.pdf',bbox_inches='tight')
		plt.clf()

		#difference between estimation and desired with matches
		plt.subplot(211)
		plt.plot(time, errors[:,0], label='$||\hat{\mathbf{p}}_{'+name[0]+name[1]+'} - \mathbf{p}_{'+name[0]+name[1]+'}^{*}||$')
		plt.ylabel('Distance $(m)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/pij_estimation_desired_matches.pdf',bbox_inches='tight')
		plt.clf()

		#difference between estimation and ground truth
		plt.plot(time, errors[:,1], label='$||\hat{\mathbf{p}}_{'+name[0]+name[1]+'} - \mathbf{p}_{'+name[0]+name[1]+'}^{GT}||$')
		plt.ylabel('Distance $(m)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/pij_estimation_gt.pdf',bbox_inches='tight')
		plt.clf()

		#difference between estimation and ground truth with matches
		plt.subplot(211)
		plt.plot(time, errors[:,1], label='$||\hat{\mathbf{p}}_{'+name[0]+name[1]+'} - \mathbf{p}_{'+name[0]+name[1]+'}^{GT}||$')
		plt.ylabel('Distance $(m)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/pij_estimation_gt_matches.pdf',bbox_inches='tight')
		plt.clf()

		#difference between estimation and desired
		plt.plot(time, errors[:,2], label='$|\hat{\psi}_{'+name[0]+name[1]+'} - \psi_{'+name[0]+name[1]+'}^{*}|$')
		plt.ylabel('Difference $(rad)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/yawij_estimation_desired.pdf',bbox_inches='tight')
		plt.clf()

		#difference between estimation and desired with matches
		plt.subplot(211)
		plt.plot(time, errors[:,2], label='$|\hat{\psi}_{'+name[0]+name[1]+'} - \psi_{'+name[0]+name[1]+'}^{*}|$')
		plt.ylabel('Difference $(rad)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/yawij_estimation_desired_matches.pdf',bbox_inches='tight')
		plt.clf()

		#difference between estimation and ground truth
		plt.plot(time, errors[:,3], label='$|\hat{\psi}_{'+name[0]+name[1]+'} - \psi_{'+name[0]+name[1]+'}^{GT}|$')
		plt.ylabel('Difference $(rad)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/yawij_estimation_gt.pdf',bbox_inches='tight')
		plt.clf()

		#difference between estimation and ground truth with matches
		plt.subplot(211)
		plt.plot(time, errors[:,3], label='$|\hat{\psi}_{'+name[0]+name[1]+'} - \psi_{'+name[0]+name[1]+'}^{GT}|$')
		plt.ylabel('Difference $(rad)$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/yawij_estimation_gt_matches.pdf',bbox_inches='tight')
		plt.clf()

		#matching
		plt.plot(time, matching, label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/matching.pdf',bbox_inches='tight')
		plt.clf()
	
		#coordinates comparation
		labels = ['$\hat{x}_{'+name[0]+name[1]+'}$','$x_{'+name[0]+name[1]+'}^*$','$x_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,j], label=labels[j])
		plt.ylabel('$m$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/x.pdf',bbox_inches='tight')
		plt.clf()

		#coordinates comparation matches
		plt.subplot(211)
		labels = ['$\hat{x}_{'+name[0]+name[1]+'}$','$x_{'+name[0]+name[1]+'}^*$','$x_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,j], label=labels[j])
		plt.ylabel('$m$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/x_matches.pdf',bbox_inches='tight')
		plt.clf()
		
		labels = ['$\hat{y}_{'+name[0]+name[1]+'}$','$y_{'+name[0]+name[1]+'}^*$','$y_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,3+j], label=labels[j])
		plt.ylabel('$m$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/y.pdf',bbox_inches='tight')
		plt.clf()

		plt.subplot(211)
		labels = ['$\hat{y}_{'+name[0]+name[1]+'}$','$y_{'+name[0]+name[1]+'}^*$','$y_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,3+j], label=labels[j])
		plt.ylabel('$m$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/y_matches.pdf',bbox_inches='tight')
		plt.clf()

		labels = ['$\hat{z}_{'+name[0]+name[1]+'}$','$z_{'+name[0]+name[1]+'}^*$','$z_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,6+j], label=labels[j])
		plt.ylabel('$m$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/z.pdf',bbox_inches='tight')
		plt.clf()

		plt.subplot(211)
		labels = ['$\hat{z}_{'+name[0]+name[1]+'}$','$z_{'+name[0]+name[1]+'}^*$','$z_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,6+j], label=labels[j])
		plt.ylabel('$m$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/z_matches.pdf',bbox_inches='tight')
		plt.clf()

		#angle comparation
		labels = ['$\hat{\psi}_{'+name[0]+name[1]+'}$','$\psi_{'+name[0]+name[1]+'}^*$','$\psi_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,9+j], label=labels[j])
		plt.ylabel('$rad$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/yaw.pdf',bbox_inches='tight')
		plt.clf()

		plt.subplot(211)
		labels = ['$\hat{\psi}_{'+name[0]+name[1]+'}$','$\psi_{'+name[0]+name[1]+'}^*$','$\psi_{'+name[0]+name[1]+'}^{GT}$']
		for j in range(3):
			plt.plot(time, coordinates[:,9+j], label=labels[j])
		plt.ylabel('$rad$')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.subplot(212)
		plt.plot(time, matching,color='mediumorchid',label='$matches_{'+name[0]+name[1]+'}$')
		plt.ylabel('Matches')
		plt.xlabel('Time $(s)$')
		plt.grid(True)
		plt.legend(loc=0)
		plt.savefig(sub[0]+'/yaw_matches.pdf',bbox_inches='tight')
		plt.clf()


	print 'Done.'
		
