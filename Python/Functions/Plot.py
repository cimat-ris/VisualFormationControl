"""
	Wednesday July 18th, 23:43:12 2018
	@author: robotics lab (Patricia Tavares)
	@email: patricia.tavares@cimat.mx
	version: 1.0
	This code contains generic functions needed:
		- configure_plot
		- plot_cameras
		- plot_all_cameras
		- plot_quarter
"""
import matplotlib.pyplot as plt
from math import pi
from Functions.Geometric import Rodrigues

"""
	function: configure_plot
	description: configures de figure to make subplots.
	params:
		n: rows of the grid plot
		m: columns of the grid plot
		fig_w: figure width
		fig_h: figure height
		title: title of the figure
	returns:
		ax: fig with subplot (see matplotlib docs)
		fig: configured figure
"""
def configure_plot(n,m,fig_w,fig_h,title):
	fig = plt.figure(figsize=(fig_h,fig_w))
	fig.suptitle(title)
	ax = fig.add_subplot(n,m,1,projection = '3d')
	ax = fig.axes(projection='3d')
	return ax,fig

"""
	function: plot_cameras
	description: it plots the cameras pased as list with the params given.
	params:
		cameras: list cameras to plot (PlanarCamera)
		ax: axis of the plt figure
		camera_scale: scale of the camera
		axis_scale: that, axis scale for ax used for the axis related to
			the camera
		camera_color: color for the cameras, example: 'black','green','blue'..
			if you don't want to show cameras 'None'
		frame_color: color for the frame of the cameras, example: 'black','green','blue'..
			if you don't want to show frames 'None'
"""
def plot_cameras(cameras,ax,camera_scale,axis_scale,camera_color,frame_color):
	for camera in cameras:
		if camera_color != 'None':
			camera.draw_camera(ax,scale=camera_scale,color=camera_color)
		if frame_color != 'None':
			camera.draw_frame(ax,scale=axis_scale,c=frame_color)

"""
	function: plot_all_cameras
	description: plots all the cameras given as param.
	params:
		ax: fig with subplot (see matplotlib docs)
		fig: configured figure
		n_cameras: cameras used.
		final: the end position of the cameras (list of PlanarCamera)
		desired: the desired position of the cameras (list of Planar Camera)
		init: initial positions of the cameras (list of PlanarCamera)
		final_poses: array with poses of final cameras (returned by function random cameras, and used in the process)
		desired_poses: array with the desired poses (returned by function line_formation/circle_formation)
		x: trayectory for every camera in x axis (n x k )
		y: trayectory for every camera in y axis (n x k)
		z: trayectory for every camera in z axis (n x k) where n=n_cameras and k=iterations made
		axis_scale: scale for the axis (frame) of every camera)
		camera_scale: scale to control de size of the camera in the subplot.
		bounds: the given bounds for the 3d graph [lower,upper], because we wanted it to be square
		so we can appreciate the formation.
	returns:
		ax: fig with subplot (see matplotlib docs)
		fig: configured figure
"""
def plot_all_cameras(ax,fig,n_cameras,final,desired,init,final_poses,desired_poses,x,y,z,axis_scale,camera_scale,bounds):
	ax.set_xlim3d(bounds[0],bounds[1])
	ax.set_ylim3d(bounds[0],bounds[1])
	ax.set_zlim3d(bounds[0],bounds[1])
	ax.set_xlabel("$w_x$")
	ax.set_ylabel("$w_y$")
	ax.set_zlabel("$w_z$")
	ax.set_aspect('auto')
	ax.grid(True)

	#plot cameras
	plot_cameras(desired,ax,camera_scale,axis_scale,'violet','None')
	plot_cameras(final,ax,camera_scale,axis_scale,'blue','None')
	plot_cameras(init,ax,camera_scale,axis_scale,'red','None')

	#plot trayectories
	for i in range(n_cameras):
		ax.plot(x[i],y[i],z[i], label=str(i+1))
		plt.grid(True)
		plt.legend(loc=2,prop={'size': 10})

	return ax,fig

"""
	function: plot_quarter
	description: plots a quarter of a 4x4 plot.
	params:
		ax: fig with subplot (see matplotlib docs)
		fig: configured figure
		place: corresponding place [2,2,X] with X as the number of the place
		x: the x data, list or numpy array [ite elements]
		y: the y data, numpy array [ite x n] where n can be the amount
			of data to represent. For example, if we have 100 iterations
			with the average velocity in x, y z, it will be [100 x 3]
			so, it will plot the x, y, z velocities along the 100 iterations.
		label: the label for the y data, if its more than one, it should be a list.
		ite: the number of iterations made.
		size: size of the label
	returns:
		ax: fig with subplot (see matplotlib docs)
		fig: configured figure
"""
def plot_quarter(ax,fig,place,x,y,label,ite,size,labels, location=1):
	ax = fig.add_subplot(place[0],place[1],place[2])
	plt.xlabel(labels[0])
	plt.ylabel(labels[1])

	#if we only have 1 data
	if y.shape == (ite,):
		ax.plot(x,y,label=label)
	else:
		#if we have more
		data = y.shape[1]
		for i in range(data):
			ax.plot(x,y[:,i],label=label[i])

	ax.grid(True)
	if size == None:
		plt.legend(loc=location)
	else:
		plt.legend(loc=location,prop={'size': size})
	return ax,fig
