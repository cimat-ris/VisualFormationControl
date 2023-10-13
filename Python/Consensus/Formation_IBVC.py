# -*- coding: utf-8 -*-
"""
    2018
    @author: E Chávez Aparicio (Bloodfield)
    @email: edgar.chavez@cimat.mx
    version: 1.0
    This code cointains a visual based control
    of a camera using the descriptors
"""

#   Math
import random
import numpy as np
from numpy.linalg import inv, svd, norm
from numpy import sin, cos, pi
from scipy.optimize import minimize_scalar

#   Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from numpy.random import rand, randint
from scipy.stats import gaussian_kde

#   Image
#import cv2
import shutil, os
import csv

#   Custom
import graph as gr
import camera as cm
import controller as ctr
import myplots as mp

#   Interfacing 
import sys
import time

#case 1
#SceneP=[[-0.5],
#[-0.5],
#[0  ]] 
##case 2
#SceneP=[[-0.5, -0.5],
#[-0.5,  0.5],
#[0,   0.2]] 
##case 3
#SceneP=[[0,   -0.5,  0.5],
#[-0.5,  0.5, 0],
##[0.,  0., 0.]]
##[0.0,  -0.2, 0.5]]
##[0.0,  0.2, 0.3]]
#[0.0,  -0.0, 0.0]]
#case 4
SceneP=[[-0.5, -0.5, 0.5,  0.5],
[-0.5,  0.5, 0.5, -0.5],
[0,    0.2, 0.3,  -0.1]]           
#[0,    0.0, 0.0,  0.0]] 
#case 5
#SceneP=[[-0.5, -0.5, 0.5, 0.5, 0.1],
#[-0.5, 0.5, 0.5, -0.5, -0.3],
#[0, 0.0, 0.0,  -0.0, 0.0]]
##[0, 0.2, 0.3, -0.1, 0.1]]
##case 6
#SceneP=[[-0.5, -0.5, 0.5, 0.5, 0.1, -0.1],
#[-0.5, 0.5, 0.5, -0.5, -0.3, 0.2],
#[0, 0.0, 0.0, -0.0, 0.0, 0.0]]
##[0, 0.2, 0.3, -0.1, 0.1, 0.15]]
##otherwise
#SceneP=[[-0.5, -0.5, 0.5, 0.5],
#[-0.5, 0.5, 0.5, -0.5],
#[0, 0.2, 0.3, -0.1]]
colorNames = ["colorPalette_A.npz",
             "colorPalette_B.npz"]
             
logFile = ""

def write2log(text):
    global logFile
    try:
        with open(logFile,'a') as logH:
            logH.write(text)
    except IOError:
        print("Logfile Error."+logFile+" OUT")

def circle(n_agents,
           r= 1,
           T = np.zeros(3),
           angs = None):
    
    T = T.reshape((3,1))
    Objective = np.zeros((6,n_agents))
    step = 2*pi/n_agents
    ang_arange = np.arange(0,2*pi,step)-step/2.0
    Objective[0,:] = r*cos(ang_arange)
    Objective[1,:] = r*sin(ang_arange)
    Objective[3,:] = pi
    
    #Objective = Objective[:,[1,2,3,0]]
    
    if angs is None:
        Objective[:3,:] += T
        return Objective
    
    R = cm.rot(angs[2],'z') 
    R = R @ cm.rot(angs[1],'y')
    R = R @ cm.rot(angs[0],'x')
    for i in range(4):
        _R = cm.rot(Objective[5,i],'z') 
        _R = _R @ cm.rot(Objective[4,i],'y')
        _R = _R @ cm.rot(Objective[3,i],'x')
        
        _R = R @ _R
        [Objective[3,i], Objective[4,i], Objective[5,i]] = ctr.get_angles(_R)
        
    Objective[:3,:] = _R @ Objective[:3,:]
    Objective[:3,:] += T
    
    return Objective

def Z_select(depthOp, agent, P, Z_set, p0, pd, j):
    Z = np.ones((1,P.shape[1]))
    if depthOp ==1:
        Z = agent.camera.Preal @ P
        Z = Z[2,:]
    #elif depthOp == 6:
        #Z = agent.camera.p[2]*np.ones(P.shape[1])
        #Z = Z-P[2,:]
    #elif depthOp ==2: # 
        #Z = p0[2,j]*Z
        #Z = Z-P[2,:]
    #elif depthOp == 3: # distancia al valor inicial
        #Z = pd[2,j]*Z
        #Z = Z-P[2,:]
    #elif depthOp == 4: # fijo al valor de Z_set
        #Z = Z_set * np.ones(P.shape[1])
    #elif depthOp == 5: # equitativo, = promedio
        #tmp = agent.camera.p[2]-np.mean(P[2,:])
        #Z = Z*tmp
    else:
        write2log("ERR: Invalid depthOp\n")
        return None
    return Z

################################################################################
################################################################################
#
#   Plots Aux
#
################################################################################
################################################################################

def plot_3Dcam(ax, camera,
               positionArray,
               init_configuration,
               end_configuration,
               desired_configuration,
               color, i,
               camera_scale    = 0.02):
    
    #ax.plot(positionArray[0,:],
    ax.scatter(positionArray[0,:],
            positionArray[1,:],
            positionArray[2,:],
            label = str(i),
            color = color,
            s = 0.1 ) # Plot camera trajectory
    
    camera.pose(end_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='red')
    ax.text(camera.p[0],camera.p[1],camera.p[2],str(i))
    camera.pose(desired_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='brown')
    ax.text(camera.p[0],camera.p[1],camera.p[2],str(i))
    camera.pose(init_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='blue')
    ax.text(camera.p[0],camera.p[1],camera.p[2],str(i))
    ax.set_xlabel("$w_x$")
    ax.set_ylabel("$w_y$")
    ax.set_zlabel("$w_z$")
    ax.grid(True)

def view3D(directory,
           xLimit = None,
           yLimit = None,
           zLimit = None):
    
    #   load
    fileName = directory + "/data3DPlot.npz"
    npzfile = np.load(fileName)
    P = npzfile['P']
    pos_arr = npzfile['pos_arr']
    p0 = npzfile['p0']
    pd = npzfile['pd']
    end_position = npzfile['end_position']
    
    n_agents = p0.shape[1]
    
    #   Plot
    
    #colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    npzfile = np.load("general.npz")
    colors = npzfile["colors"]
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    name = directory+"/3Dplot"
    #fig.suptitle(label)
    #ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    ax.scatter(P[0,:], P[1,:], P[2,:], s = 3)
    for i in range(n_agents):
        cam = cm.camera()
        cam.pose(end_position[i,:])
        plot_3Dcam(ax, cam,
                pos_arr[i,:,:],
                p0[:,i],
                end_position[i,:],
                pd[:,i],
                color = colors[i],
                i = i,
                camera_scale    = 0.02)
    
    ##f_size = 1.1*max(abs(p0[:2,:]).max(),abs(pd[:2,:]).max())
    #lfact = 1.1
    #x_min = lfact*min(p0[0,:].min(),pd[0,:].min())
    #x_max = lfact*max(p0[0,:].max(),pd[0,:].max())
    #y_min = lfact*min(p0[1,:].min(),pd[1,:].min())
    #y_max = lfact*max(p0[1,:].max(),pd[1,:].max())
    #z_max = lfact*max(p0[2,:].max(),pd[2,:].max())
    
    #if not xLimit is None:
        #ax.set_xlim(xLimit[0],xLimit[1])
    #if not yLimit is None:
        #ax.set_ylim(yLimit[0],yLimit[1])
    #if not zLimit is None:
        #ax.set_zlim(zLimit[0],zLimit[1])
    
    
   
    
    lfact = 1.1
    x_min = min(p0[0,:].min(),
                pd[0,:].min(),
                pos_arr[:,0,-1].min(),
                P[0,:].min())
    x_max = max(p0[0,:].max(),
                pd[0,:].max(),
                pos_arr[:,0,-1].max(),
                P[0,:].max())
    y_min = min(p0[1,:].min(),
                pd[1,:].min(),
                pos_arr[:,1,-1].min(),
                P[1,:].min())
    y_max = max(p0[1,:].max(),
                pd[1,:].max(),
                pos_arr[:,1,-1].max(),
                P[1,:].max())
    z_max = max(p0[2,:].max(),
                pd[2,:].max(),
                pos_arr[:,2,-1].max(),
                P[2,:].max())
    z_min = min(p0[2,:].min(),
                pd[2,:].min(),
                pos_arr[:,2,-1].min(),
                P[2,:].min())
    
    width = x_max - x_min
    height = y_max - y_min
    depth = z_max - z_min
    sqrfact = max(width,height,depth)
    
    x_min -= (sqrfact - width )/2
    x_max += (sqrfact - width )/2
    y_min -= (sqrfact - height )/2
    y_max += (sqrfact - height )/2
    z_min -= (sqrfact - depth )/2
    z_max += (sqrfact - depth )/2
    ax.set_xlim(x_min,x_max)
    ax.set_ylim(y_min,y_max)
    ax.set_zlim(z_min,z_max)
    
    
    fig.legend( loc=1)
    plt.show()
    plt.close()



################################################################################
################################################################################
#
#   Review
#
################################################################################
################################################################################

def agent_review(directory = "", 
                 agentId = 0):
    
    #   load
    fileName = directory + "/data3DPlot.npz"
    npzfile = np.load(fileName)
    P = npzfile['P']
    pos_arr = npzfile['pos_arr']
    p0 = npzfile['p0']
    pd = npzfile['pd']
    end_position = npzfile['end_position']
    
    n_agents = p0.shape[1]
    
    #   Plot
    
    #colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    npzfile = np.load("general.npz")
    colors = npzfile["colors"]
    
    #   Transoformaciones necesarias 
    
    j = agentId
    state = pos_arr[:,:,-1].T
    
    #   Traslación
    T = state[:3,j]
    state[:3,:] = state[:3,:] - np.c_[T,T,T,T]
    T = pd[:3,j]
    pd[:3,:] = pd[:3,:] - np.c_[T,T,T,T]
    
    #   Escala
    esc = np.linalg.norm(state[:3,:],axis = 0).max()
    state[:3,:] /= esc
    esc = np.linalg.norm(pd[:3,:],axis = 0).max()
    pd[:3,:] /= esc
    
    #   Rotación
    R_end =  cm.rot(state[5,j],'z') 
    R_end = R_end @ cm.rot(state[4,j],'y')
    R_end = R_end @ cm.rot(state[3,j],'x')
    R_ref =  cm.rot(pd[5,j],'z') 
    R_ref = R_ref @ cm.rot(pd[4,j],'y')
    R_ref = R_ref @ cm.rot(pd[3,j],'x')
    
    R  =  R_ref @ R_end.T
    state[:3,:] =  R@state[:3,:]
    
    
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    name = directory+"/3Dplot"
    
    for i in range(n_agents):
        _R = cm.rot(state[5,i],'z') 
        _R = _R @ cm.rot(state[4,i],'y')
        _R = _R @ cm.rot(state[3,i],'x')
        _R = R @ _R
        [state[3,i], state[4,i], state[5,i]] = ctr.get_angles(_R)
        
        camera = cm.camera()
        
        camera.pose(state[:,i])
        camera.draw_camera(ax, scale=0.2, color='red')
        ax.text(camera.p[0],camera.p[1],camera.p[2],str(i))
        camera.pose(pd[:,i])
        camera.draw_camera(ax, scale=0.2, color='brown')
        ax.text(camera.p[0],camera.p[1],camera.p[2],str(i))
    
    
    plt.savefig(name+'_rev_'+str(j)+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    return
################################################################################
################################################################################
#
#   Error measure
#
################################################################################
################################################################################

#   Error if state calculation
#   It assumes that the states in the array are ordered and are the same
#   regresa un error de traslación y orientación

def error_state(reference,  agents, gdl = 1, name= None):
    
    n = reference.shape[1]
    state = np.zeros((6,n))
    for i in range(len(agents)):
        state[:,i] = agents[i].camera.p
    
    #   Obten centroide
    centroide_ref = reference[:3,:].sum(axis=1)/n
    centroide_state = state.sum(axis=1)/n
    
    #   Centrar elementos
    new_reference = reference.copy()
    new_reference[0] -= centroide_ref[0]
    new_reference[1] -= centroide_ref[1]
    new_reference[2] -= centroide_ref[2]
    new_reference[:3,:] /= norm(new_reference[:3,:],axis = 0).mean()
    
    new_state = state.copy()
    new_state[0] -= centroide_state[0]
    new_state[1] -= centroide_state[1]
    new_state[2] -= centroide_state[2]
    
    M = new_state[:3,:].T.reshape((n,1,3))
    D = new_reference[:3,:].T.reshape((n,3,1))
    H = D @ M
    H = H.sum(axis = 0)
    
    U, S, VH = svd(H)
    R = VH.T @ U.T
    
    #   Caso de Reflexión
    if np.linalg.det(R) < 0.:
        VH[2,:] = -VH[2,:]
        R = VH.T @ U.T
    
    #   Actualizando Orientación de traslaciones
    #   Para la visualización se optó por usar
    #   \bar p_i = R.T p_i en vez de \bar p^r_i = R p*_i
    new_state[:3,:] = R.T @ new_state[:3,:]
    
    #   Actualizando escala y Obteniendo error de traslaciones
    f = lambda r : (norm(new_reference[:3,:] - r*new_state[:3,:],axis = 0)**2).sum()/n
    r_state = minimize_scalar(f, method='brent')
    t_err = f(r_state.x)
    t_err = np.sqrt(t_err)
    
    new_state[:3,:] = r_state.x * new_state[:3,:]
    
    
    #   Actualizando rotaciones
    rot_err = np.zeros(n)
    if gdl ==1:
        for i in range(n):
            
            #   update new_state
            _R = R.T @ agents[i].camera.R
            [new_state[3,i], new_state[4,i], new_state[5,i]] = ctr.get_angles(_R)
            agents[i].camera.pose(new_state[:,i])
            
            #   Get error
            _R =  cm.rot(new_reference[3,i],'x') @ agents[i].camera.R.T
            _R = cm.rot(new_reference[4,i],'y') @ _R
            _R = cm.rot(new_reference[5,i],'z') @ _R
            _arg = (_R.trace()-1.)/2.
            if abs(_arg) < 1.:
                rot_err[i] = np.arccos(_arg)
            else:
                rot_err[i] = np.arccos(np.sign(_arg))
                
    else:
        #   Get error
        [Rtheta,Rphi,Rpsi] = ctr.get_angles(R)
        rot_err = state[5,:] - reference[5,:]
        rot_err -= Rpsi
        rot_err[rot_err < -pi] += 2*pi
        rot_err[rot_err > pi] -= 2*pi
        rot_err_plot = rot_err.copy()
        
        
    #   RMS
    rot_err = rot_err**2
    rot_err = rot_err.sum()/n
    rot_err = np.sqrt(rot_err)
    
    
    if name is None:
        #   recovering positions
        for i in range(n):
            agents[i].camera.pose(state[:,i])
        
        return [t_err, rot_err]
    
     ##   Plot
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #   TODO plot Z parallel for gdl == 2
    if gdl ==1 :
        for i in range(n):
            agents[i].camera.draw_camera(ax, scale=0.2, color='red')
            ax.text(agents[i].camera.p[0],agents[i].camera.p[1],agents[i].camera.p[2],str(i))
            agents[i].camera.pose(new_reference[:,i])
            agents[i].camera.draw_camera(ax, scale=0.2, color='brown')
            ax.text(agents[i].camera.p[0],agents[i].camera.p[1],agents[i].camera.p[2],str(i))
            agents[i].camera.pose(state[:,i])
    else:
        for i in range(n):
            #   new_state
            new_state[3:,i] = np.array([pi,0.,rot_err_plot[i]])
            agents[i].camera.pose(new_state[:,i])
            agents[i].camera.draw_camera(ax, scale=0.2, color='red')
            ax.text(agents[i].camera.p[0],agents[i].camera.p[1],agents[i].camera.p[2],str(i))
            
            #   new_reference
            new_reference[3:,i] = np.array([pi,0.,new_reference[5,i]])
            agents[i].camera.pose(new_reference[:,i])
            agents[i].camera.draw_camera(ax, scale=0.2, color='brown')
            ax.text(agents[i].camera.p[0],agents[i].camera.p[1],agents[i].camera.p[2],str(i))
            agents[i].camera.pose(state[:,i])
            
            
            
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    return [t_err, rot_err]


#   Difference between agents
#   It assumes that the states in the array are ordered and are the same
#   regresa un error de traslación y orientación
def error_state_equal(  agents, gdl = 1, name = None):
    
    n = len(agents)
    state = np.zeros((6,n))
    for i in range(n):
        state[:,i] = agents[i].camera.p
    
    reference = np.average(state,axis =1)
    
    t_err = reference[:3].reshape((3,1)) - state[:3,:]
    t_err =  (norm(t_err,axis = 0)**2).sum()/n
    rit_err = None
    if gdl == 1:
        rot_err = reference[3:].reshape((3,1)) - state[3:,:]
        rot_err =  (norm(rot_err,axis = 0)**2).sum()/n
    else:
        rot_err = reference[5] - state[5,:]
        rot_err =  (rot_err**2).sum()/n
    
    if name is None:
        return [t_err, rot_err]
    
    #   Plot
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for i in range(n):
        agents[i].camera.draw_camera(ax, scale=0.2, color='red')
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    return [t_err, rot_err]
    
################################################################################
################################################################################
#
#   Experiment Base
#
################################################################################
################################################################################

def experiment(directory = "0",
               lamb = 1.0,
               k_int = 0,
               int_res = None,
               depthOp=1,
               Z_set = 1.0,
               gdl = 1,
               dt = 0.05,
               t_end = 10.0,
               zOffset = 0.0,
               leader = None,
               case_interactionM = 1,
               adjMat = None,
               p0 = None,
               pd = None,
               P = np.array(SceneP), 
               gamma0 = None,
               gammaInf = None,
               gammaSteep = 5.,
               intGamma0 = None,
               intGammaInf = None,
               intGammaSteep = 5.,
               setRectification = False,
               set_derivative = False,
               atTarget = False,
               tanhLimit = False,
               midMarker = False,
               enablePlot = True,
               modified = [],
               repeat = False):
    
    #   Referencia de selecciones
    
    #Depth estimation for interaction matrix
    #   1-Updated, 2-Initial, 3-Referencia, 4-Arbitrary fixed, 5-Average
    depthOp_dict = {1:"Updated at each step",
                    2:"Distance at the begining",
                    3:"Distance between reference and points",
                    4:"Arbirtrary uniform value Z_set",
                    5:"Uniform value as the mean of all points",
                    6:"Height as depth"}
    
    #   interaction matrix used for the control
    #   1- computed each step 2- Computed at the end 3- average
    case_interactionM_dict = {1:"Computed at each step",
                              2:"Computed at the end",
                              3:"Average between 1 and 2"}
    
    #   Controladores
    #   1  - IBVC   2 - Montijano
    control_type = 1
    control_type_dict = {1:"Image based visual control",
                         2:"Montijano"}
    case_controlable_dict = {1:"6 Degrees of freedom",
                             2:"4 Degrees of freedom",
                             3:"3 Degrees of freedom"}
    
    set_consensoRef = True
    if repeat:
        npzfile = np.load(directory+'/data.npz')
        P = npzfile["P"]
        n_points = P.shape[1] #Number of image points
        p0 = npzfile["p0"]
        if (p0 == .0).all():
            set_consensoRef = False
        n_agents = p0.shape[1] #Number of agents
        pd = npzfile["pd"]
        if npzfile.__contains__('adjMat'):
            adjMat = npzfile["adjMat"]
        else:
            if adjMat is None:
                #   Make a complete graph
                adjMat = []
                for i in range(n_agents):
                    tmp = [1 for j in range(n_agents)]
                    tmp[i] = 0
                    adjMat.append(tmp)
                
        
        
        #   BEGIN point modification 
        if len(modified) > 0:
            if modified.__contains__("nPoints"):
                kP = 30
                #kP = 16
                xRange = [-2,2]
                yRange = [-2,2]
                zRange = [-1,0]
                # zRange = [-2,-1]
                P = np.random.rand(3,kP)
                P[0,:] = xRange[0] + P[0,:]*(xRange[1]-xRange[0])
                P[1,:] = yRange[0] + P[1,:]*(yRange[1]-yRange[0])
                P[2,:] = zRange[0] + P[2,:]*(zRange[1]-zRange[0])
                #O[2,:] = 0.0
                #P = np.r_[P,np.zeros((1,30)),np.ones((1,30))]
                P = np.r_[P,np.ones((1,kP))]
                n_points = P.shape[1]

            if modified.__contains__( "height"):
                p0[2,:] += 2.

            
        #   END point modification 
        
        #   BEGIN TEST
        
        #P[2,:] = 0.
        #p0[2,:] += 1.
        #pd[2,:] += 1.
        #pd[2,0] += 1.
        #pd[2,2] += 1.
        #P[:,1] = np.array([2.5,-1,-0.23240457,1])
        #P = np.c_[P,np.array([-0.9,0.5,-0.1,1]).reshape((4,1))]
        #n_points += 1
        #P[:,3] = np.array([-0.32326981])
        
        #p0[3,:] = pi
        #p0[4,:] = 0.
        #p0[5,:] = pi
        #p0[5,:] *= 0.8
        #p0[:3,:] = pd[:3,:].copy()
        
        #testAng =  np.pi/4
        #R = cm.rot(testAng,'y') 
        
        #P[:3,:] =  R @ P[:3]
        
        #for i in range(4):
            #_R = cm.rot(p0[5,i],'z') 
            #_R = _R @ cm.rot(p0[4,i],'y')
            #_R = _R @ cm.rot(p0[3,i],'x')
            
            #_R = R @ _R
            #[p0[3,i], p0[4,i], p0[5,i]] = ctr.get_angles(_R)
        #p0[:3,:] = R @ p0[:3,:]
        
        
        ##   rotando la referencia
        
        #for i in range(4):
            #_R = cm.rot(pd[5,i],'z') 
            #_R = _R @ cm.rot(pd[4,i],'y')
            #_R = _R @ cm.rot(pd[3,i],'x')
            
            #_R = R @ _R
            #[pd[3,i], pd[4,i], pd[5,i]] = ctr.get_angles(_R)
        #pd[:3,:] = R @ pd[:3,:]
        
        #   END TEST
    else:
        #   3D scenc points
        n_points = P.shape[1] #Number of image points
        P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
        
        #   Adjusto case of reference and initial conditions
        if p0 is None and pd is None:
            write2log("ERR: Reference and initial porsitions not provided\n")
            return
        elif p0 is None:
            set_consensoRef = True
            p0 = pd.copy()
        elif pd is None:
            set_consensoRef = False
            pd = np.zeros(p0.shape)
        n_agents = p0.shape[1] 
        
        
        #   Graphs
        if adjMat is None:
            #   Make a complete graph
            adjMat = []
            for i in range(n_agents):
                tmp = [1 for j in range(n_agents)]
                tmp[i] = 0
                adjMat.append(tmp)
        
        np.savez(directory+'/data.npz',
                 P = P,
                p0 = p0,
                pd = pd,
                adjMat = adjMat)
    
    #   Adjust p0. pd angles
    #print(pd)
    #print(p0)
    while (pd[3:,:] > pi).any():
        pd[3:,:][pd[3:,:]> pi] -= 2*pi
    while (pd[3:,:] < -pi).any():
        pd[3:,:][pd[3:,:]< -pi] += 2*pi
        
    while (p0[3:,:] > pi).any():
        p0[3:,:][p0[3:,:]> pi] -= 2*pi
    while (p0[3:,:] < -pi).any():
        p0[3:,:][p0[3:,:]< -pi] += 2*pi
    #print(pd)
    #print(p0)
    #print("out")
    #   Conectivity graph
    G = gr.graph(adjMat)
    L = G.laplacian()
    #G.plot()
    
    [U,S,V]=svd(L)
    lam_n=S[0]
    lam_2=S[-2]
    alpha=2./(lam_n+lam_2)
    A_ds=np.eye(n_agents)-alpha*L
    
    #   Agents array
    k_int_start = 1.
    if int_res is None:
        k_int_start = k_int
    else:
        k_int_start = 0.
    agents = []
    for i in range(n_agents):
        cam = cm.camera()
        agents.append(ctr.agent(cam,pd[:,i],p0[:,i],P,
                                k_int = k_int_start,
                                gamma0 = gamma0,
                                gammaInf = gammaInf,
                                gammaSteep = gammaSteep,
                                intGamma0 = intGamma0,
                                intGammaInf = intGammaInf,
                                intGammaSteep = intGammaSteep,
                                setRectification = setRectification,
                                set_consensoRef = set_consensoRef,
                                set_derivative = set_derivative))
    
    #   Check initial params 
    for i in range(n_agents):
        if agents[i].count_points_in_FOV(P) != n_points:
            write2log("ERR: invalid configuration\n")
            #return None
    
    #   INIT LOOP
    
    t=0
    steps = int((t_end-t)/dt + 1.0)
    #steps = 20
    #   case selections
    if case_interactionM > 1:
        for i in range(n_agents):
            #   Depth calculation
            Z = Z_select(depthOp, agents[i], P,Z_set,p0,pd,i)
            if Z is None:
                print("ERR: Invalid depth selection")
                return None
            agents[i].set_interactionMat(Z,gdl)
    
    if control_type == 2:
        delta_pref = np.zeros((n_agents,n_agents,6,1))
        for i in range(n_agents):
            for j in range(n_agents):
                delta_pref[i,j,:,0] = pd[:,j]-pd[:,i]
        gamma =  p0[2,:]
    
    depthFlags =  [0 for i in range(n_agents)]
    
    #   Storage variables
    #t_array = np.arange(t,t_end+dt,dt)
    t_array = np.linspace(t,t_end,steps)
    #t_array = np.arange(t,dt*steps,dt)
    err_array = np.zeros((n_agents,2*n_points,steps))
    serr_array = np.zeros((2,steps))
    U_array = np.zeros((n_agents,6,steps))
    
    desc_arr = np.zeros((n_agents,2*n_points,steps))
    pos_arr = np.zeros((n_agents,6,steps))
    #svdProy_p = np.zeros((n_agents,2*n_points,steps))
    svdProy = np.zeros((n_agents,2*n_points,steps))
    
    if gdl == 1:
        sinv_store = np.zeros((n_agents,6,steps))
        s_store = np.zeros((n_agents,6,steps))
    elif gdl == 2:
        sinv_store = np.zeros((n_agents,4,steps))
        s_store = np.zeros((n_agents,4,steps))
    elif gdl == 3:
        sinv_store = np.zeros((n_agents,3,steps))
        s_store = np.zeros((n_agents,3,steps))
    FOVflag = False
    state_err = error_state(pd,agents,gdl = gdl)
    
    #   Print simulation data:
    logText = "------------------BEGIN-----------------"
    logText += '\n' +"Laplacian matrix: "
    logText += '\n' +str(L)
    logText += '\n' +str(A_ds)
    if set_consensoRef:
        logText += '\n' +"Reference states"
        logText += '\n' +str(pd)
    else:
        logText += '\n' +"No reference states"
    logText += '\n' +"Initial states"
    logText += '\n' +str(p0)
    logText += '\n' +"Scene points"
    logText += '\n' +str(P[:3,:])
    logText += '\n' +"Number of points = "+str(n_points)
    logText += '\n' +"Number of agents = "+str(n_agents)
    if zOffset != 0.0:
        logText += '\n' +"Z offset for starting conditions = "+str(zOffset)
    logText += '\n' +"Time range = ["+str(t)+", "+str(dt)+", "+str(t_end)+"]"
    logText += '\n' +"\t Control lambda = "+str(lamb)
    if k_int != 0.:
        logText += '\n' +"\t Control Integral gain = "+str(k_int)
        if not int_res is None:
            logText += '\n' +"\t Control Integral refreshing treshold = "+str(int_res)
    if set_derivative:
        logText += '\n' +"\t Derivative component enabled"
    if not gamma0 is None:
        logText += '\n' +"\t Adaptative gain: ["+str(gamma0)+", "+ str(gammaInf)+","+str(gammaSteep)+"]"
    if not intGamma0 is None:
        logText += '\n' +"\t Adaptative gain: ["+str(intGamma0)+", "+ str(intGammaInf)+","+str(intGammaSteep)+"]"
    if tanhLimit:
        logText += '\n' +"Hyperbolic tangent limit enabled"
    logText += '\n' +"Depth estimation = "+depthOp_dict[depthOp]
    if depthOp == 4:
        logText += '\n' +"\t Estimated depth set at : "+str(Z_set)
    logText += '\n' +"Interaction matrix = "+case_interactionM_dict[case_interactionM]
    logText += '\n' +"Control selection = "+control_type_dict[control_type]
    logText += '\n' +"Controllable case = "+case_controlable_dict[gdl]
    logText += '\n' +"Directory = "+str( directory)
    if setRectification:
        logText += '\n' + "Rectification enabled"
        
    write2log(logText+'\n')
    
    
    
    
    #   LOOP
    for i in range(steps):
        
        
        #   Error:
        
        error = np.zeros((n_agents,2*n_points))
        
        for j in range(n_agents):
            error[j,:] = agents[j].error
        #print(error)
        error = L @ error
        
        #   save data
        #err_array[:,:,i] = error.copy()
        if set_derivative:
            for Li in range(n_agents):
                for Lj in range(n_agents):
                    if Li != Lj:
                        error[Li,:] -= L[Li,Lj]*agents[Lj].dot_s_current_n
        
        #   Leader 
        if not leader is None:
            error[leader,:] = agents[leader].error
        err_array[:,:,i] = error.copy()
        
        if set_consensoRef:
            [serr_array[0,i], serr_array[1,i]] = error_state(pd,agents, gdl = gdl)
        else:
            [serr_array[0,i], serr_array[1,i]] = error_state_equal(agents, gdl = gdl)
        
        ####   Image based formation
        if control_type ==2:
            H = ctr.get_Homographies(agents)
        #   Get control
        for j in range(n_agents):
            #print("agent "+str(j))
            #   Detección de choque del plano de cámara y los puntos de escena
            if agents[j].count_points_in_FOV(P, enableMargin=False) != n_points:
                if depthFlags[j] == 0. :
                    depthFlags[j] = i
                FOVflag = True
                break
            
            if not int_res is None:
                if norm(error[j,:])/n_points < int_res:
                    agents[j].k_int = k_int
                else:
                    agents[j].k_int = 0.
                    agents[j].reset_int()
            
            #   save data 
            desc_arr[j,:,i] = agents[j].s_current.T.reshape(2*n_points)
            pos_arr[j,:,i] = agents[j].camera.p.T.copy()
            
            #   Depth calculation
            Z = Z_select(depthOp, agents[j], P,Z_set,p0,pd,j)
            if Z is None:
                write2log("ERR: Z returns None\n")
                return None
            
            #   Control
            if control_type == 1:
                args = {"deg":G.deg[j] , 
                        "control_sel":case_interactionM,
                        "error": error[j,:],
                        "gdl":gdl,
                        "dt":dt}
            elif control_type == 2:
                args = {"H" : H[j,:,:,:],
                        "delta_pref" : delta_pref[j,:,:,:],
                        "Adj_list":G.list_adjacency[j][0],
                        "gamma": gamma[j]}
            else:
                write2log("ERR: invalid control selection\n")
                return None
            
            U  = agents[j].get_control(control_type,lamb,Z,args)
            sinv_store[j,:,i] = agents[j].s_inv
            s_store[j,:,i] = agents[j].s
            if tanhLimit:
                U = np.tanh(U)
            
            #U *= np.array([.5,.5,.5,1.,1.,1.]) 
            #if t<2:
                #U *= np.array([0,0,0,1.,1.,1.]) 
            #else:
                #U *= np.array([1.,1.,1.,0,0,0]) 
            
            #   Perturbaciones en U
            #U[1] -= 0.1*(np.sin(t/5))*min(norm(error[j,:]),1.)
            #U[5] -= 0.1
            
            #   Save error proyection in SVD:
            svdProy[j,:,i] = agents[j].vh_inv@error[j,:]/norm(error[j,:])
            
            if U is None:
                write2log("ERR: Invalid U control\n")
                return None
            
            
            U_array[j,:,i] = U
            #if j == 0:
            agents[j].update(U,dt,P, Z)
        
        #   Update
        t += dt
        
        if control_type ==2:
            gamma = A_ds @ gamma #/ 10.0
        
        if FOVflag:
            if i-1 >= 0:
                err_array[:,:,i] = err_array[:,:,i-1]
                serr_array[:,i] = serr_array[:,i-1]
                for idx in range(j,n_agents):
                    U_array[idx,:,i] = U_array[idx,:,i-i]
                    desc_arr[idx,:,i] = desc_arr[idx,:,i-1]
                    pos_arr[idx,:,i] = pos_arr[idx,:,i-1]
                    sinv_store[idx,:,i] = sinv_store[idx,:,i-1]
                    s_store[idx,:,i] = s_store[idx,:,i-1]
                    svdProy[idx,:,i] = svdProy[idx,:,i-1]
            break
        
    
    ##  Final data
    
    logText = "----------------------"
    logText += '\n' +"Simulation final data"
    
    #ret_err = norm(error_p,axis=1)/n_points
    ret_err = norm(error,axis=1)/n_points
    #ret_err = norm(err_array[:,:,-1],axis=1)/n_points
    for j in range(n_agents):
        logText += '\n' +"|Error_"+str(j)+"|= "+str(ret_err[j])
        logText += '\n \t' +"Error_"+str(j)
        logText += '\n' +str(error[j,:])
        #logText += '\n' +str(err_array[j,:,-1])
    for j in range(n_agents):
        logText += '\n' +"X_"+str(j)+" = "+str(agents[j].camera.p)
    for j in range(n_agents):
        logText += '\n' +"V_"+str(j)+" = "+str(U_array[j,:,-1])
    logText += '\n' +"CMRS = "+str(norm(ret_err)/n_agents)
    logText += '\n' +"Mean inital heights = "+str(np.mean(p0[2,:]))
    logText += '\n' +"Mean camera heights = " +str(np.mean(np.r_[p0[2,:],pd[2,:]]))
    
    write2log(logText+'\n')
    
    ##  Trim data if needed
    if FOVflag:
        trim = max(depthFlags)
        if trim ==0:
            return [ret_err, [0.,0.,0.,0.], FOVflag]
        trim += 1
        t_array = t_array[:trim] 
        err_array = err_array[:,:,:trim]
        serr_array = serr_array[:,:trim]
        U_array = U_array[:,:,:trim] 
        desc_arr = desc_arr[:,:,:trim]
        pos_arr = pos_arr[:,:,:trim]
        svdProy = svdProy[:,:,:trim]
        s_store = s_store[:,:,:trim]
        sinv_store = sinv_store[:,:,:trim]
    logText = "Valores propios (Matrix de Interacción) = \n"
    logText += str(s_store[:,:,-1]) + '\n'
    logText += "Valores propios (Matrix de Interacción Inversa) = \n"
    logText += str(sinv_store[:,:,-1]) + '\n'
    logText += "Proy. Error sobrevectores propios de la inversa) = \n"
    logText += str(svdProy[:,:6,-1]) + '\n'
    write2log(logText+'\n')
    
    ####   Plot
    
    
    #   Space errors
    new_agents = []
    end_position = np.zeros((n_agents,6))
    for i in range(n_agents):
        cam = cm.camera()
        end_position[i,:] = agents[i].camera.p.copy()
        new_agents.append(ctr.agent(cam,pd[:,i],end_position[i,:],P))
    if set_consensoRef:
        state_err +=  error_state(pd,new_agents, gdl = gdl, 
                                  name = directory+"/3D_error")
    else:
        state_err +=  error_state_equal(new_agents, gdl = gdl,
                                        name = directory+"/3D_error")
    
    logText = 'Errores de consenso iniciales\n'
    logText += str(err_array[:,:,0])
    logText += '\n' +str(err_array[:,:,0].max())
    
    logText += '\n' +"State error = "+str(state_err)
    if FOVflag:
        logText += '\n' +"WARNING : Cammera plane hit scene points: "+str( depthFlags)
    if (pos_arr[:,2,:] < 0.).any():
        logText += '\n' +"WARNING : Possible plane colition to ground"
    logText += '\n' +"-------------------END------------------"
    
    write2log(logText+'\n'+'\n')
    
    # Colors setup
    #n_colors = max(n_agents,2*n_points)
    #colors = randint(0,255,3*n_colors)/255.0
    #colors = colors.reshape((n_colors,3))
    npzfile = np.load("general.npz")
    colors = npzfile["colors"]
    
    #   Save 3D plot data:
    np.savez(directory + "/data3DPlot.npz",
            P = P, pos_arr=pos_arr, p0 = p0,
            pd = pd, end_position = end_position )
    
    if not enablePlot:
        return [ret_err, state_err, FOVflag]
    
    #   Camera positions in X,Y 
    mp.plot_position(pos_arr,
                    pd,
                    lfact = 1.1,
                    #colors = colors,
                    name = directory+"/Cameras_trayectories")
    
    #   3D plots
    fig, ax = plt.subplots(ncols = 2,
                           frameon=False,
                           figsize=(8,6),
                            gridspec_kw={'width_ratios': [3,1]})
    #fig = plt.figure(frameon=False, figsize=(5,3))
    ax[0].axis('off')
    ax[0] = fig.add_subplot(1, 2, 1, projection='3d')
    name = directory+"/3Dplot"
    #axplot(P[0,:], P[1,:], P[2,:], 'o')
    ax[0].scatter(P[0,:], P[1,:], P[2,:], s = 20)
    if gdl ==1:
        for i in range(n_agents):
            plot_3Dcam(ax[0], agents[i].camera,
                        pos_arr[i,:,:],
                        p0[:,i],
                        end_position[i,:],
                        pd[:,i],
                        color = colors[i],
                        i = i,
                        camera_scale    = 0.06)
    elif gdl == 2:
        for i in range(n_agents):
            init = p0[:,i]
            end = end_position[i,:]
            ref = pd[:,i]
            init[3] = end[3] = ref[3] = pi
            init[4] = end[4] = ref[4] = 0
            plot_3Dcam(ax[0], agents[i].camera,
                        pos_arr[i,:,:],
                        init,
                        end,
                        ref,
                        color = colors[i],
                        i = i,
                        camera_scale    = 0.02)
            
    #   Plot cammera position at intermediate time
    if midMarker:
        step_sel = int(pos_arr.shape[2]/2)
        ax[0].scatter(pos_arr[:,0,step_sel],
                pos_arr[:,1,step_sel],
                pos_arr[:,2,step_sel],
                    marker = '+',s = 200, color = 'black')
    
    lfact = 1.1
    x_min = min(p0[0,:].min(),
                pd[0,:].min(),
                pos_arr[:,0,-1].min(),
                P[0,:].min())
    x_max = max(p0[0,:].max(),
                pd[0,:].max(),
                pos_arr[:,0,-1].max(),
                P[0,:].max())
    y_min = min(p0[1,:].min(),
                pd[1,:].min(),
                pos_arr[:,1,-1].min(),
                P[1,:].min())
    y_max = max(p0[1,:].max(),
                pd[1,:].max(),
                pos_arr[:,1,-1].max(),
                P[1,:].max())
    z_max = max(p0[2,:].max(),
                pd[2,:].max(),
                pos_arr[:,2,-1].max(),
                P[2,:].max())
    z_min = min(p0[2,:].min(),
                pd[2,:].min(),
                pos_arr[:,2,-1].min(),
                P[2,:].min())
    
    width = x_max - x_min
    height = y_max - y_min
    depth = z_max - z_min
    sqrfact = max(width,height,depth)
    
    x_min -= (sqrfact - width )/2
    x_max += (sqrfact - width )/2
    y_min -= (sqrfact - height )/2
    y_max += (sqrfact - height )/2
    z_min -= (sqrfact - depth )/2
    z_max += (sqrfact - depth )/2
    ax[0].set_xlim(x_min,x_max)
    ax[0].set_ylim(y_min,y_max)
    ax[0].set_zlim(z_min,z_max)
    
    #ax = fig.add_subplot(1, 2, 2)
    symbols = []
    labels = [str(i) for i in range(n_agents)]
    for i in range(n_agents):
        symbols.append(mpatches.Patch(color=colors[i%colors.shape[0]]))
    ax[1].legend(symbols,labels, loc=7)
    ax[1].axis('off')
    
    #fig.legend( loc=1)
    plt.savefig(name+'.pdf')#,bbox_inches='tight')
    plt.close()
    
    #   Descriptores x agente
    #       Predicted endpoints
    pred = np.zeros((n_agents,2*n_points))
    for i in range(n_agents):
        pred[i,:] = agents[i].s_ref.T.reshape(2*n_points) - desc_arr[i,:,0]
    avrE = pred.mean(axis = 0)
    for i in range(n_agents):
        pred[i,:] = desc_arr[i,:,0] + ( pred[i,:] - avrE)
    
    for i in range(n_agents):
        mp.plot_descriptors(desc_arr[i,:,:],
                            agents[i].camera.iMsize,
                            agents[i].s_ref,
                            #colors,
                            pred[i,:],
                            #enableLims = False,
                            name = directory+"/Image_Features_"+str(i),
                            label = "Image Features")
    
    #   Error de formación
    mp.plot_time(t_array,
                serr_array[:,:],
                #colors,
                ylimits = [-0.1,1.1],
                ref = 0.1,
                refLab = "Threshold",
                name = directory+"/State_Error",
                label = "Formation Error",
                labels = ["translation","rotation"])
    
    #   Errores x agentes
    for i in range(n_agents):
        mp.plot_time(t_array,
                    err_array[i,:,:],
                    #colors,
                    ylimits = [-1,1],
                    name = directory+"/Features_Error_"+str(i),
                    label = "Features Error")
    
    #   Errores x agentes
    tmp = norm(err_array,axis = 1) / n_agents
    mp.plot_time(t_array,
                tmp,
                #colors,
                ref = int_res,
                refLab = "Integral threshold",
                ylimits = [-1,1],
                name = directory+"/Norm_Feature_Error",
                label = "Features Error")
    
    #   Velocidaes x agente
    for i in range(n_agents):
        mp.plot_time(t_array,
                    U_array[i,:,:],
                    #colors,
                    #ylimits = [-1,1],
                    ylimits = [-.1,.1],
                    name = directory+"/Velocidades_"+str(i),
                    label = "Velocities",
                    labels = ["X","Y","Z","Wx","Wy","Wz"])
    
    #   Posiciones x agente
    for i in range(n_agents):
        mp.plot_time(t_array,
                    pos_arr[i,:3,:],
                    #colors,
                    name = directory+"/Traslaciones_"+str(i),
                    label = "Translations",
                    labels = ["X","Y","Z"])
        mp.plot_time(t_array,
                    pos_arr[i,3:,:],
                    #colors,
                    name = directory+"/Angulos_"+str(i),
                    label = "Angles",
                    labels = ["Roll","Pitch","yaw"],
                    module = [-pi,pi])
    
    #   Valores propios normal
    for i in range(n_agents):
        mp.plot_time(t_array,
                    s_store[i,:,:],
                    #colors,
                    ylimits = [.0,10.1],
                    name = directory+"/ValoresPR_"+str(i),
                    label = "Non zero singular value magnitudes",
                    labels = ["0","1","2","3","4","5"])
    
    #   Valores propios inversa
    for i in range(n_agents):
        mp.plot_time(t_array,
                    sinv_store[i,:,:],
                    #colors,
                    ylimits = [.0,10.1],
                    name = directory+"/ValoresP_"+str(i),
                    label = "Non zero singular value magnitudes",
                    labels = ["0","1","2","3","4","5"])
                    #limits = [[t_array[0],t_array[-1]],[0,20]])
    
    
    
    #   Proyección de error sobre vectores propios
    #labels = [str(i) for i in range(2*n_points)]
    labels =  ["0","1","2","3","4","5"]
    for i in range(n_agents):
        mp.plot_time(t_array,
                    #svdProy[i,:,:],
                    svdProy[i,:6,:],
                    #colors,
                    name = directory+"/Proy_et_VH_"+str(i),
                    label = "Proy($e$) over $V^h$ ",
                    labels = labels)
                    #limits = [[t_array[0],t_array[-1]],[0,20]])
    
    return [ret_err, state_err, FOVflag]



################################################################################
################################################################################
#
#   Experiment Series
#
################################################################################
################################################################################

###########################################################################
#
#   Experiment random references and initial conditions
#
###########################################################################

def scene(modify = [],
          inP = None,
          inpd = None,
          inp0 = None,
          dirBase = "",
          n_agents = 4, 
          n_points = 30,
          refCirc = None,
          Px=[-2,2],
            Py=[-2,2],
            Pz=[-1,0],
            X=[-2,2],
            Y=[-2,2],
            Z=[0,3],
            roll=[pi,pi],
            pitch=[0,0],
            yaw=[-pi,pi]):
    
    P = None
    pd = None
    p0 = None
    
    name = dirBase.rstrip('/')+'/data.npz'
    if os.path.isfile(name):
        npzfile = np.load(name)
        if npzfile.__contains__('P'):
            P = npzfile['P']
        if npzfile.__contains__('pd'):
            pd = npzfile['pd']
        if npzfile.__contains__('p0'):
            p0 = npzfile['p0']
    else:
        
        modify = ['P','pd','p0']
    #print("log P")
    if modify.__contains__("P"):
        if inP is None:
            P = np.random.rand(3,n_points)
            P[0,:] = Px[0]+ P[0,:]*(Px[1]-Px[0])
            P[1,:] = Py[0]+ P[1,:]*(Py[1]-Py[0])
            P[2,:] = Pz[0]+ P[2,:]*(Pz[1]-Pz[0])
            P =  np.r_[P,np.ones((1,n_points))]
        else:
            P = inP.copy()
    #print("log Pd")
    if modify.__contains__("pd"):
        if inpd is None:
            pd = np.zeros((6,n_agents))
            offset = np.array([X[0],X[0],Z[0],roll[0],pitch[0],yaw[0]])
            dRange = np.array([X[1],X[1],Z[1],roll[1],pitch[1],yaw[1]])
            dRange -= offset
            
            
            
            for i in range(n_agents):
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)

                while agent.count_points_in_FOV(P) != n_points :
                    tmp = np.random.rand(6)
                    tmp = offset + tmp*dRange
                    cam = cm.camera()
                    agent = ctr.agent(cam, tmp, tmp,P)

                pd[:,i] = tmp.copy()
        else:
            pd = inpd.copy()
    #print("log P0")
    if modify.__contains__("p0"):
        if inp0 is None:
            p0 = np.zeros((6,n_agents))
            offset = np.array([X[0],X[0],Z[0],roll[0],pitch[0],yaw[0]])
            dRange = np.array([X[1],X[1],Z[1],roll[1],pitch[1],yaw[1]])
            dRange -= offset
            #print(dRange)
            
            
            for i in range(n_agents):
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)

                while agent.count_points_in_FOV(P) != n_points :
                    tmp = np.random.rand(6)
                    tmp = offset + tmp*dRange
                    cam = cm.camera()
                    agent = ctr.agent(cam, tmp, tmp,P)

                p0[:,i] = tmp.copy()
        else :
            p0 = inp0.copy()
    
    #   Adjust P when needed
    agents = []
    for i in range(n_agents):
        cam = cm.camera()
        agents.append(ctr.agent(cam,pd[:,i],p0[:,i],P))
    
    testing = False
    for i in range(n_agents):
        if agents[i].count_points_in_FOV(P) != n_points:
            testing = True
    #print("log")
    while testing:
        testing = False
        
        P = np.random.rand(3,n_points)
        P[0,:] = Px[0]+ P[0,:]*(Px[1]-Px[0])
        P[1,:] = Py[0]+ P[1,:]*(Py[1]-Py[0])
        P[2,:] = Pz[0]+ P[2,:]*(Pz[1]-Pz[0])
        P =  np.r_[P,np.ones((1,n_points))]
        
        for i in range(n_agents):
            if agents[i].count_points_in_FOV(P) != n_points:
                testing = True
    
    #write2log(name + '\n')
    np.savez(name,
            P = P,
            p0 = p0,
            pd = pd)
    return




def colorPalette(name = "colors.npz", n_colors = 30):
    
    colors = randint(0,255,3*n_colors)/255.0
    colors = colors.reshape((n_colors,3))
    
    np.savez(name, colors = colors)
    return




def experiment_frontParallel(nReps = 1,
                     t_end = 800,
                     intGamma0 = None,
                    intGammaInf = None,
                    intGammaSteep = 5.,
                    gamma0 = None,
                    gammaInf = None,
                    gammaSteep = 5.,
                     n_points = None,
                    dirBase = "",
                    node = 0,
                    gdl = 1,
                    Flat = True,
                    setRectification = False,
                    repeat = False,
                    enablePlotExp = False):
    
    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    arr_n_points = np.zeros(nReps)
    mask = np.zeros(nReps)
    #Misscount = 0
    
    Pz = [-1,0]
    if Flat:
        Pz = [0,0]
    
    logText = '\n' +"Local test directory = "+ str(dirBase)
    write2log(logText + '\n')
    
    for k in range(nReps):
        
        #   Modifica aleatoriamente lospuntos de escena
        if not repeat:
            nP = None
            if n_points is None:
                nP = random.randint(4,11)
            else:
                nP = n_points
            scene(modify = ["P"],
                Pz = Pz,
                    dirBase = dirBase+str(k+node*nReps),
                    n_agents = 4, 
                    #n_points = random.randint(4,11))
                    n_points = nP)
        
        write2log("CASE "+str(k)+'\n')
        ret = experiment(directory=dirBase+str(node*nReps+k),
                         gamma0 = gamma0,
                    gammaInf = gammaInf,
                    gammaSteep = gammaSteep ,
                    intGamma0 = intGamma0,
                    intGammaInf = intGammaInf,
                    intGammaSteep = intGammaSteep ,
                         repeat = True,
                         gdl = gdl,
                        t_end = t_end,
                        setRectification = setRectification,
                        enablePlot = enablePlotExp)
        #print(ret)
        [var_arr[:,k], errors, FOVflag] = ret
        var_arr_et[k] = errors[0]
        var_arr_er[k] = errors[1]
        var_arr_2[k] = errors[2]
        var_arr_3[k] = errors[3]
        if FOVflag:
            mask[k] = 1
        
    #   Save data
    
    np.savez(dirBase+'data_'+str(node)+'.npz',
                n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            arr_n_points = arr_n_points,
            mask = mask)
            #Misscount = Misscount)

#   BEGIN
def experiment_repeat(nReps = 1,
                     t_end = 100,
                    dirBase = "",
                    setRectification = False,
                    gdl = 1,
                    leader = None,
                    intGamma0 = None,
                    intGammaInf = None,
                    intGammaSteep = 5.,
                    gamma0 = None,
                    gammaInf = None,
                    gammaSteep = 5.,
                    enablePlotExp = True):
    
    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    arr_n_points = np.zeros(nReps)
    mask = np.zeros(nReps)
    #Misscount = 0
    
    logText = '\n' +"Local test directory = "+ str(dirBase)
    write2log(logText + '\n')
    
    for k in range(nReps):
        
        write2log("CASE "+str(k)+'\n')
        ret = experiment(directory=dirBase+str(k),
                    gamma0 = gamma0,
                    gammaInf = gammaInf,
                    gammaSteep = gammaSteep ,
                    intGamma0 = intGamma0,
                    intGammaInf = intGammaInf,
                    intGammaSteep = intGammaSteep ,
                    setRectification = setRectification,
                    gdl = gdl,
                    leader = leader,
                    #set_derivative = True,
                    #tanhLimit = True,
                    #depthOp = 4, Z_set = 1.,
                    t_end = t_end,
                    repeat = True,
                    enablePlot = enablePlotExp)
        #print(ret)
        [var_arr[:,k], errors, FOVflag] = ret
        var_arr_et[k] = errors[0]
        var_arr_er[k] = errors[1]
        var_arr_2[k] = errors[2]
        var_arr_3[k] = errors[3]
        if FOVflag:
            mask[k] = 1
            #Misscount += 1
        
    #   Save data
    
    np.savez(dirBase+'data.npz',
                n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            arr_n_points = arr_n_points,
            mask = mask)
            #Misscount = Misscount)
    
#   END

def experiment_all_random(nReps = 100,
                          conditions = 1,
                          pFlat = False,
                          dirBase = "",
                          nPconst = None,
                          enablePlotExp = True):

    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    arr_n_points = np.zeros(nReps)
    mask = np.zeros(nReps)
    nP = 0
    #Misscount = 0

    conditionsDict = {1:"Random",
                      2:"Circular plus a rigid transformation",
                      3:"Circular, fixed",
                      4:"Circular, plus a perturbation"}

    logText = "Test series, References = "+str( conditionsDict[conditions])
    logText += '\n' +"Test series, pFlat = "+str( pFlat)
    logText += '\n' +"Test series, nP conf = "+str( nPconst)
    write2log(logText+'\n')

    for k in range(nReps):
        #   Points
        # Range
        xRange = [-2,2]
        yRange = [-2,2]
        zRange = [-1,0]

        if nPconst is None :
            nP = random.randint(4,11)
        else:
            nP = nPconst
        arr_n_points[k] = nP
        #nP = 4
        #nP = 10
        P = np.random.rand(3,nP)
        P[0,:] = xRange[0]+ P[0,:]*(xRange[1]-xRange[0])
        P[1,:] = yRange[0]+ P[1,:]*(yRange[1]-yRange[0])
        P[2,:] = zRange[0]+ P[2,:]*(zRange[1]-zRange[0])
        if pFlat :
            P[2,:] = 0.
        Ph = np.r_[P,np.ones((1,nP))]

        #   Cameras
        # Range
        xRange = [-2,2]
        yRange = [-2,2]
        zRange = [0,3]

        p0 = np.zeros((6,n_agents))
        pd = np.zeros((6,n_agents))

        offset = np.array([xRange[0],yRange[0],zRange[0],-np.pi,-np.pi,-np.pi])
        dRange = np.array([xRange[1],yRange[1],zRange[1],np.pi,np.pi,np.pi])
        dRange -= offset
        for i in range(n_agents):

            tmp = np.random.rand(6)
            tmp = offset + tmp*dRange
            tmp[3] = pi
            tmp[4] = 0.
            cam = cm.camera()
            agent = ctr.agent(cam, tmp, tmp,P)

            while agent.count_points_in_FOV(Ph) != nP :
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                tmp[3] = pi
                tmp[4] = 0.
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)

            p0[:,i] = tmp.copy()

            #   BEGIN referencias aleatorias
            if conditions == 1:
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                tmp[3] = pi
                tmp[4] = 0.
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)

                while agent.count_points_in_FOV(Ph) != nP:
                    tmp = np.random.rand(6)
                    tmp = offset + tmp*dRange
                    tmp[3] = pi
                    tmp[4] = 0.
                    cam = cm.camera()
                    agent = ctr.agent(cam, tmp, tmp,P)

                pd[:,i] = tmp.copy()
            #   END Referencias aleatorias
        ##   BEGIN referencias fijas, + s T R
        #if conditions == 2:
            #visibilityTest = False
            ##pd = None
            #while not visibilityTest:
                #T = 4*(np.random.rand(3)-np.array([.5,.5,0.]))
                #angs = 2*np.pi*(np.random.rand(3)-0.5)
                #s = 2*(np.random.rand(1)-0.5)
                #pd = circle(n_agents,s,T = T , angs = angs)
                #visibilityTest = True
                #for i in range(n_agents):
                    #cam = cm.camera()
                    #agent = ctr.agent(cam, tmp, pd[:,i],P)
                    #visibilityTest = visibilityTest and (agent.count_points_in_FOV(Ph) == nP)


        ##   END Referencias fijas s R T

        ##  BEGIN Referencias fijas
        if conditions == 3:
            pd = circle(n_agents,1,T = np.array([0.,0.,1.]))
        ##  END Referencias fijas

        ##   BEGIN referencias con perturbación
        #if conditions == 4:
            #pd = circle(n_agents,1,T = np.array([0.,0.,1.]))
            #dRange = np.array([0.1,0.1,0.1,0.1,0.1,0.1])
            #for i in range(n_agents):
                #tmp = np.random.rand(6)-0.5
                #tmp = pd[:,i] + tmp*dRange*2
                #cam = cm.camera()
                #agent = ctr.agent(cam, tmp, tmp,P)

                #while agent.count_points_in_FOV(Ph) != nP:
                    #tmp = np.random.rand(6)-0.5
                    #tmp = pd[:,i] + tmp*dRange*2
                    #cam = cm.camera()
                    #agent = ctr.agent(cam, tmp, tmp,P)
                #pd[:,i] = tmp.copy()
        ##   END Referencias con perturbación


        write2log("CASE "+str(k)+'\n')
        ret = experiment(directory=dirBase+str(k),
                #k_int = 0.1,
                    pd = pd,
                    p0 = p0,
                    P = P,
                    #set_derivative = True,
                    #tanhLimit = True,
                    #depthOp = 4, Z_set = 1.,
                    t_end = 100,
                    enablePlot = enablePlotExp)
        #print(ret)
        [var_arr[:,k], errors, FOVflag] = ret
        var_arr_et[k] = errors[0]
        var_arr_er[k] = errors[1]
        var_arr_2[k] = errors[2]
        var_arr_3[k] = errors[3]
        if FOVflag:
            mask[k] = 1

    #   Save data

    np.savez(dirBase+'data.npz',
                n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            arr_n_points = arr_n_points,
            mask = mask)
            #Misscount = Misscount)

def experiment_3D_min(nReps = 100,
                          dirBase = "",
                          node = 0,
                          sel_case = 0,
                          enablePlotExp = True):

    gamma0 = None
    gammaInf = None
    intGamma0 = None
    intGammaInf = None
    if sel_case%2 == 1:
        gamma0 = 3.
        gammaInf = .1
        intGamma0 = 0.1
        intGammaInf = 0.01
    
    
    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    arr_nP = np.zeros(nReps)
    arr_n_points = np.zeros(nReps)
    arr_trial_heat_t = -np.ones((nReps,100-4))
    arr_trial_heat_ang = -np.ones((nReps,100-4))
    arr_trial_heat_cons = -np.ones((nReps,100-4))
    mask = np.zeros(nReps)
    nP = 4
    #Misscount = 0


    logText = "Test series, minimun 3D points required = "
    write2log(logText+'\n')

    for k in range(nReps):

        #   Points
        nP = 4
        scene(modify = ["P"],
                dirBase = dirBase+str(k+node*nReps),
                n_points = nP)


        errors = [0.2,0.2,0.2,0.2]
        ret = None
        
        while((errors[2] > 0.1 or errors[3]> 0.1) and nP < 100):


            write2log("CASE "+str(k)+'\n')
            t = time.localtime()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", t)
            write2log("TIMESTAMP "+current_time+'\n')
            ret = experiment(directory=dirBase+str(k+node*nReps),
                        repeat = True,
                        gamma0 = gamma0,
                        gammaInf = gammaInf,
                        intGamma0 = intGamma0,
                        intGammaInf = intGammaInf,
                        #set_derivative = True,
                        #tanhLimit = True,
                        #depthOp = 4, Z_set = 1.,
                        t_end = 800,
                        enablePlot = enablePlotExp)
            if ret is None:
                write2log("ERR: ret out of experiment\n")
                return None
            [arr_tmp, errors, FOVflag] = ret
            
            #   si no falla, guardar datos de error
            if not FOVflag:
                arr_trial_heat_t[k,nP-4] = errors[2]
                arr_trial_heat_ang[k,nP-4] = errors[3]
                arr_trial_heat_cons[k,nP-4] = norm(arr_tmp)

            if (errors[2] > 0.1 or errors[3]> 0.1):
                #   Points
                arr_n_points[k] = nP
                nP += 1

                scene(modify = ["P"],
                    dirBase = dirBase+str(k+node*nReps),
                    n_points = nP)
        if ret is None:
            write2log("ERR: ret out of while at "+str(errors)+" " +str(nP)+'\n')
        else:    
            [var_arr[:,k], errors, FOVflag] = ret
            var_arr_et[k] = errors[0]
            var_arr_er[k] = errors[1]
            var_arr_2[k] = errors[2]
            var_arr_3[k] = errors[3]
            arr_nP[k] = nP
        if FOVflag:
            mask[k] = 1

    #   Save data

    np.savez(dirBase+'data_'+str(node)+'.npz',
                n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            arr_n_points = arr_n_points,
            mask = mask,
             arr_nP= arr_nP,
             arr_trial_heat_t = arr_trial_heat_t,
             arr_trial_heat_ang = arr_trial_heat_ang,
             arr_trial_heat_cons = arr_trial_heat_cons)


def plot_minP_data(dirBase, nodos, nReps, colorFile = "default.npz"):

    data = np.array([])
    mask = np.array([])
    arr_trial_heat_t = np.zeros((1,100-4))
    arr_trial_heat_ang = np.zeros((1,100-4))
    arr_trial_heat_cons = np.zeros((1,100-4))
    for k in range(nodos):
        name=dirBase.rstrip('/')+'/data_'+str(k)+'.npz'
        npzfile = np.load(name)
        data = np.concatenate((data,npzfile["arr_nP"]))
        mask = np.concatenate((mask,npzfile["mask"]))
        arr_trial_heat_t = np.r_[arr_trial_heat_t,npzfile["arr_trial_heat_t"]]
        arr_trial_heat_ang = np.r_[arr_trial_heat_ang,npzfile["arr_trial_heat_ang"]]
        arr_trial_heat_cons = np.r_[arr_trial_heat_cons,npzfile["arr_trial_heat_cons"]]
    
    arr_trial_heat_t = arr_trial_heat_t[1:,:]
    arr_trial_heat_ang = arr_trial_heat_ang[1:,:]
    arr_trial_heat_cons = arr_trial_heat_cons[1:,:]
    
    #   Masking
    if (np.count_nonzero(mask == 1.) == mask.shape[0]):
        print("Simulations : No data to process")
        return
    
    write2log("\nnreps = "+str(data))
    failed = np.count_nonzero(data == 100)
    write2log("\n96 reps = "+str(failed)+'\n')
    data = data[mask==0.]
    
    npzfile = np.load(colorFile)
    colors = npzfile["colors"]
    
    ##  Histogram min points
    fig, ax = plt.subplots()
    fig.suptitle("Number of 3D points needed histogram")
    #counts, bins = np.histogram(data)
    #plt.stairs(counts, bins)
    plt.hist(data, edgecolor = colors[0], fill = False)
    plt.tight_layout()
    plt.savefig(dirBase+'minPoints_Histogram.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()

    ##  Histogram min points zoom
    fig, ax = plt.subplots()
    fig.suptitle("Number of 3D points needed histogram")
    
    filtered = data[data != 100]
    #nbins = filtered.max() - filtered.min() +1
    #nbins = int(nbins)
    #binrange = (filtered.min()-0.5, filtered.max()+0.5)
    nbins = 51
    binrange = (3.5, 50.5)
    
    plt.hist(data,
             edgecolor = colors[0],
             fill = False,
             bins = nbins ,
             range = binrange)
             #bins = 47 ,
             #range = (3.5,50.5))
    plt.xticks(np.arange(4, 51, 2.0))
    #plt.xticks(np.arange(int(filtered.min()), int(filtered.max())+1, 2.0))
    ax.grid(axis='x')
    plt.tight_layout()
    plt.savefig(dirBase+'minPoints_Histogram_zoom.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   Heatmaps de errores p/experimento

    #       previous end detect
    h1 = arr_trial_heat_t.copy()
    h1[h1 < 0] = 100
    h2 = arr_trial_heat_ang.copy()
    h2[h2 < 0] = 100
    test = - np.ones(100, dtype = np.int16)
    test2 = - np.ones(100, dtype = np.int16)
    for i in range(nodos*nReps):
        for j in range(96):
            if h1[i,j] < 0.1 and h2[i,j] < 0.1:
                test[i] = j
            if h1[i,j] < 0.2 and h2[i,j] < 0.2 and test2[i] <0:
                test2[i] = j
    
    #       Traslacion
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Trial errors (T) heatmap", size = 9)
    
    h = arr_trial_heat_t.copy()
    h[h < 0] = 0
    ax.invert_yaxis()
    x = [str(i) for i in range(100-4)]
    y = [str(i) for i in range(100)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize = 2.5)
    ax.set_yticklabels(y, fontsize = 2.5)
    ax.set_xlabel("Trial", size =4)
    ax.set_ylabel("Translation error", size = 4)
    ax.grid(axis='y', linewidth=0.1)
    ax.set_axisbelow(True)
    im = ax.imshow(h, #norm=mcolors.PowerNorm(gamma=0.5),
                   cmap = "Purples", vmax = 1.1, vmin = 0)
    fig.colorbar(im, ax = ax, shrink = 0.8)
    for j in range(len(y)):
        if test[j] != -1:
            text = ax.text( test[j], j , 
                           "X", ha="center", va="center", 
                           color="k", fontsize =4)
            text = ax.text( test2[j], j , 
                           "X", ha="center", va="center", 
                           color="g", fontsize =4)
        for i in range(len(x)):
            if arr_trial_heat_t[j,i] < 0.:
                text = ax.text( i, j , 
                           "-", ha="center", va="center", 
                           color="r", fontsize =4)
    
    plt.tight_layout()
    plt.savefig(dirBase+'ErrorT_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #       Traslacion
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Trial errors (R) heatmap", size = 9)
    
    h = arr_trial_heat_ang.copy()
    h[h < 0] = 0
    ax.invert_yaxis()
    x = [str(i) for i in range(100-4)]
    y = [str(i) for i in range(nodos*nReps)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize = 2.5)
    ax.set_yticklabels(y, fontsize = 2.5)
    ax.set_xlabel("Trial", size =4)
    ax.set_ylabel("Rotation error", size = 4)
    ax.grid(axis='y', linewidth=0.1)
    ax.set_axisbelow(True)
    im = ax.imshow(h, #norm=mcolors.PowerNorm(gamma=0.5),
                   cmap = "Purples", vmax = pi, vmin = 0)
    fig.colorbar(im, ax = ax, shrink = 0.8)
    for j in range(len(y)):
        if test[j] != -1:
            text = ax.text( test[j], j , 
                           "X", ha="center", va="center", 
                           color="k", fontsize =4)
            text = ax.text( test2[j], j , 
                           "X", ha="center", va="center", 
                           color="g", fontsize =4)
        for i in range(len(x)):
            if arr_trial_heat_t[j,i] < 0.:
                text = ax.text( i, j , 
                           "-", ha="center", va="center", 
                           color="r", fontsize =4)
    
    plt.tight_layout()
    plt.savefig(dirBase+'ErrorR_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #       Traslacion
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle("Trial errors (Consensus) heatmap", size = 9)
    
    h = arr_trial_heat_cons.copy()
    h[h < 0] = 0
    ax.invert_yaxis()
    x = [str(i) for i in range(100-4)]
    y = [str(i) for i in range(nodos*nReps)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize = 2.5)
    ax.set_yticklabels(y, fontsize = 2.5)
    ax.set_xlabel("Trial", size =4)
    ax.set_ylabel("Consensus error", size = 4)
    ax.grid(axis='y', linewidth=0.1)
    ax.set_axisbelow(True)
    im = ax.imshow(h, #norm=mcolors.PowerNorm(gamma=0.5),
                   cmap = "Purples", vmax = 0.2, vmin = 0)
    fig.colorbar(im, ax = ax, shrink = 0.8)
    for j in range(len(y)):
        if test[j] != -1:
            text = ax.text( test[j], j , 
                           "X", ha="center", va="center", 
                           color="k", fontsize =4)
            text = ax.text( test2[j], j , 
                           "X", ha="center", va="center", 
                           color="g", fontsize =4)
        for i in range(len(x)):
            if arr_trial_heat_t[j,i] < 0.:
                text = ax.text( i, j , 
                           "-", ha="center", va="center", 
                           color="r", fontsize =4)
    
    plt.tight_layout()
    plt.savefig(dirBase+'ErrorC_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    


def experiment_angles(nReps = 100,
                    dirBase = "",
                    k_int = 0.,
                    intGamma0 = None,
                    intGammaInf = None,
                    intGammaSteep = 5.,
                    gamma0 = None,
                    gammaInf = None,
                    gammaSteep = 5.,
                    ang = 0,
                    enablePlotExp = True):

    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    arr_n_points = np.zeros(nReps)
    mask = np.zeros(nReps)
    #Misscount = 0

    #   Testing ranges


    logText = "Local testing"
    logText += '\n' +"Local test directory = "+ str(dirBase)
    write2log(logText + '\n')

    for k in range(nReps):
        FOVflag = True

        #   Points
        # Range
        PxRange = [-1,1]
        PyRange = [-1,1]
        PzRange = [-1,0]

        nP = 30
        arr_n_points[k] = nP
        P = np.random.rand(2,nP)
        P[0,:] = PxRange[0]+ P[0,:]*(PxRange[1]-PxRange[0])
        P[1,:] = PyRange[0]+ P[1,:]*(PyRange[1]-PyRange[0])
        P = np.r_[P,-np.ones((1,nP))]

        R = cm.rot(ang,'y')
        P = R @ P

        Ph = np.r_[P,np.ones((1,nP))]

        #   Conifgs cameras
        xRange = [-2,2]
        yRange = [-2,2]
        zRange = [0,3]
        offset = np.array([xRange[0],yRange[0],zRange[0],-np.pi,-np.pi,-np.pi])
        dRange = np.array([xRange[1],yRange[1],zRange[1],np.pi,np.pi,np.pi])
        dRange -= offset


        pd = np.zeros((6,n_agents))
        p0 = np.zeros((6,n_agents))
        for i in range(n_agents):

            #   References

            tmp = np.random.rand(6)
            tmp = offset + tmp*dRange
            tmp[3] = pi
            tmp[4] = 0.
            cam = cm.camera()
            agent = ctr.agent(cam, tmp, tmp,P)

            while agent.count_points_in_FOV(Ph) != nP:
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                tmp[3] = pi
                tmp[4] = 0.
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)

            pd[:,i] = tmp.copy()

            #   Initial conds

            tmp = np.random.rand(6)
            tmp = offset + tmp*dRange
            tmp[3] = pi
            tmp[4] = 0.
            cam = cm.camera()
            agent = ctr.agent(cam, tmp, tmp,P)

            while agent.count_points_in_FOV(Ph) != nP:
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                tmp[3] = pi
                tmp[4] = 0.
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)

            p0[:,i] = tmp.copy()







        write2log("CASE "+str(k)+'\n')
        ret = experiment(directory=dirBase+str(k),
                    k_int = k_int,
                    gamma0 = gamma0,
                    gammaInf = gammaInf,
                    gammaSteep = gammaSteep ,
                    intGamma0 = intGamma0,
                    intGammaInf = intGammaInf,
                    intGammaSteep = intGammaSteep ,
                    pd = pd,
                    p0 = p0,
                    P = P,
                    #set_derivative = True,
                    #tanhLimit = True,
                    #depthOp = 4, Z_set = 1.,
                    t_end = 100,
                    enablePlot = enablePlotExp)
        #print(ret)
        [var_arr[:,k], errors, FOVflag] = ret
        var_arr_et[k] = errors[0]
        var_arr_er[k] = errors[1]
        var_arr_2[k] = errors[2]
        var_arr_3[k] = errors[3]
        if FOVflag:
            mask[k] = 1
            #Misscount += 1

    #   Save data

    np.savez(dirBase+'data.npz',
                n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            arr_n_points = arr_n_points,
            mask = mask)
            #Misscount = Misscount)
    


def experiment_rep_3D(nReps = 1,
                     t_end = 100,
                    dirBase = "",
                    node = 0,
                    intGamma0 = None,
                    intGammaInf = None,
                    intGammaSteep = 5.,
                    gamma0 = None,
                    gammaInf = None,
                    gammaSteep = 5.,
                    enablePlotExp = True):
    
    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    arr_n_points = np.zeros(nReps)
    mask = np.zeros(nReps)
    #Misscount = 0
    
    logText = '\n' +"Local test directory = "+ str(dirBase)
    write2log(logText + '\n')
    
    for k in range(nReps):
        
        write2log("CASE "+str(k)+'\n')
        ret = experiment(directory=dirBase+str(node*nReps+k),
                         modified = ["nPoints"],
                    gamma0 = gamma0,
                    gammaInf = gammaInf,
                    gammaSteep = gammaSteep ,
                    intGamma0 = intGamma0,
                    intGammaInf = intGammaInf,
                    intGammaSteep = intGammaSteep ,
                    #set_derivative = True,
                    #tanhLimit = True,
                    #depthOp = 4, Z_set = 1.,
                    t_end = t_end,
                    repeat = True,
                    enablePlot = enablePlotExp)
        #print(ret)
        [var_arr[:,k], errors, FOVflag] = ret
        var_arr_et[k] = errors[0]
        var_arr_er[k] = errors[1]
        var_arr_2[k] = errors[2]
        var_arr_3[k] = errors[3]
        if FOVflag:
            mask[k] = 1
            #Misscount += 1
        
    #   Save data
    
    np.savez(dirBase+'data_'+str(node)+'.npz',
                n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            arr_n_points = arr_n_points,
            mask = mask)
            #Misscount = Misscount)
            

#   sets up the inital, reference and pounts for local experiments
#   actual experiments run with experiment_repeat
def experiment_local(pFlat = False,
                    n_points = None,
                    dirBase = "",
                    caseId = 0,
                    dArray = np.zeros((2,1)),
                    activatepdCirc = True):
    
    #   Scene 0
    n_agents = 4
    
    Pz = [-1,0]
    if pFlat:
        Pz = [0,0]
    
    nP = None
    if n_points is None:
        nP = random.randint(4,11)
    else:
        nP = n_points
        
        
    if activatepdCirc:
        scene(modify = ["P","pd","p0"],
        inpd  = circle(4,1,T=np.array([0.,0.,2.])),
        Pz = Pz,
            dirBase = dirBase+"0/"+str(caseId),
            n_agents = 4, 
            n_points = nP)    
    else:
        scene(modify = ["P","pd","p0"],
            Pz = Pz,
                dirBase = dirBase+"0/"+str(caseId),
                n_agents = 4, 
                n_points = nP)
    
    #   Log
    logText = "Local testing setup "
    logText += '\n' +"Test series, pFlat = "+ str( pFlat)
    logText += '\n' +"Local test directory = "+ str(dirBase)
    logText += '\n' +"Case ID = "+ str(caseId)
    write2log(logText + '\n')
    
    for k in range(dArray.shape[1]):
        
        #   Log
        write2log("CASE "+str(k)+'\n')
        logText = '\n' +"Testing ranges:"
        logText += '\n' +"\t State delta / 2 = "+ str(dArray[:,k])
        write2log(logText+'\n')
        
        #   leer datos de data.npz 
        
        npzfile = np.load(dirBase+"0/"+str(caseId)+"/data.npz")
        P = npzfile['P']
        pd = npzfile['pd']
        p0 = npzfile['p0']
        
        ##   Cameras
        
        p0 = np.zeros((6,n_agents))
        
        
        offset = -dArray[:,k]
        dRange = dArray[:,k].copy()
        dRange -= offset
        
        #   Initial conditions
        #   Z > 0 
        #   FOV de cuadre
        for i in range(n_agents):
            
            tmp = np.random.rand(6)
            tmp = pd[:,i] + offset + tmp*dRange
            
            cam = cm.camera()
            agent = ctr.agent(cam, tmp, tmp,P)
            
            while agent.count_points_in_FOV(P) != nP or tmp[2]<0.:
                tmp = np.random.rand(6)
                tmp = pd[:,i] + offset + tmp*dRange
                
                
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)
            
            p0[:,i] = tmp.copy()
            
        name = dirBase+str(k)+"/"+str(caseId)+"/data.npz"
        np.savez(name,
            P = P,
            p0 = p0,
            pd = pd)
        
       
        
        
    
def experiment_plots(dirBase = "", kSets = 0, colorFile = "default.npz"):
    
    n_agents = 4    #   TODO: generalizar
    var_arr = np.array([[],[],[],[]])
    var_arr_2 = np.array([])
    var_arr_3 = np.array([])
    var_arr_et = np.array([])
    var_arr_er = np.array([])
    arr_n_points = np.array([])
    mask = np.array([])
    
    npzfile = np.load(colorFile)
    colors = npzfile["colors"]
    
    if kSets != 0:
        data = np.array([])
        for k in range(kSets):
            name=dirBase+'data_'+str(k)+'.npz'
            npzfile = np.load(name)
            var_arr = np.c_[var_arr,npzfile['var_arr']]
            var_arr_2 = np.concatenate((var_arr_2, npzfile['var_arr_2']))
            var_arr_3 = np.concatenate((var_arr_3, npzfile['var_arr_3']))
            var_arr_et = np.concatenate((var_arr_et, npzfile['var_arr_et']))
            var_arr_er = np.concatenate((var_arr_er, npzfile['var_arr_er']))
            mask = np.concatenate((mask, npzfile['mask']))
            
    else:
        npzfile = np.load(dirBase+'data.npz')
        n_agents = npzfile['n_agents']
        var_arr = npzfile['var_arr']
        var_arr_2 = npzfile['var_arr_2']
        var_arr_3 = npzfile['var_arr_3']
        var_arr_et = npzfile['var_arr_et']
        var_arr_er = npzfile['var_arr_er']
        arr_n_points = npzfile['arr_n_points']
        mask = npzfile['mask']
        #Misscount = npzfile['Misscount']
    nReps = var_arr.shape[1]
    
    
    logText = "Simulations that failed = "
    logText += str(mask.sum()) 
    logText += " / "+ str(nReps)
    write2log(logText + '\n')
    
    etsup = np.where((var_arr_2 > var_arr_et) &
                    (mask == 0.))[0]
    ersup = np.where((var_arr_3 > var_arr_er) &
                    (mask == 0.))[0]
    esup = np.where((var_arr_3 > var_arr_er) &
                    (var_arr_2 > var_arr_et) &
                    (mask == 0.))[0]
    
    Succsesful = np.where((var_arr_3 < 0.1) &
                    (var_arr_2 < 0.1) &
                    (mask == 0.))[0]
    UnSuccsesful = np.where((var_arr_3 >= 0.1) &
                    (var_arr_2 >= 0.1) &
                    (mask == 0.))[0]
    
    #   Masking
    write2log("Failed repeats = "+str( np.where(mask == 1)[0]))
    logText = '\nSuccsesful cases = ' + str(Succsesful.shape[0])
    logText += '\n' +"Simulations that increased et = "+ str( etsup.shape[0])+  " / "+ str( nReps)
    logText += '\n' +str(etsup)
    logText += '\n' +"Simulations that increased er = "+ str( ersup.shape[0])+  " / "+ str( nReps)
    logText += '\n' +str(ersup)
    logText += '\n' +"Simulations that increased both err = "+ str( esup.shape[0])+  " / "+ str( nReps)
    logText += '\n' +str(esup)
    logText += "\nUnsuccsesful cases" 
    logText += '\n' +str(UnSuccsesful)
    logText += '\n'
    
    write2log(logText + '\n')
    if (np.count_nonzero(mask == 1.) == mask.shape[0]):
        print("Simulations : No data to process")
        return
    var_arr = var_arr[:,mask==0.]
    var_arr_2 = var_arr_2[mask==0.]
    var_arr_3 = var_arr_3[mask==0.]
    var_arr_et = var_arr_et[mask==0.]
    var_arr_er = var_arr_er[mask==0.]
        
    
        
        
    np.set_printoptions(linewidth=np.inf)
    
    
    nReps = var_arr.shape[1]
    ref_arr = np.arange(nReps)
    
    #   Plot data
    
    ##      Simulation - wise consensus error
    fig, ax = plt.subplots()
    fig.suptitle("Consensus error by experiment")
    #plt.ylim([-2.,2.])
    for i in range(n_agents):
        ax.scatter(ref_arr, var_arr[i,:], 
                marker = "x", alpha = 0.5,color=colors[i])
    
    plt.yscale('logit')
    plt.tight_layout()
    plt.savefig(dirBase+'Consensus error_byExp.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ###   Kernel density consensus
    #fig, ax = plt.subplots()
    #fig.suptitle("Error de consenso (Densidad por Kernels)")
    ##plt.ylim([-2.,2.])
    
    var_arr = norm(var_arr,axis = 0)/n_agents
    #density = gaussian_kde(var_arr)
    #dh = var_arr.max() - var_arr.min()
    #xs = np.linspace(var_arr.min()-dh*0.1,var_arr.max()+dh*0.1,200)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    #plt.plot(xs,density(xs),color=colors[0])
    #plt.scatter(var_arr,np.zeros(nReps),
                #marker = "|", alpha = 0.5,color=colors[0])
    #plt.tight_layout()
    #plt.savefig(dirBase+'Consensus error_Kernel_density.pdf',bbox_inches='tight')
    ##plt.show()
    #plt.close()
    
    logText = "\nAverage consensus image error" 
    logText += '\n' +str(var_arr)
    logText += '\n'
    
    write2log(logText + '\n')
    
    ##  Histogram consensus
    
    fig, ax = plt.subplots()
    fig.suptitle("Consensus error histogram")
    #counts, bins = np.histogram(var_arr)
    #plt.stairs(counts, bins)
    plt.hist(var_arr,
             #bins=1,
             edgecolor = colors[0],
             fill = False)
    #plt.tight_layout()
    plt.savefig(dirBase+'Consensus error_Histogram.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
    ##  Histogram consensus Zoom
    fig, ax = plt.subplots()
    fig.suptitle("Consensus error histogram (zoom)")
    #counts, bins = np.histogram(var_arr[var_arr < 0.5])
    #plt.stairs(counts, bins)
    plt.hist(var_arr[var_arr < 0.01],
             edgecolor = colors[0],
             fill = False)
    plt.tight_layout()
    plt.savefig(dirBase+'Consensus error_Histogram_zoom.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
    
    ##  Simulation - wise formation error
    fig, ax = plt.subplots()
    fig.suptitle("Formation state error")
    ax.scatter(ref_arr,var_arr_2, label = "Posición",
            marker = ".", alpha = 0.5,color = colors[0])
    ax.scatter(ref_arr,var_arr_3, label = "Rotación",
            marker = "x", alpha = 0.5,color=colors[1])
    fig.legend( loc=2)
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_byExp.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Scatter err T,R
    fig, ax = plt.subplots()
    fig.suptitle("State error")
    ax.scatter(var_arr_et,var_arr_er, 
            marker = "*", alpha = 0.5,color = colors[1])
    ax.scatter(var_arr_2,var_arr_3, 
            marker = "*", alpha = 0.5,color = colors[0])
    for i in range(nReps):
        ax.plot([var_arr_et[i],var_arr_2[i]],
                [var_arr_er[i],var_arr_3[i]],
                alpha = 0.5, color = colors[0],
                linewidth = 0.5)
    ax.set_xlabel("Translation")
    ax.set_ylabel("Rotation")
    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,3.1))
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_Scatter.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
    
    ##   Kernel plots formation error
    
    #fig, ax = plt.subplots()
    #fig.suptitle("Errores de traslación")
    #density = gaussian_kde(var_arr_2)
    #dh = var_arr_2.max() - var_arr_2.min()
    #xs = np.linspace(var_arr_2.min()-dh*0.1,var_arr_2.max()+dh*0.1,200)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    #plt.plot(xs,density(xs),color=colors[0])
    #plt.scatter(var_arr_2,np.zeros(nReps),
                #marker = "|", alpha = 0.5,color=colors[0])
    
    #plt.tight_layout()
    #plt.savefig(dirBase+'Formation error_Kernel_desity_T.pdf',bbox_inches='tight')
    ##plt.show()
    #plt.close()
    
    #fig, ax = plt.subplots()
    #fig.suptitle("Errores de rotación")
    #density = gaussian_kde(var_arr_3)
    #dh = var_arr_3.max() - var_arr_3.min()
    #xs = np.linspace(var_arr_3.min()-dh*0.1,var_arr_3.max()+dh*0.1,200)
    #density.covariance_factor = lambda : .25
    #density._compute_covariance()
    #plt.plot(xs,density(xs),color=colors[1])
    #plt.scatter(var_arr_3,np.zeros(nReps),
                #marker = "|", alpha = 0.5,color=colors[1])
    
    #plt.tight_layout()
    #plt.savefig(dirBase+'Formation error_Kernel_desity_R.pdf',bbox_inches='tight')
    ##plt.show()
    #plt.close()
    
    ##  Heatmap
    h, x, y, img = plt.hist2d(var_arr_2,var_arr_3, 
                              #cmap = "Purples",
                              #norm =  mcolors.LogNorm(),
                              #norm=mcolors.PowerNorm(gamma=0.5),
                                #cmap='PuBu_r',
                              range = [[0., 1.],[0.,3.15]])
    fig, ax = plt.subplots()
    fig.suptitle("Final formation error heatmap")
    
    h = h.T
    ax.imshow(h, norm=mcolors.PowerNorm(gamma=0.5),
                                cmap='Purples')
                                #vmax = 100, vmin = 0)
    ax.invert_yaxis()
    x = [format((x[i+1]+x[i])/2,'.2f') for i in range(x.shape[0]-1)]
    y = [format((y[i+1]+y[i])/2,'.2f') for i in range(y.shape[0]-1)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)
    ax.set_xlabel("Translation")
    ax.set_ylabel("Rotation")
    for i in range(len(x)):
        for j in range(len(y)):
            text = ax.text(j, i, h[i, j],
                        ha="center", va="center", color="k")

    
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Heatmap Zoom
    var_arr_3 = var_arr_3[var_arr_2 < 0.2]
    var_arr_2 = var_arr_2[var_arr_2 < 0.2]
    var_arr_2 = var_arr_2[var_arr_3 < 0.2]
    var_arr_3 = var_arr_3[var_arr_3 < 0.2]
    h, x, y, img = plt.hist2d(var_arr_2,var_arr_3, cmap = "Purples")
    fig, ax = plt.subplots()
    fig.suptitle("Final formation error heatmap (Zoom)")
    
    h = h.T
    ax.imshow(h, norm=mcolors.PowerNorm(gamma=0.5),
                                cmap='Purples')
                                #vmax = 100, vmin = 0)
    ax.invert_yaxis()
    x = [format((x[i+1]+x[i])/2,'.2f') for i in range(x.shape[0]-1)]
    y = [format((y[i+1]+y[i])/2,'.2f') for i in range(y.shape[0]-1)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.set_yticklabels(y)
    ax.set_xlabel("Error de traslación")
    ax.set_ylabel("Error de rotación")
    for i in range(len(x)):
        for j in range(len(y)):
            text = ax.text(j, i , h[i, j],
                        ha="center", va="center", color="k")

    
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_heatmap_zoom.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Scatter err T,R Zoom
    fig, ax = plt.subplots()
    fig.suptitle("Errores de estado (Zoom)")
    ax.scatter(var_arr_2,var_arr_3, 
            marker = "*", alpha = 0.5,color = colors[0])
    ax.set_xlabel("Translation")
    ax.set_ylabel("Rotation")
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_Scatter_zoom.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()


#   Grafica las relaciones de errores inicial y final, RvsT
def plot_error_stats(nReps = 100,
                     dirBase = "",
                     colorFile = "default.pnz"):
    #   TODO: actualizar con masking
    arr_err = np.zeros(nReps)
    
    for k in range(nReps):
        directory=dirBase+str(k)
        npzfile = np.load(directory+'/data.npz')
        P = npzfile["P"]
        n_points = P.shape[1] #Number of image points
        p0 = npzfile["p0"]
        n_agents = p0.shape[1] #Number of agents
        pd = npzfile["pd"]
        adjMat = npzfile["adjMat"]
        tmp=np.zeros(n_agents)
        for i in range(n_agents):
            cam = cm.camera()
            agent = ctr.agent(cam,pd[:,i],p0[:,i],P)
            tmp[i] = np.linalg.norm(agent.s_ref_n)
        arr_err[k] = np.linalg.norm(tmp)/ n_agents
    
    npzfile = np.load(dirBase+'data.npz')
    n_agents = npzfile['n_agents']
    var_arr = npzfile['var_arr']
    var_arr_2 = npzfile['var_arr_2']
    var_arr_3 = npzfile['var_arr_3']
    var_arr_et = npzfile['var_arr_et']
    var_arr_er = npzfile['var_arr_er']
    #Misscount = npzfile['Misscount']
    mask = npzfile['mask']
    nReps = var_arr.shape[1]
    
    var_arr = norm(var_arr,axis = 0)/n_agents
    
    #colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    npzfile = np.load(colorFile)
    colors = npzfile["colors"]
    
    ##  Scatter err T, Ce
    fig, ax = plt.subplots()
    fig.suptitle("Consensus error evolution for each final translation error")
    ax.scatter(var_arr_2,arr_err, 
            marker = "*", alpha = 0.5,color = colors[0],
            label= "Initial")
    ax.scatter(var_arr_2,var_arr, 
            marker = "*", alpha = 0.5,color = colors[1],
            label= "Final")
    for i in range(nReps):
        ax.plot([var_arr_2[i],var_arr_2[i]],
                [arr_err[i],var_arr[i]],
                alpha = 0.5, color = colors[1],
                linewidth = 0.5)
    
    fig.legend( loc=1)
    ax.set_xlabel("Translation error")
    ax.set_ylabel("Consensus error")
    #plt.xlim((-0.1,1.1))
    #plt.ylim((-0.1,20))
    plt.tight_layout()
    plt.savefig(dirBase+'Translation vs consensus.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Scatter err R, Ce
    fig, ax = plt.subplots()
    fig.suptitle("Consensus error evolution for each final rotation error")
    ax.scatter(var_arr_3,arr_err, 
            marker = "*", alpha = 0.5,color = colors[0],
            label= "Initial")
    ax.scatter(var_arr_3,var_arr, 
            marker = "*", alpha = 0.5,color = colors[1],
            label= "Final")
    for i in range(nReps):
        ax.plot([var_arr_3[i],var_arr_3[i]],
                [arr_err[i],var_arr[i]],
                alpha = 0.5, color = colors[1],
                linewidth = 0.5)
    
    fig.legend( loc=1)
    ax.set_xlabel("Rotation error")
    ax.set_ylabel("Consensus error")
    #plt.xlim((-0.1,1.1))
    #plt.ylim((-0.1,3.1))
    plt.tight_layout()
    plt.savefig(dirBase+'Rotation vs consensus.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Scatter err init, end
    fig, ax = plt.subplots()
    fig.suptitle("Trastation error evolution fvs each final translation error")
    ax.scatter(var_arr_et,var_arr_2, 
            marker = "*", alpha = 0.5,color = colors[0],
            label= "Initial")
    ax.scatter(var_arr_et,var_arr_et, 
            marker = "*", alpha = 0.5,color = colors[1],
            label= "Final")
    for i in range(nReps):
        ax.plot([var_arr_et[i],var_arr_et[i]],
                [var_arr_2[i],var_arr_et[i]],
                alpha = 0.5, color = colors[1],
                linewidth = 0.5)
    
    fig.legend( loc=1)
    ax.set_xlabel("Translation error")
    ax.set_ylabel("Translation error")
    #plt.xlim((-0.1,1.1))
    #plt.ylim((-0.1,3.1))
    plt.tight_layout()
    plt.savefig(dirBase+'Translation error evolution vs final tralation.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Scatter err end, init
    fig, ax = plt.subplots()
    fig.suptitle("Rotation error evolution fvs each final rotation error")
    ax.scatter(var_arr_er,var_arr_3, 
            marker = "*", alpha = 0.5,color = colors[0],
            label= "Translation")
    ax.scatter(var_arr_er,var_arr_er, 
            marker = "*", alpha = 0.5,color = colors[1],
            label= "Rotation")
    for i in range(nReps):
        ax.plot([var_arr_er[i],var_arr_er[i]],
                [var_arr_3[i],var_arr_er[i]],
                alpha = 0.5, color = colors[1],
                linewidth = 0.5)
    
    fig.legend( loc=1)
    ax.set_xlabel("Rotation error")
    ax.set_ylabel("Rotation error")
    #plt.xlim((-0.1,1.1))
    #plt.ylim((-0.1,3.1))
    plt.tight_layout()
    plt.savefig(dirBase+'Translation error evolution vs final tralation.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ###  Scatter err end, init
    #fig, ax = plt.subplots()
    #fig.suptitle("Errores traslación final vs Formación inicial")
    #ax.scatter(var_arr_3,var_arr_et, 
            #marker = "*", alpha = 0.5,color = colors[0],
            #label= "Traslacion")
    #ax.scatter(var_arr_3,var_arr_er, 
            #marker = "*", alpha = 0.5,color = colors[1],
            #label= "Rotación")
    #for i in range(nReps):
        #ax.plot([var_arr_3[i],var_arr_3[i]],
                #[var_arr_et[i],var_arr_er[i]],
                #alpha = 0.5, color = colors[1],
                #linewidth = 0.5)
    
    #fig.legend( loc=1)
    #ax.set_xlabel("Error de Rotación final")
    #ax.set_ylabel("Error de Formación inicial")
    ##plt.xlim((-0.1,1.1))
    ##plt.ylim((-0.1,3.1))
    #plt.tight_layout()
    #plt.savefig(dirBase+'Error traslación vs EF.pdf',bbox_inches='tight')
    ##plt.show()
    #plt.close()
    
    ###  Scatter err init, end
    #fig, ax = plt.subplots()
    #fig.suptitle("Errores traslación final vs Formación inicial")
    #ax.scatter(var_arr_2,var_arr_et, 
            #marker = "*", alpha = 0.5,color = colors[0],
            #label= "traslación")
    #ax.scatter(var_arr_2,var_arr_er, 
            #marker = "*", alpha = 0.5,color = colors[1],
            #label= "Rotación")
    #for i in range(nReps):
        #ax.plot([var_arr_2[i],var_arr_2[i]],
                #[var_arr_et[i],var_arr_er[i]],
                #alpha = 0.5, color = colors[1],
                #linewidth = 0.5)
    
    #fig.legend( loc=1)
    #ax.set_xlabel("Error de Traslación final")
    #ax.set_ylabel("Error de Formación inicial")
    ##plt.xlim((-0.1,1.1))
    ##plt.ylim((-0.1,3.1))
    #plt.tight_layout()
    #plt.savefig(dirBase+'Error rotacion vs EF.pdf',bbox_inches='tight')
    ##plt.show()
    #plt.close()



def plot_tendencias(nReps = 20,
                    dirBase = "",
                    colorFile = "default.npz"):
    
    n_agents = 4
    #colors = (randint(0,255,3*3)/255.0).reshape((3,3))
    npzfile = np.load(colorFile)
    colors = npzfile["colors"]
    lw = 0.5
    
    ref = np.arange(nReps)+1
    count_mask = np.zeros(nReps)
    
    cons_err_max = np.zeros(nReps)
    cons_err_avr = np.zeros(nReps)
    cons_err_min = np.zeros(nReps)
    
    tErr_err_max = np.zeros(nReps)
    tErr_err_avr = np.zeros(nReps)
    tErr_err_min = np.zeros(nReps)
    
    rErr_err_max = np.zeros(nReps)
    rErr_err_avr = np.zeros(nReps)
    rErr_err_min = np.zeros(nReps)
    
    data_cons = []
    data_tErr = []
    data_rErr = []
    
    mask_coords = []
    
    h1 = np.zeros((1,100))
    h2 = np.zeros((1,100))
    hcons = np.zeros((1,100))
    
    for k in range(nReps):
        
        npzfile = np.load(dirBase+str(k)+'/data.npz')
        #n_agents = npzfile['n_agents']
        var_arr = npzfile['var_arr']
        var_arr = norm(var_arr,axis = 0)/n_agents
        var_arr_2 = npzfile['var_arr_2']
        var_arr_3 = npzfile['var_arr_3']
        var_arr_et = npzfile['var_arr_et']
        var_arr_er = npzfile['var_arr_er']
        arr_n_points = npzfile['arr_n_points']
        #Misscount = npzfile['Misscount']
        mask = npzfile['mask']
        
        hcons = np.r_[hcons,var_arr.reshape((1,100))]
        h1 = np.r_[h1,var_arr_2.reshape((1,100))]
        h2 = np.r_[h2,var_arr_3.reshape((1,100))]
        
        logText = "Masks for step " + str(k)
        logText += "=" +str(np.where(mask == 1.)[0])
        write2log(logText + '\n')
        
        selection = mask==0. 
        #selection = mask==1. 
        #selection = np.ones(mask.shape, dtype= np.bool_)
        #selection = (mask==0. and arr_n_points > 4)
        if any(selection):
        
            var_arr = var_arr[selection]
            var_arr_2 = var_arr_2[selection]
            var_arr_3 = var_arr_3[selection]
            var_arr_et = var_arr_et[selection]
            var_arr_er = var_arr_er[selection]
            
            count_mask[k] = mask.sum()
            mask_coords.append(np.where(mask> 0)[0])
            
            data_cons.append(var_arr)
            cons_err_max[k] = var_arr.max()
            cons_err_avr[k] = var_arr.mean()
            cons_err_min[k] = var_arr.min()
            
            data_tErr.append(var_arr_2)
            tErr_err_max[k] = var_arr_2.max()
            tErr_err_avr[k] = var_arr_2.mean()
            tErr_err_min[k] = var_arr_2.min()
            
            data_rErr.append(var_arr_3)
            rErr_err_max[k] = var_arr_3.max()
            rErr_err_avr[k] = var_arr_3.mean()
            rErr_err_min[k] = var_arr_3.min()
    h1 = h1[1:,:]
    h2 = h2[1:,:]
    hcons = hcons[1:,:]
    
    #   BEGIN
    #   Errores de consenso
    #fig, ax = plt.subplots( 2,1,sharex=True,frameon=False)
    fig, ax = plt.subplots(frameon=False, figsize=(4.8,5.4))
    fig.suptitle("Consensus error by repetition sets")
    ax.imshow(count_mask.reshape((1,nReps)), cmap = "Purples", 
              #norm=mcolors.PowerNorm(gamma=0.5),
                 extent = (0+0.5,nReps+0.5, -0.00075,-0.00025),
                 aspect = 'auto',
                 vmin = 0, vmax = 100)
    
    ax.plot(ref,cons_err_min, 
            color = colors[0],
            label= "Min",
            lw = 0.5)
    ax.plot(ref,cons_err_avr, 
            color = colors[1],
            label= "Average",
            lw = 0.5)
    ax.plot(ref,cons_err_max, 
            color = colors[2],
            label= "Max",
            lw = 0.5)
    ax.plot([1,nReps],[0.1,0.1], 
            color = 'red',
            linestyle = 'dashed',
            label= "Threshold",
            lw = 0.5)
    #ax[0].violinplot(data_cons, showextrema = False)
    ax.boxplot(data_cons)
    
    fig.legend( loc=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Consensus error")
    #plt.xlim((-0.1,1.1))
    plt.ylim((-0.001,0.011))
    plt.xticks(ticks = np.arange(1,20,2),
                  labels = np.arange(1,20,2))
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    
    for i in range(nReps):
        text = ax.text(i +1,  -0.0005,str(int(count_mask[i])),
                          ha="center", va="center", 
                           color="k", fontsize =10)
    
    #plt.tight_layout()
    plt.savefig(dirBase+'Consensus.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   END
    
    #   heatmap 
    #       consensus
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Consensus errors", size = 9)
    
    h = hcons.copy()
    h[h < 0] = 0
    ax.invert_yaxis()
    x = [str(i) for i in range(100)]
    y = [str(i) for i in range(nReps)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize = 2.5)
    ax.set_yticklabels(y, fontsize = 2.5)
    ax.set_xlabel("Scene", size =4)
    ax.set_ylabel("Step", size = 4)
    ax.grid(axis='y', linewidth=0.1)
    ax.set_axisbelow(True)
    im = ax.imshow(h,# norm=mcolors.PowerNorm(gamma=0.5),
                   cmap = "Purples", vmax = 0.11, vmin = 0)
    fig.colorbar(im, ax = ax, shrink = 0.7)
    for j in range(len(y)):
        for i in mask_coords[j]:
            text = ax.text(i ,  j, 
                           "X", ha="center", va="center", 
                           color="r", fontsize =4)
    
    plt.tight_layout()
    plt.savefig(dirBase+'ErrorC_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
    #   Errores de traslación
    fig, ax = plt.subplots(frameon=False, figsize=(4.8,5.4))
    fig.suptitle("Translation error by repetition sets")
    ax.imshow(count_mask.reshape((1,nReps)), cmap = "Purples", 
              #norm=mcolors.PowerNorm(gamma=0.5),
                 extent = (0+0.5,nReps+0.5, -0.075,-0.025),
                 aspect = 'auto',
                 vmin = 0, vmax = 100)
    
    ax.plot(ref,tErr_err_min, 
            color = colors[0],
            label= "Min",
            lw = 0.5)
    ax.plot(ref,tErr_err_avr, 
            color = colors[1],
            label= "Average",
            lw = 0.5)
    ax.plot(ref,tErr_err_max, 
            color = colors[2],
            label= "Max",
            lw = 0.5)
    ax.plot([1,nReps],[0.1,0.1], 
            color = 'red',
            linestyle = 'dashed',
            label= "Threshold",
            lw = 0.5)
    
    #ax.violinplot(data_tErr, showextrema = False)
    ax.boxplot(data_tErr)
    
    fig.legend( loc=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Translation error")
    #plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.xticks(ticks = np.arange(1,20,2),
                  labels = np.arange(1,20,2))
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    
    for i in range(nReps):
        text = ax.text(i +1,  -0.05,str(int(count_mask[i])),
                          ha="center", va="center", 
                           color="k", fontsize =10)
        
    plt.tight_layout()
    plt.savefig(dirBase+'Translation.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   heatmap 
    #       Traslacion
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Translation errors", size = 9)
    
    h = h1.copy()
    h[h < 0] = 0
    ax.invert_yaxis()
    x = [str(i) for i in range(100)]
    y = [str(i) for i in range(nReps)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize = 2.5)
    ax.set_yticklabels(y, fontsize = 2.5)
    ax.set_xlabel("Scene", size =4)
    ax.set_ylabel("Step", size = 4)
    ax.grid(axis='y', linewidth=0.1)
    ax.set_axisbelow(True)
    im = ax.imshow(h,
                   cmap = "Purples", vmax = 1.1, vmin = 0)
    fig.colorbar(im, ax = ax, shrink = 0.7)
    for j in range(len(y)):
        for i in mask_coords[j]:
            text = ax.text( i ,j, 
                           "X", ha="center", va="center", 
                           color="r", fontsize =4)
    
    plt.tight_layout()
    plt.savefig(dirBase+'ErrorT_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   Errores de Rotación
    fig, ax = plt.subplots(frameon=False, figsize=(4.8,5.4))
    fig.suptitle("Rotation error by repetition sets")
    
    ax.imshow(count_mask.reshape((1,nReps)), cmap = "Purples", 
              #norm=mcolors.PowerNorm(gamma=0.5),
                 extent = (0+0.5,nReps+0.5, -0.225,-0.075),
                 aspect = 'auto',
                 vmin = 0, vmax = 100)
    
    ax.plot(ref,rErr_err_min, 
            color = colors[0],
            label= "Min",
            lw = lw)
    ax.plot(ref,rErr_err_avr, 
            color = colors[1],
            label= "Average",
            lw = lw)
    ax.plot(ref,rErr_err_max, 
            color = colors[2],
            label= "Max",
            lw = lw)
    ax.plot([1,nReps],[0.1,0.1], 
            color = 'red',
            linestyle = 'dashed',
            label= "Threshold",
            lw = 0.5)
    
    #ax.violinplot(data_rErr, showextrema = False)
    ax.boxplot(data_rErr)
    
    for i in range(nReps):
        text = ax.text(i +1,  -0.15,str(int(count_mask[i])),
                          ha="center", va="center", 
                           color="k", fontsize =10)
    
    fig.legend( loc=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rotation error")
    #plt.xlim((-0.1,1.1))
    plt.ylim((-0.3,3.2))
    plt.xticks(ticks = np.arange(1,20,2),
                  labels = np.arange(1,20,2))
    ax.grid(axis='y')
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(dirBase+'Rotation.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   heatmap 
    #       Traslacion
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Rotation errors", size = 9)
    
    h = h2.copy()
    h[h < 0] = 0
    ax.invert_yaxis()
    x = [str(i) for i in range(100)]
    y = [str(i) for i in range(nReps)]
    ax.set_yticks(np.arange(len(y)))
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x, fontsize = 2.5)
    ax.set_yticklabels(y, fontsize = 2.5)
    ax.set_xlabel("Scene", size =4)
    ax.set_ylabel("Step", size = 4)
    ax.grid(axis='y', linewidth=0.1)
    ax.set_axisbelow(True)
    im = ax.imshow(h,
                   cmap = "Purples", vmax = pi, vmin = 0)
    fig.colorbar(im, ax = ax, shrink = 0.7)
    for j in range(len(y)):
        for i in mask_coords[j]:
            text = ax.text( i ,j, 
                           "X", ha="center", va="center", 
                           color="r", fontsize =4)
    
    plt.tight_layout()
    plt.savefig(dirBase+'ErrorR_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
        
################################################################################
################################################################################
#
#   M   A   I   N
#
################################################################################
################################################################################


def test_reg():
    
    points = np.array([[-0.5, -0.5, 0.5,  0.5],
                [-0.5,  0.5, 0.5, -0.5],
                [0.,    0., 0.,  0.],
                [1, 1, 1, 1]])
    p_obj = np.array([0,0,1,pi,0,0])
    #p_current = np.array([0,0,2,pi,0,0])
    p_current = np.array([0,0,2,pi,0.2,0])
    #p_current = np.array([0,0,2,pi+0.2,0,0])
    
    
    camera = cm.camera()
    agent = ctr.agent(camera,
                 p_obj,
                 p_current,
                 points,
                 setRectification = False)
    
    #p_current = np.array([0,0,2,pi,0,0])
    p_current = np.array([0,0,2,pi,0.2,0])
    #p_current = np.array([0,0,2,pi+0.2,0,0])
    
    camera_r = cm.camera()
    agent_r = ctr.agent(camera,
                 p_obj,
                 p_current,
                 points,
                 setRectification = True)
    
    
    fig, ax = plt.subplots()
    plt.xlim([0,agent.camera.iMsize[0]])
    plt.ylim([0,agent.camera.iMsize[1]])
    
    ax.plot([agent.camera.iMsize[0]/2,agent.camera.iMsize[0]/2],
            [0,agent.camera.iMsize[0]],
            color=[0.25,0.25,0.25])
    ax.plot([0,agent.camera.iMsize[1]],
            [agent.camera.iMsize[1]/2,agent.camera.iMsize[1]/2],
            color=[0.25,0.25,0.25])
    plt.plot(agent.s_current[0,:],agent.s_current[1,:],'*b')
    plt.plot(agent.s_ref[0,:],agent.s_ref[1,:],'^b')
    
    
    plt.plot(agent_r.s_current[0,:],agent_r.s_current[1,:],'*r')
    plt.plot(agent_r.s_ref[0,:],agent_r.s_ref[1,:],'*r')
    
    plt.show()
    plt.close()
    return


def mainLocal(arg):
    
    ##  Color palettes
    #colorPalette("general.npz", n_colors = 60)
    #colorPalette("default.npz", n_colors = 4)
    #colorPalette("colorPalette_A.npz", n_colors = 4)
    #colorPalette("colorPalette_B.npz", n_colors = 4)
    #return
    
    #test_reg()
    #return 
    
    #   reset Log
    global logFile
    logFile = arg[1]
    
    try:
        with open(logFile,'w') as file:
            file.write("Log file INIT\n")
    except IOError:
        print("Logfile Error."+logFile+" OUT")
        return
    
    
    #   BEGIN   Contrajemplos
    #P = np.array([[1.,1.,-1.,-1.],
                  #[1.,-1.,-1.,1.],
                  #[0.,0.,0.,0.]])
    ##P = np.array([[1.,1.,-1.,-1.,0.5],
                  ##[1.,-1.,-1.,1.,0.1],
                  ##[0.,0.,0.,0,0.5]])
    #P *= 0.75
    #pd = circle(4,1,T=np.array([0.,0.,2.]))
    
    ####   Escalando 
    #p0 = pd.copy()
    
    #   escalando
    #p0[:3,:] *= 2.
    #p0[:2,:] *= 2.
    
    #   Traslación en Z
    #p0[2,:] += 1
    
    #   Traslación en XY
    #p0[:2,:] += np.array([[1.],[1.]])
    
    ##   Comprobando escalamiento desde un punto arbitrario en el plano z = 0
    #p0[:2,:] += np.array([[1.],[1.]])
    #p0[:3,:] *= 2.
    #p0[:2,:] -= np.array([[1.],[1.]])
    
    #   Comprobando la composición de transformaciones A x B 
    #p0[:2,:] += np.array([[1.],[1.]])
    #p0[:3,:] *= 2.
    
    #   Comprobando la composición de transformaciones B  x A
    #p0[:3,:] *= 2.
    #p0[:2,:] += np.array([[1.],[1.]])

    
    #   rotación en yaw
    
    #testAng =  np.pi
    #R = cm.rot(testAng,'z') 
    
    #for i in range(4):
        #_R = cm.rot(p0[5,i],'z') 
        #_R = _R @ cm.rot(p0[4,i],'y')
        #_R = _R @ cm.rot(p0[3,i],'x')
        
        #_R = R @ _R
        #[p0[3,i], p0[4,i], p0[5,i]] = ctr.get_angles(_R)
    #p0[:3,:] = R @ p0[:3,:]
    
    #   rotación en pitch (esta es la que falla a pesar de iniciar en formación)
    
    #testAng =  np.pi/4
    #R = cm.rot(testAng,'y') 
    
    #for i in range(4):
        #_R = cm.rot(p0[5,i],'z') 
        #_R = _R @ cm.rot(p0[4,i],'y')
        #_R = _R @ cm.rot(p0[3,i],'x')
        
        #_R = R @ _R
        #[p0[3,i], p0[4,i], p0[5,i]] = ctr.get_angles(_R)
    #p0[:3,:] = R @ p0[:3,:]
    
    
    #   rotando la referencia
    #testAng =  -np.pi/4
    #R = cm.rot(testAng,'y') 
    
    #for i in range(4):
        #_R = cm.rot(pd[5,i],'z') 
        #_R = _R @ cm.rot(pd[4,i],'y')
        #_R = _R @ cm.rot(pd[3,i],'x')
        
        #_R = R @ _R
        #[pd[3,i], pd[4,i], pd[5,i]] = ctr.get_angles(_R)
    #pd[:3,:] = R @ pd[:3,:]
    
    
    
    #   rotación en pitch en el centroide
    
    #testAng =  np.pi/4
    #R = cm.rot(testAng,'y') 
    
    #for i in range(4):
        #_R = cm.rot(p0[5,i],'z') 
        #_R = _R @ cm.rot(p0[4,i],'y')
        #_R = _R @ cm.rot(p0[3,i],'x')
        
        #_R = R @ _R
        #[p0[3,i], p0[4,i], p0[5,i]] = ctr.get_angles(_R)
    #centroide = p0[:3,:].sum(axis = 1)/4.
    #centroide = centroide.reshape((3,1))
    #p0[:3,:] = R @ (p0[:3,:]-centroide)
    #p0[:3,:] = (p0[:3,:]+centroide)
    
    #   rotación en pitch en un eje XZ
    
    #testAng =  np.pi/4
    #R = cm.rot(testAng,'y') 
    
    #for i in range(4):
        #_R = cm.rot(p0[5,i],'z') 
        #_R = _R @ cm.rot(p0[4,i],'y')
        #_R = _R @ cm.rot(p0[3,i],'x')
        
        #_R = R @ _R
        #[p0[3,i], p0[4,i], p0[5,i]] = ctr.get_angles(_R)
    #offset = np.array([1.,0.,0.])
    #offset = offset.reshape((3,1))
    #p0[:3,:] = R @ (p0[:3,:]-offset)
    #p0[:3,:] = (p0[:3,:]+offset)
    
    #   paralelizando la referencia
    
    #for i in range(4):
        #_R = cm.rot(pd[5,i],'z') 
        #_R = _R @ cm.rot(pd[4,i],'y')
        #_R = _R @ cm.rot(pd[3,i],'x')
        
        #_R = R @ _R
        #[pd[3,i], pd[4,i], pd[5,i]] = ctr.get_angles(_R)
    #offset = np.array([1.,0.,0.])
    #offset = offset.reshape((3,1))
    #pd[:3,:] = R @ (pd[:3,:]-offset)
    #pd[:3,:] = (pd[:3,:]+offset)
    
    #   rotando puntos de imagen
    #testAng =  np.pi/4
    #R = cm.rot(testAng,'y') 
    #P = R @ P
    
    #experiment(directory="counterex",
                    #t_end = 100,
                    #k_int = 0.1,
                    ##lamb = 3.,
                    #gamma0 = 5,
                    #gammaInf = 2,
                    ###int_res = 0.2,
                    ##set_derivative = True,
                    ##depthOp = 4, Z_set = 1.,
                    #pd = pd,
                    #p0 = p0,
                    #P = P)
    #view3D("counterex")
    #return
    
    #   END Contrajemplos
    
    #   BEGIN plot circle reference
    
    
    
    
    #Pxy = 0.6
    #P = [[-Pxy, -Pxy, Pxy,  Pxy],
        #[-Pxy,  Pxy, Pxy, -Pxy],
        #[0,    0.0, 0.0,  0.0]] 
    #P = np.array(P)
    #camera = cm.camera()
    #n_agents = 4
    #end_configuration = circle(n_agents,T = np.array([0,0,1]))
    #camera_scale    = 0.15
    
    ##   3D plots
    #fig = plt.figure()
    #ax = plt.axes(projection='3d')
    #ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    
    #for i in range(n_agents):
        #camera.pose(end_configuration[:,i])
        #camera.draw_camera(ax, scale=camera_scale, color='brown')
        #ax.text(camera.p[0],camera.p[1],camera.p[2],str(i))
        #ax.text(camera.p[0],camera.p[1],camera.p[2],str(i))
        
    #ax.set_xlabel("$w_x$")
    #ax.set_ylabel("$w_y$")
    #ax.set_zlabel("$w_z$")
    #ax.grid(True)
    
    #dxy = 1.2
    #ax.set_xlim(-dxy,dxy)
    #ax.set_ylim(-dxy,dxy)
    #ax.set_zlim(-0.5,1.9)
    
    #plt.savefig('TauFormation.pdf',bbox_inches='tight')
    #plt.close()
    
    #return
    #   END plot circle reference
    
    #   BEGIN UI [r | v | Simple | 0-4]
    
    #  Arg parser and log INIT
    selector = arg[2]
    
    #   Desktop
    if selector == 'r':
        #scene(modify = ["P"],
              #Pz = [-1,0],
                #dirBase = arg[3],
                #n_agents = 4, 
                #n_points = 30)
        ret = experiment(directory=arg[3],
                    t_end = 400,
                    #setRectification = True,
                    #gdl = 2,
                    #leader = 0,
                    #lamb = 0.1,
                    ##modified =["nPoints"],
                    #gamma0 = 3.,
                    #gammaInf = .1,
                    #intGamma0 = 0.1,
                    #intGammaInf = 0.01,
                    #intGammaSteep = 5,
                    #set_derivative = True,
                    #tanhLimit = True,
                    #k_int = 0.1,
                    #int_res = 0.2,
                    repeat = True)
        print(ret)
        #view3D(arg[3])
        return
    if selector == 'v':
        view3D(arg[3])
        return
    
    
    
    return
    
    #   END UI
    
    
    
    # 
    
    #   BEGIN singular repeats
    #n = 84
    ##n = 10
    ##n = 22
    ####n = 91
    ####n=37
    ###   97, 89
    ###n= 56
    ##localdir = "/home/est_posgrado_edgar.chavez/Consenso/W_01_locales/localRand_PIG/1/64"
    #experiment(directory=str(n),
    ##experiment(directory=localdir,
                #modified = ["nPoints"],
                #t_end = 800,
                ##gamma0 = 5.,
                ##gammaInf = 2.,
                ##intGamma0 = 0.2,
                ##intGammaInf = 0.05,
                ##intGammaSteep = 5,
                ##set_derivative = True,
                ##tanhLimit = True,
                ##k_int = 0.1,
                #repeat = True)
    #view3D(str(n))
    ##view3D(localdir)
    #return
    
    #experiment(directory="local/0/53",
                #t_end = 200,
                ##gammaInf = 2.,
                ##gamma0 = 5.,
                ##set_derivative = True,
                ##tanhLimit = True,
                #k_int = 0.1,
                #repeat = True)
    #view3D("local/0/53")
    #return
    
    #   Revisando casos de falla con ganancia adaptativa conn FOV simple
    #for i in [15, 19, 21, 43, 46, 58, 62, 63, 73, 76, 89, 98]:
    #for i in [46, 63, 73, 89]:
        #experiment(directory=str(i),
                #t_end = 100,
                #gammaInf = 2.,
                #gamma0 = 5.,
                #enablePlot = False,
                #repeat = True)
        #view3D(str(i))
    #return
    
    ##  Test gamma
    #n= 31
    #experiment(directory=str(n),
                #t_end = 100,
                ##gammaInf = 0.1,
                ##gamma0 = 1.,
                ##tanhLimit = True,
                ##k_int = 0.1,
                #repeat = True)
    #view3D(str(n))
    #return
    
    #   Incremento solo de traslación 3D
    #n = 25
    #experiment(directory=str(n),
                #t_end = 100,
                #repeat = True)
    #view3D(str(n))
    
    
    #   Revisando casos de éxito con puntos planos
    #for i in [7, 37, 41, 65, 67, 75]:
        #experiment(directory=str(i),
                #t_end = 100,
                #repeat = True)
        ##view3D(str(i))
    #return
    
    ##   Revisando casos que mejoran con el integral
    #for i in [36,60,71,85]:
        #experiment(directory=str(i),
                #t_end = 100,
                #k_int = 0.1,
                #repeat = True)
        #view3D(str(i))
    #return
    
    ##   Incremento solo de rotacion 3D
    #n = 52
    #experiment(directory=str(n),
                #t_end = 100,
                #repeat = True)
    #view3D(str(n))
    #return 

    #   Incremento de ambos errores en el caso 3D
    #for i in [32,34]:
        #experiment(directory=str(i),
                #t_end = 100,
                #repeat = True)
        #view3D(str(i))
    #return

    
    ##   Incremento de ambos errores del caso plano
    #for i in [8, 10, 13, 17, 19, 20]:
        #experiment(directory=str(i),
                #t_end = 100,
                #repeat = True)
        #view3D(str(i))
    #return

    ##   VIEWER
    #view3D('3', xLimit = [-10,10], yLimit = [-10,10],zLimit = [0,20])
    ##view3D('5')
    ##view3D('20')
    ##return
    
    ##  Casos patologicos
    #   Estados cuasiestacionarios
    #view3D('local/0/31')
    #view3D('local/0/22')
    #view3D('local/0/18')
    #view3D('local/0/20')
    #view3D('local/0/9')
    
    #   Aumenta alguno de los dos errores
    #view3D('local/1/54')
    #view3D('local/1/84')
    
    #   Aumentan los dos errores
    #view3D('local/5/33')
    #return 
    
    
    #name = "11"
    #name = "72"
    #experiment(directory=name,
                    #t_end =200,
                    #lamb = 3.,
                    ##gammaInf = 2.,
                    ##gamma0 = 5.,
                    ##set_derivative = True,
                    ##depthOp = 4, Z_set = 0.1,
                    ##tanhLimit = True,
                    ##k_int = 0.1,
                    ##int_res = 0.2,
                    #repeat = True)
    #view3D(name)
    #return
    
    #   END Singular repeats
    
    
    #   BEGIN repeat folder experiments
    
    #name = "pNFlat_P04DOF_Int/"
    #experiment_repeat(nReps = 100,
                      #dirBase = name,
                      #k_int = 0.1 ,
                        #enablePlotExp = False)
    #experiment_plots(dirBase = name)
    #return 

    #   END repeat folder experiments
    
    
    
    
    #   BEGIN Experimento singular
    
    #   Pruebas con rotación del mundo CASO DE FALLA
    
    #   Rotaciones de Prueba
    #testAng =  0 #np.pi /2
    #refRot = np.array([testAng,0.,0.])
    #R = cm.rot(refRot[2],'z') 
    #R = R @ cm.rot(refRot[1],'y')
    #R = R @ cm.rot(refRot[0],'x')
    
    ##   Posiciones iniciales
    #p0 = [[-0.48417528,  1.07127934,  1.05383249, -0.02028547],
        #[ 1.5040017,   0.26301641, -0.2127149,  -0.35572372],
        #[ 1.07345242,  0.77250055,  1.15142682,  1.4490757 ],
        #[ 3.14159265,  3.14159265,  3.14159265,  3.14159265],
        #[ 0.      ,    0.   ,      -0.   ,       0.        ],
        #[-0.30442168, -1.3313259,  -1.5302976,   1.4995989 ]]
    #p0 = np.array(p0)
    ##p0[2,:] = 1.
    #n_agents = p0.shape[1]
    
    #for i in range(4):
        #_R = cm.rot(p0[5,i],'z') 
        #_R = _R @ cm.rot(p0[4,i],'y')
        #_R = _R @ cm.rot(p0[3,i],'x')
        
        #_R = R @ _R
        #[p0[3,i], p0[4,i], p0[5,i]] = ctr.get_angles(_R)
    #p0[:3,:] = R @ p0[:3,:]
    
    #p0[5,:] *= 0.
    ##p0[2,:] += 1.5
    ##p0[:,[2,0]] = p0[:,[0,2]]
    
    ##   Reference
    #pd = circle(n_agents,r= 1.,h = 1.)
    ##pd[2,:] += 1.
    #pd[2,:] = np.array([1.,1.1,1.2,1.3])
    #pd[5,0] += 0.2
    #pd[5,2] -= 0.2
    #pd[3,0] += 0.2
    #pd[4,2] -= 0.2
    
    ##for i in range(n_agents):
        ##_R = cm.rot(pd[5,i],'z') 
        ##_R = _R @ cm.rot(pd[4,i],'y')
        ##_R = _R @ cm.rot(pd[3,i],'x')
        
        ##_R = R @ _R
        ##[pd[3,i], pd[4,i], pd[5,i]] = ctr.get_angles(_R)
        
    ##pd[:3,:] = R @ pd[:3,:]
    
    ## atTarget:
    ##pd = p0.copy()
    ##p0 = pd.copy()
    
    ##   Puntos 3D de escena
    #P = np.array(SceneP)
    ##P = R @ P
    
    ##   Experiment
    #ret = experiment(directory='2',
               ##k_int = 0.1,
                #pd = pd,
                #p0 = p0,
                #P = P,
                ##set_derivative = True,
                ##tanhLimit = True,
                ##depthOp = 4, Z_set = 1.,
                #t_end = 100)
                ##t_end = 4.2)
                ##t_end = 100)
                ##repeat = True)
                
    #print(ret)
    #view3D('2')
    #return
    #   END Experimento singular
    
    
def main(arg):
    
    ##  Color palettes
    #colorPalette("general.npz", n_colors = 60)
    #colorPalette("default.npz", n_colors = 4)
    #colorPalette("colorPalette_A.npz", n_colors = 4)
    #colorPalette("colorPalette_B.npz", n_colors = 4)
    #return
    
    #test_reg()
    #return 
    
    #   reset Log
    global logFile
    logFile = arg[1]
    
    try:
        with open(logFile,'w') as file:
            file.write("Log file INIT\n")
    except IOError:
        print("Logfile Error."+logFile+" OUT")
        return
    
    #   BEGIN read npz
    
    ##   TODO: test the consenus of hte good cases 
    ##directory = arg[3]
    ##npzfile = np.load(directory+'/data.npz')
    ##arr = []
    #for i in range(20):
        #npzfile = np.load(str(i)+'/data.npz')
        ##npzfile = np.load('data_'+str(i)+'.npz')
        ##print(npzfile['var_arr_2'].max())
        ##print(npzfile['var_arr_3'].max())
        #tmp = npzfile['var_arr']
        #et = npzfile['var_arr_2']
        #ew = npzfile['var_arr_3']
        #mask = npzfile['mask']
        #tmp = np.linalg.norm(tmp,axis = 0)/4
        ##if i == 15:
            ##print(tmp)
        #tmp = tmp[mask == 0]
        #et = et[mask == 0]
        #ew = ew[mask == 0]
        #tmp_max = tmp.max()
        ##print(tmp_max)
        #idx = np.argmax(tmp)
        ##print(idx)
        ##idx = idx[0]
        #print('ec = ' + str(tmp[idx]) + '\tet = ' +str(et[idx]) + '\tew = ' +str(ew[idx]))
        ##print('--')
        
        ##if i == 16:
            ##print(tmp)
        
        ##   Min 3d analysis
        ##npzfile = np.load('data_'+str(i)+'.npz')
        ##arr_nP = npzfile['arr_nP']
        ##mask = npzfile['mask']
        ##arr_nP = np.array(arr_nP, np.int64)
        ##arr_nP -= 4
        ##arr_trial_heat_cons = npzfile['arr_trial_heat_cons']
        ##for j in range(4):
            ###if mask[j] == 0. and arr_nP[j] <95:
            ##if mask[j] == 0. :
                ##if arr_nP[j] == 96:
                    ##arr_nP[j] = 95
                ##arr.append(arr_trial_heat_cons[j,arr_nP[j]])
        ##print(arr_nP)
        ##print(arr_trial_heat_cons[:,arr_nP])
        ##print('--')
       
    ##print(arr)
    ##print(max(arr))
    #return
    
    #   END read npz
    
    #   BEGIN plot fail cases comparison 
    
    #Names = ["W_04_localCirc_Rectification_P/",
             #"W_04_localRand_Rectification_P/",
             #"W_04_localCirc_Rectification_PIG/",
             #"W_04_localRand_Rectification_PIG/"]
    #root = "/home/est_posgrado_edgar.chavez/Consenso/"
    
    #nReps = 20
    #count_mask = np.zeros((4,nReps),dtype = np.int_)
    #for i in range(len(Names)):
        #dirBase = root + Names[i]
        #for k in range(nReps):
            
            #npzfile = np.load(dirBase+str(k)+'/data.npz')
            #mask = npzfile['mask']
            #count_mask[i,k] = mask.sum()
    
    #from matplotlib.ticker import MaxNLocator, FuncFormatter
    
    #fig, ax = plt.subplots(frameon=False)
    #fig.suptitle("Failed cases by step")
    ##ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #ref = np.arange(nReps)+1
    #ax.set_xticks(ref)
    
    
    #npzfile = np.load("default.npz")
    #colors = npzfile["colors"]
    #ax.plot(ref,count_mask[0,:], color = colors[0], label= "EHR-P")#,lw = 0.5)
    #ax.plot(ref,count_mask[1,:], color = colors[1], label= "DHR-P")#,lw = 0.5)
    #ax.plot(ref,count_mask[2,:], color = colors[2], label= "EHR-PI-AG")#,lw = 0.5)
    #ax.plot(ref,count_mask[3,:], color = colors[3], label= "DHR-PI-AG")#,lw = 0.5)
    
    
    #fig.legend( loc=2)
    #ax.set_xlabel("Step")
    #ax.set_ylabel("Failed simulations")
    ##plt.xlim((-0.1,1.1))
    ##plt.ylim((-0.01,0.11))
    #ax.grid(axis='y')
    #ax.set_axisbelow(True)
    
    
    #plt.tight_layout()
    #plt.savefig('Rectification_count.pdf',bbox_inches='tight')
    ##plt.show()
    #plt.close()
    #return 

    #   END plot fail cases comparison 
    
    #   BEGIN Front parallel Part: set Circular
    
    
    
    ##   Local tests  
    
    
    #root = "/home/est_posgrado_edgar.chavez/Consenso/"
    #Names = ["W_04_FrontParallel_Circular_Flat/",
             #"W_04_FrontParallel_Circular_NFlat/",
             #"W_04_FrontParallel_Circular_NFlat_PIG/"]
    #nReps = 5
    #nodes = 20
    
    ##   Plot
    #if arg[2] == 'p':
        #for i in range(len(Names)):
            #dirBase = root + Names[i]
            #colorFile = colorNames[int(i>0)]
            #experiment_plots(dirBase =  dirBase,
                            #kSets = nodes,
                            #colorFile = colorFile)
            
        #return
    
    #job = int(arg[3])
    #sel_case = int(arg[4])
    
    ##  process
    #dirBase = root + Names[sel_case]
    #colorFile = colorNames[int(sel_case>0)]
    
    #logText = "Set = "+ dirBase+'\n'
    #write2log(logText)
    
    #logText = "Job = "+str(job)+'\n'
    #write2log(logText)
    
    #if sel_case < 2:
        #experiment_frontParallel(nReps = nReps,
                                 #repeat = True, #   Only if already created
                     #t_end = 800,
                    #dirBase = dirBase,
                    #node = job,
                    #Flat = (sel_case == 0))
    #else:
        #experiment_frontParallel(nReps = nReps,
                                 #repeat = True,
                        #t_end = 800,
                        #gamma0 = 3.,
                        #gammaInf = .1,
                        #intGamma0 = 0.1,
                        #intGammaInf = 0.01,
                        #intGammaSteep = 5,
                        #dirBase = dirBase,
                        #node = job)
    #write2log("LOG FINISHED \n")
    #return 
    
    
    #   END Front parallel Part: set Circular
    #   BEGIN Front parallel Part: set Random
    
    
    
    ##   Local tests  
    
    #root = "/home/est_posgrado_edgar.chavez/Consenso/"
    #Names = ["W_04_FrontParallel_Random_Flat/",
             #"W_04_FrontParallel_Random_NFlat/",
             #"W_04_FrontParallel_Random_NFlat_PIG/"]
    #nReps = 5
    #nodes = 20
    
    ##   Plot
    #if arg[2] == 'p':
        #for i in range(len(Names)):
            #dirBase = root + Names[i]
            #colorFile = colorNames[int(i>0)]
            #experiment_plots(dirBase =  dirBase,
                            #kSets = nodes,
                            #colorFile = colorFile)
            
        #return
    
    #job = int(arg[3])
    #sel_case = int(arg[4])
    
    ##  process
    #dirBase = root + Names[sel_case]
    #colorFile = colorNames[int(sel_case>0)]
    
    #logText = "Set = "+ dirBase+'\n'
    #write2log(logText)
    
    #logText = "Job = "+str(job)+'\n'
    #write2log(logText)
    
    #if sel_case < 2:
        #experiment_frontParallel(nReps = nReps,
                        #t_end = 800,
                        #n_points = 30,
                        #repeat = True, #   Only if already created
                        #dirBase = dirBase,
                        #node = job,
                        #Flat = (sel_case == 0))
    #else:
        #experiment_frontParallel(nReps = nReps,
                        #t_end = 800,
                         #gamma0 = 3.,
                        #gammaInf = .1,
                        #intGamma0 = 0.1,
                        #intGammaInf = 0.01,
                        #repeat = True,
                        #dirBase = dirBase,
                        #node = job)
    #write2log("LOG FINISHED \n")
    #return 
    
    
    #   END Front parallel Part: set Random
    
    
    ##   BEGIN CLUSTER Local
    
    #    Local tests  
    series = int(arg[3])
    
    root = "/home/est_posgrado_edgar.chavez/Consenso/"
    
    ##   Local
    if series == 0:
        Names = ["W_05_localCirc_P/",
                "W_05_localRand_P/",
                "W_05_localCirc_PIG/",
                "W_05_localRand_PIG/"]
        setRectification = False
        gdl = 1
        leader = None
        
    
    # Local + rectification
    if series == 1:
        Names = ["W_05_localCirc_P_r/",
                "W_05_localRand_P_r/",
                "W_05_localCirc_PIG_r/",
                "W_05_localRand_PIG_r/"]
        setRectification = True
        gdl = 2
        leader = None
    
    
    ## Local + leader
    if series == 2:
        Names = ["W_05_localCirc_P_l/",
                "W_05_localRand_P_l/",
                "W_05_localCirc_PIG_l/",
                "W_05_localRand_PIG_l/"]
        setRectification = False
        gdl = 1
        leader = 0
    
    
    ###   Plot part
    if arg[2] == 'p':
        #i = 0
        #colorFile = colorNames[i%2]
        #plot_tendencias(dirBase = root + Names[i], 
                        #colorFile = colorFile)
        #return
        
        for i in range(len(Names)):
            colorFile = colorNames[i%2]
            plot_tendencias(dirBase = root + Names[i], 
                            colorFile = colorFile)
            
        return
    
    #   Proc part
    job = int(arg[3])
    sel = int(arg[4])
    i = job
    
    #  process
    name = Names[sel]
    colorFile = colorNames[sel%2]
    
    logText = "Set = "+root + name+'\n'
    write2log(logText)
    
    logText = "Repetition = "+str(i)+'\n'
    write2log(logText)
    #   Proporcional
    if sel == 0 or sel == 1:
        experiment_repeat(nReps = 100,
                          t_end = 800,
                          setRectification = setRectification,
                          gdl = gdl,
                          leader = leader,
                        dirBase = root + name+str(i)+"/",
                        enablePlotExp= False)
    #   PI AG
    if sel == 2 or sel ==3:
        experiment_repeat(nReps = 100,
                          t_end = 800,
                          setRectification = setRectification,
                          gdl = gdl,
                          leader = leader,
                        gamma0 = 3.,
                        gammaInf = .1,
                        intGamma0 = 0.1,
                        intGammaInf = 0.01,
                        dirBase = root + name+str(i)+"/",
                        enablePlotExp= False)
    #experiment_plots(dirBase = root + name+str(i)+"/", 
                     #colorFile = colorFile)
    
    write2log("LOG FINISHED \n")
    return
    
    #   END Cluster local
    #   BEGIN Cluster local: SETUP
    
    #root = "/home/est_posgrado_edgar.chavez/Consenso/"
    
    ##   Span:
    ##   local
    ##names = ["W_05_localCirc_Ref/",
             ##"W_05_localRand_Ref/"]
    ##dArray = np.r_[np.linspace(0.1,2.,20).reshape((1,20)),
                    ##np.linspace(0.1,2.,20).reshape((1,20)),
                    ##np.linspace(0.1,2.,20).reshape((1,20)),
                    ##np.linspace(0.1,np.pi/2,20).reshape((1,20)),
                    ##np.linspace(0.1,np.pi/2,20).reshape((1,20)),
                    ##np.linspace(0.1,np.pi/2,20).reshape((1,20))]
    ###   Reference
    #names = ["W_05_localCirc_Ref_r/",
             #"W_05_localRand_Ref_r/"]
    #dArray = np.r_[np.linspace(0.1,2.,20).reshape((1,20)),
                    #np.linspace(0.1,2.,20).reshape((1,20)),
                    #np.linspace(0.1,2.,20).reshape((1,20)),
                    #np.linspace(0.1,np.pi/6,20).reshape((1,20)),
                    #np.zeros((1,20)),
                    #(np.pi/2) * np.ones(20).reshape((1,20))]
             
    #for i in range(100):
        #experiment_local(n_points = 30,
                    #dirBase = root+names[0],
                    #caseId = i,
                    #dArray = dArray,
                    #activatepdCirc = True)    
    
    #for i in range(100):
        #experiment_local(n_points = 30,
                    #dirBase = root+names[1],
                    #caseId = i,
                    #dArray = dArray,
                    #activatepdCirc = False)    
        
    #return
    
    
    #   END Cluster local: SETUP
    
    #   BEGIN   testing required points

    
    #Names= ["W_04_nPoints_Circular/",
            #"W_04_nPoints_Circular_PIG/",
            #"W_04_nPoints_Random/",
            #"W_04_nPoints_Random_PIG/"]
    
    #root = "/home/est_posgrado_edgar.chavez/Consenso/"
    
    #nReps = 4   # por nodo
    ##nReps = 5   # por nodo
    #nodos = 25
    ##nodos = 20
    
    ##   Plot part
    #if arg[2] == 'p':
        #i = 2
        #dirBase = root + Names[i]
        #colorFile = colorNames[int(np.floor(i/2))]
        #plot_minP_data(dirBase, nodos,nReps, colorFile = colorFile)
        #experiment_plots(dirBase = dirBase, kSets = nodos, colorFile = colorFile)
        #return 
        #for i in range(len(Names)):
            #dirBase = root + Names[i]
            #colorFile = colorNames[int(np.floor(i/2))]
            #plot_minP_data(dirBase, nodos,nReps, colorFile = colorFile)
            #experiment_plots(dirBase = dirBase, kSets = nodos, colorFile = colorFile)
        #return
    
    ##  process
    #job = int(arg[3])
    #sel_case = int(arg[4])
    
    #dirBase = root + Names[sel_case]
    #logText = "Set = "+str(job)+'\n'
    #write2log(logText)
    #experiment_3D_min(nReps = nReps,
                    #dirBase = dirBase,
                    #node = job,
                    #sel_case = sel_case,
                    #enablePlotExp = False)
    
    #write2log("LOG FINISHED \n")
    
    #return

    #   END testing required points
    
    return

if __name__ ==  "__main__":
    
    if os.getlogin( ) == 'bloodfield':
        mainLocal(sys.argv)
    else:
        main(sys.argv)
