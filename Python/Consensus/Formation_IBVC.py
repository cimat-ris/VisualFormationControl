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
    elif depthOp == 6:
        Z = agent.camera.p[2]*np.ones(P.shape[1])
        Z = Z-P[2,:]
    elif depthOp ==2: # 
        Z = p0[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 3: # distancia al valor inicial
        Z = pd[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 4: # fijo al valor de Z_set
        Z = Z_set * np.ones(P.shape[1])
    elif depthOp == 5: # equitativo, = promedio
        tmp = agent.camera.p[2]-np.mean(P[2,:])
        Z = Z*tmp
    else:
        print("Invalid depthOp")
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
               desired_configuration,
               color, i,
               camera_scale    = 0.02):
    
    #ax.plot(positionArray[0,:],
    ax.scatter(positionArray[0,:],
            positionArray[1,:],
            positionArray[2,:],
            label = str(i),
            color = color) # Plot camera trajectory
    
    camera.draw_camera(ax, scale=camera_scale, color='red')
    camera.pose(desired_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='brown')
    camera.pose(init_configuration)
    camera.draw_camera(ax, scale=camera_scale, color='blue')
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
    
    colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    name = directory+"/3Dplot"
    #fig.suptitle(label)
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    for i in range(n_agents):
        cam = cm.camera()
        cam.pose(end_position[i,:])
        plot_3Dcam(ax, cam,
                pos_arr[i,:,:],
                p0[:,i],
                pd[:,i],
                color = colors[i],
                i = i,
                camera_scale    = 0.02)
    
    #f_size = 1.1*max(abs(p0[:2,:]).max(),abs(pd[:2,:]).max())
    lfact = 1.1
    x_min = lfact*min(p0[0,:].min(),pd[0,:].min())
    x_max = lfact*max(p0[0,:].max(),pd[0,:].max())
    y_min = lfact*min(p0[1,:].min(),pd[1,:].min())
    y_max = lfact*max(p0[1,:].max(),pd[1,:].max())
    z_max = lfact*max(p0[2,:].max(),pd[2,:].max())
    
    if not xLimit is None:
        ax.set_xlim(xLimit[0],xLimit[1])
    if not yLimit is None:
        ax.set_ylim(yLimit[0],yLimit[1])
    if not zLimit is None:
        ax.set_zlim(zLimit[0],zLimit[1])
    
    fig.legend( loc=1)
    plt.show()
    plt.close()



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

def error_state(reference,  agents, name= None):
    
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
    for i in range(n):
        
        #   update new_state
        _R = R.T @ agents[i].camera.R
        [new_state[3,i], new_state[4,i], new_state[5,i]] = ctr.get_angles(_R)
        agents[i].camera.pose(new_state[:,i])
        
        #   Get error
        _R =  cm.rot(new_reference[3,i],'x') @ agents[i].camera.R.T
        _R = cm.rot(new_reference[4,i],'y') @ _R
        _R = cm.rot(new_reference[5,i],'z') @ _R
        rot_err[i] = np.arccos((_R.trace()-1.)/2.)
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
    
    for i in range(n):
        agents[i].camera.draw_camera(ax, scale=0.2, color='red')
        agents[i].camera.pose(new_reference[:,i])
        agents[i].camera.draw_camera(ax, scale=0.2, color='brown')
        agents[i].camera.pose(state[:,i])
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    return [t_err, rot_err]


#   Difference between agents
#   It assumes that the states in the array are ordered and are the same
#   regresa un error de traslación y orientación
def error_state_equal(  agents, name = None):
    
    n = len(agents)
    state = np.zeros((6,n))
    for i in range(n):
        state[:,i] = agents[i].camera.p
    
    reference = np.average(state,axis =1)
    
    t_err = reference[:3].reshape((3,1)) - state[:3,:]
    t_err =  (norm(t_err,axis = 0)**2).sum()/n
    rot_err = reference[3:].reshape((3,1)) - state[3:,:]
    rot_err =  (norm(rot_err,axis = 0)**2).sum()/n
    
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
               depthOp=1,
               Z_set = 1.0,
               gdl = 1,
               dt = 0.05,
               t_end = 10.0,
               zOffset = 0.0,
               case_interactionM = 1,
               adjMat = None,
               p0 = None,
               pd = None,
               P = np.array(SceneP), 
               set_derivative = False,
               atTarget = False,
               tanhLimit = False,
               midMarker = False,
               enablePlot = True,
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
        adjMat = npzfile["adjMat"]
    else:
        #   3D scenc points
        n_points = P.shape[1] #Number of image points
        P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
        
        #   Adjusto case of reference and initial conditions
        if p0 is None and pd is None:
            print("Reference and initial porsitions not provided")
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
        
    #   Conectivity graph
    G = gr.graph(adjMat)
    L = G.laplacian()
    G.plot()
    
    [U,S,V]=svd(L)
    lam_n=S[0]
    lam_2=S[-2]
    alpha=2./(lam_n+lam_2)
    A_ds=np.eye(n_agents)-alpha*L
    
    #   Agents array
    agents = []
    for i in range(n_agents):
        cam = cm.camera()
        agents.append(ctr.agent(cam,pd[:,i],p0[:,i],P,
                                k_int = k_int,
                                set_consensoRef = set_consensoRef,
                                set_derivative = set_derivative))
    
    #   Check initial params 
    for i in range(n_agents):
        if any(Z_select(1, agents[i], P, Z_set, p0, pd, i) < 0.):
            print("invalid configuration")
            return None
    
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
                print("Invalid depth selection")
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
    svdProy_p = np.zeros((n_agents,2*n_points,steps))
    svdProy = np.zeros((n_agents,2*n_points,steps))
    if gdl == 1:
        s_store = np.zeros((n_agents,6,steps))
    elif gdl == 2:
        s_store = np.zeros((n_agents,4,steps))
    elif gdl == 3:
        s_store = np.zeros((n_agents,3,steps))
    FOVflag = False
    state_err = error_state(pd,agents)
    
    #   Print simulation data:
    print("------------------BEGIN-----------------")
    print("Laplacian matrix: ")
    print(L)
    print(A_ds)
    if set_consensoRef:
        print("Reference states")
        print(pd)
    else:
        print("No reference states")
    print("Initial states")
    print(p0)
    print("Scene points")
    print(P[:3,:])
    print("Number of points = "+str(n_points))
    print("Number of agents = "+str(n_agents))
    if zOffset != 0.0:
        print("Z offset for starting conditions = "+str(zOffset))
    print("Time range = ["+str(t)+", "+str(dt)+", "+str(t_end)+"]")
    print("\t Control lambda = "+str(lamb))
    if k_int != 0.:
        print("\t Control Integral gain = "+str(k_int))
    if set_derivative:
        print("\t Derivative component enabled")
    if tanhLimit:
        print("Hyperbolic tangent limit enabled")
    print("Depth estimation = "+depthOp_dict[depthOp])
    if depthOp == 4:
        print("\t Estimated depth set at : "+str(Z_set))
    print("Interaction matrix = "+case_interactionM_dict[case_interactionM])
    print("Control selection = "+control_type_dict[control_type])
    print("Controllable case = "+case_controlable_dict[gdl])
    print("Directory = ", directory)
    
    #   LOOP
    for i in range(steps):
        
        #print("loop",i, end="\r")
        
        #   Error:
        
        error = np.zeros((n_agents,2*n_points))
        error_p = np.zeros((n_agents,2*n_points))
        for j in range(n_agents):
            error[j,:] = agents[j].error
            error_p[j,:] = agents[j].error_p
        error = L @ error
        if set_derivative:
            for j in range(n_agents):
                error[j,:] = error[j,:]+G.deg[j]*agents[j].dot_s_current_n
        error_p = L @ error_p
        
        #   save data
        #print(error)
        err_array[:,:,i] = error_p
        if set_consensoRef:
            [serr_array[0,i], serr_array[1,i]] = error_state(pd,agents)
        else:
            [serr_array[0,i], serr_array[1,i]] = error_state_equal(agents)
        
        ####   Image based formation
        if control_type ==2:
            H = ctr.get_Homographies(agents)
        #   Get control
        for j in range(n_agents):
            
            #   save data 
            desc_arr[j,:,i] = agents[j].s_current.T.reshape(2*n_points)
            pos_arr[j,:,i] = agents[j].camera.p.T.copy()
            
            #   Depth calculation
            Z = Z_select(depthOp, agents[j], P,Z_set,p0,pd,j)
            if Z is None:
                return None
            
            #   Control
            if control_type == 1:
                args = {"deg":G.deg[j] , 
                        "control_sel":case_interactionM,
                        "error": error[j,:],
                        "gdl":gdl}
            elif control_type == 2:
                args = {"H" : H[j,:,:,:],
                        "delta_pref" : delta_pref[j,:,:,:],
                        "Adj_list":G.list_adjacency[j][0],
                        "gamma": gamma[j]}
            else:
                print("invalid control selection")
                return None
            
            #s = None
            U,u, s, vh  = agents[j].get_control(control_type,lamb,Z,args)
            s_store[j,:,i] = s
            if tanhLimit:
                U = 0.3*np.tanh(U)
            
            #   Detección de choque del plano de cámara y los puntos de escena
            if agents[j].count_points_in_FOV(P) != n_points:
                if depthFlags[j] == 0. :
                    depthFlags[j] = i
                U =  np.array([0.,0.,0.,0.,0.,0.])
                FOVflag = True
                break
            
            #   Save error proyection in SVD:
            svdProy[j,:,i] = vh@error[j,:]/norm(error[j,:])
            svdProy_p[j,:,i] = vh@error_p[j,:]/norm(error_p[j,:])
            
            if U is None:
                print("Invalid U control")
                return None
            
            
            U_array[j,:,i] = U
            agents[j].update(U,dt,P, Z)
            
        
        #   Update
        t += dt
        if control_type ==2:
            gamma = A_ds @ gamma #/ 10.0
        
        if FOVflag:
            break
        
    
    ##  Final data
    
    print("----------------------")
    print("Simulation final data")
    
    print(error_p)
    ret_err = norm(error_p,axis=1)
    for j in range(n_agents):
        print(error[j,:])
        print("|Error_"+str(j)+"|= "+str(ret_err[j]))
    for j in range(n_agents):
        print("X_"+str(j)+" = "+str(agents[j].camera.p))
    for j in range(n_agents):
        print("V_"+str(j)+" = "+str(U_array[j,:,-1]))
    
    print("Mean inital heights = ",np.mean(p0[2,:]))
    print("Mean camera heights = ",np.mean(np.r_[p0[2,:],pd[2,:]]))
    
    ##  Trim data if needed
    if FOVflag:
        trim = max(depthFlags)
        t_array = t_array[:trim] 
        err_array = err_array[:,:,:trim]
        serr_array = serr_array[:,:trim]
        U_array = U_array[:,:,:trim] 
        desc_arr = desc_arr[:,:,:trim]
        pos_arr = pos_arr[:,:,:trim]
        svdProy_p = svdProy_p[:,:,:trim]
        svdProy = svdProy[:,:,:trim]
        s_store = s_store[:,:,:trim]
    
    ####   Plot
    
    
    #   Space errors
    new_agents = []
    end_position = np.zeros((n_agents,6))
    for i in range(n_agents):
        cam = cm.camera()
        end_position[i,:] = agents[i].camera.p.copy()
        new_agents.append(ctr.agent(cam,pd[:,i],end_position[i,:],P))
    if set_consensoRef:
        state_err +=  error_state(pd,new_agents,directory+"/3D_error")
    else:
        state_err +=  error_state_equal(new_agents,directory+"/3D_error")
        
    print("State error = "+str(state_err))
    if FOVflag:
        print("WARNING : Cammera plane hit scene points: ", depthFlags)
    if (pos_arr[:,2,:] < 0.).any():
        print("WARNING : Possible plane colition to ground")
    print("-------------------END------------------")
    print()
    #print(err_array)
    #print(U_array)
    #print(pos_arr)
    # Colors setup
    n_colors = max(n_agents,2*n_points)
    colors = randint(0,255,3*n_colors)/255.0
    colors = colors.reshape((n_colors,3))
    
    if not enablePlot:
        return [ret_err, state_err, FOVflag]
    
    #   Camera positions in X,Y 
    mp.plot_position(pos_arr,
                    pd,
                    lfact = 1.1,
                    colors = colors,
                    name = directory+"/Cameras_trayectories")
    
    #   3D plots
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    name = directory+"/3Dplot"
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    for i in range(n_agents):
        plot_3Dcam(ax, agents[i].camera,
                pos_arr[i,:,:],
                p0[:,i],
                pd[:,i],
                color = colors[i],
                i = i,
                camera_scale    = 0.02)
    #   Plot cammera position at intermediate time
    if midMarker:
        step_sel = int(pos_arr.shape[2]/2)
        ax.scatter(pos_arr[:,0,step_sel],
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
    sqrfact = max(width,
                        height,
                        depth)
    
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
    plt.savefig(name+'.pdf',bbox_inches='tight')
    plt.close()
    
    #   Save 3D plot data:
    np.savez(directory + "/data3DPlot.npz",
             P = P, pos_arr=pos_arr, p0 = p0,
             pd = pd, end_position = end_position )
    
    #   Descriptores x agente
    for i in range(n_agents):
        mp.plot_descriptors(desc_arr[i,:,:],
                            agents[i].camera.iMsize,
                            agents[i].s_ref,
                            colors,
                            name = directory+"/Image_Features_"+str(i),
                            label = "Image Features")
    
    #   Error de formación
    mp.plot_time(t_array,
                serr_array[:,:],
                colors,
                ylimits = [-0.1,1.1],
                name = directory+"/State_Error_"+str(i),
                label = "Formation Error",
                labels = ["traslation","rotation"])
    
    #   Errores x agentes
    for i in range(n_agents):
        mp.plot_time(t_array,
                    err_array[i,:,:],
                    colors,
                    ylimits = [-1,1],
                    name = directory+"/Features_Error_"+str(i),
                    label = "Features Error")
    
    #   Velocidaes x agente
    for i in range(n_agents):
        mp.plot_time(t_array,
                    U_array[i,:,:],
                    colors,
                    ylimits = [-1,1],
                    name = directory+"/Velocidades_"+str(i),
                    label = "Velocidades",
                    labels = ["X","Y","Z","Wx","Wy","Wz"])
    
    #   Posiciones x agente
    for i in range(n_agents):
        mp.plot_time(t_array,
                    pos_arr[i,:3,:],
                    colors,
                    name = directory+"/Traslaciones_"+str(i),
                    label = "Traslaciones",
                    labels = ["X","Y","Z"])
        mp.plot_time(t_array,
                    pos_arr[i,3:,:],
                    colors,
                    name = directory+"/Angulos_"+str(i),
                    label = "Angulos",
                    labels = ["Roll","Pitch","yaw"])
    
    #   Valores propios
    for i in range(n_agents):
        mp.plot_time(t_array,
                    s_store[i,:,:],
                    colors,
                    name = directory+"/ValoresP_"+str(i),
                    label = "Valores propios (SVD)",
                    labels = ["0","1","2","3","4","5"])
                    #limits = [[t_array[0],t_array[-1]],[0,20]])
    
    #   Proyección del error proporcional en los vectores principales 
    for i in range(n_agents):
        mp.plot_time(t_array,
                    svdProy_p[i,:,:],
                    colors,
                    name = directory+"/Proy_eP_VH_"+str(i),
                    label = "Proy($e_p$) en $V^h$ ",
                    labels = ["0","1","2","3","4","5","6","7"])
                    #limits = [[t_array[0],t_array[-1]],[0,20]])
    
    #   Valores propios
    for i in range(n_agents):
        mp.plot_time(t_array,
                    svdProy[i,:,:],
                    colors,
                    name = directory+"/Proy_et_VH_"+str(i),
                    label = "Proy($e$) en $V^h$ ",
                    labels = ["0","1","2","3","4","5","6","7"])
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


def experiment_repeat(nReps = 1,
                      dirBase = "",
                      k_int = 0,
                      intMatSel = 1,
                      enablePlotExp = True):
    
    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    mask = np.zeros(nReps)
    Misscount = 0
    
    intMatSelDict = {1:"Real",
                     2:"Constant depth"}
    print("test series, Base Directory = ",dirBase)
    print("Test series, Interaction Matrix = ", intMatSelDict[intMatSel])
    print("Test series, Integral gain = ", k_int)
        
    for k in range(nReps):
        if intMatSel  == 1:
            ret = experiment(directory=dirBase+str(k),
                            k_int = k_int,
                                t_end = 100,
                                enablePlot = enablePlotExp,
                                repeat = True)
        if intMatSel  == 4:
            ret = experiment(directory=dirBase+str(k),
                            k_int = k_int,
                                depthOp = 4, Z_set = 1.,
                                t_end = 100,
                                enablePlot = enablePlotExp,
                                repeat = True)
        [var_arr[:,k], errors, FOVflag] = ret
        var_arr_et[k] = errors[0]
        var_arr_er[k] = errors[1]
        var_arr_2[k] = errors[2]
        var_arr_3[k] = errors[3]
        
        if FOVflag:
            mask[k] = 1
            Misscount += 1
    
    np.savez(dirBase+'data.npz',
            n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            mask = mask,
            Misscount = Misscount)
    
    
def experiment_all_random(nReps = 100, 
                          conditions = 1,
                          pFlat = False,
                          dirBase = "",
                          nP = 4,
                          enablePlotExp = True):
    
    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    var_arr_et = np.zeros(nReps)
    var_arr_er = np.zeros(nReps)
    Misscount = 0
    
    conditionsDict = {1:"Random",
                      2:"Circular plus a rigid transformation",
                      3:"Circular, fixed",
                      4:"Circular, plus a perturbation"}
    
    print("Test series, References = ", conditionsDict[conditions])
    print("Test series, pFlat = ", pFlat)
    print("Test series, nP = ", nP)
    
    for k in range(nReps):
        FOVflag = True
        while FOVflag:
            #   Points
            # Range
            xRange = [-2,2]
            yRange = [-2,2]
            zRange = [-1,0]
            
            #nP = random.randint(4,11)
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
                cam = cm.camera()
                agent = ctr.agent(cam, tmp, tmp,P)
                
                while agent.count_points_in_FOV(Ph) != nP :
                    tmp = np.random.rand(6)
                    tmp = offset + tmp*dRange
                    cam = cm.camera()
                    agent = ctr.agent(cam, tmp, tmp,P)
                
                p0[:,i] = tmp.copy()
                
                #   BEGIN referencias aleatorias
                if conditions == 1:
                    tmp = np.random.rand(6)
                    tmp = offset + tmp*dRange
                    cam = cm.camera()
                    agent = ctr.agent(cam, tmp, tmp,P)
                    
                    while agent.count_points_in_FOV(Ph) != nP:
                        tmp = np.random.rand(6)
                        tmp = offset + tmp*dRange
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
                
            
            print("CASE ",k)
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
                Misscount += 1
        
    #   Save data
    
    np.savez(dirBase+'data.npz',
                n_agents = n_agents,
            var_arr = var_arr,
            var_arr_2 = var_arr_2,
            var_arr_3 = var_arr_3,
            var_arr_et = var_arr_et,
            var_arr_er = var_arr_er,
            mask = np.zeros(nReps),
            Misscount = Misscount)
    
def experiment_plots(dirBase = ""):
    
    npzfile = np.load(dirBase+'data.npz')
    n_agents = npzfile['n_agents']
    var_arr = npzfile['var_arr']
    var_arr_2 = npzfile['var_arr_2']
    var_arr_3 = npzfile['var_arr_3']
    var_arr_et = npzfile['var_arr_et']
    var_arr_er = npzfile['var_arr_er']
    Misscount = npzfile['Misscount']
    mask = npzfile['mask']
    nReps = var_arr.shape[1]
    
    print("Simulations that failed = ",Misscount+mask.sum()," / ", nReps+Misscount)
    
    etsup = np.where((var_arr_2 > var_arr_et) &
                     (mask == 0.))[0]
    ersup = np.where((var_arr_3 > var_arr_er) &
                     (mask == 0.))[0]
    esup = np.where((var_arr_3 > var_arr_er) &
                    (var_arr_2 > var_arr_et) &
                     (mask == 0.))[0]
    
    
    print("Simulations that increased et = ", etsup.shape[0], " / ", nReps)
    print(etsup)
    print("Simulations that increased er = ", ersup.shape[0], " / ", nReps)
    print(ersup)
    print("Simulations that increased both err = ", esup.shape[0], " / ", nReps)
    print(esup)
    
    #   Masking
    print("Failed repeats = ", np.where(mask == 1))
    var_arr = var_arr[:,mask==0.]
    var_arr_2 = var_arr_2[mask==0.]
    var_arr_3 = var_arr_3[mask==0.]
    var_arr_et = var_arr_et[mask==0.]
    var_arr_er = var_arr_er[mask==0.]
    nReps = var_arr.shape[1]
    ref_arr = np.arange(nReps)
    
    #   Plot data
    
    colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    
    ##      Simulation - wise consensus error
    fig, ax = plt.subplots()
    fig.suptitle("Error de consenso por experimento")
    #plt.ylim([-2.,2.])
    for i in range(n_agents):
        ax.scatter(ref_arr, var_arr[i,:], 
                marker = "x", alpha = 0.5,color=colors[i])
    
    plt.yscale('logit')
    plt.tight_layout()
    plt.savefig(dirBase+'Consensus error_byExp.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##   Kernel density consensus
    fig, ax = plt.subplots()
    fig.suptitle("Error de consenso (Densidad por Kernels)")
    #plt.ylim([-2.,2.])
    
    var_arr = norm(var_arr,axis = 0)/n_agents
    density = gaussian_kde(var_arr)
    dh = var_arr.max() - var_arr.min()
    xs = np.linspace(var_arr.min()-dh*0.1,var_arr.max()+dh*0.1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs),color=colors[0])
    plt.scatter(var_arr,np.zeros(nReps),
                marker = "|", alpha = 0.5,color=colors[0])
    plt.tight_layout()
    plt.savefig(dirBase+'Consensus error_Kernel_density.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Histogram consensus
    fig, ax = plt.subplots()
    fig.suptitle("Histograma de error de consenso")
    counts, bins = np.histogram(var_arr)
    plt.stairs(counts, bins)
    plt.tight_layout()
    plt.savefig(dirBase+'Consensus error_Histogram.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
    ##  Histogram consensus Zoom
    fig, ax = plt.subplots()
    fig.suptitle("Histograma de error de consenso (zoom)")
    counts, bins = np.histogram(var_arr[var_arr < 0.5])
    plt.stairs(counts, bins)
    plt.tight_layout()
    plt.savefig(dirBase+'Consensus error_Histogram_zoom.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
    
    ##  Simulation - wise formation error
    fig, ax = plt.subplots()
    fig.suptitle("Errores de estado por experimento")
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
    fig.suptitle("Errores de estado")
    ax.scatter(var_arr_et,var_arr_er, 
            marker = "*", alpha = 0.5,color = colors[1])
    ax.scatter(var_arr_2,var_arr_3, 
            marker = "*", alpha = 0.5,color = colors[0])
    for i in range(nReps):
        ax.plot([var_arr_et[i],var_arr_2[i]],
                [var_arr_er[i],var_arr_3[i]],
                alpha = 0.5, color = colors[0],
                linewidth = 0.5)
    ax.set_xlabel("Error de traslación")
    ax.set_ylabel("Error de rotación")
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_Scatter.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    
    
    #   Kernel plots formation error
    
    fig, ax = plt.subplots()
    fig.suptitle("Errores de traslación")
    density = gaussian_kde(var_arr_2)
    dh = var_arr_2.max() - var_arr_2.min()
    xs = np.linspace(var_arr_2.min()-dh*0.1,var_arr_2.max()+dh*0.1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs),color=colors[0])
    plt.scatter(var_arr_2,np.zeros(nReps),
                marker = "|", alpha = 0.5,color=colors[0])
    
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_Kernel_desity_T.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    fig, ax = plt.subplots()
    fig.suptitle("Errores de rotación")
    density = gaussian_kde(var_arr_3)
    dh = var_arr_3.max() - var_arr_3.min()
    xs = np.linspace(var_arr_3.min()-dh*0.1,var_arr_3.max()+dh*0.1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs),color=colors[1])
    plt.scatter(var_arr_3,np.zeros(nReps),
                marker = "|", alpha = 0.5,color=colors[1])
    
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_Kernel_desity_R.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Heatmap
    h, x, y, img = plt.hist2d(var_arr_2,var_arr_3)
    fig, ax = plt.subplots()
    fig.suptitle("Heatmap de errores de formación")
    
    h = h.T
    ax.imshow(h)
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
            text = ax.text(j, i, h[i, j],
                        ha="center", va="center", color="w")

    
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_heatmap.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Heatmap Zoom
    var_arr_3 = var_arr_3[var_arr_2 < 0.2]
    var_arr_2 = var_arr_2[var_arr_2 < 0.2]
    var_arr_2 = var_arr_2[var_arr_3 < 0.2]
    var_arr_3 = var_arr_3[var_arr_3 < 0.2]
    h, x, y, img = plt.hist2d(var_arr_2,var_arr_3)
    fig, ax = plt.subplots()
    fig.suptitle("Heatmap de errores de formación (Zoom)")
    
    h = h.T
    ax.imshow(h)
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
                        ha="center", va="center", color="w")

    
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_heatmap_zoom.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    ##  Scatter err T,R Zoom
    fig, ax = plt.subplots()
    fig.suptitle("Errores de estado (Zoom)")
    ax.scatter(var_arr_2,var_arr_3, 
            marker = "*", alpha = 0.5,color = colors[0])
    ax.set_xlabel("Error de traslación")
    ax.set_ylabel("Error de rotación")
    plt.tight_layout()
    plt.savefig(dirBase+'Formation error_Scatter_zoom.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    

################################################################################
################################################################################
#
#   M   A   I   N
#
################################################################################
################################################################################

    
def main():
    
    #   REPEAT
    #experiment(directory="6",
                #t_end = 100,
                #repeat = True)
    #view3D('6')
    #return

    ##   VIEWER
    #view3D('3', xLimit = [-10,10], yLimit = [-10,10],zLimit = [0,20])
    ##view3D('5')
    ##view3D('20')
    ##view3D('26')
    #return 
    
    ##  TODO
    ##  Exprimentos con n de puntos fijos
    ##  Experimentos con puntos PZ = 0
    ##  Experimento con caso 2 'transformaciones
    ##  + Integrador con el anterior 
    ##  + Profundidades constantes con el anterior
    ##  Mayor número de agentes
    
    k_int = 0.1
    intMatSel = 1
    
    #   Experimento con puntos planos, 10 puntos, all random
    #experiment_all_random(nReps = 100, 
                          #conditions = 1,
                          #pFlat = True,
                          #dirBase = "rand_flat_10/",
                          #nP = 10,
                          #enablePlotExp= False)
    #experiment_repeat(nReps = 100,
                      #dirBase = "rand_flat_10/",
                      #k_int = k_int ,
                      #intMatSel = intMatSel,
                        #enablePlotExp = False)
    experiment_plots(dirBase = "rand_flat_10/")
    
    
    #   Experimento con puntos planos, 4 puntos, all random
    #experiment_all_random(nReps = 100, 
                          #conditions = 1,
                          #pFlat = True,
                          #dirBase = "rand_flat_4/",
                          #nP = 4,
                          #enablePlotExp= False)
    experiment_repeat(nReps = 100,
                      dirBase = "rand_flat_4/",
                      k_int = k_int ,
                      intMatSel = intMatSel,
                        enablePlotExp = False)
    experiment_plots(dirBase = "rand_flat_4/")
    
    
    #   Experimento con  10 puntos, all random
    #experiment_all_random(nReps = 100, 
                          #conditions = 1,
                          #dirBase = "rand_10/",
                          #nP = 10,
                          #enablePlotExp= False)
    experiment_repeat(nReps = 100,
                      dirBase = "rand_10/",
                      k_int = k_int ,
                      intMatSel = intMatSel,
                        enablePlotExp = False)
    experiment_plots(dirBase = "rand_10/")
    
    
    #   Experimento con  4 puntos, all random
    #experiment_all_random(nReps = 100, 
                          #conditions = 1,
                          #dirBase = "rand_4/",
                          #nP = 4,
                          #enablePlotExp= False)
    experiment_repeat(nReps = 100,
                      dirBase = "rand_4/",
                      k_int = k_int ,
                      intMatSel = intMatSel,
                        enablePlotExp = False)
    experiment_plots(dirBase = "rand_4/")
    
    #   Experimento con  10 puntos, circle
    #experiment_all_random(nReps = 100, 
                          #conditions = 3,
                          #dirBase = "circ_10/",
                          #nP = 10,
                          #enablePlotExp= False)
    experiment_repeat(nReps = 100,
                      dirBase = "circ_10/",
                      k_int = k_int ,
                      intMatSel = intMatSel,
                        enablePlotExp = False)
    experiment_plots(dirBase = "circ_10/")
    
    
    #   Experimento con  4 puntos, circle
    #experiment_all_random(nReps = 100, 
                          #conditions = 3,
                          #dirBase = "circ_4/",
                          #nP = 4,
                          #enablePlotExp= False)
    experiment_repeat(nReps = 100,
                      dirBase = "circ_4/",
                      k_int = k_int ,
                      intMatSel = intMatSel,
                        enablePlotExp = False)
    experiment_plots(dirBase = "circ_4/")
    
    
    #   Experimento con  10 puntos planos, circle
    #experiment_all_random(nReps = 100, 
                          #conditions = 3,
                          #pFlat = True,
                          #dirBase = "circ_flat_10/",
                          #nP = 10,
                          #enablePlotExp= False)
    experiment_repeat(nReps = 100,
                      dirBase = "circ_flat_10/",
                      k_int = k_int ,
                      intMatSel = intMatSel,
                        enablePlotExp = False)
    experiment_plots(dirBase = "circ_flat_10/")
    
    
    #   Experimento con  4 puntos planos, circle
    #experiment_all_random(nReps = 100, 
                          #conditions = 3,
                          #pFlat = True,
                          #dirBase = "circ_flat_4/",
                          #nP = 4,
                          #enablePlotExp= False)
    experiment_repeat(nReps = 100,
                      dirBase = "circ_flat_4/",
                      k_int = k_int ,
                      intMatSel = intMatSel,
                        enablePlotExp = False)
    experiment_plots(dirBase = "circ_flat_4/")
    
    
    return
    
    #   Experimento exhaustivo de posiciones y referencias random
    #experiment_all_random(nReps = 100, 
                          #enablePlotExp= False)
    
    #   Experimento anterior con profundidad constante
    #experiment_repeat(nReps = 10,
                      #enablePlotExp = False)
    
    #   Plot data
    #experiment_plots(dirBase = "rand_flat_10/")
    
    #return
    
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
    
    
    

if __name__ ==  "__main__":
    main()
