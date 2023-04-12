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

def circle(n_agents,r,h):
    Objective = np.zeros((6,n_agents))
    step = 2*pi/n_agents
    ang_arange = np.arange(0,2*pi,step)-step/2.0
    Objective[0,:] = r*cos(ang_arange)
    Objective[1,:] = r*sin(ang_arange)
    Objective[2,:] = h
    Objective[3,:] = pi
    #Objective[5,:] = ang_arange
    
    #Objective = Objective[:,[1,2,3,0]]
    
    return Objective

def Z_select(depthOp, agent, P, Z_set, p0, pd, j):
    Z = np.ones((1,P.shape[1]))
    if depthOp ==1:
        #   TODO creo que esto está mal calculado
        #M = np.c_[ agent.camera.R.T, -agent.camera.R.T @ agent.camera.p[:3] ]
        #Z = M @ P
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

def plot_err_consenso(ref_arr,
                      arr_error,
                      colors,
                      labels,
                      limits = None,
                      enableShow = False,
                      title = "Error de consenso",
                      filename = "ConsensusError.pdf"):
    
    #   Variantes
    n = arr_error.shape[0]
    #   agentes
    m = arr_error.shape[0]
    
    fig, ax = plt.subplots()
    fig.suptitle(title)
    
    symbols = []
    for i in range(n):
        for j in range(m):
            ax.scatter(ref_arr,arr_error[i,j,:],
                        marker = "x", alpha = 0.5, color=colors[i])
        symbols.append(mpatches.Patch(color=colors[i]))
        
    
    fig.legend(symbols,labels, loc=1)
    
    plt.yscale('logit')
    if not limits is None:
        plt.ylim(limits)
    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')
    if enableShow:
        plt.show()
    plt.close()

def plot_err_formacion(ref_arr,
                      arr_error,
                      colors,
                      labels,
                      limits = None,
                      enableShow = False,
                      title = "Error de formación",
                      filename = "FormationError.pdf"):
    #   Variantes
    n = arr_error.shape[0]
    
    fig, ax = plt.subplots()
    fig.suptitle(title)
    
    symbols = []
    for i in range(n):
        ax.scatter(ref_arr,arr_error[i,0,:],
                   marker='.', label  = labels[i]+ " Tras", 
                   alpha = 0.5, color = colors[i])
        ax.scatter(ref_arr,arr_error[i,1,:],
                   marker='*', label  = labels[i]+ " Rot", 
                   alpha = 0.5, color = colors[i])
    
    fig.legend( loc=1)
    if not limits is None:
        plt.ylim(limits)
    plt.tight_layout()
    plt.savefig(filename,bbox_inches='tight')
    if enableShow:
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
        #state[:3,i] = agents[i].camera.p
        #state[3,i] = agents[i].camera.roll
        #state[4,i] = agents[i].camera.pitch
        #state[5,i] = agents[i].camera.yaw
    
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
    #new_state[:3,:] = new_state[:3,:] - centroide_state
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
    #new_state[:3,:] /= norm(new_state[:3,:],axis = 0).mean()
    f = lambda r : (norm(new_reference[:3,:] - r*new_state[:3,:],axis = 0)**2).sum()/n
    r_state = minimize_scalar(f, method='brent')
    t_err = f(r_state.x)
    t_err = np.sqrt(t_err)
    #print("scale diff = ",norm(new_state[:3,:],axis = 0).mean(),'*',  r_state.x)
    #print("scale diff = ",norm(new_state[:3,:],axis = 0).mean()*  r_state.x)
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
        #state[:3,i] = agents[i].camera.p
        #state[3,i] = agents[i].camera.roll
        #state[4,i] = agents[i].camera.pitch
        #state[5,i] = agents[i].camera.yaw
    
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
               repeat = None):
    
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
        
        
        #   Parameters
        
        case_n = 1  #   Case selector if needed
        
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
    #G.plot()
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
            #pos_arr[j,:,i] = np.r_[agents[j].camera.p.T ,
                                   #agents[j].camera.roll,
                                   #agents[j].camera.pitch,
                                   #agents[j].camera.yaw]
            
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
            #Z_test = Z_select(1, agents[j], P,Z_set,p0,pd,j)
            #if any(Z_test < 0.):
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
                break
            
            
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
    
    ret_err = norm(error_p,axis=1)
    for j in range(n_agents):
        print(error[j,:])
        print("|Error_"+str(j)+"|= "+str(ret_err[j]))
    for j in range(n_agents):
        print("X_"+str(j)+" = "+str(agents[j].camera.p))
    for j in range(n_agents):
        print("V_"+str(j)+" = "+str(U_array[j,:,-1]))
    #for j in range(n_agents):
        #print("Angles_"+str(j)+" = "+str(agents[j].camera.roll)+
              #", "+str(agents[j].camera.pitch)+", "+str(agents[j].camera.yaw))
    
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
        #end_position[i,:] = np.r_[agents[i].camera.p,
                                  #agents[i].camera.roll,
                                  #agents[i].camera.pitch,
                                  #agents[i].camera.yaw]
        new_agents.append(ctr.agent(cam,pd[:,i],end_position[i,:],P))
    if set_consensoRef:
        state_err = error_state(pd,new_agents,directory+"/3D_error")
    else:
        state_err = error_state_equal(new_agents,directory+"/3D_error")
        
    print("State error = "+str(state_err))
    if FOVflag:
        print("WARNING : Cammera plane hit scene points: ", depthFlags)
    print("-------------------END------------------")
    print()
    #print(err_array)
    #print(U_array)
    #print(pos_arr)
    # Colors setup
    n_colors = max(n_agents,2*n_points)
    colors = randint(0,255,3*n_colors)/255.0
    colors = colors.reshape((n_colors,3))
    
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
    
    return [ret_err, state_err[0],state_err[1], FOVflag]



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




def experiment_all_random(nReps = 100, repeat = False):
    
    n_agents = 4
    var_arr = np.zeros((n_agents,nReps))
    var_arr_2 = np.zeros(nReps)
    var_arr_3 = np.zeros(nReps)
    mask = np.zeros(nReps)
    
    for k in range(nReps):
        #   Points
        # Range
        xRange = [-2,2]
        yRange = [-2,2]
        zRange = [-1,0]
        
        nP = random.randint(4,11)
        P = np.random.rand(3,nP)
        P[0,:] = xRange[0]+ P[0,:]*(xRange[1]-xRange[0])
        P[1,:] = yRange[0]+ P[1,:]*(yRange[1]-yRange[0])
        P[2,:] = zRange[0]+ P[2,:]*(zRange[1]-zRange[0])
        Ph = np.r_[P,np.ones((1,nP))]
        
        #   Cameras
        # Range
        xRange = [-2,2]
        yRange = [-2,2]
        zRange = [0,3]
        
        nC = 4
        p0 = np.zeros((6,nC))
        pd = np.zeros((6,nC))
        
        offset = np.array([xRange[0],yRange[0],zRange[0],-np.pi,-np.pi,-np.pi])
        dRange = np.array([xRange[1],yRange[1],zRange[1],np.pi,np.pi,np.pi])
        dRange -= offset
        for i in range(nC):
            
            tmp = np.random.rand(6)
            tmp = offset + tmp*dRange
            cam = cm.camera()
            agent = ctr.agent(cam, np.zeros(6), tmp,P)
            #Z = Z_select(1, agent, Ph,None,None, None,None)
            while agent.count_points_in_FOV(Ph) != nP :
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                cam = cm.camera()
                agent = ctr.agent(cam, np.zeros(6), tmp,P)
                #Z = Z_select(1, agent, Ph,None,None, None,None)
            
            p0[:,i] = tmp.copy()
            
            tmp = np.random.rand(6)
            tmp = offset + tmp*dRange
            cam = cm.camera()
            agent = ctr.agent(cam, np.zeros(6), tmp,P)
            #Z = Z_select(1, agent, Ph,None,None, None,None)
            while agent.count_points_in_FOV(Ph) != nP:
                tmp = np.random.rand(6)
                tmp = offset + tmp*dRange
                cam = cm.camera()
                agent = ctr.agent(cam, np.zeros(6), tmp,P)
                #Z = Z_select(1, agent, Ph,None,None, None,None)
            
            pd[:,i] = tmp.copy()
        
        ret = experiment(directory=str(k),
                #k_int = 0.1,
                    pd = pd,
                    p0 = p0,
                    P = P,
                    #set_derivative = True,
                    #tanhLimit = True,
                    #depthOp = 4, Z_set = 1.,
                    t_end = 100,
                    repeat = repeat)
        [var_arr[:,k], var_arr_2[k], var_arr_3[k], FOVflag] = ret
        if FOVflag:
            mask[i] = 1
        
    
    #   Plot data
    
    fig, ax = plt.subplots()
    fig.suptitle("Error de consenso")
    #plt.ylim([-2.,2.])
    
    colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    
    var_arr = norm(var_arr,axis = 0)
    density = gaussian_kde(var_arr)
    dh = var_arr.max() - var_arr.min()
    xs = np.linspace(var_arr.min()-dh*0.1,var_arr.max()+dh*0.1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs),color=colors[0])
    #for i in range(n_agents):
        #ax.plot(ref_arr,var_arr[i,:] , color=colors[i])
    
    #plt.yscale('logit')
    plt.tight_layout()
    plt.savefig('Consensus error.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    fig, ax = plt.subplots()
    fig.suptitle("Errores de estado")
    #ax.plot(ref_arr,var_arr_2, label = "Posición")
    #ax.plot(ref_arr,var_arr_3, label = "Rotación")
    #fig.legend( loc=2)
    density = gaussian_kde(var_arr_2)
    dh = var_arr_2.max() - var_arr_2.min()
    xs = np.linspace(var_arr_2.min()-dh*0.1,var_arr_2.max()+dh*0.1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs),color=colors[0])
    
    density = gaussian_kde(var_arr_3)
    dh = var_arr_3.max() - var_arr_3.min()
    xs = np.linspace(var_arr_3.min()-dh*0.1,var_arr_3.max()+dh*0.1,200)
    density.covariance_factor = lambda : .25
    density._compute_covariance()
    plt.plot(xs,density(xs),color=colors[1])
    
    plt.tight_layout()
    plt.savefig('Formation error.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    print("Total number of simulations = "+str(nReps))



###########################################################################
#
#   Experiment Initial condiditon variation
#
###########################################################################




def experiment_initalConds(justPlot = False,
                          midMarker = False,
                          noRef = True, # graph WoRef
                          set_derivative = True,
                          r = 0.8,
                        n = 1,
                        k_int = 0.1,
                        t_f = 100):
    
    #   For grid
    #n = n**2
    #sqrtn = int(np.sqrt(n))
    
    if justPlot:
        
        #   Load
        arr_error = np.load('arr_err.npy')
        arr_epsilon = np.load('arr_epsilon.npy')
        n = arr_epsilon.shape[2]
        ref_arr = np.arange(n)
        variantes = arr_error.shape[0]
        
    else:
        
        variantes = 1
        if k_int != 0.:
            variantes *= 2
        if noRef:
            variantes *= 2
        if set_derivative:
            variantes *= 2
        ref_arr = np.arange(n)
        #   4 variantes X 4 agentes X n repeticiones
        arr_error = np.zeros((variantes,4,n))
        #   4 variantes X 2 componentes X n repeticiones
        arr_epsilon = np.zeros((variantes,2,n))
        idx = 0
        for i in range(n):
            
            print(" ---  Case ", i, " RP--- ")
            
            ret = None
            while ret is None:
                
                #   Aleatorio
                #p0=[[0.8,0.8,-0.8,-.8],
                    #[-0.8,0.8,0.8,-0.8],
                p0=[[-0.8,0.8,0.8,-.8],
                    [0.8,0.8,-0.8,-0.8],
                    #[1.4,0.8,1.2,1.6],
                    [1.2,1.2,1.2,1.2],
                    [np.pi,np.pi,np.pi,np.pi],
                    #[0.1,0.1,-0.1,0.1],
                    [0.,0.,-0.,0.],
                    [0,0,0,0]]
                
                p0 = np.array(p0)
                p0[:3,:] += r *2*( np.random.rand(3,p0.shape[1])-0.5)
                p0[2,p0[2,:]<0.6] = 0.6
                #p0[3,:] += 2*( np.random.rand(p0.shape[1])-0.5) * np.pi /7 #4
                #p0[4,:] += 2*( np.random.rand(1,p0.shape[1])-0.5) * np.pi / 4
                p0[5,:] += 2*( np.random.rand(p0.shape[1])-0.5) * np.pi /2
                
                ##   Grid
                #p0=[[0.8,0.8,-0.8,-.8],
                    #[-0.8,0.8,0.8,-0.8],
                ##p0=[[-0.8,0.8,0.8,-.8],
                    ##[0.8,0.8,-0.8,-0.8],
                    ##[1.4,0.8,1.2,1.6],
                    #[1.2,1.2,1.2,1.2],
                    #[np.pi,np.pi,np.pi,np.pi],
                    ##[0.1,0.1,-0.1,0.1],
                    #[0.,0.,-0.,0.],
                    #[0,0,0,0]]
                
                #p0 = np.array(p0)
                #p0[0,:] += 1.*2*(int(i/sqrtn)-sqrtn/2)
                #p0[1,:] += 1.*(i%sqrtn-sqrtn/2)
                
                
                
                #   Con referencia P
                ret = experiment(directory=str(idx),
                                 midMarker = midMarker,
                            h = 1 ,
                            r = 1.,
                            #tanhLimit = True,
                            #depthOp = 4, Z_set=2.,
                            depthOp = 1,
                            p0 = p0,
                            t_end = t_f)
            
            arr_error[idx%variantes,:,i] = ret[0]
            arr_epsilon[idx%variantes,0,i] = ret[1]
            arr_epsilon[idx%variantes,1,i] = ret[2]
            idx += 1
            
            #   Sin referencia P
            if noRef:
                print(" ---  Case ", i, " P--- ")
                ret = experiment(directory=str(idx),
                            midMarker = midMarker,
                            h = 1 ,
                            r = 1.,
                            #tanhLimit = True,
                            #depthOp = 4, Z_set=2.,
                            depthOp = 1,
                            p0 = p0,
                            set_consensoRef = False,
                            t_end = t_f)
                [arr_error[idx%variantes,:,i], arr_epsilon[idx%variantes,0,i], arr_epsilon[idx%variantes,1,i]] = ret
                idx += 1
            
            if k_int != 0.:
                #   Con referencia PI
                print(" ---  Case ", i, " RPI--- ")
                ret = experiment(directory=str(idx),
                            midMarker = midMarker,
                            k_int =k_int,
                            h = 1 ,
                            r = 1.,
                            #tanhLimit = True,
                            #depthOp = 4, Z_set=2.,
                            depthOp = 1,
                            p0 = p0,
                            t_end = t_f)
                [arr_error[idx%variantes,:,i], arr_epsilon[idx%variantes,0,i], arr_epsilon[idx%variantes,1,i]] = ret
                idx += 1
                
                #   Sin referencia PI
                if noRef:
                    print(" ---  Case ", i, " PI--- ")
                    ret = experiment(directory=str(idx),
                                midMarker = midMarker,
                                k_int = k_int,
                                h = 1 ,
                                r = 1.,
                                #tanhLimit = True,
                                #depthOp = 4, Z_set=2.,
                                depthOp = 1,
                                p0 = p0,
                                set_consensoRef = False,
                                t_end = 20)
                    [arr_error[idx%variantes,:,i], arr_epsilon[idx%variantes,0,i], arr_epsilon[idx%variantes,1,i]] = ret
                    idx += 1
                    
            if set_derivative:
                #   Con referencia PD
                print(" ---  Case ", i, " RPD--- ")
                ret = experiment(directory=str(idx),
                            midMarker = midMarker,
                            set_derivative = set_derivative,
                            h = 1 ,
                            r = 1.,
                            #tanhLimit = True,
                            #depthOp = 4, Z_set=2.,
                            depthOp = 1,
                            p0 = p0,
                            t_end = t_f)
                [arr_error[idx%variantes,:,i], arr_epsilon[idx%variantes,0,i], arr_epsilon[idx%variantes,1,i]] = ret
                idx += 1
                
                #   Sin referencia PD
                if noRef:
                    print(" ---  Case ", i, " PD--- ")
                    ret = experiment(directory=str(idx),
                                midMarker = midMarker,
                                set_derivative = set_derivative,
                                h = 1 ,
                                r = 1.,
                                #tanhLimit = True,
                                #depthOp = 4, Z_set=2.,
                                depthOp = 1,
                                p0 = p0,
                                set_consensoRef = False,
                                t_end = 20)
                    [arr_error[idx%variantes,:,i], arr_epsilon[idx%variantes,0,i], arr_epsilon[idx%variantes,1,i]] = ret
                    idx += 1
                    
            if set_derivative and k_int != 0:
                #   Con referencia PI
                print(" ---  Case ", i, " RPID--- ")
                ret = experiment(directory=str(idx),
                            midMarker = midMarker,
                            set_derivative = set_derivative,
                            k_int = k_int,
                            h = 1 ,
                            r = 1.,
                            #tanhLimit = True,
                            #depthOp = 4, Z_set=2.,
                            depthOp = 1,
                            p0 = p0,
                            t_end = t_f)
                [arr_error[idx%variantes,:,i], arr_epsilon[idx%variantes,0,i], arr_epsilon[idx%variantes,1,i]] = ret
                idx += 1
                
                #   Sin referencia PI
                if noRef:
                    print(" ---  Case ", i, " PID--- ")
                    ret = experiment(directory=str(idx),
                                midMarker = midMarker,
                                set_derivative = set_derivative,
                                k_int = k_int,
                                h = 1 ,
                                r = 1.,
                                #tanhLimit = True,
                                #depthOp = 4, Z_set=2.,
                                depthOp = 1,
                                p0 = p0,
                                set_consensoRef = False,
                                t_end = 20)
                    [arr_error[idx%variantes,:,i], arr_epsilon[idx%variantes,0,i], arr_epsilon[idx%variantes,1,i]] = ret
                    idx += 1
        np.save('arr_err.npy',arr_error)
        np.save('arr_epsilon.npy',arr_epsilon)
        
        
    #   Plot data
    colors = randint(0,255,3*arr_error.shape[1]*variantes)/255.0
    colors = colors.reshape((arr_error.shape[1]*variantes,3))
    
    #   Consenso con referencia 
    if k_int != 0. and not set_derivative:
        idx_sel = [0,1]
        labels = ["P","PI"]
    elif k_int == 0. and set_derivative:
        idx_sel = [0,1]
        labels = ["P","PD"]
    elif k_int != 0. and set_derivative:
        idx_sel = [0,1,2,3]
        labels = ["P","PI","PD","PID"]
    else:
        idx_sel = [0]
        labels = ["P"]
    
    if noRef:
        for i in range(len(idx_sel)):
            idx_sel[i] *= 2
    
    plot_err_consenso(ref_arr,
                      arr_error[idx_sel,:,:],
                      colors[idx_sel],
                      labels,
                      title = "Error de consenso con referencia",
                      filename = "ConsensusError_WRef.pdf")
    plot_err_formacion(ref_arr,
                      arr_epsilon[idx_sel,:,:],
                      colors[idx_sel],
                      labels,
                      title = "Error de formación con referencia",
                      filename = "FormationError_WRef.pdf")
    plot_err_formacion(ref_arr,
                      arr_epsilon[idx_sel,:,:],
                      colors[idx_sel],
                      labels,
                      limits = [0,0.1],
                      title = "Error de formación con referencia",
                      filename = "FormationError_WRef_zoom.pdf")
    
    #   Sin referencia
    if noRef:
        #   Consenso sin referencia
        for i in range(len(idx_sel)):
            idx_sel[i] += 1
        plot_err_consenso(ref_arr,
                      arr_error[idx_sel,:,:],
                      colors[idx_sel],
                      labels,
                      title = "Error de consenso sin referencia",
                      filename = "ConsensusError_WoRef.pdf")
        plot_err_formacion(ref_arr,
                      arr_epsilon[idx_sel,:,:],
                      colors[idx_sel],
                      labels,
                      title = "Error de formación sin referencia",
                      filename = "FormationError_WoRef.pdf")
    

################################################################################
################################################################################
#
#   M   A   I   N
#
################################################################################
################################################################################

    
def main():
    
    #   REPEAT
    experiment(directory="3",
                t_end = 100,
                repeat = True)
    return

    #   VIEWER
    view3D('3', xLimit = [-10,10], yLimit = [-10,10],zLimit = [0,20])
    #view3D('5')
    #view3D('20')
    #view3D('26')
    return 
    
    #   Experimento exhaustivo de posiciones y referencias random
    experiment_all_random(nReps = 20)
    print("Echo")
    return
    
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
    
    ##   Experimant
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
    
    ##   complete aleatory setup
    
    ##   Points
    ## Range
    #xRange = [-2,2]
    #yRange = [-2,2]
    #zRange = [-1,0]
    
    #nP = random.randint(4,11)
    #P = np.random.rand(3,nP)
    #P[0,:] = xRange[0]+ P[0,:]*(xRange[1]-xRange[0])
    #P[1,:] = yRange[0]+ P[1,:]*(yRange[1]-yRange[0])
    #P[2,:] = zRange[0]+ P[2,:]*(zRange[1]-zRange[0])
    #Ph = np.r_[P,np.ones((1,nP))]
    
    ##   Cameras
    ## Range
    #xRange = [-2,2]
    #yRange = [-2,2]
    #zRange = [0,3]
    
    #nC = 4
    #p0 = np.zeros((6,nC))
    #pd = np.zeros((6,nC))
    
    #offset = np.array([xRange[0],yRange[0],zRange[0],-np.pi,-np.pi,-np.pi])
    #dRange = np.array([xRange[1],yRange[1],zRange[1],np.pi,np.pi,np.pi])
    #dRange -= offset
    #for i in range(nC):
        
        #tmp = np.random.rand(6)
        #tmp = offset + tmp*dRange
        #cam = cm.camera()
        #agent = ctr.agent(cam, np.zeros(6), tmp,P)
        #Z = Z_select(1, agent, Ph,None,None, None,None)
        #while agent.count_points_in_FOV(Z) != nP :
            #tmp = np.random.rand(6)
            #tmp = offset + tmp*dRange
            #cam = cm.camera()
            #agent = ctr.agent(cam, np.zeros(6), tmp,P)
            #Z = Z_select(1, agent, Ph,None,None, None,None)
        
        #p0[:,i] = tmp.copy()
        
        ##tmp = np.random.rand(6)
        ##tmp = offset + tmp*dRange
        ##cam = cm.camera()
        ##agent = ctr.agent(cam, np.zeros(6), tmp,P)
        ##Z = Z_select(1, agent, Ph,None,None, None,None)
        ##while agent.count_points_in_FOV(Z) != nP:
            ##tmp = np.random.rand(6)
            ##tmp = offset + tmp*dRange
            ##cam = cm.camera()
            ##agent = ctr.agent(cam, np.zeros(6), tmp,P)
            ##Z = Z_select(1, agent, Ph,None,None, None,None)
        
        ##pd[:,i] = tmp.copy()
    
    #pd = circle(4,1,1)
    #ret = experiment(directory='1',
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
    #view3D('1')
    
    return

    ret = experiment(directory='0',
               #k_int = 0.1,
                pd = pd,
                p0 = p0,
                P = P,
                #set_derivative = True,
                #tanhLimit = True,
                #depthOp = 4, Z_set = 1.,
                t_end = 100)
    
    
    
    return
    
    ##   Experimentos de variación de altura
    #experiment_height()
    #return
    
    
    ##  Experimentos de variación de parámetros de contol
   
    #   Lambda values
    #exp_select = [0.25, 0.5, 0.75, 1., 1.5, 1.25, 1.5, 1.75, 2., 5., 10., 15.]
    #   gdl
    #exp_select = [1, 2, 3]
    #   Z_set (depthOp = 4)
    exp_select = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    exp_select = [ 2.0 ]
    
    n_agents = 4
    var_arr = np.zeros((len(exp_select),n_agents,1))
    ref_arr = np.array(exp_select)
    for i in range(len(exp_select)):
        ret = experiment(directory=str(i),
                        depthOp = 1,
                        Z_set = exp_select[i],
                        lamb = 0.1,
                        gdl = 3,
                        #zOffset = 1.0 ,
                        t_end = 20)
        var_arr[i,:,0] = ret[0]
        
    
    #   Plot data
    
    colors = (randint(0,255,3*n_agents)/255.0).reshape((n_agents,3))
    plot_err_consenso(ref_arr,
                      var_arr,
                      colors,
                      labels)
    
    

if __name__ ==  "__main__":
    main()
