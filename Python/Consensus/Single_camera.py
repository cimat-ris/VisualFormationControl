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
import numpy as np
from numpy.linalg import inv, svd
from numpy import sin, cos, pi
from scipy.optimize import minimize_scalar

#   Plot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.random import rand, randint

#   Image
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
#SceneP=[[0,   -1.,  1.],
#[-1.,  1., 0],
#[0.,0.,0.]]
#SceneP=[[0,   -0.5,  0.5],
#[-0.5,  0.5, 0],
##[0.,  0., 0.]]
#[0.0,  -0.2, 0.5]]
#[0.0,  0.2, 0.3]]
#[0.0,  -0.0, 0.0]]
#case 4
SceneP=[[-0.5, -0.5, 0.5,  0.5],
[-0.5,  0.5, 0.5, -0.5],
#[0,    0.2, 0.3,  -0.1]]           
[0,    0.0, 0.0,  0.0]] 
#case 5
#SceneP=[[-0.5, -0.5, 0.5, 0.5, 0.1],
#[-0.5, 0.5, 0.5, -0.5, -0.3],
#[0, 0.0, 0.0,  -0.0, 0.0]]
#[0, 0.2, 0.3, -0.1, 0.1]]
##case 6
#SceneP=[[-0.5, -0.5, 0.5, 0.5, 0.1, -0.1],
#[-0.5, 0.5, 0.5, -0.5, -0.3, 0.2],
#[0, 0.0, 0.0, -0.0, 0.0, 0.0]]
##[0, 0.2, 0.3, -0.1, 0.1, 0.15]]
##otherwise
#SceneP=[[-0.5, -0.5, 0.5, 0.5],
#[-0.5, 0.5, 0.5, -0.5],
#[0, 0.2, 0.3, -0.1]]




def plot_3Dcam(ax, camera,
               positionArray,
               init_configuration,
               desired_configuration,
               color,
               label = "0",
               camera_scale    = 0.02):
    
    #ax.plot(positionArray[0,:],
    ax.scatter(positionArray[0,:],
            positionArray[1,:],
            positionArray[2,:],
            label = label,
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

def view3D(directory):
    
    #   load
    fileName = directory + "/data3DPlot.npz"
    npzfile = np.load(fileName)
    P = npzfile['P']
    pos_arr = npzfile['pos_arr']
    p0 = npzfile['p0']
    pd = npzfile['pd']
    end_position = npzfile['end_position']
    
    
    #   Plot
    
    colors = (randint(0,255,3)/255.0)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    name = directory+"/3Dplot"
    #fig.suptitle(label)
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    
    cam = cm.camera()
    cam.pose(end_position)
    plot_3Dcam(ax, cam,
            pos_arr[0,:,:],
            p0,
            pd,
            color = colors,
            camera_scale    = 0.02)
    
    #f_size = 1.1*max(abs(p0[:2,:]).max(),abs(pd[:2,:]).max())
    #lfact = 1.1
    #x_min = lfact*min(p0[0,:].min(),pd[0,:].min())
    #x_max = lfact*max(p0[0,:].max(),pd[0,:].max())
    #y_min = lfact*min(p0[1,:].min(),pd[1,:].min())
    #y_max = lfact*max(p0[1,:].max(),pd[1,:].max())
    #z_max = lfact*max(p0[2,:].max(),pd[2,:].max())
    
    #ax.set_xlim(x_min,x_max)
    #ax.set_ylim(y_min,y_max)
    #ax.set_zlim(0,z_max)
    
    fig.legend( loc=1)
    plt.show()
    plt.close()

def Z_select(depthOp, agent, P, Z_set, p0, pd, j):
    Z = np.ones((1,P.shape[1]))
    if depthOp ==1: #   Real depth
        #print(P)
        #   TODO creo que esto está mal calculado
        M = np.c_[ agent.camera.R.T, -agent.camera.R.T @ agent.camera.p ]
        Z = M @ P
        Z = Z[2,:]
    elif depthOp ==2:   # Height at begining
        Z = p0[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 3:  # Height at reference
        Z = pd[2,j]*Z
        Z = Z-P[2,:]
    elif depthOp == 4:  # fixed at Z_set
        Z = Z_set * np.ones(P.shape[1])
    elif depthOp == 5:  # Height at current position
        tmp = agent.camera.p[2]
        Z = Z*tmp
    elif depthOp == 6:  # Height at current position adjusted to each Point
        Z = agent.camera.p[2]*np.ones(P.shape[1])
        Z = Z-P[2,:]
    else:
        print("Invalid depthOp")
        return None
    return Z

################################################################################
################################################################################
#
#   Single Experiment
#
################################################################################
################################################################################
    
    
def experiment(directory = "0",
               h  = 2.0,
               lamb = 1.0,
               depthOp=1,
               Z_set = 1.0,
               gdl = 1,
               t0 = 0.0,
               dt = 0.05,
               t_end = 10.0,
               zOffset = 0.0,
               case_interactionM = 1,
               control_type = 1,
               p0 = np.array([0.,0.,1.,np.pi,0.,0.]),
               pd = np.array([0.,0.,1.,np.pi,0.,0.]),
               P = np.array(SceneP), 
               nameTag = "",
               set_derivative = False,
               tanhLimit = False,
               verbose = False):
    
    #   Data
    n_points = P.shape[1] #Number of image points
    P = np.r_[P,np.ones((1,n_points))] # change to homogeneous
    
    
    #   Parameters
    
    case_n = 1  #   Case selector if needed
    directed = False
    
    #Depth estimation for interaction matrix
    #   1-Updated, 2-Initial, 3-Referencia, 4-Arbitrary fixed, 5-Average
    depthOp_dict = {1:"Updated at each step",
                    2:"Distance at the begining",
                    3:"Distance between reference and points",
                    4:"Arbirtrary uniform value Z_set",
                    5:"Uniform Z camera coordenate position",
                    6:"Height as depth"}
    
    #   interaction matrix used for the control
    #   1- computed each step 2- Computed at the end 3- average
    case_interactionM_dict = {1:"Computed at each step",
                              2:"Computed at the end",
                              3:"Average between 1 and 2"}
    
    #   Controladores
    #   1  - IBVC   2 - Montijano
    control_type_dict = {1:"Image based visual control",
                         2:"Montijano",
                         3:"Position based visual control"}
    case_controlable_dict = {1:"6 Degrees of freedom",
                             2:"4 Degrees of freedom",
                             3:"3 Degrees of freedom"}
    
    
    
    #   Agents array
    cam = cm.camera()
    agent = ctr.agent(cam,pd,p0,P,set_derivative= set_derivative)
        
    #   INIT LOOP
    
    t=t0
    steps = int((t_end-t)/dt + 1.0)
    
    #   case selections
    if case_interactionM > 1:
        #   Depth calculation
        Z = Z_select(depthOp, agent, P,Z_set,p0,pd,i)
        if Z is None:
            return
        agent.set_interactionMat(Z,gdl)
    
    if control_type ==3:
        refR = cm.rot(pd[5],'z') 
        refR = refR @ cm.rot(pd[4],'y')
        refR = refR @ cm.rot(pd[3],'x')
        
    #if control_type == 2:
        #delta_pref = np.zeros((n_agents,n_agents,6,1))
        #for i in range(n_agents):
            #for j in range(n_agents):
                #delta_pref[i,j,:,0] = pd[:,j]-pd[:,i]
        #gamma =  p0[2,:]
    
    #   Storage variables
    t_array = np.arange(t,t_end+dt,dt)
    steps = t_array.shape[0]
    err_array = np.zeros((1,2*n_points,steps))
    U_array = np.zeros((1,6,steps))
    desc_arr = np.zeros((1,2*n_points,steps))
    pos_arr = np.zeros((1,3,steps))
    if gdl == 1:
        s_store = np.zeros((1,6,steps))
    elif gdl == 2:
        s_store = np.zeros((1,4,steps))
    elif gdl == 3:
        s_store = np.zeros((1,3,steps))
    
    #   Print simulation data:
    print("------------------BEGIN-----------------")
    print("Reference state")
    print(pd)
    print("Initial state")
    print(p0)
    print("Scene points")
    print(P[:3,:])
    print("Number of points = "+str(n_points))
    print("Time range = ["+str(t)+", "+str(dt)+", "+str(t_end)+"]")
    print("Control lambda = "+str(lamb))
    print("Depth estimation = "+depthOp_dict[depthOp])
    if depthOp == 4:
        print("\t Estimated theph set at : "+str(Z_set))
    print("Interaction matrix = "+case_interactionM_dict[case_interactionM])
    print("Control selection = "+control_type_dict[control_type])
    print("Controllable case = "+case_controlable_dict[gdl])
    print("Total steps = "+str(steps))
    
    #   LOOP
    for i in range(steps):
    #for i in range(2):
        
        if verbose:
            print("loop",i, end="\r")
        
        #   Error:
        error = agent.error.copy().reshape(2*n_points)
        
        #   save data
        #print(error)
        err_array[0,:,i] = error
        
        
        ####   Image based formation
        #if control_type ==2:
            #H = ctr.get_Homographies(agents)
        #   Get control
        
        
        #   save data 
        desc_arr[0,:,i] = agent.s_current.T.reshape(2*n_points)
        pos_arr[0,:,i] = agent.camera.p
        
        #   Depth calculation
        Z = Z_select(depthOp, agent, P,Z_set,p0,pd,0)
        if Z is None:
            return
        
        #   Control
        if control_type == 1:
            args = {"deg":1 , 
                    "control_sel":case_interactionM,
                    "error": error,
                    "gdl":gdl}
        elif control_type == 3:
            realT = pd[:3]-agent.camera.p
            reatT = agent.camera.R.T @ realT
            args = {"p1": agent.s_current.T,
                    "p2": agent.s_ref.T,
                    "K": agent.camera.K,
                    "realR": refR.T @ agent.camera.R,
                    "realT": reatT}
        #elif control_type == 2:
            #args = {"H" : H[j,:,:,:],
                    #"delta_pref" : delta_pref[j,:,:,:],
                    #"Adj_list":G.list_adjacency[j][0],
                    #"gamma": gamma[j]}
        else:
            print("invalid control selection")
            return
        
        #s = None
        Z_test =  Z_select(1, agent, P,Z_set,p0,pd,0)
        if (any(Z_test < 0.)):
            print("Image plane Colision")
            U = np.zeros(6)
        else:
            U,u,s,vh  = agent.get_control(control_type,lamb,Z,args)
            
            s_store[0,:,i] = s
            if tanhLimit:
                #U = 0.5*np.tanh(U)
                U[:3] = 0.5*np.tanh(U[:3])
                U[3:] = 0.3*np.tanh(U[3:])
            #print(s)
            #U[abs(U) > 0.2] = np.sign(U)[abs(U) > 0.2]*0.2
            #U_sel = abs(U[3:]) > 0.2
            #U[3:][U_sel] = np.sign(U[3:])[U_sel]*0.2
            
            if any(abs(U) > 5.):
                print("U = ",U)
                print("p = ",agent.camera.p)
                print("theta = ",
                    agent.camera.roll,
                    agent.camera.pitch,
                    agent.camera.yaw)
                print("(u,v) pix = ",agent.s_current)
                print("(u,v) nrom = ",agent.s_current_n)
                L=ctr.Interaction_Matrix(agent.s_current_n,Z,gdl)
                print("L = ",L)
                print("L = ",np.linalg.pinv(L))
        
        if U is None:
            print("Invalid U control")
            break
        
        #   Test image in straigt line
        #U = 0.2*(pd-np.r_[agent.camera.p.T,np.pi,np.zeros(2)])
        
        
        U_array[0,:,i] = U
        agent.update(U,dt,P, Z)
        
    
        #   Update
        t += dt
        #if control_type ==2:
            #gamma = A_ds @ gamma #/ 10.0
        
    
    ##  Final data
    pf = np.r_[agent.camera.p.T ,
               agent.camera.roll,
               agent.camera.pitch,
               agent.camera.yaw]
    
    print("----------------------")
    print("Simulation final data")
    
    ret_err=np.linalg.norm(error)
    print(error)
    print("|Error|= "+str(ret_err))
    print("X = "+str(agent.camera.p))
    print("Angles = "+str(agent.camera.roll)+
            ", "+str(agent.camera.pitch)+", "+str(agent.camera.yaw))
    
    end_position = np.r_[agent.camera.p,
                         agent.camera.roll,
                         agent.camera.pitch,
                         agent.camera.yaw]
    np.savez(directory + "/data3DPlot.npz",
             P = P, pos_arr=pos_arr, p0 = p0,
             pd = pd, end_position = end_position )
    
    ####   Plot
    # Colors setup
    
    #        RANDOM X_i
    colors = (randint(0,255,3*2*n_points)/255.0).reshape((2*n_points,3))
    
    print("-------------------END------------------")
    print()
    
    
    pd = pd.reshape((6,1))
    p0 = p0.reshape((6,1))
    #   Camera positions (init, end, ref) 
    lfact = 1.1
    mp.plot_position(pos_arr,
                    pd,
                    lfact,
                    colors,
                    name = directory+"/Cameras_trayectories")
                    #label = "Positions")
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    label = "World setting"
    name = directory+"/3Dplot"
    #fig.suptitle(label)
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    #agents[i].count_points_in_FOV(P)
    plot_3Dcam(ax, agent.camera,
            pos_arr[0,:,:],
            p0[:,0],
            pd[:,0],
            color = colors[0],
            camera_scale    = 0.02)
    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()
    
    #   Descriptores (init, end, ref) x agente
    #print("Agent: "+nameTag)
    #L = ctr.Interaction_Matrix(agent.s_current_n,Z,gdl)
    #A = L.T@L
    #if np.linalg.det(A) != 0:
        #print("Matriz de interacción (L): ")
        #print(L)
        #L = inv(A) @ L.T
        #print("Matriz de interacción pseudoinversa (L+): ")
        #print(L)
        #print("Error de consenso final (e): ")
        #print(error)
        #print("velocidades resultantes (L+ @ e): ")
        #print(L@error)
        #print("Valores singulares al final (s = SVD(L+)): ")
        #print(s_store[0,:,-1])
    mp.plot_descriptors(desc_arr[0,:,:],
                        agent.camera.iMsize,
                        agent.s_ref,
                        colors,
                        name = directory+"/Image_Features_"+nameTag,
                        label = "Image Features")
    
    #   Errores 
    mp.plot_time(t_array,
                err_array[0,:,:],
                colors,
                name = directory+"/Features_Error_"+nameTag,
                label = "Features Error")
    
    #   Velocidaes x agente
    mp.plot_time(t_array,
                U_array[0,:,:],
                colors,
                name = directory+"/Velocidades_"+nameTag,
                label = "Velocidades",
                labels = ["X","Y","Z","Wx","Wy","Wz"])
    
    #   Valores propios
    mp.plot_time(t_array,
                s_store[0,:,:],
                colors,
                name = directory+"/ValoresP_"+nameTag,
                label = "Valores propios (SVD)",
                labels = ["0","1","2","3","4","5"])
                #limits = [[t_array[0],t_array[-1]],[0,20]])
    
    agent.camera.pose(pf)
    return ret_err, pos_arr[0,:,:], agent.camera

################################################################################
################################################################################
#
#   Experiment Series
#
################################################################################
################################################################################

def experiment_mesh(enableShow = False):
    
    ##   SETUP
    
    #   alpha
    alpha = 0 # np.pi/2
    Ralpha = cm.rot(alpha,'y')
    
    #   PD
    pd = np.array([0.,0.,1,np.pi,alpha,0.])
    pd[:3,] = Ralpha @ pd[:3,]
    
    #   P
    P = np.array(SceneP)      #   Scene points
    P = cm.rot(0.00,"y") @ P 
    P = Ralpha @ P 
    
    #   P0
    x = np.linspace(-2,2,5)
    y = np.linspace(-2,2,5)
    _x, _y = np.meshgrid(x,y)
    n = _x.size
    
    p0 = np.array([_x.reshape(n),
                     _y.reshape(n),
                     2.*np.ones(n),
                     np.pi*np.ones(n),
                     0.0*np.ones(n),
                     1.*np.ones(n)])
    p0[:3,:] = Ralpha @ p0[:3,:]
    
    for i in  range(p0.shape[1]):
        _R = cm.rot(p0[5,i],'z') 
        _R = _R @ cm.rot(p0[4,i],'y')
        _R = _R @ cm.rot(p0[3,i],'x')
        
        _R = cm.rot(alpha,'y') @ _R
        [p0[3,i], p0[4,i], p0[5,i]] = ctr.get_angles(_R)
    
    ##  EXPERIMENTS
    
    print("testing ",n," repeats")
    pos_arr = []
    cam_arr = []
    for i in range( n):
        err, _pos_arr, _cam = experiment(directory=str(i),
                                lamb = 1.,
                                gdl = 1,
                                P=P,
                                pd = pd,
                                p0 = p0[:,i],
                                #tanhLimit = True,
                                #depthOp = 4, Z_set=2.,
                                t_end =.6)
        pos_arr.append(_pos_arr)
        cam_arr.append(_cam)
    
    ##  Prepare data for plot
    colors = (randint(0,255,3*n)/255.0).reshape((n,3))
    n_points = P.shape[1] #Number of image points
    
    #   PLOT
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    name = "3Dplot"
    ax.plot(P[0,:], P[1,:], P[2,:], 'o')
    for i in range( n):
        plot_3Dcam(ax, cam_arr[i],
                pos_arr[i],
                p0[:,i],
                pd,
                color = colors[i],
                label = str(i),
                camera_scale    = 0.02)
    ax.set_xlim(-2.5,2.5)
    ax.set_ylim(-2.5,2.5)
    ax.set_zlim(-1,3)
    fig.legend( loc=2)
    plt.savefig(name+'.pdf',bbox_inches='tight')
    if enableShow:
        plt.show()
    plt.close()

def experiment_alpha():
    
    #   Pruebas individuales
    alpha =  np.pi/4
    Ralpha = cm.rot(alpha,'y')
    
    pd = np.array([0.,0.,1,np.pi,alpha,0.])
    pd[:3] = Ralpha @ pd[:3]
    
    P = np.array(SceneP)      #   Scene points
    P = Ralpha @ P 
    
    p0 = np.array([1., 1., 2., np.pi, 0., 4.])
    p0[:3] = Ralpha @ p0[:3]
    
    _R = cm.rot(p0[5],'z') 
    _R = _R @ cm.rot(p0[4],'y')
    _R = _R @ cm.rot(p0[3],'x')
    _R =  Ralpha @ _R
    [p0[3], p0[4], p0[5]] = ctr.get_angles(_R)
    
    experiment(directory='0',
               control_type = 1,
                lamb = 1.,
                gdl = 1,
                h = 1. ,
                p0 = p0,
                P = P,
                pd = pd,
                #depthOp = 4, Z_set=1.,
                t_end = 10.)
    
    return
################################################################################
################################################################################
#
#   M   A   I   N
#
################################################################################
################################################################################

def main():
        
    #experiment_alpha()
    #view3D("0")
    #return
    
    #   Pruebas En Mesh
    #experiment_mesh()
    #experiment_mesh(enableShow = True)
    #return
    
    #   Prueba libre
    
    pd = np.array([0.,0.,1,np.pi,0.,0.])
    
    P = np.array(SceneP)      #   Scene points
    
    #p0 = np.array([1., 1., 2., np.pi, 0., 4.])
    p0 = np.array([ 1., 0., 2., np.pi, -0., 3.])
    #p0 = np.array([ 0.98835021,
                   #-0.06295688, 
                   #1.96015421,
                   ##3.,
                   #-3.1294862608035783,
                   #0.02036460213558716, 
                   #0.9158529015192103])
    #p0 = np.array([ 3.42588971, -1.63160462,  5.17080969,
                   #-2.7985927198830716, 
                   #-0.16907152649798748,
                   #0.3938874659944802])
    
    experiment(directory='0',
               control_type = 1,
                lamb = 1.,
                gdl = 1,
                h = 1. ,
                p0 = p0,
                P = P,
                pd = pd,
                #depthOp = 4, Z_set=2.,
                t_end = 10)
    view3D("0")


if __name__ ==  "__main__":
    main()
