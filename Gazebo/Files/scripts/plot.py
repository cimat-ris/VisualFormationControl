
#   libs
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from numpy import pi, sin, cos
from numpy.linalg import norm, svd
from scipy.optimize import minimize_scalar

#   sys
import sys
import os

#   Custom
import camera as cm


#   Configs

get_type={2:np.float16,
          4:np.float32,
          8:np.float64}

#   Aux fun


def get_angles(R, prev_angs= None):
    #print(R)
    if (R[2,0] < 1.0):
    #if (R[2,0] < 0.995):
        if R[2,0] > -1.0:
        #if R[2,0] > -0.995:
            pitch = np.arcsin(-R[2,0])
            if not( prev_angs is None):
                #print(prev_angs)
                #print("pitch = ", pitch)
                pitch_alt = np.sign(pitch) *(pi - abs(pitch))
                #print("pitch_alt = ", pitch_alt)
                delta_pitch = abs(pitch-prev_angs[1])
                #print("delta_p = ", delta_pitch)
                if delta_pitch > pi:
                    delta_pitch = 2*pi-delta_pitch
                    #print("delta_p = ", delta_pitch)
                delta_pitch2 = abs(pitch_alt-prev_angs[1])
                #print("delta_p2 = ", delta_pitch2)
                if delta_pitch2 > pi:
                    delta_pitch2 = 2*pi-delta_pitch2
                    #print("delta_p2 = ", delta_pitch2)
                if delta_pitch2 < delta_pitch:
                    pitch = pitch_alt
                #print("pitch = ", pitch)
            cp = cos(pitch)
            yaw = np.arctan2(R[1,0]/cp,R[0,0]/cp)
            roll = np.arctan2(R[2,1]/cp,R[2,2]/cp)
        else:
            #print("WARN: rotation not uniqe C1")
            pitch = np.pi/2.
            if prev_angs is None:
                yaw = -np.arctan2(-R[1,2],R[1,1])
                roll = 0.
            else:
                tmp = np.arctan2(-R[1,2],R[1,1])
                roll = prev_angs[0]
                yaw = roll - tmp
                if yaw > pi:
                    yaw -= 2*pi
                if yaw < -pi:
                    yaw += 2*pi
                #tmp = np.arctan2(-R[1,2],R[1,1])
                #yaw = prev_angs[2]
                #roll = yaw + tmp
                #if roll > pi:
                    #roll -= 2*pi
                #if roll < -pi:
                    #roll += 2*pi

    else:
        #print("WARN: rotation not uniqe C2")
        pitch = -np.pi/2.
        if prev_angs is None:
            yaw = np.arctan2(-R[1,2],R[1,1])
            roll = 0.
        else:
            tmp = np.arctan2(-R[1,2],R[1,1])
            roll = prev_angs[0]
            yaw = tmp - roll
            if yaw > pi:
                yaw -= 2*pi
            if yaw < -pi:
                yaw += 2*pi
            #tmp = np.arctan2(-R[1,2],R[1,1])
            #yaw = prev_angs[2]
            #roll = tmp - yaw
            #if roll > pi:
                #roll -= 2*pi
            #if roll < -pi:
                #roll += 2*pi

    return [roll, pitch, yaw]

#   My plots
def plot_time(ax, t_array,
              var_array,
              ref = None,
              module = None):

    n = var_array.shape[0]

    npzfile = np.load("general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]



    symbols = []
    for i in range(n):
        ax.plot(t_array,var_array[i,:] , color=colors[i%nColors], lw = 0.6 )
        symbols.append(mpatches.Patch(color=colors[i%nColors]))

    if not ref is None:
        ax.plot([t_array[0],t_array[-1]],[ref,ref],
                'k--', alpha = 0.5)
        symbols.append(mpatches.Patch(color='k'))
        labels.append(refLab)
    if not module is None:
        for i in range(len(module)):
            ax.plot([t_array[0],t_array[-1]],[module[i],module[i]],
                'r--', lw = 0.5)
        symbols.append(mpatches.Patch(color='r'))
        labels.append("Limits")

    return symbols




def plot_descriptors(ax,descriptors_array,
                     # camera_iMsize,
                     # s_ref,
                    #colors,
                    pred,
                    # enableLims = True,
                    # name="time_plot",
                    # label="Variable"):
                    ):
    n = descriptors_array.shape[0]/2
    #print(n)
    n = int(n)

    npzfile = np.load("general.npz")
    colors = npzfile["colors"]
    nColors = colors.shape[0]

    # fig, ax = plt.subplots()
    # fig.suptitle(label)
    # if enableLims:
    #     plt.xlim([0,camera_iMsize[0]])
    #     plt.ylim([0,camera_iMsize[1]])

    # ax.plot([camera_iMsize[0]/2,camera_iMsize[0]/2],
    #         [0,camera_iMsize[0]],
    #         color=[0.25,0.25,0.25])
    # ax.plot([0,camera_iMsize[1]],
    #         [camera_iMsize[1]/2,camera_iMsize[1]/2],
    #         color=[0.25,0.25,0.25])

    # symbols = []

    ref =  descriptors_array[:,0].copy()
    descriptors_array = descriptors_array[:,1:]


    for i in range(n):
        ax.plot(descriptors_array[2*i,0],descriptors_array[2*i+1,0],
                '*',color=colors[i%nColors])
        ax.plot(descriptors_array[2*i,-1],descriptors_array[2*i+1,-1],
                'o',color=colors[i%nColors])
        ax.plot(descriptors_array[2*i,:],descriptors_array[2*i+1,:],
                color=colors[i%nColors])
        ax.plot(ref[2*i],ref[2*i+1],marker='^',color=colors[i%nColors])
        #print(descriptors_array[2*i,-1],descriptors_array[2*i+1,-1])

        #   Hip = predicted endpoints
        ax.plot(pred[2*i],pred[2*i+1],
                'x',color=colors[i%nColors])

    ax.plot(ref[[0,2,4,6,0]],ref[[1,3,5,7,1]],
                color='k', lw = 0.4)
    ax.plot(pred[[0,2,4,6,0]],pred[[1,3,5,7,1]],
                color='k', lw = 0.4)
    ax.plot(descriptors_array[[0,2,4,6,0],-1],descriptors_array[[1,3,5,7,1],-1],
                color='k', lw = 0.4)
    ax.plot(descriptors_array[[0,2,4,6,0],0],descriptors_array[[1,3,5,7,1],0],
                color='k', lw = 0.4)


    # symbols = [mlines.Line2D([0],[0],marker='*',color='k'),
    #            mlines.Line2D([0],[0],marker='o',color='k'),
    #            # mlines.Line2D([0],[0],marker='^',color='k'),
    #            # mlines.Line2D([0],[0],marker='x',color='k'),
    #            mlines.Line2D([0],[0],linestyle='-',color='k')]
    # # labels = ["Start","End","reference","Predicted","trayectory"]
    # labels = ["Start","End","trayectory"]
    # fig.legend(symbols,labels, loc=1)
    #
    # plt.tight_layout()
    # plt.savefig(name+'.pdf',bbox_inches='tight')
    # #plt.show()
    # plt.close()

def error_state(reference,  camera, gdl = 1, name= None):

    n = reference.shape[1]
    state = np.zeros((6,n))
    for i in range(len(camera)):
        state[:,i] = camera[i].p

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

    # print("centered ref and state")
    # print(new_reference)
    # print(new_state)

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
            _R = R.T @ camera[i].R
            [new_state[3,i], new_state[4,i], new_state[5,i]] = get_angles(_R)
            camera[i].pose(new_state[:,i])

            #   Get error
            _R =  cm.rot(new_reference[3,i],'x') @ camera[i].R.T
            _R = cm.rot(new_reference[4,i],'y') @ _R
            _R = cm.rot(new_reference[5,i],'z') @ _R
            _arg = (_R.trace()-1.)/2.
            if abs(_arg) < 1.:
                rot_err[i] = np.arccos(_arg)
            else:
                rot_err[i] = np.arccos(np.sign(_arg))

    else:
        #   Get error
        [Rtheta,Rphi,Rpsi] = get_angles(R)
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
            camera[i].pose(state[:,i])

        return [t_err, rot_err]

     ##   Plot

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    #   TODO plot Z parallel for gdl == 2
    if gdl ==1 :
        for i in range(n):
            camera[i].draw_camera(ax, scale=0.2, color='red')
            ax.text(camera[i].p[0],camera[i].p[1],camera[i].p[2],str(i))
            camera[i].pose(new_reference[:,i])
            camera[i].draw_camera(ax, scale=0.2, color='brown')
            ax.text(camera[i].p[0],camera[i].p[1],camera[i].p[2],str(i))
            camera[i].pose(state[:,i])
    else:
        for i in range(n):
            #   new_state
            new_state[3:,i] = np.array([pi,0.,rot_err_plot[i]])
            camera[i].pose(new_state[:,i])
            camera[i].draw_camera(ax, scale=0.2, color='red')
            ax.text(camera[i].p[0],camera[i].p[1],camera[i].p[2],str(i))

            #   new_reference
            new_reference[3:,i] = np.array([pi,0.,new_reference[5,i]])
            camera[i].pose(new_reference[:,i])
            camera[i].draw_camera(ax, scale=0.2, color='brown')
            ax.text(camera[i].p[0],camera[i].p[1],camera[i].p[2],str(i))
            camera[i].pose(state[:,i])



    plt.savefig(name+'.pdf',bbox_inches='tight')
    #plt.show()
    plt.close()

    return [t_err, rot_err]



 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------
 #      -----------------------------------------------------------



##  Plot task

def plotErrors(directory, n_agents):
    for i in range(n_agents):
        name = directory + str(i)+'/error.dat'
        length = os.path.getsize(name)
        # print("length = ", length)
        ArUcoTab = {}

        with open(name, 'rb') as fileH:
            #   header
            row_bytes = np.fromfile(fileH, dtype=np.int32, count= 1)
            # row_bytes = row_bytes[0]
            # print("row_bytes = ", row_bytes)
            rows = (length-4) / (8*10)
            rows = int(np.floor(rows))

            # Read the data into a NumPy array
            for j in range(rows):
                timestamp = np.fromfile(fileH,
                                    dtype=np.float64,
                                    count = 1)
                ArUcoId = np.fromfile(fileH,
                                    dtype=np.int32,
                                    count = 1)
                ArUcoId = ArUcoId[0]
                errorRow = np.fromfile(fileH,
                                    # dtype=get_type[elemSize],
                                    dtype = np.float32,
                                    # dtype = np.double,
                                    count = 8)
                row = np.concatenate((timestamp,errorRow))
                # print(timestamp)
                # print(ArUcoId)
                # print(errorRow)
                if ArUcoId in ArUcoTab.keys():
                    ArUcoTab[ArUcoId] = np.r_[ArUcoTab[ArUcoId], row.reshape((1,9))]
                else:
                    ArUcoTab[ArUcoId] = row.reshape((1,9))
        # return
        #  plot errors
        fig, ax = plt.subplots()
        fig.suptitle("Feature errors")
        for ki in ArUcoTab.keys():
            symbols = plot_time(ax, ArUcoTab[ki][:,0],ArUcoTab[ki][:,1:].T)

        # ylimits = [.0,10.1]
        # plt.ylim((ylimits[0],ylimits[1]))

        plt.tight_layout()
        plt.savefig(directory + str(i)+"/errores.pdf",bbox_inches='tight')
        #plt.show()
        plt.close()


def plotFeatures(directory, n_agents):

    agentsDicts = []

    for i in range(n_agents):
        name = directory + str(i)+'/arUcos.dat'
        length = os.path.getsize(name)
        # print("length = ", length)
        ArUcoTab = {}

        with open(name, 'rb') as fileH:
            #   header
            # row_bytes = row_bytes[0]
            # print("row_bytes = ", row_bytes)
            rows = length / (8*4+4)
            rows = int(np.floor(rows))

            # Read the data into a NumPy array
            for j in range(rows):
                ArUcoId = np.fromfile(fileH,
                                    dtype=np.int32,
                                    count = 1)
                ArUcoId = ArUcoId[0]
                Features = np.fromfile(fileH,
                                    # dtype=get_type[elemSize],
                                    dtype = np.float32,
                                    # dtype = np.double,
                                    count = 8)
                # print(ArUcoId)
                # print(Features)
                if ArUcoId in ArUcoTab.keys():
                    ArUcoTab[ArUcoId] = np.c_[ArUcoTab[ArUcoId], Features.reshape((8,1))]
                else:
                    ArUcoTab[ArUcoId] = Features.reshape((8,1))
        agentsDicts.append(ArUcoTab)

    #   Calculate targets



    #  plot errors
    camera_iMsize = [1024, 1024]
    for i in range(n_agents):

        ArUcoTab=agentsDicts[i]

        #   Predicted (grafo conmpletamente conexo)
        arUcoPred = {}
        for ki in ArUcoTab.keys():
            pred = np.zeros((n_agents,8))
            for j in range(n_agents):
                if ki in agentsDicts[j].keys():
                    pred[j,:] = agentsDicts[j][ki][:,1] - agentsDicts[j][ki][:,0]
            pred = pred.mean(axis = 0)
            pred = agentsDicts[i][ki][:,0] + pred
            arUcoPred[ki] = pred.copy()

        fig, ax = plt.subplots()
        fig.suptitle("Feature errors")
        plt.xlim([0,camera_iMsize[0]])
        plt.ylim([0,camera_iMsize[1]])
        ax.plot([camera_iMsize[0]/2,camera_iMsize[0]/2],
            [0,camera_iMsize[0]],
            color=[0.25,0.25,0.25])
        ax.plot([0,camera_iMsize[1]],
            [camera_iMsize[1]/2,camera_iMsize[1]/2],
            color=[0.25,0.25,0.25])

        for ki in ArUcoTab.keys():
            plot_descriptors(ax, ArUcoTab[ki], pred= arUcoPred[ki] )

        symbols = [mlines.Line2D([0],[0],marker='*',color='k'),
               mlines.Line2D([0],[0],marker='o',color='k'),
               mlines.Line2D([0],[0],marker='^',color='k'),
               # mlines.Line2D([0],[0],marker='x',color='k'),
               mlines.Line2D([0],[0],linestyle='-',color='k')]
        labels = ["Start","End","reference","trayectory"]
        fig.legend(symbols,labels, loc=1)
        plt.tight_layout()
        plt.savefig(directory + str(i)+"/Image_Features.pdf",bbox_inches='tight')
        #plt.show()
        plt.close()

def plotVelocities(directory, n_agents):
    allStates = []
    for i in range(n_agents):
        name = directory + str(i)+'/partial.dat'
        length = os.path.getsize(name)
        # print("length = ", length)

        with open(name, 'rb') as fileH:
            #   header
            np.fromfile(fileH, dtype=np.int32, count= 1)
            # rows = (length-4) / row_bytes
            rows = (length-4) / (8*7)
            # print("rows = ",rows)
            rows = int(np.floor(rows))
            # print("rows = ",rows)
            time = np.zeros(rows)
            velocities = np.zeros((6,rows))
            states = np.zeros((6,rows))

            # Read the data into a NumPy array
            for j in range(rows):
                time[j] = np.fromfile(fileH,
                                    dtype=np.float64,
                                    count = 1)
                states[:,j] = np.fromfile(fileH,
                                    dtype = np.float32,
                                    count = 6)
                velocities[:,j] = np.fromfile(fileH,
                                    dtype = np.float32,
                                    count = 6)

                # print(time[j])
                # print(states[:,j])
                # print(velocities[:,j])
            allStates.append(np.r_[time.reshape((1,states.shape[1])),states])

        # return
        #  plot errors
        fig, ax = plt.subplots()
        fig.suptitle("Velocities")
        symbols = plot_time(ax, time,velocities)
        plt.savefig(directory + str(i)+"/Velocities.pdf",bbox_inches='tight')
        plt.close()

    return allStates

def plotFError(directory,n_agents, allStates,pd):


    P = np.array([[1, 0, 0],
                  [1, 1, 0],
                  [0, 0, 0]]) # Dummy points
    #   Error at the begining
    camera = []
    for i in range(n_agents):
        tmp = allStates[i][1:,0]
        tmp[3] += pi
        cam = cm.camera()
        cam.pose(tmp)
        camera.append(cam)
    ret = error_state(pd,  camera, gdl = 1, name= directory+"Err_0.pdf")
    print("Error_0 = ",ret)

    #   Error at the end
    camera = []
    for i in range(n_agents):
        tmp = allStates[i][1:,-1]
        tmp[3] += pi
        cam = cm.camera()
        cam.pose(tmp)
        camera.append(cam)
    ret = error_state(pd,  camera, gdl = 1, name= directory+"Err_fin.pdf")
    print("Error_fin = ",ret)


def main(arg):
    n_agents = 4
    EHR = np.zeros((5,n_agents))
    DHR = np.zeros((5,n_agents))

    EHR[:,0]=np.array([0.8, 3.2, 2.1, 0, 0])
    DHR[:,0]=np.array([0.8, 3.2, 2.6, -10, 0])
    EHR[:,1]=np.array([0.8, 1.2, 2.1, 0, 0.1])
    DHR[:,1]=np.array([1.0, 1.4, 2.1, 15, 0])
    EHR[:,2]=np.array([2.8, 1.2, 2.1, 0, 0.2])
    DHR[:,2]=np.array([2.8, 1.2, 3.1, 0, 0])
    EHR[:,3]=np.array([2.8, 3.2, 2.1, 0, 0.3])
    DHR[:,3]=np.array([2.8, 3.2, 2.1, 10, 0])

    pd=np.array([[0.,0.,0.,0.],
                [0.,0.,0.,0.],
                [0.,0.,0.,0.],
                [pi,pi,pi,pi],
                [0.,0.,0.,0.],
                [0.,0.,0.,0.]])


    directory = arg[1].rstrip('/')+'/'

    if "EHR" in directory:
        pd[:3,:] = EHR[:3,:]
        pd[5,:] = EHR[3,:]

    if "DHR" in directory:
        pd[:3,:] = DHR[:3,:]
        pd[5,:] = DHR[3,:]

    # #   BEGIN ERRORS
    print("Ploting IMAGE ERRORS ERRORS")
    plotErrors(directory,n_agents)
    #
    # #   END ERRORS
    # #   BEGIN VELOCITIES / POSITION
    print("Ploting VELOCITIES / POSITION")
    allStates = plotVelocities(directory,n_agents)

    #   END VELOCITIES / POSITION
    print("Ploting FORMATION ERRORS")
    #   BEGIN FORMATION ERROR
    plotFError(directory,n_agents, allStates,pd)
    #   END FORMATION ERROR


    #   BEGIN Features

    print("Ploting FEATURES")
    plotFeatures(directory,n_agents)

    #   END FEATURES


if __name__ ==  "__main__":

    main(sys.argv)
