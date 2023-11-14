import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import sys
# import re
import os

from numpy import pi

#   Configs

get_type={2:np.float16,
          4:np.float32,
          8:np.float64}


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

##  Plot task

def plotErrors(directory, n_agents):
    for i in range(n_agents):
        name = directory + str(i)+'/error.dat'
        length = os.path.getsize(name)
        print("length = ", length)
        ArUcoTab = {}

        with open(name, 'rb') as fileH:
            #   header
            row_bytes = np.fromfile(fileH, dtype=np.int32, count= 1)
            # row_bytes = row_bytes[0]
            print("row_bytes = ", row_bytes)
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
        print("length = ", length)
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

        print("DB: 1")
        for ki in ArUcoTab.keys():
            plot_descriptors(ax, ArUcoTab[ki], pred= arUcoPred[ki] )

        symbols = [mlines.Line2D([0],[0],marker='*',color='k'),
               mlines.Line2D([0],[0],marker='o',color='k'),
               mlines.Line2D([0],[0],marker='^',color='k'),
               # mlines.Line2D([0],[0],marker='x',color='k'),
               mlines.Line2D([0],[0],linestyle='-',color='k')]
        labels = ["Start","End","reference","trayectory"]
        fig.legend(symbols,labels, loc=1)
        print("DB: 2")
        plt.tight_layout()
        plt.savefig(directory + str(i)+"/Image_Features.pdf",bbox_inches='tight')
        #plt.show()
        plt.close()

def plotVelocities(directory, n_agents):
    allStates = []
    for i in range(n_agents):
        name = directory + str(i)+'/partial.dat'
        length = os.path.getsize(name)
        print("length = ", length)

        with open(name, 'rb') as fileH:
            #   header
            np.fromfile(fileH, dtype=np.int32, count= 1)
            # rows = (length-4) / row_bytes
            rows = (length-4) / (8*7)
            print("rows = ",rows)
            rows = int(np.floor(rows))
            print("rows = ",rows)
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

# def plotFError(directory,n_agents, allStates):
#
#     #   filter by time
#
#     allStatesFiltered = np.zeros((n_agents,6,1))
#
#     for j in range(allStates[0].shape[1]):
#         _filter = [-1]*n_agents
#         for i in range(1,n_agents):
#             tmp = np.where(allStates[0][0,j] == allStates[i][0,:])
#             if tmp.shape[0] > 0 :
#                 _filter[i] = tmp[0]
#         if all(_filter != -1):
#
#
#     for i in range(1,n_agents):
#         allStates[i] = allStates[i][:,select]




def main(arg):
    n_agents = 4
    pd=np.array([[0.,0.,0.,0.],
                [0.,0.,0.,0.],
                [0.,0.,0.,0.],
                [pi,pi,pi,pi],
                [0.,0.,0.,0.],
                [0.,0.,0.,0.]])

    directory = arg[1].rstrip('/')+'/'

    # #   BEGIN ERRORS
    # print("Ploting ERRORS")
    # plotErrors(directory,n_agents)
    #
    # #   END ERRORS
    # #   BEGIN VELOCITIES / POSITION
    # print("Ploting VELOCITIES / POSITION")
    # allStates = plotVelocities(directory,n_agents)

    #   END VELOCITIES / POSITION

    #   BEGIN FORMATION ERROR
    # plotFError(directory,n_agents, allStates)
    #   END FORMATION ERROR


    #   BEGIN Features

    print("Ploting FEATURES")
    plotFeatures(directory,n_agents)

    #   END FEATURES


if __name__ ==  "__main__":

    main(sys.argv)
