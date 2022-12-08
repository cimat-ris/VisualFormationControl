/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
March 1st, 2019
This code is used to save and access information about the desired formation and
its communication.
*/

/*********************************************************************************** c++ libraries */
#include <fstream>
#include <iostream>
#include <math.h>

/******************************************************************************** OpenCv libraries */
#include <opencv2/core.hpp>

/***************************************************************************** Hand made libraries*/
#include "Auxiliar.hpp"
#include "Geometry.hpp"
#include "DirectedGraph.hpp"

using namespace std;
using namespace cv;

class Formation{

	public: 		
		Formation(int n_agents, int controller_type); //constructor
		~Formation(); //destructor
		void readLaplacian(string dir);//function to read the laplacian for this formation
		int *getNeighbors(int i, int *n_neigh); //to obtain the neighbors of a given agent
		void initDesiredRelativePoses(int shape,double width); //to calculate the desired formation poses
		double **getA(); //to return the doubly stochastic matrix A
		double **getXAster();//to return the desired relative X 
		double **getYAster(); //to return the desired relative Y
		double **getZAster();//to return the desired relative Z
		double **getYawAster();//to return the desired relative Yaw
		DirectedGraph getGraph();//returns the associated graph to compute communication standards
	private:
		//-----------------------------------------------------atributes
		int n_agents;//number of agents
		int controller_type;//controller used 
		int **L;//Laplacian matrix
		double **A = nullptr; //doubly stochasthic matrix, sometimes needed
		double  **x_aster, **y_aster, **z_aster, **yaw_aster; //for the desired formation
		
		//------------------------------------------------------- private functions
		void computeA();//computes the doubly stochasthic matrix for altitude consensus (when needed)		
};
