#include "Formation.hpp"

/*
	Function: Constructor
	Descriptions: creates memory for the variables to be used
	params:
		n: number of agents in the formation
*/
Formation::Formation(int n_agents, int controller_type){
	this->n_agents = n_agents; 
	this->controller_type = controller_type;
	L = (int **) createMatrix(sizeof(int),sizeof(int*),n_agents,n_agents);
	x_aster = (double**) createMatrix(sizeof(double),sizeof(double*),n_agents,n_agents);
	y_aster = (double**) createMatrix(sizeof(double),sizeof(double*),n_agents,n_agents);
	z_aster = (double**) createMatrix(sizeof(double),sizeof(double*),n_agents,n_agents);
	yaw_aster = (double**) createMatrix(sizeof(double),sizeof(double*),n_agents,n_agents);
}

/*
	Function: Destructor
	Description: Destroys all the instanciated objets
*/
Formation::~Formation(){	
	freeMatrix((char**)L,n_agents);
	freeMatrix((char**)x_aster,n_agents);
	freeMatrix((char**)y_aster,n_agents);
	freeMatrix((char**)z_aster,n_agents);
	freeMatrix((char**)yaw_aster,n_agents);
	if(needsAltitudeConsensus(controller_type)) freeMatrix((char**)A,n_agents);
}

/*
	Function: readLaplacian
	description: reads the laplacian from a file and saves it in L (nxn matrix)
	params: 
		dir: path of the folder containing the laplacian file
*/
void Formation::readLaplacian(string dir){
	fstream inFile;
	string file("Laplacian.txt");

	int x=0,i=0,j=0;
	//open file
	inFile.open(dir+file);
	//read file
	while (inFile >> x) {
		L[i][j] = x;

		j++; if(j==n_agents){i++;j=0;}
	}

	//close file
	inFile.close();

	if(needsAltitudeConsensus(controller_type)) computeA();
}

/*
	function: getNeighbors
	description: gets the neighbors from the given agent using the laplacian
	params:
		i: integer representing the label of the drone executing the algorithm
		n_neigh: pointer to the variable with the number of neighbors
	returns:
		array with the id of every neighbor
*/
int *Formation::getNeighbors(int i, int *n_neigh){
	*n_neigh = abs(L[i][i]);
	int count=0;
	int *neighbors = new int[abs(L[i][i])];

	//search in the corresponding laplacian row
	for(int j=0;j<n_agents;j++){
		if(i!=j && L[i][j]!=0){
			neighbors[count] = j;
			count++;
		}
	}
			
	return neighbors;
}

/*
	Function: initDesiredRelativePoses
	description: inits the desired relative poses between agents
		using the laplacian and amount of agents
	params:
		shape: Desired shape, 0= circle, 1=line, 2: 3D circle
		width: longitud (for lines) or radius for 3D and circle
*/
void Formation::initDesiredRelativePoses(int shape, double width){
	double x[n_agents],y[n_agents],z[n_agents],yaw[n_agents];//poses of every agent
	//init everything with zero
	for(int i=0;i<n_agents;i++)
		for(int j=0;j<n_agents;j++){
			x_aster[i][j]=0.0;
			y_aster[i][j]=0.0;
			z_aster[i][j]=0.0;
			yaw_aster[i][j]=0.0;
		}

	//make circle
	double scaled = 0,number=(double)n_agents;

	//if we want a 3D formation
	if(shape==2) scaled=0.5;
	if(shape==0 || shape==2){
		for(int i=0;i<n_agents;i++){
			double place = (double) i;
			x[i] = width/2.0*cos((2.0*M_PI/number)*place);
			y[i] = width/2.0*sin((2.0*M_PI/number)*place);
			z[i] = scaled*place;
			yaw[i] = 0;
		}
	}else if(shape==1){
		for(int i=0;i<n_agents;i++){
			double place = (double) i;
			x[i] = width*place/(number-1.0); 
			y[i] = 0;
			z[i] = 0;
			yaw[i] = 0;
		}
	}
	
	//obtain the relative poses
	for(int i=0;i<n_agents;i++)
		for(int j=0;j<n_agents;j++)
			if(i!=j && L[i][j]!=0){
				Mat R_i = rotationZ(yaw[i]); Mat R_j = rotationZ(yaw[j]); Mat R_ij = R_i.t()*R_j;		
				Vec3d p_i(x[i],y[i],z[i]);
				Vec3d p_j(x[j],y[j],z[j]);

				Vec3f angles = rotationMatrixToEulerAngles(R_ij); yaw_aster[i][j] = angles[2];
				Mat p_ij = R_i.t()*Mat(p_j-p_i); 
				x_aster[i][j] = p_ij.at<double>(0,0);
				y_aster[i][j] = p_ij.at<double>(1,0);
				z_aster[i][j] = p_ij.at<double>(2,0);
			}	
}

/*
	Function: getA
	description: returns the pointer to A matrix
*/
double **Formation::getA(){
	return A;
}

/*
	Function: getXAster
	description: returns the pointer to x_aster matrix
*/
double **Formation::getXAster(){
	return x_aster;
}

/*
	Function: getYAster
	description: returns the pointer to y_aster matrix
*/
double **Formation::getYAster(){
	return y_aster;
}

/*
	Function: getZAster
	description: returns the pointer to z_aster matrix
*/
double **Formation::getZAster(){
	return z_aster;
}

/*
	Function: getYawAster
	description: returns the pointer to yaw_aster matrix
*/
double **Formation::getYawAster(){
	return yaw_aster;
}

/*
	function: getGraph
	description: returns the graph associated to this Formation for further computations
*/
DirectedGraph Formation::getGraph(){
	DirectedGraph g(n_agents);
	for(int i=0;i<n_agents;i++){
		for(int j=0;j<n_agents;j++){
			if(i!=j&&L[i][j]!=0){
				g.addEdge(i,j);
			}
		}
	}
	return g;
}

/*
	Function: computeA
	description: computes the double stochastic matrix A, needed for altitude consensus
*/
void Formation::computeA(){
	//create matrix
	A = (double**) createMatrix(sizeof(double),sizeof(double*),n_agents,n_agents);		

	//create needed matrices
	Mat S, Lap(n_agents, n_agents, CV_64F);
	for(int i=0;i<n_agents;i++)
		for(int j=0;j<n_agents;j++)
			Lap.at<double>(i,j) = (double) L[i][j];

	//compute svd
	SVD::compute(Lap,S);

	double alpha = 2.0/(S.at<double>(0,0)*S.at<double>(0,0)+S.at<double>(n_agents-2,0)*S.at<double>(n_agents-2,0));
	
	for(int i=0;i<n_agents;i++)
		for(int j=0;j<n_agents;j++)
			if(L[i][j]!=0 && i!=j) A[i][j] = alpha;
			else if(i==j) A[i][j] = 1.0 - (-L[i][j])*alpha;	
}
