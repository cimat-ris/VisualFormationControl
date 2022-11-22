/******************************************************************************** OpenCv libraries */
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

/*********************************************************************************** c++ libraries */
#include <iostream>
#include <vector>

/***************************************************************************** Hand made libraries*/
#include "Auxiliar.hpp"
#include "Geometry.hpp"

using namespace std;
using namespace cv;

class Controller{
	public:
		Controller();//constructor
		~Controller();//destructor
		//-----------------------------------set and get
		void setDesiredRelativePoses(double **x_aster,double **y_aster, double **z_aster, double **yaw_aster);//to set the desired relative poses
		void setProperties(int label, int n_agents, int n_neigh, int controller_type, double Kv, double Kw, double **A, string input_dir, string output_dir, int gamma_file);//to set some properties of the controller
		void resetVelocities();//equals velocities to 0		
		void getVelocities(double *Vx, double *Vy, double *Vz, double *Wz);//return computed velocities
		void setGamma(int label, double value);//sets the value of gamma for the given agent
		//---------------------------- processing
		void compute(int matches,int j,Mat &GC,double *pose_i, double *pose_j,double *R, double *t, vector<vector<KeyPoint>> kp_j,vector<vector<KeyPoint>> kp_i);//compute velocities
		//-----------------------------flags
		int haveComputedVelocities(double *et, double *epsi);//to verify everything has been computed
		int gammaInitialized();//verifies if the gamma vector has been initialized
        void camera_norm(Mat &  points);
	private:
		//--------------------------------------------attributes
		int label;//id for this quadrotor
		int n_neigh, n_agents; //number of neighbors and agents
		int controller_type;//controller type
		Mat K = Mat(3,3,CV_64F); //calibration matrix
		double Vx=0.0,Vy=0.0,Vz=0.0,Wz=0.0;//velocities
		double Kv,Kw;//gains
		double **x_aster,**y_aster,**z_aster,**yaw_aster;//desired relative poses
		double **A, *gamma;//doubly stochastic matrix and vector with altitudes

		//----------------------------------- additional attributes
		string input_dir, output_dir; //to write files
		double **xf, **yf,**zf,**yawf; //filters
		double *errors_t, *errors_psi; //to save the errors for all the neighbors
		double *pose_i, *pose_j;//for plotting

		//-----------------------------flags
		int ERRORS_SET = 0; //to verify if the array errors have been allocated
		int **DONE; //to check if  we have computed the pose for fist time
		int FILTERS_SET = 0; //if we have allocated DONE;xf,yf,zf and yawf
		int GAMMA_SET = 0; //if we have alocated gamma vector
		
		//-------------------------------- CONTROLLERS
		void PBFCHD(int matches,int i, int j, Mat &H); //position based formation contr with homography decomposition
		void PBFCED(int matches,int i, int j, Mat &E,double *R, double *t); //position based formation contr with homography decomposition
		void PBFCHDSA(int matches,int i, int j, Mat &H);//position based formatio  control with homography decomposition scale aware
		void PBFCEDSA(int matches,int i, int j, Mat &E, double *R, double *t);//position based formatio  control with essential decomposition scale aware
		void RBFCHD(int matches,int i, int j, Mat &H); //image based formation control with h. and rigidity matrix
		void RBFCED(int matches,int i, int j, Mat &E,double *R, double *t); //image based formation control with e. and rigidity matrix
		void IBFCH(int matches, int i, int j, Mat &H);//image based formation control with h. Montijano		
		void IBFCE(int matches, int i, int j, Mat &E,double *R, double *t);//image based formation control with essential
		void EBFC(int matches,int i, int j, Mat &F); //Epipoles Based Formation Control
        void IBFCF(int matches, int i, int j, vector<vector<KeyPoint>> kp_j,vector<vector<KeyPoint>> kp_i); // Visual controller from Chaumette implementacion 

		//--------------------------------- extra functions for controllers
		void getError(int matches,int i, int j, double hat_x, double hat_y, double hat_z, double hat_yaw);//to obtain the error between every pair
		int choose_first_homography_decomposition(int matches, int i,int j,vector<Mat> &t, vector<Mat> &r, vector<Mat> &n); //to choose the first homography decomposition
		void filter_pose(int i, int j, vector<Mat> &r, vector<Mat> &t,double *p, double *yaw);//to filter the pose with previous iterations
		void filter_pose_essential(int i, int j, Mat &R1, Mat &R2, Mat &t, double *p, double *yaw);//to filter essential decomposition
		void filter_pose_epipoles(int i, int j, double *e, double *p, double *yaw); //to filter epipoles
		void readK();
};
Mat interaction_Mat(Mat pointmatrix,double Z);
Mat  Moore_Penrose_PInv(Mat L,double & det);
