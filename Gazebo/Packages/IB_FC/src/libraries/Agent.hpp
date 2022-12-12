/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
March 1st, 2019
This code is used to save and access information about an agent, its callbacks, neighbors,
 and pose.
*/

/******************************************************************************** OpenCv libraries */
#include <cv_bridge/cv_bridge.h>

/*********************************************************************************** ROS libraries*/
#include <ros/ros.h>
#include "tf/transform_datatypes.h"
#include <image_transport/image_transport.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/conversions.h>
#include <IB_FC/gamma.h>

/*********************************************************************************** c++ libraries */
#include <string>
#include <sys/stat.h>

/***************************************************************************** Hand made libraries*/
#include "Auxiliar.hpp"
#include "Processor.hpp"
#include "Geometry.hpp"
#include "Controller.hpp"

#ifndef DIRECTEDGRAPH_H
	#include "DirectedGraph.hpp"
#endif

using namespace std;

class Agent{

	public: 
		Processor processor; //imageprocessing and other cv stuff		
		Agent(string name); //constructor
		~Agent(); //destructor

		//--------------------------------------------------------------- Setter and getter
		int getLabel();  //get the label for this drone
		int *getNNeighPtr(); //gets the ptr to the number of agents variable
		void setNeighbors(int *neighbors); //an array with the labels of every neighbor
		void setDesiredRelativePoses(double **x_aster,double **y_aster, double **z_aster, double **yaw_aster); //to set the desired relative poses in controller
		void setControllerProperties(int controller_type, double Kv, double Kw, double **A, string input_dir, string output_dir, int gamma_file, int n_agents, int matching, int communication_type); //to set some properties		
		void resetVelocities(); //reset the velocitites from previous matchings		
		int *communicationSends(int *n_sends,DirectedGraph g);//to obtain an array with the neighbors that will receive my image_description
		int *communicationReceives(int *n_receives);//to obtain an array with the neighbors that will send me the computed constraint 
		IB_FC::geometric_constraint sendGeometricConstraintMsg(int index);//send the corresponding geometric constraint

		//---------------------------------------------------------------- callbacks
		void getPose(const geometry_msgs::Pose::ConstPtr& msg); //callback to obtain pose from sensors
		void processImage(const sensor_msgs::Image::ConstPtr& msg); //callback to process image from camera
		IB_FC::image_description getImageDescriptionID(); //to send the msg with image description
		trajectory_msgs::MultiDOFJointTrajectory move(double dt); //to calculate the new pose of the agent
		void getImageDescription(const IB_FC::image_description::ConstPtr& msg);//computes constraint and velocities from msg		
		void getGeometricConstraint(const IB_FC::geometric_constraint::ConstPtr& msg); //obtains geometric constraint from message and process it
		IB_FC::gamma getGamma();//get the value gamma for this agent, ready to send as message
		void setGamma(const IB_FC::gamma::ConstPtr& msg);//sets the gamma value for the agent specified in the message

		//-----------------------------------------------------------------flags	
		int haveComputedVelocities(double *et, double *epsi); //verify we have computed the geometric constraint
        bool incompleteComputedVelocities();
		int isUpdated(); //to verify if we have updated poses at least once to integrate		
		int gammaInitialized();//verifies if the gamma vector is completely set		
        
        //  utilidades para leer imagen
        void imageRead(const string & file);
        bool imgEmpty();
        
	private:
		//-------------------------------------------------------------- Attributes 
		Controller controller; //control object for this drone		
		int label;//the label that corresponds to this drone
		string name; //number of the agent in string
		int n_neigh; //number of neighbors	
		int *neighbors; //array with the neihbors and its order
		double X, Y, Z, Yaw;//pose of this drone
		int controller_type = 0; //label of the 
		int communication_type = 0; //if it uses brute force to communicate or optima
		double *pose, *pose_j; //pose of this drone for plotting gt
		trajectory_msgs::MultiDOFJointTrajectory position_msg; //to send mesage with new position		

		//-------------------------------------------------------------- Flags
		int SHOW_MATCHING = 0;//save images for matching
		int NEIGHBORS_SET = 0; //to know if we have give memory to neighbors 
		int POSE_UPDATED = 0; //flag to know if the pose has been updated for the first time
		int GAMMA_FILE = 0; //controller code to obtain the first gamma aproximation
};
