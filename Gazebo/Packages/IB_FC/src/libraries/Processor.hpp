/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
March 1st, 2019
This code is used to save and access information about the computer vision process.
*/

/*********************************************************************************** ROS libraries*/
#include <IB_FC/image_description.h>
#include <IB_FC/key_point.h>
#include <IB_FC/descriptors.h>
#include <IB_FC/geometric_constraint.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>

/******************************************************************************** OpenCv libraries */
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

#define RATIO 0.7 

/*********************************************************************************** c++ libraries */
#include <vector>

/***************************************************************************** Hand made libraries*/
#include "Auxiliar.hpp"
#ifndef DIRECTEDGRAPH_H
	#include "DirectedGraph.hpp"
#endif


using namespace std;
using namespace cv;

class Processor{

	public:
		Processor();//constructor
		~Processor();//destructor
	
		//setters and getters
		void takePicture(Mat img); //stores the image
		IB_FC::image_description getImageDescription();//returns the image description ready to send as msg
		IB_FC::geometric_constraint getGM(int index);//to obtain the msg for the corresponding pair
		int *BRCommunicationSends(int label, int n_neigh, int *neighbors, int *n_s);//describe the labelssage this drone will send
		int *BRCommunicationReceives(int label, int n_neigh, int *neighbors, int *n_r);//describe the messages this drone will receive
		int *OptCommunicationSends(DirectedGraph g, int label, int *n_s);//describe the labelssage this drone will send
		int *OptCommunicationReceives(int label, int n_neigh, int *neighbors, int *n_r);//describe the messages this drone will receive
		void setProperties(int label, int n_agents, int matching, int controller_type,string input_dir,string output_dir);//sets the controller type

		//computing
		void detectAndCompute(double *pose);//process the actual image to obtain key points and descriptors
		Mat getGeometricConstraint(const IB_FC::image_description::ConstPtr& msg, int *you, double *pose_i, double *pose_j, int *SUCCESS, int *n_matches,double *R, double *t); //
		void matchingCallback(const sensor_msgs::Image::ConstPtr& msg);
// 	private:
		//------------------------------------------attributes
		int label;//id of the quadrotor
		int controller_type;//code of the controller
		int *s, *r, ns=0, nr=0; //to kwow what it receives and sends
		int count = 0;//to label the images if show matching is enabled
		string input_dir; //directory to get the calibration matrix
		string output_dir; //directory to write results

		//------------------------------------------------ for computing and processing		
		Mat descriptors;//descriptors
		Mat K = Mat(3,3,CV_64F);//calibration matrix
		Mat img;//image from camera		
		vector<KeyPoint> kp; // the keypoints of this agent
		vector<vector<DMatch>> matchesNeighbors; //the matches between its neighbors
		vector<vector<KeyPoint>> kp_j, kp_i; //key points from neighbors
		vector<vector<Point2f>> pi,pj; //key points from neighbors
		IB_FC::image_description id; //for image description message
		vector<IB_FC::geometric_constraint> gm;//to send homography	made with every neighbor
		
		//------------------------------------------------- orb params
		int nfeatures=100;
		int nlevels=8;
		int edgeThreshold=20;
		int firstLevel=0;
		int WTA_K=2;
		int scoreType=cv::ORB::HARRIS_SCORE;
		int patchSize=31;
		int fastThreshold=20;
		float scaleFactor=1.2f;
		Ptr<ORB> orb;

		//------------------------------------------- flags
		int SHOW_MATCHING = 0; //to know if ew need to visualize matching
		int SEND_RECEIVE_SET = 0; //to know if we have set the way to communicate
		int COMPUTED = 0; //to check if the drone has process the image so it can be matched with neighbors
		int DONE_SET = 0; //to check if the drone processed the decomposition the first time (epipolar geometry)
		int *DONE; //flags to verify the first essential decomposition

		void readK(); //reads the calibration matrix from a file
		Mat H_from_points(vector<Point2f> &fp,vector<Point2f> &tp, int normalized);
};
