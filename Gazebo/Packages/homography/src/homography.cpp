/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares & Gerardo Rodriguez.
November 20th, 2017
This ROS code is used to connect rotors_simulator hummingbird's camera
and process the images to obtain the homography.
*/

/******************************************************* ROS libraries*/
#include <ros/ros.h>
#include "tf/transform_datatypes.h"
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/conversions.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

/**************************************************** OpenCv Libraries*/
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

/*************************************************** c++ libraries */
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>

#include "utils_geom.h"
#include "utils_io.h"
#include "utils_vc.h"
#include "utils_img.h"

/* Declaring namespaces */
using namespace cv;
using namespace std;

/* Declaring callbacks and other functions*/
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
void poseCallback(const geometry_msgs::Pose::ConstPtr& msg);

/* Declaring objects to receive messages */
sensor_msgs::ImagePtr image_msg;

/* Workspace definition from CMake */
string workspace = WORKSPACE;

// Visual control parameters
vc_parameters params;

// Visual control state
vc_state state;

// Control vector
vc_control control;

// Desired configuration
vc_desired_configuration desired_configuration;

// Result of the matching operation
vc_homograpy_matching_result matching_result;

int selected  = 0; //to see if we have already choose a decomposition

Mat t_best;
Mat R_best; //to save the rot and trans

/* Main function */
int main(int argc, char **argv){

	/***************************************************************************************** INIT */
	ros::init(argc,argv,"homography");
	ros::NodeHandle nh;
	params.load(nh);


	image_transport::ImageTransport it(nh);

	/************************************************************* CREATING PUBLISHER AND SUBSCRIBER */
	image_transport::Subscriber image_sub = it.subscribe("/hummingbird/camera_nadir/image_raw",1,imageCallback);
	image_transport::Publisher image_pub = it.advertise("matching",1);
	ros::Rate rate(40);

	/************************************************************************** OPENING DESIRED IMAGE */
	string image_dir = "/src/homography/src/desired.png";
	desired_configuration.img = imread(workspace+image_dir,IMREAD_COLOR);
	if(desired_configuration.img.empty()) {
		 cerr <<  "[ERR] Could not open or find the reference image" << std::endl ;
		 return -1;
	}

	Ptr<ORB> orb = ORB::create(params.nfeatures,params.scaleFactor,params.nlevels,params.edgeThreshold,params.firstLevel,params.WTA_K,params.scoreType,params.patchSize,params.fastThreshold);
	orb->detect(desired_configuration.img, desired_configuration.kp);
	orb->compute(desired_configuration.img,desired_configuration.kp, desired_configuration.descriptors);

	/******************************************************************************* MOVING TO A POSE */
	ros::Publisher pos_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>("/hummingbird/command/trajectory",1);
	ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::Pose>("/hummingbird/ground_truth/pose",1,poseCallback);

	/**************************************************************************** data for graphics */
	vector<float> vel_x; vector<float> vel_y; vector<float> vel_z; vector<float> vel_yaw;
	vector<float> errors; vector<float> time;

	/******************************************************************************* CYCLE START*/
	while(ros::ok()){
		//get a msg
		ros::spinOnce();

		if(!state.initialized){rate.sleep(); continue;} //if we havent get the new pose

		//save data
		time.push_back(state.t);errors.push_back((float)matching_result.mean_feature_error);
		vel_x.push_back(control.Vx);
    vel_y.push_back(control.Vy);
    vel_z.push_back(control.Vz);
    vel_yaw.push_back(control.Vyaw);

		// Do we continue?
		if(matching_result.mean_feature_error < params.feature_threshold)
			break;

		// Publish image of the matching
		image_pub.publish(image_msg);

    // Update state with the current control
    auto new_pose = state.update(control);

    // Create message for the pose
    trajectory_msgs::MultiDOFJointTrajectory msg;
    // Prepare msg
    msg.header.stamp=ros::Time::now();
    mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(new_pose.first, new_pose.second, &msg);

    // Publish msg
    pos_pub.publish(msg);

		rate.sleep();
	}

	//save data
	string file_folder = "/src/homography/src/data/";
	writeFile(errors, workspace+file_folder+"errors.txt");
	writeFile(time, workspace+file_folder+"time.txt");
	writeFile(vel_x, workspace+file_folder+"Vx.txt");
	writeFile(vel_y, workspace+file_folder+"Vy.txt");
	writeFile(vel_z, workspace+file_folder+"Vz.txt");
	writeFile(vel_yaw, workspace+file_folder+"Vyaw.txt");

	return 0;
}

/*
	function: imageCallback
	description: uses the msg image and converts it to and opencv image to obtain the kp and
	descriptors, it is done if the drone has moved to the defined position. After that the resulting image and velocities are published.
	params:
		msg: ptr to the msg image.
*/

void imageCallback(const sensor_msgs::Image::ConstPtr& msg){

	try{
		Mat img=cv_bridge::toCvShare(msg,"bgr8")->image;

    // Call to function estimating the homography
    if (compute_homography(img,params,desired_configuration,matching_result)<0)
      return;
		/************************************************************* Prepare message */
		image_msg = cv_bridge::CvImage(std_msgs::Header(),sensor_msgs::image_encodings::BGR8,matching_result.img_matches).toImageMsg();
		image_msg->header.frame_id = "matching_image";
		 image_msg->width = matching_result.img_matches.cols;
		image_msg->height = matching_result.img_matches.rows;
		image_msg->is_bigendian = false;
		image_msg->step = sizeof(unsigned char) * matching_result.img_matches.cols*3;
		image_msg->header.stamp = ros::Time::now();

		// Decompose homography*/
		vector<Mat> Rs;
		vector<Mat> Ts;
		vector<Mat> Ns;
		decomposeHomographyMat(matching_result.H,params.K,Rs,Ts,Ns);

    // Select decomposition
    select_decomposition(Rs,Ts,Ns,matching_result,selected,R_best,t_best);

		/**********Computing velocities in the axis*/
		//velocities from homography decomposition
		control.Vx = (float) t_best.at<double>(0,1);
		control.Vy = (float) t_best.at<double>(0,0);
		control.Vz = (float) t_best.at<double>(0,2); //due to camera framework
		//velocities from homography decomposition and euler angles.
		Vec3f angles = rotationMatrixToEulerAngles(R_best);
		control.Vyaw = (float) -angles[2];//due to camera framework

		if(state.initialized)
		  cout << "---------------->\nVx: " << control.Vx << " Vy: " << control.Vy << " Vz: " << control.Vz << " Wz: " << control.Vyaw << " average error: " << matching_result.mean_feature_error <<  endl;

	}catch (cv_bridge::Exception& e){
		 ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	 }

}

/*
	Function: PoseCallback
	description: get the ppose info from the groundtruth of the drone and uses it in simulation
	params: message with pose info
*/
void poseCallback(const geometry_msgs::Pose::ConstPtr& msg){
	// Creating quaternion
	tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
	// Creatring rotation matrix ffrom quaternion
	tf::Matrix3x3 mat(q);
	//obtaining euler angles
	double roll, pitch, yaw;
	mat.getEulerYPR(yaw, pitch, roll);
	//saving the data obtained
	state.Roll = (float) roll; state.Pitch = (float) pitch;

	//setting the position if its the first time
	if(!state.initialized){
    state.set_gains(params);
    state.initialize((float) msg->position.x,(float) msg->position.y,(float) msg->position.z,yaw);
	}
}
