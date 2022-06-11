/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares & Gerardo Rodriguez.
November 20th, 2017
This ROS code is used to connect rotors_simulator hummingbird's camera
and process the images to obtain the homography.
*/

/******************************************************* ROS libraries*/
#include <ros/ros.h>
#include "tf/transform_datatypes.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/conversions.h>

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

/* defining  ratio for flann*/
#define RATIO 0.7

/* Declaring namespaces */
using namespace cv;
using namespace std;

/* Declaring callbacks and other functions*/
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
void poseCallback(const geometry_msgs::Pose::ConstPtr& msg);
Vec3f rotationMatrixToEulerAngles(Mat &R);
void writeFile(vector<float> &vec, const string& name);
double my_abs(double a);

/* Declaring objetcs to receive messages */
sensor_msgs::ImagePtr image_msg;

/* CHANGE THE NAME OF MY WORKSPCE TO YOUR WORKSPACE */
string workspace = "/home/jbhayet/catkin_ws";

/* defining where the drone will move and integrating system*/
float X = 0, Y = 0 ,Z = 0, Yaw = 0, Pitch = 0, Roll = 0;
float Vx = 0, Vy = 0, Vz= 0, Vyaw = 0;
float dt = 0.025;
float Kv = 0.75, Kw = 1.1; //gains 0.75 1.1
double feature_threshold = 0.5; //in pixels
double mean_feature_error = 1e10;/* Error on the matched kp */
int updated = 0; //of the pose has been updated
float t = 0;//time
int selected  = 0; //to see if we have already choose a decomposition
float sign = 1.0;

/* declaring detector params */
int nfeatures=250;
float scaleFactor=1.2f;
int nlevels=8;
int edgeThreshold=15; // Changed default (31);
int firstLevel=0;
int WTA_K=2;
cv::ORB::ScoreType scoreType=cv::ORB::HARRIS_SCORE;
int patchSize=31;
int fastThreshold=20;

/* Declaring data for the desired pos: image, descriptors and keypoints of the desired pose */
Mat desired_descriptors;
vector<KeyPoint> desired_kp;
Mat desired_img;

/* Matrix for camera calibration*/
Mat K;
Mat t_best;
Mat R_best; //to save the rot and trans

/* Main function */
int main(int argc, char **argv){

	/***************************************************************************************** INIT */
	ros::init(argc,argv,"homography");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);

	/************************************************************* CREATING PUBLISHER AND SUBSCRIBER */
	image_transport::Subscriber image_sub = it.subscribe("/hummingbird/camera_nadir/image_raw",1,imageCallback);
	image_transport::Publisher image_pub = it.advertise("matching",1);
	ros::Rate rate(40);

	/************************************************************************** OPENING DESIRED IMAGE */
	string image_dir = "/src/homography/src/desired.png";
	desired_img = imread(workspace+image_dir,IMREAD_COLOR);
	Ptr<ORB> orb = ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
   	orb->detect(desired_img, desired_kp);
	orb->compute(desired_img,desired_kp, desired_descriptors);

	/******************************************************************************* MOVING TO A POSE */
	ros::Publisher pos_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>("/hummingbird/command/trajectory",1);
	ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::Pose>("/hummingbird/ground_truth/pose",1,poseCallback);

	/******************************************************************************* CAMERA MATRIX */
	K = Mat(3,3, CV_64F, double(0));
	K.at<double>(0,0) = 241.4268236;
	K.at<double>(1,1) = 241.4268236;
	K.at<double>(0,2) = 376.0;
	K.at<double>(1,2) = 240.0;
	K.at<double>(2,2) = 1.0;

	cout << "Calibration Matrix " << endl << K << endl;

	/**************************************************************************** data for graphics */
	vector<float> vel_x; vector<float> vel_y; vector<float> vel_z; vector<float> vel_yaw;
	vector<float> errors; vector<float> time;

	/******************************************************************************* BUCLE START*/
	while(ros::ok()){
		//get a msg
		ros::spinOnce();

		if(updated == 0){rate.sleep(); continue;} //if we havent get the pose

		//save data
		time.push_back(t);errors.push_back((float)mean_feature_error);
		vel_x.push_back(Vx);vel_y.push_back(Vy);vel_z.push_back(Vz);vel_yaw.push_back(Vyaw);

		//do we conitnue?
		if(mean_feature_error < feature_threshold)
			break;

		//publish image of the matching
		image_pub.publish(image_msg);
		t+=dt;
		//integrating
		X = X + Kv*Vx*dt;
		Y = Y + Kv*Vy*dt;
		Z = Z + Kv*Vz*dt;
		Yaw = Yaw + Kw*Vyaw*dt;
    cout << X << " " << Y << " " << Z << endl;

		//create message for the pose
		trajectory_msgs::MultiDOFJointTrajectory msg;
		Eigen::VectorXd position; position.resize(3);
		position(0) = X; position(1) = Y; position(2) = Z;

		// prepare msg
		msg.header.stamp=ros::Time::now();
		mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(position, Yaw, &msg);

		//publish
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

		/*************************************************************KP*/
		Mat descriptors; vector<KeyPoint> kp; //declaring kp and descriptors for actual image

		/*************************************************************Creatring ORB*/
		Ptr<ORB> orb = ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);
		orb->detect(img, kp);
		orb->compute(img, kp, descriptors);

		/************************************************************* Using flann for matching*/
		FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
  		vector<vector<DMatch>> matches;
 		matcher.knnMatch(desired_descriptors,descriptors,matches,2);

		/************************************************************* Processing to get only goodmatches*/
		vector<DMatch> goodMatches;

		for(int i = 0; i < matches.size(); ++i){
	    	if (matches[i][0].distance < matches[i][1].distance * RATIO)
	        	goodMatches.push_back(matches[i][0]);
		}

		/************************************************************* Findig homography */
		 //-- transforming goodmatches to points
  		vector<Point2f> p1;
  		vector<Point2f> p2;

  		for(int i = 0; i < goodMatches.size(); i++){
			//-- Get the keypoints from the good matches
			p1.push_back(desired_kp[goodMatches[i].queryIdx]. pt);
			p2.push_back(kp[goodMatches[i].trainIdx].pt);
		}

		//computing error
		Mat a = Mat(p1); Mat b = Mat(p2);
		mean_feature_error = norm(a,b)/(float)p1.size();

		//finding homography
		Mat H = findHomography(p1, p2 ,RANSAC, 0.5);
    if (H.rows==0)
      return;
		/************************************************************* Draw matches */
		Mat img_matches = Mat::zeros(img.rows, img.cols * 2, img.type());
		drawMatches(desired_img, desired_kp, img, kp,
					goodMatches, img_matches,
					Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

		/************************************************************* Prepare message */
		image_msg = cv_bridge::CvImage(std_msgs::Header(),sensor_msgs::image_encodings::BGR8,img_matches).toImageMsg();
		image_msg->header.frame_id = "matching_image";
	 	image_msg->width = img_matches.cols;
	  image_msg->height = img_matches.rows;
	  image_msg->is_bigendian = false;
		image_msg->step = sizeof(unsigned char) * img_matches.cols*3;
	  image_msg->header.stamp = ros::Time::now();
    cout << "9 " << endl;
    cout << H << endl;
		/************************************************************* descomposing homography*/
		vector<Mat> Rs;
		vector<Mat> Ts;
		vector<Mat> Ns;
		decomposeHomographyMat(H,K,Rs,Ts,Ns);
    cout << "10 " << endl;

		//////////////////////////////////////////////////////////// Selecting one decomposition the first time
		if(selected == 0){
			//to store the matrix Rotation and translation that best fix
			//constructing the extrinsic parameters matrix for the actual image
			Mat P2 = Mat::eye(3, 4, CV_64F);

			double th = 0.1, nz = 1.0; //max value for z in the normal plane
			//preparing the points for the test
			vector<Point2f> pp1; vector<Point2f> pp2; pp1.push_back(p1[0]);pp2.push_back(p2[0]);

			//for every rotation matrix
			for(int i=0;i<Rs.size();i++){
				//constructing the extrinsic parameters matrix for the desired image
				Mat P1; hconcat(Rs[i],Ts[i],P1);
				//to store the result
				Mat p3D;
				triangulatePoints(P1,P2,pp1,pp2,p3D); //obtaining 3D point
				//transforming to homogeneus
				Mat point(4,1,CV_64F);
				point.at<double>(0,0) = p3D.at<float>(0,0) /p3D.at<float>(3,0);
				point.at<double>(1,0) = p3D.at<float>(1,0) /p3D.at<float>(3,0);
				point.at<double>(2,0) = p3D.at<float>(2,0) /p3D.at<float>(3,0);
				point.at<double>(3,0) = p3D.at<float>(3,0) /p3D.at<float>(3,0);
				//verify if the point is in front of the camera. Also if is similar to [0 0 1] o [0 0 -1]
				//grving preference to the frist

				//cout << " Punto " << point << endl<< "  t " << Ts[i] <<endl<< " n " << Ns[i] << endl<<"----------"<<endl;
				if(point.at<double>(2,0) >= 0.0 && my_abs(my_abs(Ns[i].at<double>(2,0))-1.0) < th ){
					if(nz > 0){
						Rs[i].copyTo(R_best);
						Ts[i].copyTo(t_best);
						nz = Ns[i].at<double>(2,0);
						selected = 1;
					}
				}
			}
			//process again, it is probably only in z axiw rotation, and we want the one with the highest nz component
			if (selected == 0){
				double max = -1;
				for(int i=0;i<Rs.size();i++){
					//constructing the extrinsic parameters matrix for the desired image
					Mat P1; hconcat(Rs[i],Ts[i],P1);
					//to store the result
					Mat p3D;
					triangulatePoints(P1,P2,pp1,pp2,p3D); //obtaining 3D point
					//transforming to homogeneus
					Mat point(4,1,CV_64F);
					point.at<double>(0,0) = p3D.at<float>(0,0) /p3D.at<float>(3,0);
					point.at<double>(1,0) = p3D.at<float>(1,0) /p3D.at<float>(3,0);
					point.at<double>(2,0) = p3D.at<float>(2,0) /p3D.at<float>(3,0);
					point.at<double>(3,0) = p3D.at<float>(3,0) /p3D.at<float>(3,0);

					if(point.at<double>(2,0) >= 0.0 && my_abs(Ns[i].at<double>(2,0)) > max){
						Rs[i].copyTo(R_best);
						Ts[i].copyTo(t_best);
						max = my_abs(Ns[i].at<double>(2,0));
						selected = 1;
					}
				}
			}

			//cout << "selected " << endl;
			//cout << t_best << endl;
			//if not of them has been selected
			//now, we are not going to do everything again
		}else{//if we already selected one, select the closest to that one
			double min_t = 1e8, min_r = 1e8;
			Mat t_best_for_now, r_best_for_now;
			//choose the closest to the previous one
			for(int i=0;i<Rs.size();i++){
				double norm_diff_rot = norm(Rs[i],R_best);
				double norm_diff_t = norm(Ts[i],t_best);
				if(norm_diff_rot < min_r){ Rs[i].copyTo(r_best_for_now); min_r=norm_diff_rot; }
				if(norm_diff_t < min_t){ Ts[i].copyTo(t_best_for_now); min_t=norm_diff_t; }
			}
			//save the best but dont modify it yet
			r_best_for_now.copyTo(R_best);
			t_best_for_now.copyTo(t_best);
		}

		/********************************************************* calculing velocities in the axis*/
    cout << "11 " << endl;

		//velocities from homography decomposition
		Vx = (float) t_best.at<double>(0,1);
		Vy = (float) t_best.at<double>(0,0);
		Vz =  (float) t_best.at<double>(0,2); //due to camera framework
		//velocities from homography decomposition and euler angles.
		Vec3f angles = rotationMatrixToEulerAngles(R_best);
		Vyaw = (float) -angles[2];//due to camera framework

		if(updated == 1)
		cout << "---------------->\nVx: " << Vx << " Vy: " << Vy << " Vz: " << Vz << " Wz: " << Vyaw << " average error: " << mean_feature_error <<  endl;

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
	//creating quaternion
	tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
	//creatring rotation matrix ffrom quaternion
	tf::Matrix3x3 mat(q);
	//obtaining euler angles
	double roll, pitch, yaw;
	mat.getEulerYPR(yaw, pitch, roll);
	//saving the data obtained
	Roll = (float) roll; Pitch = (float) pitch;

	//setting the position if its the first time
	if(updated == 0){
		X = (float) msg->position.x;
		Y = (float) msg->position.y;
		Z = (float) msg->position.z;
		Yaw = (float) yaw;
		updated = 1;
		cout << "Init pose" << endl << "X: " << X << endl << "Y: " << Y << endl << "Z: " << Z << endl;
		cout << "Roll: " << Roll << endl << "Pitch: " << Pitch << endl << "Yaw: " << Yaw << endl;
	}
}

/*
	function: rotationMatrixToEulerAngles
	params:
		R: rotation matrix
	result:
		vector containing the euler angles
	function taken from : https://www.learnopencv.com/rotation-matrix-to-euler-angles/
*/
Vec3f rotationMatrixToEulerAngles(Mat &R){
    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );
    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular){
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }else{
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);
}

/*
	Function: writeFile
	description: Writes the vect given as param into a file with the specified name
	params: std:vector containing the info and file name
*/
void writeFile(vector<float> &vec, const string& name){
	ofstream myfile;
  	myfile.open(name);
	for(int i=0;i<vec.size();i++)
		myfile << vec[i] << endl;

	myfile.close();
}

/*
	function: abs
	description: dummy function to get and absolute value.
	params: number to use in the function.
*/
double my_abs(double a){
	if(a<0)
		return -a;
	return a;
}
