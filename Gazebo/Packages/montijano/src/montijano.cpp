/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
February 5th, 2019
This ROS code is used to connect rotors_simulator hummingbird's camera 
and process some homography-based montijano control.
*/

/*********************************************************************************** ROS libraries*/
#include <ros/ros.h>
#include "tf/transform_datatypes.h"
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/conversions.h>
#include <std_msgs/UInt8MultiArray.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <montijano/image_description.h>
#include <montijano/key_point.h>
#include <montijano/descriptors.h>
#include <montijano/geometric_constraint.h>

/*********************************************************************************** OpenCv Libraries*/
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

/*********************************************************************************** c++ libraries */
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <math.h>

/*********************************************************************************** Declaring namespaces*/
using namespace cv;
using namespace std;

/*********************************************************************************** Declaring callbacks and other functions*/
void geometricConstraintCallback(const montijano::geometric_constraint::ConstPtr& msg);
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
void imageDescriptionCallback(const montijano::image_description::ConstPtr& msg);
void poseCallback(const geometry_msgs::Pose::ConstPtr& msg);
Vec3d rotationMatrixToEulerAngles(Mat &R);
Mat rotationX(double roll);
Mat rotationY(double pitch);
Mat rotationZ(double yaw);
void writeFile(vector<float> &vec, char *name);
double my_abs(double a);
void initDesiredPoses(int montijano);
void readLaplacian(char *dir);
void getNeighbors(int me);
double choose(double x,double x1,double x2,double x3,double x4);

/*********************************************************************************** Computer Vision Params */
#define RATIO 0.7 /* defining  ratio for flann*/
/* declaring detector params */
Mat descriptors; vector<KeyPoint> kp,kpm; //kp and descriptors for actual image for this drone
int n_matches = 0; //matching
int nfeatures=100;//250
float scaleFactor=1.2f;
int nlevels=8;
int edgeThreshold=20; // Changed default (31);
int firstLevel=0;
int WTA_K=2;
int scoreType=cv::ORB::HARRIS_SCORE;
int patchSize=31;
int fastThreshold=20;
Ptr<ORB> orb = ORB::create(nfeatures,scaleFactor,nlevels,edgeThreshold,firstLevel,WTA_K,scoreType,patchSize,fastThreshold);

/*********************************************************************************** Declaring neighbors params*/
const int n = 3; //amount of drones in the system
int L[n][n]; //Laplacian of the system
int neighbors[n]; //array with the neighbors
int info[n],rec[n]; //neighbors comunicating the image description and receiving homgraphy
int n_info, n_rec=0; //number of neigbors communicating image info
int n_neigh = 0; //amount of neighbors
int actual; //the drone running this script
double x_aster[n][n], y_aster[n][n],z_aster[n][n],yaw_aster[n][n]; //desired poses

/*********************************************************************************** Declaring msg*/
sensor_msgs::ImagePtr image_msg; //to get image
montijano::image_description id; //for image description
montijano::geometric_constraint gm[n];//to send homography
Mat Hom[n],img; //to send homographies and get the actual image
sensor_msgs::ImagePtr matching[n]; // to send matching

/*********************************************************************************** Vars to move the actual drone*/
double X = 0.0, Y = 0.0 ,Z = 0.0, Yaw = 0.0, Pitch = 0.0, Roll = 0.0; //pose
double Vx = 0.0, Vy = 0.0, Vz= 0.0, Vyaw = 0.0; //linear velocities
double dt = 0.02; //samplig time
double Kv = 0.75, Kw = 1.1; //gains
double error_threshold = 0.1; //in meters
double mean_error = 1e10; //error in 
int updated = 0; //if the pose has been updated
float t = 0;//time
float sign = 1.0; 
int done = 0, ite = 0,d[n][n];
double xc[n][n],yc[n][n],z[n][n],yaw[n][n];

/*********************************************************************************** Camera matrix*/
Mat K;

/* Main function */
int main(int argc, char **argv){
	
	/*********************************************************************************** Verify if the dron label has been set */
	if(argc == 1){//you havent named the quadrotor to use!
		cout << "You did not name a hummingbird" <<endl;
		return 0;
	}

	/*********************************************************************************** Defining neighbors */
	string act(argv[1]);//actual neighbor in string, a value between 1 and n (inclusive)
	actual = atoi(argv[1]);//actual neighbor integer, a value between 1 and n (inclusive)
	//read the given laplacian
	readLaplacian("/home/phyrsash/patty_ws/src/montijano/src/Laplacian.txt");
	//assign the corresponding neighbors to this drone using the laplacian
	 getNeighbors(actual);
	 
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++)
			d[i][j] = 0;
	}
	/*********************************************************************************** Init node */
	ros::init(argc,argv,"montijano");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);
	
	/*********************************************************************************** Pubs and subs for the actual drone */
	ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::Pose>("/hummingbird"+act+"/ground_truth/pose",1,poseCallback);
	image_transport::Subscriber image_sub = it.subscribe("/hummingbird"+act+"/camera_nadir/image_raw",1,imageCallback);
	ros::Publisher pp = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>("/hummingbird"+act+"/command/trajectory",1);
	ros::Publisher id_pub = nh.advertise<montijano::image_description>("/hummingbird"+act+"/image_description", 1);

	/*********************************************************************************** Subscription to neighbors */
	/* Right now, every agent publishes its own key points and descriptors. An agent will subscribe to the image description from neighbors 
	with larger id label. For example, in an all connected graph with 3 drones, the agent 1 will receive keypoints and descriptors from agent 2 and 3.
	The agent 2 will receive from agent 3, and the agent 3 will not get any key points; instead, the drone 3 will receive the hompgraphy computed
	by other drones. Since the graph is known to be undirected, the connections for the graph given as example would be:
	
		- 1: receives kp from 2 and 3. Computes homographies H_{12} and H_{13}. Agent 2 and 3 do not receive kp from 1 but homographies 
		H_{12} and H_{13}, respectively.
		- 2: receives kp from 3. Computes homography H_{23}. Agent 3 receives H_{23} and agent 2 receives H_{12}.
		- 3: does not receives any key point, but receives H_{13} and H_{23}.

	this way, they all have their respective needed homographies but the load is not balanced. This is the first approach made in order to make it
	automatic but it needs to be improved. */

	vector<ros::Subscriber> subs_neighbors;
	vector<ros::Publisher> pubs_constraint;

	for(int i=0;i<n_neigh;i++){
		info[i] = 0;	
		rec[i] = 0;	
		std::string name = std::to_string(neighbors[i]);		
		if(neighbors[i] > actual){//if we get the image description, we will not send it ours but the geometric description				
			ros::Subscriber n_s = nh.subscribe<montijano::image_description>("/hummingbird"+name+"/image_description",1,imageDescriptionCallback);
			subs_neighbors.push_back(n_s);
			ros::Publisher gm_p = nh.advertise<montijano::geometric_constraint>("/hummingbird"+act+"/geometric_constraint"+name,1);
			pubs_constraint.push_back(gm_p);
			Hom[n_info] = rotationX(0);
			info[n_info] = neighbors[i]; n_info++;					
		}else{//if we send points (someone subscribe it, we will receive the geometric constraint
			ros::Subscriber n_s = nh.subscribe<montijano::geometric_constraint>("/hummingbird"+name+"/geometric_constraint" + act, 1, geometricConstraintCallback);
			subs_neighbors.push_back(n_s);		
		}
	}
	
	//define ros rate
	ros::Rate rate(20);

	/*********************************************************************************** CAMERA MATRIX, the same for everyone*/
	K = Mat(3,3, CV_64F, double(0));
	K.at<double>(0,0) = 241.4268236;
	K.at<double>(1,1) = 241.4268236;
	K.at<double>(0,2) = 376.0;
	K.at<double>(1,2) = 240.0;
	K.at<double>(2,2) = 1.0;
	initDesiredPoses(0);//load the desired poses

	/******************************************************************************* BUCLE START*/
	while(ros::ok()){
		
		//reset velocities
		Vx = 0.0; Vy = 0.0; Vz = 0.0; Vyaw = 0.0;		
			
		//get a msg	
		ros::spinOnce();
		
		//if we havent get the pose
		if(updated == 0){rate.sleep(); continue;}
		
		//publish kp and descriptors
		id_pub.publish(id);
		//publish the geometric constraints obtained
		for(int i=0;i<n_info;i++)
			pubs_constraint[i].publish(gm[i]);

		//reset the kp and descriptors calculation				
		done = 0;

		//add time
		t+=dt;	

		//do we stop?
		if(t>10.0)
			break;
		printf("---->%d %f %f %f\n",actual,Vx,Vy,Vyaw);
		
		/***********************************************MOVING THE DRONE*/		
		//moving drone
		Mat Rz = rotationZ(Yaw);
		Vec3d Vel(Vx,Vy,Vz);
		Mat p_d = Rz*Mat(Vel);//change in position

		//change in rotation
		Mat S = (Mat_<double>(3, 3) << 0,-Vyaw,0,Vyaw,0,0,0,0,0);
		Mat R_d = Rz*S; Mat R = Rz+R_d*dt; Vec3f angles = rotationMatrixToEulerAngles(R);

		X = X + p_d.at<double>(0,0)*dt;
		Y = Y + p_d.at<double>(1,0)*dt;
		Z = Z + p_d.at<double>(2,0)*dt;
		Yaw = (double) angles[2];
		
		//create message for the pose
		trajectory_msgs::MultiDOFJointTrajectory msg;
		Eigen::VectorXd position; position.resize(3); 
		position(0) = X; position(1) = Y; position(2) = Z;

		// prepare msg
		msg.header.stamp=ros::Time::now();
		mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(position, Yaw, &msg);	
		
		rate.sleep();	
		//publish position
		pp.publish(msg);		
		ite+=1;
	}

	return 0;
}

/*
	function: geometricConstraintCallback
	description: function to recive the geometric constraint computed by a neighbor as a msg
	params: 
		msg: the message
*/
void geometricConstraintCallback(const montijano::geometric_constraint::ConstPtr& msg){
	
	//geometric constraint, roll and pitch from neighbor
	double GC[3][3], r = msg->roll, p=msg->pitch;
	int ii=msg->i,jj=msg->j; //received calculation from agent i to agent j, we need the inverse
	//if we received the correct inmontijano
	if(ii!=0 && jj!=0){

		//get the homography
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				GC[i][j] = msg->constraint[i*3+j];

		//invert matrix
		Mat H = Mat(3, 3, CV_64F, GC).inv();



		//create rotation matrices to obtain the rectified matrix
		Mat RXj = rotationX(r); Mat RXi = rotationX(Roll);
		Mat RYj = rotationY(p); Mat RYi = rotationY(Pitch);
		//rectification to agent i and agent j
		Mat Hir = RXi.inv() * RYi.inv()*K.inv();
		Mat Hjr = RXj.inv() * RYj.inv()*K.inv();
		//rectified matrix
		Mat Hr = Hir*H*Hjr.inv();		

		if(d[jj][ii] == 0){
		d[jj][ii]  =1;
		xc[jj][ii] = Hr.at<double>(1,2);
		yc[jj][ii] = Hr.at<double>(0,2);
			//velocities from homography 		
		Vx += Kv*(Z*Hr.at<double>(1,2)-x_aster[jj][ii]);
		Vy +=  Kv*(Z*Hr.at<double>(0,2)-y_aster[jj][ii]);		
		Vyaw += Kw*(atan2(Hr.at<double>(1,0),Hr.at<double>(0,0))-yaw_aster[jj][ii]);
		}else{
		
		vector<Mat> rotations;
		vector<Mat> translations;
		vector<Mat> normals;
		decomposeHomographyMat(H, K, rotations,translations, normals);

		double gg2[2],min=1e10;
		for(int i =0;i<4;i++){
			double c[2],c2[2];
			for(int j=0;j<2;j++){
				c[j] = translations[i].at<double>(j,0);			
			}c2[0] = xc[jj][ii]; c2[1] =yc[jj][ii]; 
			double aux = c[1];
			c[1] = c[0];
			c[0] = aux;
			double k = sqrt((c[0]-c2[0])*(c[0]-c2[0])+(c[1]-c2[1])*(c[1]-c2[1]));	
			if(k < min){
				min = k;
				gg2[0] = c[0]; gg2[1]=c[1];
			}
		}
		xc[jj][ii] = gg2[0];
		yc[jj][ii] = gg2[1];
		Vx += Kv*(Z*xc[jj][ii]-x_aster[jj][ii]);
		Vy +=  Kv*(Z*yc[jj][ii]-y_aster[jj][ii]);	
		Vyaw += Kw*(atan2(Hr.at<double>(1,0),Hr.at<double>(0,0))-yaw_aster[jj][ii]);
		}
		
	}
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
		img=cv_bridge::toCvShare(msg,"bgr8")->image;
		
		/************************************************************ Creatring ORB*/
		kp.clear();
		orb->detect(img, kp);
		orb->compute(img, kp, descriptors);

		/************************************************************ Prepare msg to publish descriptors*/
		int rows = descriptors.rows, cols = descriptors.cols;
		id.des.cols = cols; id.des.rows = rows; 
		id.des.data.clear();//clear previous data
		
		for (int i=0; i<rows; i++)
			for (int j=0; j<cols; j++)
			    id.des.data.push_back(descriptors.at<unsigned char>(i,j));

		/************************************************************ Prepare msg to publish kp*/
		id.kps.clear();
		for(std::vector<KeyPoint>::size_type i = 0; i != kp.size(); i++){
			montijano::key_point k;
			k.angle = kp[i].angle; k.class_id = kp[i].class_id;k.octave = kp[i].octave;
			k.pt[0] = kp[i].pt.x; k.pt[1] = kp[i].pt.y; 
			k.response = kp[i].response; k.size=kp[i].size;
    			id.kps.push_back(k);
		}
		//add aditional data and say that you have calculated everything
		done = 1;
		id.autor = actual;
		id.roll = Roll;
		id.pitch = Pitch;

	}catch (cv_bridge::Exception& e){
	 	ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
   }
	
}

/* 
	Function: imageDescriptionCallback
	description: gets the key points and descriptors from neighbors
	params: message with the image description 
*/
void imageDescriptionCallback(const montijano::image_description::ConstPtr& msg){	
	int cols = msg->des.cols, rows = msg->des.rows,autor = (int) msg->autor,index=0;
	Mat dn(rows,cols,0); //matrix with the descriptors from neighbors
	vector<KeyPoint> kn; //key points with the keypoints from neighbors
	
	//fill the descriptors matrix
	for(int i=0;i<rows;i++)
		for(int j=0;j<cols;j++)		
			dn.at<unsigned char>(i,j) = msg->des.data[i*cols + j];
	//fill the kp vector
	for(std::vector<KeyPoint>::size_type i = 0; i != msg->kps.size(); i++){
			KeyPoint k;
			k.angle = msg->kps[i].angle; k.class_id = msg->kps[i].class_id;k.octave = msg->kps[i].octave;
			k.pt.x = msg->kps[i].pt[0]; k.pt.y = msg->kps[i].pt[1]; 
			k.response = msg->kps[i].response; k.size=msg->kps[i].size;
    			kn.push_back(k);
	}
	
	/************************************************************* Using flann for matching*/
	if(done==0) return; //if we dont have our own kp and descriptors

	FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
  	vector<vector<DMatch>> matches;
 	matcher.knnMatch(descriptors,dn,matches,2);
	
	/************************************************************* Processing to get only goodmatches*/
	vector<DMatch> goodMatches;

	for(int i = 0; i < matches.size(); ++i){
    	if (matches[i][0].distance < matches[i][1].distance * RATIO)
        	goodMatches.push_back(matches[i][0]);
	}

	/************************************************************* Finding homography */
	 //-- transforming goodmatches to points		
	vector<Point2f> p1; vector<Point2f> p2; n_matches = 0; vector<int> mask;

	for(int i = 0; i < goodMatches.size(); i++){
		//-- Get the keypoints from the good matches
		p1.push_back(kp[goodMatches[i].queryIdx].pt);
		p2.push_back(kn[goodMatches[i].trainIdx].pt);
		
	}

	n_matches = 0;	
	Mat H = findHomography(p1, p2 ,CV_RANSAC, 1,mask);
	
	
	/************************************************************* preparing homography message */	
	//find in which order needs to be published	
	for(int i=0;i<n_info;i++) if(info[i]==autor) index = i;

	//create a geometric constraint and fill it
	montijano::geometric_constraint cons;
	cons.roll = Roll;
	cons.pitch = Pitch;
	cons.i = actual;
	cons.j = autor;

	//fill the homography matrix
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			cons.constraint[i*3+j] = H.at<double>(i,j);
	gm[index] = cons;
	H.copyTo(Hom[index]);
	/************************************************************* computing velocities */	
	double rollj = msg->roll, pitchj = msg->pitch;//getting roll and pitch data from neighbor
	
		Mat RXj = rotationX(rollj);
		Mat RXi = rotationX(Roll);
		Mat RYj = rotationY(pitchj);
		Mat RYi = rotationY(Pitch);
		//rectify homography
		Mat Hir = RXi.inv() * RYi.inv()*K.inv();
		Mat Hjr = RXj.inv() * RYj.inv()*K.inv();
		Mat Hr = Hir*H*Hjr.inv();
	if(d[actual][autor]==0){
	d[actual][autor] = 1;
	xc[actual][autor] = Hr.at<double>(1,2); yc[actual][autor] = Hr.at<double>(0,2);
	//velocities from homography 	

	
	Vx += Kv*(Z*Hr.at<double>(1,2)-x_aster[actual][autor]);
	Vy += Kv*(Z*Hr.at<double>(0,2)-y_aster[actual][autor]);		
	Vyaw += Kw*(atan2(Hr.at<double>(1,0),Hr.at<double>(0,0))-yaw_aster[actual][autor]);//due to camera framework	
	}else{
		
		vector<Mat> rotations;
		vector<Mat> translations;
		vector<Mat> normals;
		decomposeHomographyMat(H, K, rotations,translations, normals);

double gg2[2],min=1e10;
		for(int i =0;i<4;i++){
			double c[2],c2[2];
			for(int j=0;j<2;j++){
				c[j] = translations[i].at<double>(j,0);			
			}c2[0] = xc[actual][autor]; c2[1] =yc[actual][autor]; 
			double aux = c[1];
			c[1] = c[0];
			c[0] = aux;
			double k = sqrt((c[0]-c2[0])*(c[0]-c2[0])+(c[1]-c2[1])*(c[1]-c2[1]));	
			if(k < min){
				min = k;
				gg2[0] = c[0]; gg2[1]=c[1];
			}
		}
		xc[actual][autor] = gg2[0];
		yc[actual][autor] = gg2[1];
		Vx += Kv*(Z*xc[actual][autor]-x_aster[actual][autor]);
		Vy +=  Kv*(Z*yc[actual][autor]-y_aster[actual][autor]);	
		Vyaw += Kw*(atan2(Hr.at<double>(1,0),Hr.at<double>(0,0))-yaw_aster[actual][autor]);
	}
	
}	

double choose(double x,double x1,double x2,double x3,double x4){
	double dif[4];
	dif[0] = abs(x-x1); 
	dif[1] = abs(x-x2); 
	dif[2] = abs(x-x3); 
	dif[3] = abs(x-x4); 
	
	if (dif[0]<dif[1] && dif[0] < dif[2] && dif[0] < dif[3])
		return x1;
	if (dif[1]<dif[2] && dif[1] < dif[0] && dif[1] <dif[3])
		return x2;
	if (dif[2]<dif[0] && dif[2] < dif[1] && dif[2] < dif[3])
		return x3;
	return x4;
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
	Roll = roll; Pitch = pitch; 
	
	if(updated == 0){
		//setting the position if its the first time 
		X = msg->position.x;
		Y = msg->position.y;
		Z = msg->position.z;
		Yaw = yaw;
		updated = 1;
		cout << "Init pose drone " << actual << endl << "X: " << X << endl << "Y: " << Y << endl << "Z: " << Z << endl;
		cout << "Roll: " << Roll << endl << "Pitch: " << Pitch << endl << "Yaw: " << Yaw << endl ;
		cout << "-------------" << endl;	
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
Vec3d rotationMatrixToEulerAngles(Mat &R){      
    double sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) ); 
    bool singular = sy < 1e-6; // If
 
    double x, y, z;
    if (!singular){
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }else{
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3d(x, y, z);        
}

/*
	function: rotationX
	description: creates a matrix with rotation in X axis
	params:
		roll: angle in roll
*/
Mat rotationX(double roll){
	return (Mat_<double>(3, 3) <<
                  1,          0,           0,
                  0, cos(roll), -sin(roll),
                  0, sin(roll),  cos(roll));
}

/*
	function: rotationY
	description: creates a matrix with rotation in Y axis
	params:
		pitch: angle in pitch
*/
Mat rotationY(double pitch){
	return (Mat_<double>(3, 3) <<
                  cos(pitch), 0, sin(pitch),
                  0, 1,          0,
                  -sin(pitch), 0,  cos(pitch));
}

/*
	function: rotationZ
	description: creates a matrix with rotation in Z axis
	params:
		yaw: angle in yaw
*/
Mat rotationZ(double yaw){
	return (Mat_<double>(3, 3) <<                 
                  cos(yaw), -sin(yaw), 0,
                  sin(yaw),  cos(yaw), 0,
		  0,          0,           1);
}

/*
	Function: writeFile
	description: Writes the vect given as param into a file with the specified name
	params: std:vector containing the info and file name
*/
void writeFile(vector<float> &vec, char *name){
	ofstream myfile;
  	myfile.open(name);
	for(int i=0;i<vec.size();i++)
		myfile << vec[i] << endl;

	myfile.close();
}

/* 
	function: abs
	description: dummy function to get an absolute value.
	params: number to use in the function.
*/
double my_abs(double a){
	if(a<0)
		return -a;
	return a;
}



/*
	fuction: initDesiredPoses
	description: inits the desired relative poses between agents
	using the laplacian and amount of agents
	params:
		montijano: 0=line montijano, 1: circle montijano.
*/
void initDesiredPoses(int montijano){

	//init everything with zero
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			x_aster[i][j]=0.0;
			y_aster[i][j]=0.0;
			z_aster[i][j]=0.0;
			yaw_aster[i][j]=0.0;
		}}

	if(montijano==0){		
		x_aster[0][1] =x_aster[1][2] = 1.0;
		x_aster[2][1] = x_aster[1][0] = -1.0;
		x_aster[2][0] = -2;
		x_aster[0][2] = 2.0;
	}else if(montijano==1){
		x_aster[0][1] = -1.5;
		y_aster[0][1] = 0.8660254;
		x_aster[0][2] = -1.5;
		y_aster[0][2] = -0.8660254;
		x_aster[1][0] = 1.5;
		y_aster[1][0] = -0.8660254;
		x_aster[1][2] = -6.66133815e-16;
		y_aster[1][2] =  -1.73205081e+00;
		x_aster[2][0] = 1.5;
		y_aster[2][0] = 0.8660254;
		x_aster[2][1] = 6.66133815e-16;
		y_aster[2][1] = 1.73205081e+00;
	}	
}

/*
	Function: readLaplacian()
	description: reads the laplacian from a file and saves it in L (nxn matrix)
	params: 
		dir: direction of the file containing the laplacian
*/
void readLaplacian(char *dir){
	fstream inFile;
	int x=0,i=0,j=0;
	//open file
	inFile.open(dir);
	//read file
	while (inFile >> x) {
		L[i][j] = x;
		j++; if(j==n) {i++;j=0;}
	}
	
	//close file
	inFile.close();
}

/*
	function: getNeighbors
	description: gets the neighbors from the given agent using the laplacian
	params:
		me: integer representing the label of the actual agent
*/
void getNeighbors(int me){
	//search in the corresponding laplacian row
	for(int i=0;i<n;i++)
		if(L[me][i]==-1){
			neighbors[n_neigh]=i;
			n_neigh++;
		}
}
