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

/*  CUSTOM LIBS */

#include "utils_geom.h"
#include "utils_io.h"
#include "utils_vc.h"
#include "utils_img.h"
#include "multiagent.h"

/*********************************************************************************** Declaring namespaces*/
using namespace cv;
using namespace std;

/*********************************************************************************** Declaring callbacks */
void geometricConstraintCallback(const montijano::geometric_constraint::ConstPtr& msg);
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
void imageDescriptionCallback(const montijano::image_description::ConstPtr& msg);
void poseCallback(const geometry_msgs::Pose::ConstPtr& msg);

/* OTHER Libs functions*/

// void initDesiredPoses(int montijano);
void getNeighbors(int me, int ** L );


/*********************************************************************************** Computer Vision Params */
montijano_parameters params ;

/* declaring detector params */
Mat descriptors; vector<KeyPoint> kp,kpm; //kp and descriptors for actual image for this drone

/*********************************************************************************** Declaring neighbors params*/
const int n = 3; //amount of drones in the system
int neighbors[n]; //array with the neighbors
int info[n],rec[n]; //neighbors comunicating the image description and receiving homgraphy
int n_info, n_rec=0; //number of neigbors communicating image info
int n_neigh = 0; //amount of neighbors
int actual; //the drone running this script

/*********************************************************************************** Declaring msg*/
sensor_msgs::ImagePtr image_msg; //to get image
montijano::image_description id; //for image description
montijano::geometric_constraint gm[n];//to send homography
Mat Hom[n],img; //to send homographies and get the actual image

/*  state and control   */

montijano_state state;
multiagent_state multiagent;
montijano_control control;

int ite = 0;

Ptr<ORB> orb;

/* Main function */
int main(int argc, char **argv){
    
    /*  LOADING STUFF   */

    int ** L = readLaplacian(WORKSPACE "/src/montijano/src/Laplacian.txt",n);
    ros::init(argc,argv,"montijano");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    
    multiagent.load(nh,"0");
    params.load(nh);
     orb = ORB::create(params.nfeatures,params.scaleFactor,params.nlevels,params.edgeThreshold,params.firstLevel,params.WTA_K,params.scoreType,params.patchSize,params.fastThreshold);

     
	/*********************************************************************************** Verify if the dron label has been set */
	if(argc == 1){//you havent named the quadrotor to use!
		cout << "You did not name a hummingbird" <<endl;
		return 0;
	}
    
	/*********************************************************************************** Defining neighbors */
	string act(argv[1]);//actual neighbor in string, a value between 1 and n (inclusive)
	actual = atoi(argv[1]);//actual neighbor integer, a value between 1 and n (inclusive)
	//read the given laplacian

	//assign the corresponding neighbors to this drone using the laplacian
	 getNeighbors(actual, L);
	 
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++)
			multiagent.d[i][j] = 0;
	}

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

	
	/******************************************************************************* BUCLE START*/
	while(ros::ok()){
		
		//reset velocities
		control.Vx = 0.0; control.Vy = 0.0; control.Vz = 0.0; control.Vyaw = 0.0;		
			
		//get a msg	
		ros::spinOnce();
		
		//if we havent get the pose
		if(state.updated == 0){rate.sleep(); continue;}
		
		//publish kp and descriptors
		id_pub.publish(id);
		//publish the geometric constraints obtained
		for(int i=0;i<n_info;i++)
			pubs_constraint[i].publish(gm[i]);

		//reset the kp and descriptors calculation				
		state.done = 0;

		//add time
// 		state.t+=params.dt;	

		//do we stop?
		if(state.t>10.0)
			break;
		printf("---->%d %f %f %f\n",actual,control.Vx,control.Vy,control.Vyaw);
		
		/***********************************************MOVING THE DRONE*/		
		//moving drone
		Mat Rz = rotationZ(state.Yaw);
		Vec3d Vel(control.Vx,control.Vy,control.Vz);
		Mat p_d = Rz*Mat(Vel);//change in position

		//change in rotation
		Mat S = (Mat_<double>(3, 3) << 0,-control.Vyaw,0,control.Vyaw,0,0,0,0,0);
		Mat R_d = Rz*S; Mat R = Rz+R_d*params.dt; Vec3f angles = rotationMatrixToEulerAngles(R);

        
        //  Update
        control.Vx = p_d.at<double>(0,0);
        control.Vy = p_d.at<double>(1,0);
        control.Vz = p_d.at<double>(2,0);
        control.Vyaw = (double) angles[2];
        state.update(control);
		
		//create message for the pose
		trajectory_msgs::MultiDOFJointTrajectory msg;
		Eigen::VectorXd position; position.resize(3); 
		position(0) = state.X; position(1) = state.Y; position(2) = state.Z;

		// prepare msg
		msg.header.stamp=ros::Time::now();
		mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(position, state.Yaw, &msg);	
		
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
void geometricConstraintCallback(const montijano::geometric_constraint::ConstPtr& msg ){
	
	//geometric constraint, roll and pitch from neighbor
	double GC[3][3];
    double  r = msg->roll, p=msg->pitch;
	int ii=msg->i,jj=msg->j; //received calculation from agent i to agent j, we need the inverse
	//if we received the correct inmontijano
	if(ii!=0 && jj!=0){

		//get the homography
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				GC[i][j] = msg->constraint[i*3+j];

		//invert matrix
		Mat H = Mat(3, 3, CV_64F, GC).inv();
            multiagent.update( r,p, params, state,control,  H, ii,  jj);

        
		
	}
}

/* 
	function: imageCallback
	description: uses the msg image and converts it to and opencv image to obtain the kp and 
	descriptors, it is done if the drone has moved to the defined position. After that the resulting image and velocities are published.
	params: 
		msg: ptr to the msg image.
*/
void imageCallback(const sensor_msgs::Image::ConstPtr& msg ){	
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
		state.done = 1;
		id.autor = actual;
		id.roll = state.Roll;
		id.pitch = state.Pitch;

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
	if(state.done==0) return; //if we dont have our own kp and descriptors

	FlannBasedMatcher matcher(new flann::LshIndexParams(20, 10, 2));
  	vector<vector<DMatch>> matches;
 	matcher.knnMatch(descriptors,dn,matches,2);
	
	/************************************************************* Processing to get only goodmatches*/
	vector<DMatch> goodMatches;

	for(int i = 0; i < matches.size(); ++i){
    	if (matches[i][0].distance < matches[i][1].distance * params.flann_ratio)
        	goodMatches.push_back(matches[i][0]);
	}

	/************************************************************* Finding homography */
	 //-- transforming goodmatches to points		
	vector<Point2f> p1; vector<Point2f> p2; 
    vector<int> mask;

	for(int i = 0; i < goodMatches.size(); i++){
		//-- Get the keypoints from the good matches
		p1.push_back(kp[goodMatches[i].queryIdx].pt);
		p2.push_back(kn[goodMatches[i].trainIdx].pt);
		
	}

	Mat H = findHomography(p1, p2 ,RANSAC, 1,mask);
	
	
	/************************************************************* preparing homography message */	
	//find in which order needs to be published	
	for(int i=0;i<n_info;i++) if(info[i]==autor) index = i;

	//create a geometric constraint and fill it
	montijano::geometric_constraint cons;
	cons.roll = state.Roll;
	cons.pitch = state.Pitch;
	cons.i = actual;
	cons.j = autor;

	//fill the homography matrix
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			cons.constraint[i*3+j] = H.at<double>(i,j);
	gm[index] = cons;
	H.copyTo(Hom[index]);
    
    double rollj = msg->roll, pitchj = msg->pitch;
    multiagent.update( rollj, pitchj, params,state, control,  H,  autor,  actual);
	
	
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
	state.Roll = roll; state.Pitch = pitch; 
	
	if(!state.updated ){
        
        state.set_gains(params);
        state.initialize((float) msg->position.x,(float) msg->position.y,(float) msg->position.z,yaw);

	}	
}





/*
	function: getNeighbors
	description: gets the neighbors from the given agent using the laplacian
	params:
		me: integer representing the label of the actual agent
*/
void getNeighbors(int me, int ** L ){
	//search in the corresponding laplacian row
	for(int i=0;i<n;i++)
		if(L[me][i]==-1){
			neighbors[n_neigh]=i;
			n_neigh++;
		}
}
