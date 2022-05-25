/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares & Gerardo Rodriguez.
November 20th, 2017
This ROS code is used to connect rotors_simulator hummingbird's data and
move it to a desired pose.
*/

/******************************************************* ROS libraries*/
#include <ros/ros.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/conversions.h>
#include <geometry_msgs/PointStamped.h>

/****************************************************** c++ libraries */
#include <random>
#include <string>
#include <iostream>

void positionCallback(const geometry_msgs::PointStamped::ConstPtr& msg);
geometry_msgs::PointStamped pos_msg;

using namespace std;

/* Main function */
int main(int argc, char **argv){

	/***************************************************************************************** INIT */
	ros::init(argc,argv,"placing");
	ros::NodeHandle nh;
	
	if(argc < 2) return 0; //if no name was given, nothing to do here-
	string slash("/"); string name(argv[1]); string publish("/command/trajectory"); string subscribe("/ground_truth/position");
	string topic_pub = slash + name + publish; 
	string topic_sub = slash + name + subscribe;

	/************************************************************* CREATING PUBLISHER AND SUBSCRIBER*/
	ros::Publisher pos_pub = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(topic_pub,1);
	ros::Subscriber pos_sub = nh.subscribe<geometry_msgs::PointStamped>(topic_sub,1,positionCallback);
	ros::Rate rate(50);

	/************************************************************* DEFINING THE POSE */
	std::random_device rd; // obtain a random number from hardware
    std::mt19937 gen(rd()); // seed the generator
    std::uniform_real_distribution<> disXY(-3.0, 3.0); // define the range for X and Y
	std::uniform_real_distribution<> disZ(0.0, 3.0); // define the range for Z
	std::uniform_real_distribution<> disYaw(-2, 2); // define the range yaw

	float X, Y, Z, Yaw;
	X = argc > 2 ? atof(argv[2]):disXY(gen);//if x coordinate has been given
	Y = argc > 3 ? atof(argv[3]):disXY(gen);//if y coordinate has been given
	Z = argc > 4 ? atof(argv[4]):disZ(gen);//if z coordinate has been given
	Yaw = argc > 5 ? atof(argv[5]):disYaw(gen);//if yaw has been given
	cout << "Pose of the drone: " << endl << "X: " << X << "\nY: " <<Y << "\nZ: " << Z << "\nYaw: " << Yaw << endl;

	/******************************************************************************* BUCLE START*/
	while(ros::ok()){
		//get a msg
		ros::spinOnce();		
		//create message for the pose
		trajectory_msgs::MultiDOFJointTrajectory msg;
		Eigen::VectorXd position; position.resize(3); 
		position(0) = X; position(1) = Y; position(2) = Z;
		// prepare msg
		msg.header.stamp=ros::Time::now();
		mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(position, Yaw, &msg);	
		//publish
		pos_pub.publish(msg);
		//verify if it's over
		float error = (pos_msg.point.x-X)*(pos_msg.point.x-X)+(pos_msg.point.y-Y)*(pos_msg.point.y-Y)+(pos_msg.point.z-Z)*(pos_msg.point.z-Z);
		if(error < 10e-3)
			break;
		
		rate.sleep(); 
		
	}

	return 0;
}

/* 
	function: positionCallback
	description: gets the position of the drone and assigns it to the variable point_msg.
	params: ptr to msg.
*/
void positionCallback(const geometry_msgs::PointStamped::ConstPtr& msg){
	pos_msg.point.x = msg->point.x;
	pos_msg.point.y = msg->point.y;
	pos_msg.point.z = msg->point.z;
}
