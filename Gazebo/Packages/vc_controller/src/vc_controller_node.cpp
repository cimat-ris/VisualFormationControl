/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares & Gerardo Rodriguez.
November 20th, 2017
This ROS code is used to connect rotors_simulator hummingbird's camera
and process the images to obtain the homography.
*/

#include "vc_controller_node.h"

/* Declaring namespaces */
using namespace cv;
using namespace std;

/* Declaring callbacks and other functions*/
void imageCallback(const sensor_msgs::Image::ConstPtr& msg);
void poseCallback(const geometry_msgs::Pose::ConstPtr& msg);
void writeFile(vector<float> &vec, const string& name);

/* Declaring objects to receive messages */
sensor_msgs::ImagePtr image_msg;

/* Workspace definition from CMake */
string workspace = WORKSPACE;

// Visual control state
vcc::state State;


// Result of the matching operation
vcc::matching_result Matching_result;

//  Selector de control 
int contr_sel = 0;

/* Main function */
int main(int argc, char **argv){

	/***************************************************************************************** INIT */
	ros::init(argc,argv,"vc_controller");
	ros::NodeHandle nh;
	State.load(nh);
    
    nh.getParam("/vc_controller_node/select", contr_sel);

	image_transport::ImageTransport it(nh);

	/************************************************************* CREATING PUBLISHER AND SUBSCRIBER */
	image_transport::Subscriber image_sub = it.subscribe("/hummingbird/camera_nadir/image_raw",1,imageCallback);
	image_transport::Publisher image_pub = it.advertise("matching",1);
	ros::Rate rate(40);

	/************************************************************************** OPENING DESIRED IMAGE */
	string image_dir = "/src/vc_controller/src/desired.png";
	State.Desired_Configuration.img = imread(workspace+image_dir,IMREAD_COLOR);
	if(State.Desired_Configuration.img.empty()) {
		 cerr <<  "[ERR] Could not open or find the reference image" << std::endl ;
		 return -1;
	}

	// TODO: simplificar
	Ptr<ORB> orb = ORB::create(State.params.nfeatures,State.params.scaleFactor,State.params.nlevels,State.params.edgeThreshold,State.params.firstLevel,State.params.WTA_K,State.params.scoreType,State.params.patchSize,State.params.fastThreshold);

	orb->detect(State.Desired_Configuration.img, 
                State.Desired_Configuration.kp);
	orb->compute(State.Desired_Configuration.img,
                 State.Desired_Configuration.kp,
                State.Desired_Configuration.descriptors);

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

		if(!State.initialized){rate.sleep(); continue;} //if we havent get the new pose

		//save data
		time.push_back(ros::Time::now().toSec());
        errors.push_back((float)Matching_result.mean_feature_error);
		vel_x.push_back(State.Vx);
    vel_y.push_back(State.Vy);
    vel_z.push_back(State.Vz);
    vel_yaw.push_back(State.Vyaw);

		// Do we continue?
        if(Matching_result.mean_feature_error < State.params.feature_threshold)
			break;

		// Publish image of the matching
		image_pub.publish(image_msg);

    // Update State with the current control
    auto new_pose = State.update();

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
	string file_folder = "/src/vc_controller/src/data/";
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

//         funlist controllers[1] = {&homography};
        
//         if (controllers[contr_sel](img, State, Matching_result)< 0)
//             return;
    
        State.Vx = 0.;
        State.Vy = 0.;
        State.Vz = 0.;
        State.Vroll = 0.;
        State.Vpitch = 0.;
        State.Vyaw = 0.;
        if (preprocessors[contr_sel](img, State.params, State.Desired_Configuration ,Matching_result)!= 0)
            return;
    
        if (controllers[contr_sel]( State, Matching_result)!= 0)
            return;
    
        /************************************************************* Prepare message */
		image_msg = cv_bridge::CvImage(std_msgs::Header(),sensor_msgs::image_encodings::BGR8,Matching_result.img_matches).toImageMsg();
		image_msg->header.frame_id = "matching_image";
		 image_msg->width = Matching_result.img_matches.cols;
		image_msg->height = Matching_result.img_matches.rows;
		image_msg->is_bigendian = false;
		image_msg->step = sizeof(unsigned char) * Matching_result.img_matches.cols*3;
		image_msg->header.stamp = ros::Time::now();
        
        
		if(State.initialized)
		  cout << "---------------->\n" << 
		  " Vx: " << State.Vx << 
		  " Vy: " << State.Vy <<
		  " Vz: " << State.Vz << 
		  " Vroll: " << State.Vroll <<
		  " Vpitch: " << State.Vpitch << 
		  " Wyaw: " << State.Vyaw <<
		  " average error: " << Matching_result.mean_feature_error <<  endl;

	}catch (cv_bridge::Exception& e){
		 ROS_ERROR("Could not convert from '%s' to 'bgr8'.", 
                   msg->encoding.c_str());
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
	State.Roll = (float) roll; State.Pitch = (float) pitch;

	//setting the position if its the first time
	if(!State.initialized){
//     State.set_gains(params);
    State.initialize((float) msg->position.x,(float) msg->position.y,(float) msg->position.z,yaw);
	}
}

void writeFile(vector<float> &vec, const string& name){
	ofstream myfile;
		myfile.open(name);
	for(int i=0;i<vec.size();i++)
		myfile << vec[i] << endl;
	myfile.close();
}
