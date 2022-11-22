#include "Agent.hpp"

/*
	Function: constructor
	Description: creates and defines all the vars needed
	params:
		name: string with the number of the agent
*/
Agent::Agent(string name){
	this->name = name;
	label = atoi(name.c_str());
	X = 0.0; Y= 0.0 ; Z = 0.0; Yaw = 0.0;	
	pose = new double[6];
	pose_j = new double[6];
}

/*
	Function: destructor
	Description: destroys everything the constructor or methods created
*/
Agent::~Agent(){
	if(NEIGHBORS_SET) delete [] neighbors;
	delete [] pose;
	delete [] pose_j;
}

/*
	function: getLabel
	description: tells the label of agent
	returns: variable i
*/
int Agent::getLabel(){
	return label;
}

/*
	function: getNNeighPtr
	description: returns de pointer to variable that counts the number
		of neighbors for this agent
*/
int *Agent::getNNeighPtr(){
	return &n_neigh;
}

/*
	function: setNeighbors
	description: saves the neighbors labels in an array
	param: pointer with the calculated array of neighbors, this can
	be computed with formation.getNeighbors
*/
void Agent::setNeighbors(int *neighbors){
	this->neighbors = neighbors;
	NEIGHBORS_SET = 1;
}

/*
	function: setDesiredRelativePoses
	description: sets the desired poses on the controller obtained with
	function formation.setInitDesiredPoses
	params:
		x_aster: pointer to desired relative x coords
		y_aster: pointer to desired relative y coords
		z_aster: pointer to desired relative z coords
		x_aster: pointer to desired relative yaw
*/
void Agent::setDesiredRelativePoses(double **x_aster,double **y_aster, double **z_aster, double **yaw_aster){
	controller.setDesiredRelativePoses(x_aster,y_aster,z_aster,yaw_aster);
}

/*
	function: setControllerProperties
	description: sets the needed information to work with the specified controller
	params: 
		controller_type:  code of the controller
		Kv: gain for linear velocities
		Kw: gain for angular velocity
		A: doubly stochastic matrix used for some consensus
		input_dir: dir to read files from 
		output_dir: output dir to write files
		gamma_file: if we will read the initial altitudes from a file
		n_agents: number of agents
		matching: if we want to save images of matching
		communication_type: how the drones will communicate with its neighbors (brute force or optimal)
*/
void Agent::setControllerProperties(int controller_type, double Kv, double Kw, double **A, string input_dir, string output_dir,int gamma_file, int n_agents, int matching, int communication_type){
	//create folders to save the information
	string rm("rm -rf "+output_dir+name); 
	string mk("mkdir "+output_dir+name);
	system(rm.c_str()); 
	system(mk.c_str());	
	for(int i=0;i<n_neigh;i++){
		string j = to_string(neighbors[i]);
		string mkj("mkdir "+output_dir+name+"/"+j);
		system(mkj.c_str());	
	}
	//set the properties for the given controller type
	controller.setProperties(label,n_agents,n_neigh,controller_type,Kv,Kw,A, input_dir,output_dir+name+"/",gamma_file);
	this->controller_type = controller_type;
	processor.setProperties(label,n_agents,matching,controller_type,input_dir,output_dir+name+"/");
	this->GAMMA_FILE = gamma_file;
	this->communication_type = communication_type;
}

/*
	function: resetVelocities
	description: turns in 0 every velocity calculated previously by the controller
*/
void Agent::resetVelocities(){		
	controller.resetVelocities();	
}

/*
	function: communicationSends
	description: it calculates the communication with a brute force/optimal way
	given by the first way to communicate from the documentation
	params: 
		*n_sends: integer ptr ti store the number of image descripiton
		this agent will receive and then send the computed geomtric constraint
		g: a graph associated to the formation
	returns: 
		an array with the neighbors that will send the image description to this agent
*/
int *Agent::communicationSends(int *n_sends,DirectedGraph g){
	if(communication_type == 0)
		return processor.BRCommunicationSends(label, n_neigh, neighbors, n_sends);

	return processor.OptCommunicationSends(g,label, n_sends);
}

/*
	function: communicationReceives
	description: it calculates the communication with a brute force/optimal  way
	given by the first way to communicate from the documentation
	params: 
		*n_receives: integer ptr ti store the number of computed geometric
		constraint this agent will receive
		g: a graph associated to the formation
	returns: 
		an array with the neighbors that will send the geometric constraint
*/
int *Agent::communicationReceives(int *n_receives){
	if(communication_type == 0)
		return processor.BRCommunicationReceives(label, n_neigh, neighbors, n_receives);

	return processor.OptCommunicationReceives(label, n_neigh, neighbors, n_receives);
}

/*
	function: sendGeometricConstraintMsg
	description: returns the geometric constraint on the custom msg, ready to send
	params:
		index: index of the agent we want to know about. This
		index depends of the index the agent appears in the array of neighbors
		given vy the communicationSends function.
	returns:
		computed geometric constraint between both agents.
		for example, if this agent label is i=1, and its neighbor j=2,
		this will send the constraint_{1,2}

*/	
formation_control::geometric_constraint Agent::sendGeometricConstraintMsg(int index){
	return processor.getGM(index);
}

/*
	Function: getPose
	description: get the ppose info from the groundtruth of the drone and uses it in simulation
	params: message with pose info
*/
void Agent::getPose(const geometry_msgs::Pose::ConstPtr& msg){	
	//creating quaternion
	tf::Quaternion q(msg->orientation.x, msg->orientation.y, msg->orientation.z, msg->orientation.w);
	//creatring rotation matrix ffrom quaternion
	tf::Matrix3x3 mat(q);
	//obtaining euler angles
	double roll, pitch, yaw;
	mat.getEulerYPR(yaw, pitch, roll);
	//saving the data obtained
	pose[3] = roll; pose[4] = pitch; pose[5] = yaw;
	pose[0] = msg->position.x;
	pose[1] = msg->position.y;
	pose[2] = msg->position.z;
				
	/*To integrate, we only need it the first time*/
	if(!POSE_UPDATED){
		//setting the position if its the first time 
		X = msg->position.x;
		Y = msg->position.y;
		Z = msg->position.z;
		Yaw = yaw;
		POSE_UPDATED = 1;

		//if altitude consensus needed and the initial altitudes are not given
		if(!GAMMA_FILE && needsAltitudeConsensus(controller_type)) controller.setGamma(label,Z);

		cout << "Init pose drone " << label << endl << "X: " << X << endl << "Y: " << Y << endl << "Z: " << Z << endl;
		cout << "Roll: " << pose[3] << endl << "Pitch: " << pose[4] << endl << "Yaw: " << Yaw << endl ;
		cout << "-------------" << endl;	
	}	
}

/* 
	function: processImage
	description: uses the msg image and converts it to and opencv image to obtain the kp and 
	descriptors.
	params: 
		msg: ptr to the msg image.
*/
void Agent::processImage(const sensor_msgs::Image::ConstPtr& msg){	
	try{		
		processor.takePicture(cv_bridge::toCvShare(msg,"bgr8")->image);
		processor.detectAndCompute(pose);		
	}catch (cv_bridge::Exception& e){
	 	ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
   }
	
}

/*
	function: getImageDescription
	description: returns the processed image description in the format
	of a message of the type image_description, ready to send to the neighbors
*/
formation_control::image_description Agent::getImageDescription(){
	//add aditional data and say that you have calculated everything
	return processor.getImageDescription();
}

/*
	function: move
	description: integrates the agent position with the given velocities
	computed by the controller. 
	params: sampling time dt
	returns: msg with the new pose, ready to publish
*/
trajectory_msgs::MultiDOFJointTrajectory Agent::move(double dt){
	double Vx,Vy,Vz,Wz;
	controller.getVelocities(&Vx,&Vy,&Vz,&Wz);

	Mat Rz = rotationZ(Yaw);
	Vec3d Vel(Vx,Vy,Vz);
	Mat p_d = Rz*Mat(Vel);//change in position

	//change in rotation
	Mat S = (Mat_<double>(3, 3) << 0,-Wz,0,Wz,0,0,0,0,0);
	Mat R_d = Rz*S; Mat R = Rz+R_d*dt; Vec3f angles = rotationMatrixToEulerAngles(R);

	X = X + p_d.at<double>(0,0)*dt;
	Y = Y + p_d.at<double>(1,0)*dt;
	Z = Z + p_d.at<double>(2,0)*dt;
	Yaw = (double) angles[2];
	
	//create message for the pose
	Eigen::VectorXd position; position.resize(3); 
	position(0) = X; position(1) = Y; position(2) = Z;

	// prepare msg
	position_msg.header.stamp=ros::Time::now();
	mav_msgs::msgMultiDofJointTrajectoryFromPositionYaw(position, Yaw, &position_msg);
	
	return position_msg;
}

/*
	function: getImageDescription
	description: get the image description from a neighbor message and process it
	params: 
		msg: image description
*/
void Agent::getImageDescription(const formation_control::image_description::ConstPtr& msg){
	int j, SUCCESS = 0, matches = 0; double R[9], t[3];	
	//compute message and obtain geometric constraint
	Mat GM = processor.getGeometricConstraint(msg,&j,pose,pose_j,&SUCCESS,&matches,R,t);
	if(SUCCESS) //everything could be computed
		controller.compute(matches,j,GM,
                           pose,pose_j,
                            R,t,
                            processor.kp_j, processor.kp_i);		
}


/*
	Function: getGeometricConstraint
	description: obtains geometric constraint from message and computes the velocities
*/
void Agent::getGeometricConstraint(const formation_control::geometric_constraint::ConstPtr& msg){
	//geometric constraint, roll and pitch from neighbor
	int you=msg->i;
	for(int i=0;i<6;i++)
		pose_j[i] = msg->pose[i]; 
	 
	//received calculation from agent i to agent j, we need the inverse for homography and transpose for fundamental
	double M[3][3];	
	for(int i=0;i<3;i++)
		for(int j=0;j<3;j++)
			M[i][j] = msg->constraint[i*3+j];

	Mat GC;  
	if(isHomographyConsensus(controller_type)) GC =  Mat(3, 3, CV_64F, M).inv();
	else GC =  Mat(3, 3, CV_64F, M).t();

	//get the additional info if needed
	double R[9],t[3];

	if(!isHomographyConsensus(controller_type)){
		for(int i=0;i<3;i++){
			t[i] = msg->t[i];
			for(int j=0;j<3;j++)
				R[i*3+j] = msg->R[i*3+j];
		}

		Mat tij = Mat(3, 1, CV_64F, t);
		Mat Rij = Mat(3, 3, CV_64F, R);

		Mat Rji = Rij.t();//transpose rotation matrix to obtain Rji and send in message
		Mat tji=-Rji*tij;//use this matrix to obtain t_ji and send as message
		
		for(int i=0;i<3;i++){
			t[i] = tji.at<double>(i,0);
			for(int j=0;j<3;j++)
				R[i*3+j] = Rji.at<double>(i,j);
		}
	}

	int matches = msg->n_matches;
    cout << "------------- DB4 ------------- \n";
	controller.compute(matches,you,GC,pose,pose_j,R,t,processor.kp_j, processor.kp_i);	
}

/*
	function: getGamma
	description: Obtaines the value gamma_0 for this agent, ready to send as message
	returns:
		formation_control costumed message to advertise gamma value
*/
formation_control::gamma Agent::getGamma(){
	formation_control::gamma g;
	g.gamma = Z;
	g.label = label;
	return g;
}

/*
	function: setGamma
	description: sets the gamma value for the given agent given in the message
	params:
		msg: message from an agent in the network containing gamma
*/
void Agent::setGamma(const formation_control::gamma::ConstPtr& msg){
	double value = msg->gamma;
	int label = msg->label;
	controller.setGamma(label,value);
}

/*
	function: haveComputedVelocities
	description: returns 1 if the agent has computed velocities with
	the matching. this is a padlock to avoid iterations without velocities computed
	and get the information from all the neighbors. It also returns the error
	of the desired relative poses and the actual computed poses.
	params:
		et: pointer to a variable where to store the translation error.
		epsi: pointer to a variable where to store the rotation error.	
*/
int Agent::haveComputedVelocities(double *et, double *epsi){
	return controller.haveComputedVelocities(et,epsi);
}

/*
	function: isUpdated
	description: returns 1 if the drone have updated its pose initially.
	This is necesary because we move it integrating the pose. Its a padlock
*/
int Agent::isUpdated(){
	return POSE_UPDATED;
}

/*
	function: gammaInitialized
	description: verifies if the gamma vector is initialized
	returns:
		0: it is not fully initialized
		1: it is initialized
*/
int Agent::gammaInitialized(){
	return controller.gammaInitialized();
}

