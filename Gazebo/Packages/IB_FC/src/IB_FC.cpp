/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
February 5th, 2019
This ROS code is used to connect rotors_simulator hummingbird's camera 
and process some vision-based IB_FC control.
*/

/*********************************************************************************** ROS libraries*/
#include <ros/ros.h>

/*********************************************************************************** c++ libraries */
#include <string>

/***************************************************************************** Hand made libraries*/
#include "libraries/Formation.hpp"
#include "libraries/Agent.hpp"
#include "libraries/Auxiliar.hpp"

/*********************************************************************************** Vars*/
/* Main function */
int main(int argc, char **argv){

	/*********************************************************************************** Init params for algorithm */
	//algorithm params and stopping criteria
	double dt = 0.02; //sampling time
	double t = 0; //init time
	int ite = 0; //number of iterations	

	//stopping criteria using errors
	double err_t = 1000; //translation error
	double err_psi = 10000; //rotation error
	int last_time = 0; //last time we updated velocities
	vector <double> e_t; //to reserve the last 10 translation errors
	vector <double> e_psi; //to reserve the last 10 rotation errors
	int window = 10; //window to compute the average of the errors
	int size = 0; //to keep control of the average

	/*********************************************************************************** Verify if the dron label has been set */
	if(argc < 16){//you havent named the quadrotor to use or any other config!
		cout << "You did not give enough info." <<endl;
		return 0;
	}

	/*********************************************************************************** get all the given information*/
	string me_str(argv[1]); //the actual drone
	int n = atoi(argv[2]); //n drones
	int controller_type = atoi(argv[3]); //control to be used
	double Kv = atof(argv[4]), Kw = atof(argv[5]);//gains
	double th_t = atof(argv[6]), th_psi = atof(argv[7]); //thresholds
	double max_time = atof(argv[8]);//max simulation time
	int shape = atoi(argv[9]); //desired formation code
	double width = atof(argv[10]); //desired formation code
	string input_dir(argv[11]); //dir to get information 
	string output_dir(argv[12]); //dir to write information
	int gamma_file = atoi(argv[13]); //if we give gamma in a file
	int matching = atoi(argv[14]);	//if we desired matching data
	int communication_type = atoi(argv[15]); //brute force/optimal
	
	//verify the formation and controller type are correct
	if(!isValidConsensus(controller_type)){
		cout << "Incorrect controller type."<<endl; 
		return 0;
	}

	//verify the desired formation
	if(!isValidDesiredFormation(shape)){
		cout << "Incorrect desired formation." << endl;
		return 0;	
	}
	
	/*********************************************************************************** create robot object */
	Formation formation(n, controller_type);
	formation.readLaplacian(input_dir);
	formation.initDesiredRelativePoses(shape,width);

	//create agent info
	Agent this_drone(me_str);
	this_drone.setDesiredRelativePoses(formation.getXAster(),formation.getYAster(),formation.getZAster(),formation.getYawAster());
	this_drone.setNeighbors(formation.getNeighbors(this_drone.getLabel(), this_drone.getNNeighPtr()));			
	this_drone.setControllerProperties(controller_type, Kv, Kw, formation.getA(),input_dir,output_dir,gamma_file, n,matching,communication_type);

    /************************************************************************** OPENING DESIRED IMAGE */
// 	string image_dir = "reference" + me_str + ".png";
// 	this_drone.imageRead(input_dir+image_dir);
// 	if(this_drone.imgEmpty()) {
    if (this_drone.imageRead(input_dir) != n){
		 cerr <<  "[ERR] Could not open or find the reference images" << endl ;
		 return -1;
	}
    
	/*********************************************************************************** Init node */
	ros::init(argc,argv,"IB_FC");
	ros::NodeHandle nh;
	image_transport::ImageTransport it(nh);	

	/*********************************************************************************** Pubs and subs for the actual drone */
	ros::Subscriber position_subscriber = nh.subscribe<geometry_msgs::Pose>("/hummingbird"+me_str+"/ground_truth/pose",1,&Agent::getPose, &this_drone);
	image_transport::Subscriber camera_subscriber = it.subscribe("/hummingbird"+me_str+"/camera_nadir/image_raw",1,&Agent::processImage, &this_drone);
	ros::Publisher g = nh.advertise<IB_FC::gamma>("/hummingbird"+me_str+"/gamma", 1);
	ros::Publisher image_descriptor_publisher = nh.advertise<IB_FC::image_description>("/hummingbird"+me_str+"/image_description", 1);
	ros::Publisher position_publisher = nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>("/hummingbird"+me_str+"/command/trajectory",1);	
		
	/*********************************************************************************** Pubs and subs for the neighbors */
	vector<ros::Publisher> pubs_constraint; //to publish the geometric constraint to other neighbors
	vector<ros::Subscriber> subs_neighbors; //to get the image description from neighbors
	vector<ros::Subscriber> subs_gamma; //to obtain the first gamma approximation if needed
	vector<image_transport::Subscriber> subs_matching; // to obtain the image from other neighbors and create the image matching

	//gets the way this drone will communicate with the neighbors, given the param communication_type
	int ns,nr; int *s = this_drone.communicationSends(&ns,formation.getGraph());
	int *r = this_drone.communicationReceives(&nr);
	
	
	for(int i=0;i<ns;i++){
		string j = to_string(s[i]);
		ros::Subscriber n_s = nh.subscribe<IB_FC::image_description>(
            "/hummingbird"+j+"/image_description",
            1,
            &Agent::getImageDescription,
            &this_drone);
		subs_neighbors.push_back(n_s);
		cout <<  "---- " << me_str << " subscribing to " << j << endl << flush;
        ros::Publisher gm_p = nh.advertise<IB_FC::geometric_constraint>(
            "/hummingbird"+me_str+"/geometric_constraint"+j,1);
		pubs_constraint.push_back(gm_p);	
		if(matching==1){		
			image_transport::Subscriber nis = it.subscribe("/hummingbird"+j+"/camera_nadir/image_raw",1,&Processor::matchingCallback,&this_drone.processor);
			subs_matching.push_back(nis);
		}
		
	}

	for(int i=0;i<nr;i++){
		std::string j = std::to_string(r[i]);
		ros::Subscriber n_s = nh.subscribe<IB_FC::geometric_constraint>("/hummingbird"+j+"/geometric_constraint" +me_str, 1,&Agent::getGeometricConstraint, &this_drone);
		subs_neighbors.push_back(n_s);
	}

	if(gamma_file == 0 && needsAltitudeConsensus(controller_type)==1){
		for(int i=0;i<n;i++){
			if(i!=this_drone.getLabel()){
				string j = to_string(i);
				ros::Subscriber l = nh.subscribe<IB_FC::gamma>("/hummingbird"+j+"/gamma",1,&Agent::setGamma, &this_drone);
				subs_gamma.push_back(l);
			}
		}
	}	
		
	/*********************************************************************************** ros rate*/	
	ros::Rate rate(20);
	
    //  to save data
    string name(output_dir+"/"+me_str+"/");
    
	/******************************************************************************* BUCLE START*/
	while(ros::ok()){

		/******************************************************************************* 1ST PART
			Here, we need to reset the velocities, get a the msgs and determine if we have enough information 
			to start computing the controller. Since we integrate the position we need to get the position first
			and, if the altitude consensus is needed, we will need to obtain gamma_0, this is done in the first step
			when initial info is obtained, we will continue to next step and this will not be longer excuted.
			
		*/
        cout << "----- " << me_str <<  "ROS:OK -----\n" << flush;
	
// 		this_drone.resetVelocities();
		ite+=1;	
		
		//get a msg	
		ros::spinOnce();	

        //  save data
        appendToFile(name+"pose.txt",this_drone.pose, 6);
        
		//if we havent get the pose
		if(this_drone.isUpdated() == 0){rate.sleep(); continue;}	

        cout << "----- " << me_str <<  "Drone updated -----\n" << flush;
		//if we havent initialized gamma
// 		if(needsAltitudeConsensus(controller_type) == 1 && gamma_file == 0 && this_drone.gammaInitialized()==0) { g.publish(this_drone.getGamma()); rate.sleep();continue;}

		/******************************************************************************* 2nd PART
			If we already know the pose and gamma_0 (if needed), we will proceed to process and send information 
			from every camera. Every drone makes its own calculations and send them to its neighbors (key points and descriptors)
			. Depending of the communication_type, the drone will also receive the information from other neighbors and compute
			the velocity with the recived information.			
		*/

		//publish kp and descriptors of this drone
		image_descriptor_publisher.publish(this_drone.getImageDescriptionID());	

		//publish the geometric constraints obtained
// 		for(int i=0;i<ns;i++)			
// 			pubs_constraint[i].publish(this_drone.sendGeometricConstraintMsg(i));

		//if we havent computed velocities using information from all neighbors, skip to the next loop.
// 		if(this_drone.haveComputedVelocities(&err_t,&err_psi) == 0)
		if(this_drone.incompleteComputedVelocities() )
        {
        cout << "----- " << me_str <<  "velocities not  computed -----\n" << flush;
            rate.sleep();
//             if(ite-last_time > 100) 
//                 break;
//             else 
                continue;
        } 		
        cout << "----- " << me_str <<  "velocities  computed -----\n" << flush;
    
		rate.sleep();

		/******************************************************************************* 3rd PART
			After, the velocities have been computed, the drone integrates it and gets the new pose (with the move function)
			some needed calculations are computed to determine if we have finish. All this process only used information from 
			the neighbors.
		*/

		//publish position
		position_publisher.publish(this_drone.move(dt,name));
        this_drone.resetVelocities(); // 
//         cout << me_str << "------- M O V E --------\n" << flush;
		//add time
		t+=dt;
		last_time = ite;

		//compute average error for 10 iterations
// 		e_t.push_back(err_t);
// 		e_psi.push_back(err_psi);
// 		if(size > window){
// 			e_t.erase(e_t.begin());
// 			e_psi.erase(e_psi.begin());
// 		}else size++;		
// 		err_t = average(e_t,size);
// 		err_psi = average(e_psi,size);

		//print information
		cout << t <<" I'm drone "<< me_str ;
//         cout << " e_t: "<<err_t;
//         cout<< " e_psi: " << err_psi << endl;

		//do we stop?
		if(t>max_time ) break;
	}

	return 0;
}
