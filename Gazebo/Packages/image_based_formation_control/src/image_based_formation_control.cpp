/*
Intel (Zapopan, Jal), Robotics Lab (CIMAT, Gto), Patricia Tavares.
February 5th, 2019
This ROS code is used to connect rotors_simulator hummingbird's camera 
and process some vision-based image_based_formation_control control.
*/

//      ROS LIBRARIES
#include <ros/ros.h>

//  C++ LIBRARIES
#include <string>

//      CUSTOM LIBRARIES
#include "image_based_formation_control/agent.h"


int main(int argc, char **argv){

    // INITIAL PARAMETERS
    double t = 0; //init time
    int ite = 0; //number of iterations	

    //  STOPING CRITERIA
    double err_t = 1000; //translation error
    double err_psi = 10000; //rotation error
    int last_time = 0; //last time we updated velocities

    // GET ARGUMENTS
    if(argc < 5 ){
        std::cout << "Not enough arguments" <<std::endl;
        return 0;
    }
    std::string me_str(argv[1]); //the actual drone
    int controller_type = atoi(argv[2]); //control to be used
    int matching = atoi(argv[3]);	//if we desired matching data
    int verbose = atoi(argv[4]);	//if we desired verbose information
    double timeSpan = atof(argv[5]);	// simulation time

    // ROS NODE
    ros::init(argc,argv,"image_based_formation_control");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);	

    //  AGENT INIT
    fvc::agent new_agent(me_str);
    new_agent.load(nh);
    if (! new_agent.imageRead() ){
        std::cerr << 
        "[ERR2] Could not open or find the reference images" << 
        std::endl ;
        return -1;
    }

    // AGENT PUBS/SUBS

    //  POSE SUB
    ros::Subscriber new_position_subscriber =
        nh.subscribe<geometry_msgs::Pose>(
            "/hummingbird"+me_str+"/ground_truth/pose",1,
            &fvc::agent::setPose, &new_agent);
    //  IMAGE SUB
    image_transport::Subscriber new_camera_subscriber =
        it.subscribe(
            "/hummingbird"+me_str+"/camera_nadir/image_raw",1,
            &fvc::agent::processImage, &new_agent);
    
    //  CORNERS PUB
    ros::Publisher image_descriptor_publisher =
        nh.advertise<image_based_formation_control::corners>(
            "/hummingbird"+me_str+"/corners", 1);
    //  DRONE POSITION PUB
    ros::Publisher position_publisher =
        nh.advertise<trajectory_msgs::MultiDOFJointTrajectory>(
            "/hummingbird"+me_str+"/command/trajectory",1);	

    // CORNER SUBS
    std::vector<ros::Subscriber> new_subs_neighbors; 
    for(int i=0;i<new_agent.n_agents;i++){
        if(new_agent.isNeighbor(i))
        {
            std::string j = std::to_string(i);
            ros::Subscriber n_s =
                nh.subscribe<image_based_formation_control::corners>(
                    "/hummingbird"+j+"/corners",
                    1,
                    &fvc::agent::getImageDescription,
                    &new_agent);
            new_subs_neighbors.push_back(n_s);
            std::cout <<  "---- " << me_str << 
            " subscribing to " << j << std::endl <<
            std::flush;
        }
    }
    
    //  Aruco image publisher
    image_transport::Publisher image_pub = it.advertise("Aruco",1);

        
    // LOOP

    ros::Rate rate(20);
    t = ros::Time::now().toSec();
    double dt = t;
    double tf = t +timeSpan;
    while(ros::ok()){
        
        //-----------------------------------------------------------
        //  PART 1 [IF DRONE IS NOT UPDATED YET]
        //-----------------------------------------------------------
        
        if(verbose)
        std::cout << "----- " << me_str 
        <<  " ROS:OK -----\n" << std::flush;

        ite+=1;	

        //get a msg	
        ros::spinOnce();	

        //  save data
//         new_agent.save_state(t);
        
        //if we havent get the pose
        if(! new_agent.isUpdated()){rate.sleep(); continue;}	

        if(verbose)
        std::cout << "----- " << me_str << 
        " Drone updated -----\n" << std::flush;

        //-----------------------------------------------------------
        //    PART 2 [IF THE VELOCITIES HAVE NOT BEEN COMPUTED]
        //-----------------------------------------------------------
        
        //    CORNER PUBLISH
        if(!new_agent.ARUCO_COMPUTED)
        {
            if(verbose)
            std::cout << "----- " << me_str << 
            "Aruco not  computed -----\n" << std::flush;
            rate.sleep();
                continue;
        }
        if(verbose)
        std::cout << "----- " << me_str <<
        " ArUco computed -----\n" << std::flush;
        //  Manda coordenadas de ArUco
        image_descriptor_publisher.publish(new_agent.getArUco());
        //  Manda imagen al visualizador
        image_pub.publish(new_agent.image_msg);
        
//         std::cout << new_agent.corners << std::endl << std::flush;
        
        //  If the velocities are incomplete, wait
        if(new_agent.incompleteComputedVelocities() )
        {
            if(verbose)
            std::cout << "----- " << me_str << 
            "velocities not  computed -----\n" << std::flush;
            rate.sleep();
                continue;
        }
        if(verbose)
        std::cout << "----- " << me_str << 
        "velocities  computed -----\n" << std::flush;
    
        rate.sleep();

        //-----------------------------------------------------------
        //    PART 3 EXECUTE CONTROL AND RESET
        //-----------------------------------------------------------

        //    time update
        t=ros::Time::now().toSec();
        dt = ros::Time::now().toSec() -dt;
//         dt = 0.025;
        //  CONTROL EXECUTE
        new_agent.execControl(dt);
        dt = t;
        
        //  save data
        new_agent.save_state(t);

        //  PUBLISH NEW POSITION
//         if (new_agent.label !=1)
        position_publisher.publish(new_agent.getPose());

        //  RESET CONTROL VELOCITIES
//         new_agent.reset( fvc::CONTRIBUTIONS | fvc::CORNERS); 
//         new_agent.reset( fvc::CORNERS); 
//         new_agent.reset( fvc::CONTRIBUTIONS); 

        //print information
        if(verbose)
        std::cout << t <<" Control exec in drone "<<
        me_str << std::endl << std::flush;
//         rate.sleep();
        
        //do we stop?
        if(t > tf ) break;
    }

    return 0;
}

