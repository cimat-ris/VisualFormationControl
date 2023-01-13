#ifndef AGENT_H
#define AGENT_H
 
#include <iostream>
#include <vector>
#include <string>

#include <ros/ros.h>
#include <opencv2/aruco.hpp>
#define  VC_CONTROLLER_NODE_H
#include "vc_controller/vc_controller.h"
#include "tf/transform_datatypes.h"
#include <cv_bridge/cv_bridge.h>

//TODO depurar
#include <image_transport/image_transport.h>
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/conversions.h>

#include <image_based_formation_control/image_description.h>
#include <image_based_formation_control/key_point.h>
#include <image_based_formation_control/corners.h>
#include <image_based_formation_control/point2f.h>
#include <image_based_formation_control/descriptors.h>
#include <image_based_formation_control/geometric_constraint.h>
#include <sensor_msgs/image_encodings.h>

namespace fvc
{
    const char ALL = 0xff;
    const char CONTRIBUTIONS = 0x01;
    const char CORNERS = 0x02;
    
    class agent {
    public:
        
//         vcc::state  State;
        vcc::state * States = nullptr;
        int label = 0;
        bool POSITION_UPDATED = false;
        
        //  Fila correspondiente del Laplaciano
        int n_agents= 0;
        int n_neighbors= 0;
        int * neighbors = nullptr;
        double * gamma = nullptr;
        bool *velContributions = nullptr;
        
        //  Variables 
        std::vector<cv::Mat> errors;
        
        //  directorios
        std::string input_dir;
        std::string output_dir;
        
        //  aruco data
        bool ARUCO_COMPUTED=false;
        std::vector<std::vector<cv::Point2f>> aruco_refs;
        std::vector<cv::Mat> desired_img;
        std::vector<cv::Point2f> corners;
        
        //  CALLBACKS
        //callback to obtain pose from sensors
        void setPose(const geometry_msgs::Pose::ConstPtr& msg); 
        //callback to obtain image and compute
        void processImage(const sensor_msgs::Image::ConstPtr & msg);
        //callback to obtain corners from neighbors
        void getImageDescription(const image_based_formation_control::corners::ConstPtr& msg);
        
        //  SENDS
        //  ArUco corners
        image_based_formation_control::corners getArUco();
        //  Get current pose
        trajectory_msgs::MultiDOFJointTrajectory getPose();
       
        //  métodos
        void reset(char SELECT);
        //  load yaml
        void load(const ros::NodeHandle &n);
        //  Excecute controll velocities
        void execControl(double dt);
        //  check if j is a neigbor
        bool isNeighbor(int j);
        //  check if the drone pose has been updated at start
        bool isUpdated();
        //  check if the velocity array is sincomplete
        bool incompleteComputedVelocities();
        
        //  save data
        void save_state(double time);
        
        
        //  read reference images
        bool imageRead();
        
        //  constructores destructores
        agent();
        agent(std::string name);
        ~agent();
        
        
    };
    
}




#endif
