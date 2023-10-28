#ifndef AGENT_H
#define AGENT_H
 
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>

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
#include <image_based_formation_control/ArUco.h>
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
        bool NOT_INITIALIZED_FILE=true;
        
        //  Fila correspondiente del Laplaciano
        int n_agents= 0;
        int n_neighbors= 0;
        int * neighbors = nullptr;
        bool *velContributions = nullptr;

        //  Variables 
        std::vector<cv::Mat> errors;
        std::vector<cv::Mat> errors_1;
        std::vector<cv::Mat> errors_2;
        std::vector<int> ArUcos_ovelap;
        std::vector<std::vector<std::vector<cv::Point2f>>> complements;
        std::vector<std::vector<int>> complements_ids;
        
        //  Control modifications
        bool PIAG_ENABLE=false;
        bool INTEGRAL_INIT=false;
        std::vector<cv::Mat> errors_integral; // N arucos in aruco_refs_ids[label] X (4x2)
        double gamma_0, gamma_inf, gamma_d;
        double gammaIntegral_0, gammaIntegral_inf, gammaIntegral_d;

        //  directorios
        std::string input_dir;
        std::string output_dir;
        
        //  aruco data
        cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
        cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        bool ARUCO_COMPUTED=false;
        std::vector<std::vector<std::vector<cv::Point2f>>> aruco_refs;
        std::vector<std::vector<int>> aruco_refs_ids;
        std::vector<cv::Mat> desired_img;
        std::vector<std::vector<cv::Point2f> > corners;
        std::vector<int> corners_ids;
        
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
        //  Adaptive gamma
        double adaptGamma(double _gamma_0, double _gamma_inf, double _gamma_d,
                          cv::Mat _error);
        //  Error Integral
        void integrateError(double dt);
        
        //  save data
        void save_state(double time);
        
        //  read reference images
        bool imageRead();
        
        //  get idx list
        std::vector<int> getIdxArUco(int a, std::vector<std::vector<int>> list);

        //  Interface and debuging
        bool VERBOSE_ENABLE = false;
        cv::Mat image_store;
        sensor_msgs::ImagePtr image_msg;
        
        //  constructores destructores
        agent();
        agent(std::string name);
        ~agent();
        
        
    };
    
}




#endif
