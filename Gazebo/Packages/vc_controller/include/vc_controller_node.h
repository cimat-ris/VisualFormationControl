#ifndef VC_CONTROLLER_NODE_H
#define VC_CONTROLLER_NODE_H


/******************************************************* ROS libraries*/
#include <ros/ros.h>
#include "tf/transform_datatypes.h"
#include <trajectory_msgs/MultiDOFJointTrajectory.h>
#include <mav_msgs/conversions.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>

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
#include <iostream>
#include <string>
#include <vector>

/*************************************************** custom libraries */
#include "vc_controller/vc_controller.h"

#endif
