#!/bin/bash
WORKSPACE=~/catkin_ws/
cp -r Files/mygroundplane/ ~/.gazebo/models/
cp Files/ground.world ${WORKSPACE}/src/rotors_simulator/rotors_gazebo/worlds/
cp Files/hummingbird.xacro ${WORKSPACE}/src/rotors_simulator/rotors_description/urdf/hummingbird.xacro
cp -r Packages/Placing ${WORKSPACE}/src/
cp -r Packages/homography ${WORKSPACE}/src/
source ${WORKSPACE}/devel/setup.bash
