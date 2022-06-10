#!/bin/bash
WORKSPACE=~/catkin_ws/
cp -r Files/mygroundplane/ ~/.gazebo/models/
cp Files/ground.world ${WORKSPACE}/src/rotors_simulator/rotors_description/urdf/
cp -r Packages/Placing ${WORKSPACE}/src/
source ${WORKSPACE}/devel/setup.bash
