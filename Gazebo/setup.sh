#!/bin/bash
WORKSPACE=~/catkin_ws
cp -r Files/mygroundplane/ ~/.gazebo/models/
cp Files/ground.world ${WORKSPACE}/src/rotors_simulator/rotors_gazebo/worlds/
cp Files/hummingbird.xacro ${WORKSPACE}/src/rotors_simulator/rotors_description/urdf/hummingbird.xacro
cp -r Packages/Placing ${WORKSPACE}/src/
cp -r Packages/vc_controller ${WORKSPACE}/src/
cp -r Packages/formation_control ${WORKSPACE}/src/
echo "target_compile_definitions(vc_controller PUBLIC WORKSPACE=\"${WORKSPACE}\")" >> ${WORKSPACE}/src/vc_controller/CMakeLists.txt
source ${WORKSPACE}/devel/setup.bash
