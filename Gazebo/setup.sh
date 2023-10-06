#!/bin/bash
if [ -z "$1" ] || [ -z "$2" ]
then
    echo "Error, no workspace selected"
    echo "Use:"
    echo -e "\t./setup.sh [WORKSPACEPATH] [SELECT]"
    echo "SELECT:"
    echo -e "\tAll \t Complete copy/update"
    echo -e "\tFiles \t Only configuration files"
    echo -e "\tIBFC \t image_based_formation_control"
    echo -e "\tIBVS \t vc_controller"
    exit
else
    WORKSPACE="$1"
fi


if [ "$2" == "Files" ] || [ "$2" == "All" ]
then
cp Files/multiple_hummingbird_nodes_*.launch ${WORKSPACE}/src/rotors_simulator/rotors_gazebo/launch/
cp -r Files/mygroundplane/ ~/.gazebo/models/
cp Files/ground.world ${WORKSPACE}/src/rotors_simulator/rotors_gazebo/worlds/
cp Files/hummingbird.xacro ${WORKSPACE}/src/rotors_simulator/rotors_description/urdf/hummingbird.xacro
fi

if [ "$2" == "IBFC" ] || [ "$2" == "All" ]
then
cp -r Packages/image_based_formation_control ${WORKSPACE}/src/
fi

if [ "$2" == "IBVS" ] || [ "$2" == "All" ]
then
cp -r Packages/vc_controller ${WORKSPACE}/src/
echo "target_compile_definitions(vc_controller PUBLIC WORKSPACE=\"${WORKSPACE}\")" >> ${WORKSPACE}/src/vc_controller/CMakeLists.txt
fi

if [ "$2" == "All" ]
then
cp -r Packages/Placing ${WORKSPACE}/src/
fi

source ${WORKSPACE}/devel/setup.bash
