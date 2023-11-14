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
    WORKSPACE=${1%/}
fi

#   If Workspace does not exist
if test ! -d "$WORKSPACE"; then
  echo "Directory does not exists."
  exit
fi



if [ "$2" == "IBFC" ] || [ "$2" == "All" ]
then
cp -r Packages/image_based_formation_control ${WORKSPACE}/src/
# echo "# Directories, select desired input" >> ${WORKSPACE}/src/image_based_formation_control/config/params.yaml
# echo "input_dir: \"${WORKSPACE}/src/image_based_formation_control/config/\"" >> ${WORKSPACE}/src/image_based_formation_control/config/params.yaml
# echo "output_dir: \"${WORKSPACE}/image_based_formation_control/output/\"" >> ${WORKSPACE}/src/image_based_formation_control/config/params.yaml
mkdir -p ${WORKSPACE}/image_based_formation_control/output/0
mkdir -p ${WORKSPACE}/image_based_formation_control/output/1
mkdir -p ${WORKSPACE}/image_based_formation_control/output/2
mkdir -p ${WORKSPACE}/image_based_formation_control/output/3
cp Files/scripts/plot.py ${WORKSPACE}/image_based_formation_control/output
cp ../Python/Consensus/*.npz ${WORKSPACE}/image_based_formation_control/output
cp ../Python/Consensus/camera.py ${WORKSPACE}/image_based_formation_control/output
cp ../Python/Consensus/Arrow3D.py ${WORKSPACE}/image_based_formation_control/output
fi

if [ "$2" == "IBVS" ] || [ "$2" == "All" ]
then
cp -r Packages/vc_controller ${WORKSPACE}/src/
echo "target_compile_definitions(vc_controller PUBLIC WORKSPACE=\"${WORKSPACE}/\")" >> ${WORKSPACE}/src/vc_controller/CMakeLists.txt
fi

if [ "$2" == "Files" ] || [ "$2" == "All" ]
then
cp Files/launch/* ${WORKSPACE}/src/rotors_simulator/rotors_gazebo/launch/
cp -r Files/models/* ~/.gazebo/models/
cp -r Files/config/* ${WORKSPACE}/src/image_based_formation_control/config/
cp Files/worlds/* ${WORKSPACE}/src/rotors_simulator/rotors_gazebo/worlds/
cp Files/hummingbird.xacro ${WORKSPACE}/src/rotors_simulator/rotors_description/urdf/hummingbird.xacro

#   YAML
echo "# Directories, select desired input" >> ${WORKSPACE}/src/image_based_formation_control/config/params.yaml
echo "input_dir: \"${WORKSPACE}/src/image_based_formation_control/config/\"" >> ${WORKSPACE}/src/image_based_formation_control/config/params.yaml
echo "output_dir: \"${WORKSPACE}/image_based_formation_control/output/\"" >> ${WORKSPACE}/src/image_based_formation_control/config/params.yaml

fi

if [ "$2" == "All" ]
then
cp -r Packages/Placing ${WORKSPACE}/src/
fi

# source ${WORKSPACE}/devel/setup.bash
