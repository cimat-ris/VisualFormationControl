# Installation of rotors

Installation steps (**noetic**) inspired from [this post](https://github.com/ethz-asl/rotors_simulator/issues/699]):

1. Install and initialize ROS noetic desktop full, additional ROS packages, catkin-tools, and wstool:
```bash
> sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list'
> wget http://packages.ros.org/ros.key -O - | sudo apt-key add -
> sudo apt-get update
> sudo apt-get install ros-noetic-desktop-full ros-noetic-joy ros-noetic-octomap-ros ros-noetic-mavlink python3-wstool python3-catkin-tools protobuf-compiler libgoogle-glog-dev ros-noetic-control-toolbox ros-noetic-mavros
> sudo rosdep init
> rosdep update
> source /opt/ros/noetic/setup.bash
```

2. To create a ROS workspace
```bash
> mkdir -p ~/catkin_ws/src
> cd ~/catkin_ws/src
> catkin_init_workspace  # initialize your catkin workspace
> wstool init
> wget https://raw.githubusercontent.com/ethz-asl/rotors_simulator/master/rotors_hil.rosinstall
> wstool merge rotors_hil.rosinstall
> wstool update
```

3. To install rotors_simulator
```bash
> cd ~/catkin_ws/src
> git clone https://github.com/ethz-asl/rotors_simulator.git
```

4. Build your workspace 
```bash
> cd ~/catkin_ws/
> catkin build
```

5. Add sourcing to your .bashrc file
```bash
> echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
> source ~/.bashrc
```

# Installation of the visual formation control code

1. Download the repository and go into the Gazebo directory
```bash
> git clone git clone git@github.com:cimat-ris/VisualFormationControl.git
> cd Gazebo
```

2. Edit the WORKSPACE variable in setup.sh 

3. Run the setup.sh script. It will copy the code into the work directory, install the MAV model with camera, the ground plane image etc.
```bash
> bash setup.sh
```

4. Compile in the workspace directory
```bash
> catkin build
```

# Test the Placing package
In a first terminal,
```bash
> roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=hummingbird
```
In a second terminal,
```bash
> rosrun placing placing hummingbird 0.1 0.6 2.0 1.0
```

# Test the image acquisition
In a first terminal,
```bash
> roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=hummingbird  world_name:=ground
```
In a second terminal,
```bash
> rosrun placing placing hummingbird 0.1 0.6 2.0 1.0
```
In a third terminal,
```bash
> rosrun rqt_image_view rqt_image_view
```

You should see the image taken by the MAV in the rqt_image_view.

# Test the homography-based control
In a first terminal,
```bash
> roslaunch rotors_gazebo mav_hovering_example.launch mav_name:=hummingbird  world_name:=ground
```
In a second terminal,
```bash
> rosrun placing placing hummingbird 0.1 0.6 2.0 1.0
```
In a third terminal,
```bash
> rosrun rqt_image_view rqt_image_view
```
In a fourth terminal,
```bash
> roslaunch vc_controller  homography.launch
```

#   test image_based_formation_control

In terminal 1: Launch gazebo whit drones:
```bash
roslaunch rotors_gazebo multiple_hummingbird_nodes_3.launch world_name:=ArUco_2
```

In terminal 2: Launch thecontrol:
```bash
roslaunch image_based_formation_control controllers.launch
```

The second controll has a default set of arguments in the file `controllers.launch`
However, the control can be changed in the launch command:

```bash
roslaunch image_based_formation_control controllers.launch control:=1 matching:=0 verbose:=0
```

Such controllers are:

case 1: PBFCHD(matches,label,j,GC); break;
Position based using homography decomposition

case 2: PBFCED(matches,label,j,GC,R,t); break;
Position based using essential decomposition

case 3: PBFCHDSA(matches,label, j, GC); break;
PBFCHD + escale aware

case 4: PBFCEDSA(matches,label,j,GC,R,t); break; (NOT_IMPLEMENTED yet)
PBFCED + escale aware

case 5: RBFCHD(matches,label,j,GC); break; (NOT_IMPLEMENTED yet)
Rigidity based using homography decomposition

case 6: RBFCED(matches,label,j,GC,R,t); break; (NOT_IMPLEMENTED yet)
Rigidity based using essential decomposition


case 7: IBFCH(matches,label,j,GC); break;
Image-Based Formation Homography

case 8: IBFCE(matches,label,j,GC,R,t); break; (NOT_IMPLEMENTED yet)
Image-Based Formation Essential

case 9: EBFC(matches, label, j, GC); break;
Epipole Based 


Formation type can be changed in the same manner:

```bash
roslaunch formation_control controllers.launch desired_formation:=1
```

case 0: circle
case 1: line
case 2: 3D circle 

How to use:
https://www.overleaf.com/read/zqzqrrgbwrqc






