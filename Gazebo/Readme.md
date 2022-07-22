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
> roslaunch homography  homography.launch
```

How to use:
https://www.overleaf.com/read/zqzqrrgbwrqc






