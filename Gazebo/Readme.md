
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

How to use:
https://www.overleaf.com/read/zqzqrrgbwrqc





