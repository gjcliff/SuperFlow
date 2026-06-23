# SuperFlow

~TODO: add instructions~  
~TODO: add gps~  
~TODO: add magnetometer~  
~TODO: add camera~
TODO: make frontend for slam  
TODO: connect frontend with isam2 backend  
TODO: create pointcloud, visualize  
TODO: tune  


Point tracking and loop closure with SuperFlow
[![Watch the video](https://img.youtube.com/vi/EPN2-01Xpwc/maxresdefault.jpg)](https://youtu.be/EPN2-01Xpwc)

SuperFlow: Barebone SLAM Frontend with SuperPoint and OpticalFlow:
[![Watch the video](https://img.youtube.com/vi/RQ9UBKWL5aE/maxresdefault.jpg)](https://youtu.be/RQ9UBKWL5aE)

State estimation with GPS, IMU, and magnetometer
[![Watch the video](https://img.youtube.com/vi/rOaBWvPd-hU/maxresdefault.jpg)](https://youtu.be/rOaBWvPd-hU)

## dev setup
we are working with px4 v1.16

the default docs will take you to the docs for the main branch, but that is not
what we want to do.

*https://docs.px4.io/v1.16/en/dev_setup/dev_env_linux_ubuntu*  
vs.  
*https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu*  

we always want to look at pages with v1.16 in the URL

### install ros2 jazzy  
https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html

### setup ros2 workspace  
make a workspace:  
```bash  
mkdir -p ~/repos/ws/src/  
cd ~/repos/ws/src/  
git clone -b v1.16.1 https://github.com/PX4/px4_msgs.git  
cd ~/repos/ws/  
colcon build # takes ~3 min  
```

### setup this project
this virtual environment trick only works because i added:
```bash
[build_scripts]
executable = /usr/bin/env python3
```
to the setup.cfg file

```bash
cd ~/repos/ws/src/ &&
    git clone https://github.com/gjcliff/px4_slam.git
cd ~/repos/ws/
python -m venv .venv --system-site-packages
touch .venv/COLCON_IGNORE
source .venv/bin/activate
cd ~/repos/ws/src/px4_slam && \
    rm -rf build install log && \
    pip install -r requirements.txt && \
    cd ~/repos/ws && \
    colcon build
```

### setup px4  
follow instructions here:  
https://docs.px4.io/v1.16/en/dev_setup/dev_env_linux_ubuntu

in a nutshell:  
```bash  
cd ~/repos/
git clone -b release/v1.16 https://github.com/PX4/PX4-Autopilot.git --recursive
bash ./PX4-Autopilot/Tools/setup/ubuntu.sh
sudo reboot
```

add these topics to ```PX4-Autopilot/src/modules/uxrce-dds-client/dds_topics.yaml```:
```bash
  - topic: /fmu/out/sensor_gps
    type: px4_msgs::msg::SensorGps

  - topic: /fmu/out/vehicle_magnetometer
    type: px4_msgs::msg::VehicleMagnetometer
```

build
```bash
cd ~/repos/PX4-Autopilot
make px4_sitl
```

install uxrce-dds-bridge
```bash
sudo snap install micro-xrce-dds-agent --edge # specifically this version, if via snap
```

### running everything
you'll need three terminals

terminal 1:
```bash
# run this in any directory
micro-xrce-dds-agent udp4 -p 8888                                                                                                                                                [18:38:07]
```

terminal 2:
```bash
cd ~/repos/PX4-Autopilot
make px4_sitl gz_x500
```

terminal 3:
```bash
cd ~/repos/ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 launch px4_slam slam.launch.py
```
