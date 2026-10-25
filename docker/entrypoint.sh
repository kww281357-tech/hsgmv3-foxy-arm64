#!/bin/bash
set -eo pipefail
source /opt/ros/foxy/setup.bash
source /opt/unitree_ws/install/setup.bash
source /opt/hsgm/ros2_ws/install/setup.bash
export CYCLONEDDS_URI="file:///tmp/hsgm-cyclonedds.xml"
python3 /opt/hsgm/docker/configure_dds.py
exec "$@"
