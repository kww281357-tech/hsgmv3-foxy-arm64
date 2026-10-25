# syntax=docker/dockerfile:1
# Build with: docker buildx build --platform linux/arm64 --load -t hsgmv3:foxy-arm64 .
ARG BASE_IMAGE=ros:foxy-ros-base-focal
FROM ${BASE_IMAGE} AS runtime
ARG TARGETARCH
RUN test "$TARGETARCH" = arm64
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git ca-certificates python3-colcon-common-extensions \
    python3-numpy python3-scipy python3-opencv python3-pil python3-yaml \
    ros-foxy-rmw-cyclonedds-cpp ros-foxy-rosidl-generator-dds-idl \
    ros-foxy-geometry-msgs ros-foxy-rosidl-default-generators libyaml-cpp-dev \
    && rm -rf /var/lib/apt/lists/*
# Unitree's Foxy procedure: build CycloneDDS WITHOUT sourcing ROS first.
ARG CYCLONEDDS_REF=9995905bce6c4cf9f740d6438bbf7fcfd1c83dfd
ARG RMW_CYCLONEDDS_REF=c12abc56983204f1d91f2d839d394528c7b29b42
ARG UNITREE_ROS2_REF=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88
WORKDIR /opt/unitree_ws
RUN mkdir src && git clone https://github.com/eclipse-cyclonedds/cyclonedds src/cyclonedds \
    && git -C src/cyclonedds checkout "$CYCLONEDDS_REF" \
    && colcon build --packages-select cyclonedds --cmake-args -DBUILD_TESTING=OFF
RUN git clone https://github.com/ros2/rmw_cyclonedds src/rmw_cyclonedds \
    && git -C src/rmw_cyclonedds checkout "$RMW_CYCLONEDDS_REF" \
    && git clone https://github.com/unitreerobotics/unitree_ros2 /tmp/unitree_ros2 \
    && git -C /tmp/unitree_ros2 checkout "$UNITREE_ROS2_REF" \
    && mkdir -p /usr/share/doc/unitree_ros2 \
    && cp /tmp/unitree_ros2/LICENSE /usr/share/doc/unitree_ros2/LICENSE \
    && cp -a /tmp/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go src/unitree_go \
    && source /opt/ros/foxy/setup.bash && source install/local_setup.bash \
    && colcon build --packages-select rmw_cyclonedds_cpp unitree_go --cmake-args -DBUILD_TESTING=OFF \
    && mkdir -p /usr/share/doc/cyclonedds /usr/share/doc/rmw_cyclonedds \
    && cp src/cyclonedds/LICENSE* /usr/share/doc/cyclonedds/ \
    && cp src/rmw_cyclonedds/LICENSE* /usr/share/doc/rmw_cyclonedds/ \
    && rm -rf /tmp/unitree_ros2 build log src
WORKDIR /opt/hsgm
COPY src ./src
COPY config ./config
COPY tests ./tests
COPY scripts ./scripts
COPY requirements.txt LICENSE ./
COPY ros2_ws ./ros2_ws
RUN source /opt/ros/foxy/setup.bash && source /opt/unitree_ws/install/setup.bash \
    && cd ros2_ws && colcon build --cmake-args -DBUILD_TESTING=OFF
COPY docker/entrypoint.sh /hsgm-entrypoint.sh
COPY docker/configure_dds.py /opt/hsgm/docker/configure_dds.py
COPY docker/healthcheck.py /opt/hsgm/docker/healthcheck.py
RUN chmod +x /hsgm-entrypoint.sh && useradd --create-home --uid 10001 hsgm
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ROS_DOMAIN_ID=0 ROS_LOCALHOST_ONLY=1 \
    GO2_INTERFACE=lo HSGM_STATUS_PATH=/tmp/hsgm-status.json PYTHONUNBUFFERED=1
USER hsgm
# No HTTP service exists. DDS discovery is not a single published TCP port.
HEALTHCHECK --interval=15s --timeout=3s --start-period=15s --retries=3 \
    CMD ["python3", "/opt/hsgm/docker/healthcheck.py"]
ENTRYPOINT ["/hsgm-entrypoint.sh"]
CMD ["ros2", "run", "hsgm_go2_monitor", "monitor"]
