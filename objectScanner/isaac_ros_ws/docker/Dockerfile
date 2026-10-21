FROM nvidia/cuda:13.0.0-base-ubuntu24.04 AS trt_runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV ROS_DISTRO=jazzy
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

SHELL ["/bin/bash", "-c"]

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        locales \
        lsb-release \
        wget \
    && locale-gen en_US en_US.UTF-8

# Pin Isaac ROS to release 4.5, which matches Ubuntu 24.04/Jazzy,
# CUDA 13.0, and the host's NVIDIA 580 driver.
RUN curl -fsSL https://isaac.download.nvidia.com/isaac-ros/repos.key \
        | gpg --dearmor > /usr/share/keyrings/nvidia-isaac-ros.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-isaac-ros.gpg] https://isaac.download.nvidia.com/isaac-ros/release-4.5 noble main" \
        > /etc/apt/sources.list.d/nvidia-isaac-ros.list \
    && wget -q https://repo.download.nvidia.com/jetson/jetson-ota-public.asc \
        -O /tmp/nvidia-jetson.asc \
    && gpg --dearmor < /tmp/nvidia-jetson.asc \
        > /usr/share/keyrings/nvidia-jetson.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-jetson.gpg] http://repo.download.nvidia.com/jetson/x86_64/noble r38.4 main" \
        > /etc/apt/sources.list.d/nvidia-jetson.list \
    && curl -L -o /tmp/ros2-apt-source.deb \
        https://github.com/ros-infrastructure/ros-apt-source/releases/download/1.1.0/ros2-apt-source_1.1.0.noble_all.deb \
    && dpkg -i /tmp/ros2-apt-source.deb \
    && rm -f /tmp/nvidia-jetson.asc /tmp/ros2-apt-source.deb

# Isaac ROS 4.5 was built against CUDA 13.0 and TensorRT 10.13. Without
# these pins, NVIDIA's rolling repository currently resolves the unversioned
# TensorRT dependency to TensorRT 11/CUDA 13.4, which is ABI-incompatible.
COPY isaac-ros-cuda-13-0.pref /etc/apt/preferences.d/
COPY isaac-ros-tensorrt-13-0.pref /etc/apt/preferences.d/
COPY tensorrt-runtime.control /tmp/tensorrt-runtime.control

# Isaac ROS declares a dependency on NVIDIA's full `tensorrt` development
# metapackage. FoundationPose only needs the runtime libraries and trtexec in
# this prebuilt image; installing the full metapackage would add about 3.9 GB
# of headers, static libraries, samples, and Windows engine-builder resources.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        libnvinfer10=10.13.3.9-1+cuda13.0 \
        libnvinfer-bin=10.13.3.9-1+cuda13.0 \
        libnvinfer-dispatch10=10.13.3.9-1+cuda13.0 \
        libnvinfer-lean10=10.13.3.9-1+cuda13.0 \
        libnvinfer-plugin10=10.13.3.9-1+cuda13.0 \
        libnvinfer-vc-plugin10=10.13.3.9-1+cuda13.0 \
        libnvonnxparsers10=10.13.3.9-1+cuda13.0 \
    && mkdir -p /tmp/tensorrt-runtime/DEBIAN \
    && cp /tmp/tensorrt-runtime.control /tmp/tensorrt-runtime/DEBIAN/control \
    && dpkg-deb --build /tmp/tensorrt-runtime /tmp/tensorrt-runtime.deb \
    && dpkg -i /tmp/tensorrt-runtime.deb \
    && rm -rf /tmp/tensorrt-runtime /tmp/tensorrt-runtime.deb \
        /tmp/tensorrt-runtime.control

RUN apt-mark hold tensorrt

FROM trt_runtime AS final

# Isaac ROS 4.5 pip-shim packages inspect this setting in their Debian
# pre-install scripts.  NVIDIA's own container layer installs the CLI first
# and marks the environment as docker-activated before installing ROS nodes.
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        -o Dpkg::Options::=--force-confold \
        isaac-ros-cli \
    && cp /etc/isaac-ros-cli/docker/patches/docker-environment.conf \
        /etc/isaac-ros-cli/environment.conf

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
        -o Dpkg::Options::=--force-confold \
        jq \
        ros-jazzy-isaac-ros-examples \
        ros-jazzy-isaac-ros-foundationpose \
        ros-jazzy-ros-base \
        ros-jazzy-rosbag2 \
        ros-jazzy-rosbag2-storage-mcap \
        tar

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc

WORKDIR /workspaces/isaac_ros-dev
CMD ["bash"]
