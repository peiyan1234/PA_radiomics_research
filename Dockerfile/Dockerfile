From nvidia/cuda:11.2.0-devel-ubuntu20.04

LABEL maintainer "Li Pei-Yan"

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update                                                                                                                             && \
    apt-get install -y wget                                                                                                                    && \ 
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn8_8.0.4.30-1+cuda11.0_amd64.deb        && \    
    wget https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/libcudnn8-dev_8.0.4.30-1+cuda11.0_amd64.deb    && \    
    dpkg -i libcudnn8_8.0.4.30-1+cuda11.0_amd64.deb                                                                                            && \    
    dpkg -i libcudnn8-dev_8.0.4.30-1+cuda11.0_amd64.deb                                                                                        && \    
    apt-get install -y vim gedit firefox                                                                                                       && \
    apt-get install -y screen                                                                                                                  && \
    apt-get install -y libreoffice                                                                                                             && \
    apt-get install -y python3-pip                                                                                                             && \
    apt-get install -y build-essential cmake unzip                                                                                             && \
    apt-get install -y ffmpeg                                                                                                                  && \  
    apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev libv4l-dev                                && \
    apt-get install -y libjpeg-dev libpng-dev libtiff-dev                                                                                      && \
    apt-get install -y libxvidcore-dev libx264-dev                                                                                             && \
    apt-get install -y libgtk-3-dev                                                                                                            && \
    apt-get install -y libatlas-base-dev gfortran                                                                                              && \
    apt-get install -y libopenblas-dev                                                                                                         && \
    apt-get install -y python3-dev                                                                                                             && \
    apt-get install -y python3-tk                                                                                                              && \
    apt-get install -y tree                                                                                                                    && \
    apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler                                       && \
    apt-get install -y --no-install-recommends libboost-all-dev                                                                                && \
    apt-get install -y libgflags-dev libgoogle-glog-dev liblmdb-dev                                                                            && \
    apt-get install -y libopenexr-dev                                                                                                          && \
    apt-get install -y curl dirmngr apt-transport-https lsb-release ca-certificates                                                            && \
    curl -sL https://deb.nodesource.com/setup_12.x | bash -                                                                                    && \
    apt-get install -y nodejs                                                                                                                  && \
    apt-get install -y swig libeigen3-dev                                                                                                      && \
    apt-get install -y cpio                                                                                                                    && \
    apt-get update --fix-missing                                                                                                               && \
    apt-get install -y gtk2.0                                                                                                                  && \
    apt-get install -y xorg openbox                                                                                                            && \
    apt-get install -y linux-headers-5.4.0-58-generic                                                                                          && \
    apt-get install -y libstdc++6 build-essential                                                                                              && \
    apt-get install -y libopenexr-dev git-all                                                                                                  && \
    apt-get install -y python3-pyqt5

ENV Opencv_version="3.4.12"                                                 

RUN cd ~                                                                                                                                       && \
    wget -O opencv-${Opencv_version}.zip https://github.com/opencv/opencv/archive/${Opencv_version}.zip                                        && \
    unzip opencv-${Opencv_version}.zip                                                                                                         && \
    mv opencv-${Opencv_version} opencv                                                                                                         && \
                                                                                                                                                  \
    wget -O opencv_contrib-${Opencv_version}.zip https://github.com/opencv/opencv_contrib/archive/${Opencv_version}.zip                        && \
    unzip opencv_contrib-${Opencv_version}.zip                                                                                                 && \
    mv opencv_contrib-${Opencv_version} opencv_contrib                                                                                         && \
                                                                                                                                                  \
    python3 -m pip install -U pip                                                                                                              && \
    python3 -m pip install numpy                                                                                                               && \
    python3 -m pip install scipy decorator matplotlib scikit-image scikit-learn ipython                                                        && \
    python3 -m pip install easydict opencv-python pyinstaller                                                                                  && \
    python3 -m pip install protobuf                                                                                                            && \
    python3 -m pip install mxnet                                                                                                               && \
    python3 -m pip install gluoncv                                                                                                             && \
    python3 -m pip install pysimplegui                                                                                                         && \
    python3 -m pip install sourcedefender       									                                                           && \
    python3 -m pip install torch torchvision										                                                           && \
    python3 -m pip install pyyaml                                                                                                              && \
    python3 -m pip install tensorflow												                                                           && \
    python3 -m pip install emcee nose pyrfr jinja2 pybind11  					                                                               && \
    python3 -m pip install terminaltables                                                                                                      && \
    python3 -m pip install test-generator                                                                                                      && \
    python3 -m pip install defusedxml                                                                                                          && \
    python3 -m pip install pynrrd pyradiomics                                                                                                  && \
                                                                                                                                                  \
    mkdir -p ~/opencv/build                                                                                                                    && \
    cd ~/opencv/build                                                                                                                          && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE                           \
          -D CMAKE_INSTALL_PREFIX=/usr/local                    \
          -D INSTALL_PYTHON_EXAMPLES=ON                         \
          -D INSTALL_C_EXAMPLES=OFF                             \
          -D OPENCV_ENABLE_NONFREE=OFF                          \
          -D BUILD_opencv_xfeatures2d=OFF                       \
          -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
          -D BUILD_EXAMPLES=ON ..                                                                                                              && \
    nproc                                                                                                                                      && \
    make -j8                                                                                                                                   && \
    make install                                                                                                                               && \
    ldconfig                                                                                                                                   && \
    pkg-config --modversion opencv                                                                                                             && \
    cd ~                                                                                                                                       && \
    rm opencv-${Opencv_version}.zip opencv_contrib-${Opencv_version}.zip                                                                       && \
    rm -rf opencv opencv_contrib                            

CMD ["/bin/bash"]
