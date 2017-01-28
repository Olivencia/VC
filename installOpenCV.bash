#!/bin/bash

echo "Installing packages...";
sudo apt-get update;
sudo apt-get -y upgrade;
sudo apt-get install build-essential -y;
sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y;
sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev -y;
sudo apt-get install unzip g++ -y;
echo "Packages installed.";

echo "Begin to compile OpenCV 3.1.0";
cd $HOME;
if [ ! -f "opencv-3.1.0.zip" ];
	then
	wget https://github.com/Itseez/opencv/archive/3.1.0.zip -O "opencv-3.1.0.zip";
	unzip opencv-3.1.0.zip;
fi

cd opencv-3.1.0;
sudo cmake -DWITH_IPP=ON . && sudo make -j $(nproc) && sudo make install;
sudo cp 3rdparty/ippicv/unpack/ippicv_lnx/lib/intel64/libippicv.a /usr/local/lib/;
sudo echo "/usr/local/lib" > /etc/ld.so.conf.d/opencv.conf;
sudo ldconfig -v;
sudo updatedb;

echo "OpenCV 3.1.0 installed";