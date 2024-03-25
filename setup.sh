#!/bin/bash

# Check if Eigen3 is installed
if ! dpkg -l | grep -q libeigen3-dev; then
    echo "Eigen3 not found. Installing..."
    sudo apt install libeigen3-dev
fi

# Check if OpenCV is installed
if ! dpkg -l | grep -q opencv; then
    echo "OpenCV not found. Installing..."
    sudo apt install libopencv-dev
fi

# Check if Python development headers are installed
if ! dpkg -l | grep -q python3-dev; then
    echo "Python development headers not found. Installing..."
    sudo apt install python3-dev
fi

# Check if FFTW is installed
if ! dpkg -l | grep -q fftw; then
    echo "FFTW not found. Installing..."
    sudo apt install fftw3
fi

# Additional steps if required

echo "Additional dependencies installed."
