#!/usr/bin/env bash
pip show numpy
pip install numpy==1.16.1
!pip show numpy

git clone https://github.com/shaoanlu/faceswap-GAN.git
cd faceswap-GAN

pip install moviepy
pip install keras_vggface

pip install imageio-ffmpeg