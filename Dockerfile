FROM andrewosh/binder-base

# Approach based on that demonstrated in binder-project/example-dockerfile-two

MAINTAINER Brendt Wohlberg <brendt@ieee.org>

USER root

# Add wget library required by pyfftw
RUN apt-get update
RUN apt-get install -y wget libfftw3-dev

USER main

# Get required data file
RUN wget -P sporco/data/kodak http://r0k.us/graphics/kodak/kodak/kodim23.png
RUN wget -P sporco/data/standard http://homepages.cae.wisc.edu/~ece533/images/monarch.png

# Install requirements
RUN /home/main/anaconda/envs/python3/bin/pip install -r requirements.txt

# Install sporco
RUN /home/main/anaconda/envs/python3/bin/python setup.py install
