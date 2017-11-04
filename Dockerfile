FROM andrewosh/binder-base

# Approach based on that demonstrated in binder-project/example-dockerfile-two

MAINTAINER Brendt Wohlberg <brendt@ieee.org>

USER root

# Add library required by pyfftw
RUN apt-get update
RUN apt-get install -y libfftw3-dev

USER main

# Install requirements
#RUN /home/main/anaconda/envs/python3/bin/pip install -r requirements.txt

# Install sporco
#RUN /home/main/anaconda/envs/python3/bin/python setup.py install
