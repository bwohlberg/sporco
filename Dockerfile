FROM andrewosh/binder-base

# Approach based on that demonstrated in binder-project/example-dockerfile-two

MAINTAINER Brendt Wohlberg <brendt@ieee.org>

USER root

# Add library required by pyfftw
RUN apt-get update
RUN apt-get install -y libfftw3-dev

USER main

# Install requirements
RUN /home/main/anaconda/envs/python3/bin/pip install future numpy scipy pillow matplotlib pyfftw

# The notebooks should be run from the current version of sporco in the
# GitHub repo, but since the correct configuration for mybinder.org is
# not documented, for now we just install the latest release using pip
RUN /home/main/anaconda/envs/python3/bin/pip install sporco
