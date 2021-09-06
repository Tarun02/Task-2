FROM amazonlinux

ARG PYTHON=python3.9
ARG PYTHON_PIP=python3-pip
ARG PIP=pip3
ARG PYTHON_VERSION=3.9.6

RUN yum update -y

# This block of code is for installing python on the container
# =====
RUN yum install -y gcc openssl-devel gzip libffi-devel wget tar make bzip2-devel xz-devel
RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
RUN tar xvf Python-$PYTHON_VERSION.tgz \
 && cd Python-$PYTHON_VERSION \
 && ./configure && make && make install \
 && make && make install && rm -rf ../Python-$PYTHON_VERSION*

RUN ${PIP} --no-cache-dir install --upgrade \
    pip \
    setuptools
# ======

# Creating a directory for storing the model artifacts
RUN mkdir artifacts

# Copying the model artifacts from the host to the container
COPY artifacts artifacts

# Copying the code folder to the container
COPY code source_code

# Install the required packages based on the requirements.txt file
RUN pip install -r /source_code/requirements.txt

# Creating a directory to store the prediction results
RUN mkdir predictions

# Sending the prediction input to the container
COPY prediction_input.json input.json

ENTRYPOINT ["python3", "/source_code/inference_script.py", "/input.json"]