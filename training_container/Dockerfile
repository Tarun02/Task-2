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

# Creating the directory to store the code
RUN mkdir source_code

# Copying the code from host to the container
COPY code source_code

# Install the required packages based on the requirements.txt file
RUN pip install -r /source_code/requirements.txt

# Creating a directory to store the data
RUN mkdir data

# Copying the data from host to the container
COPY data data

# Creating an output directory to store the model.pkl and features.pkl file
RUN mkdir output

ENTRYPOINT ["python3", "/source_code/training_script.py"]