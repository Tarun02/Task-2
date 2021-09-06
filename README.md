# Task-2

# Containerization of the training and inference

The repo consists of two folders:
<ol>
<li>Training Container</li>
<li>Inference Container</li>
</ol>

# Training container

You can build the container from the Dockerfile by executing the following command:

> docker build -t \<name-of-the-container\> .

You have to be in the directory of the Dockerfile to build the container.

Then you can train the model by running the container with the following command:

> docker run \<name-of-the-container>\

Now to get the .pkl file from the container you have to run the following command:

> docker cp \`docker ps -alq\` /output .

The **docker ps -alq** will give you the container id of the latest run.

# Inference container

The process is same as of building the training container but one main difference is that the input to prediction should be in the file named **prediction_input.json**.

This file will be copied to the container to get the predictions. 

Once the container has been build and run we can get the output of the predictions by running the same command:

> docker cp \`docker ps -alq\` /predictions .


# Different models

To build different model you can change the training script in the container but please make sure that you cannot change the name of the files.

