FROM ubuntu:20.04

# Install General Requirements
RUN apt-get update && \
        apt-get install -y --no-install-recommends \
        python3.9 \
        python3-pip \
        python3-pandas \
        python3-sklearn \
        software-properties-common

# Create a /work directory within the container, copy everything from the
# build directory and switch there.
RUN mkdir /work
COPY . /work
WORKDIR /work

RUN pip install --user -e .

# Make sure your build script is executable. Generally, always check that 
# all your scripts are made executable, forgetting this is a common error. 
# Another frequent error is having Windows style line endings in .sh files. 
RUN chmod +x build.sh
RUN ./build.sh

# Again, test and train scripts should be executable within the container.
RUN chmod +x test.sh
RUN chmod +x train.sh

# Don't add any CMD or ENTRYPOINT!
