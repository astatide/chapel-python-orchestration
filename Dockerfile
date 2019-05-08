# This is a dockerfile for debugging; I need valgrind, but mac doesn't have it.

FROM chapel/chapel

# Update it.
RUN apt-get update
RUN apt-get install -y valgrind
RUN apt-get install -y python3-dev python3
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-scipy
RUN apt-get install -y python3-pip
RUN pip3 install tensorflow keras
ENV CHPL_TASKS=fifo CHPL_MEM=cstdlib CHPL_COMM=none
# build yggdrasil, ya big bitch.

make
