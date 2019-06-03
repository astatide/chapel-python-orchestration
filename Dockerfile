# This is a dockerfile for debugging; I need valgrind, but mac doesn't have it.

FROM debian:9.5

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    curl \
    gcc \
    g++ \
    perl \
    python \
    python-dev \
    python-setuptools \
    libgmp10 \
    libgmp-dev \
    locales \
    bash \
    make \
    mawk \
    file \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Update it.
RUN apt-get update
#RUN apt-get install -y valgrind
RUN apt-get install -y python3-dev python3
RUN apt-get install -y python3-numpy
RUN apt-get install -y python3-scipy
RUN apt-get install -y python3-pip
RUN pip3 install tensorflow keras
#ENV CHPL_TASKS=fifo CHPL_MEM=cstdlib CHPL_COMM=none
ENV CHPL_TASKS=fifo CHPL_MEM=cstdlib CHPL_COMM=ugni
# build yggdrasil, ya big bitch.

RUN apt-get install -y libzmq3-dev

ENV CHPL_VERSION 1.20.0
ENV CHPL_HOME    /opt/chapel/$CHPL_VERSION
ENV CHPL_GMP     system

RUN mkdir -p /opt/chapel \
    #&& wget -q -O - https://github.com/chapel-lang/chapel/releases/download/$CHPL_VERSION/chapel-$CHPL_VERSION.tar.gz | tar -xzC /opt/chapel --transform 's/chapel-//' \
    # We need to work from master
    && git clone https://github.com/chapel-lang/chapel.git /opt/chapel/$CHPL_VERSION \
    && make -C $CHPL_HOME \
    && make -C $CHPL_HOME chpldoc test-venv mason \
    && make -C $CHPL_HOME cleanall

# Copy in the modifications to ZMQ.
COPY ZMQmod.patch /opt/chapel/$CHPL_VERSION/modules/packages
#RUN wget https://gist.githubusercontent.com/ajoshpratt/4d2b41fd570747cd2d5df26189ece60b/raw/d69145ec7da09da699bb4e912499356e573f8fd4/ZMQmod.patch \
RUN cd /opt/chapel/$CHPL_VERSION/modules/packages \
    && git apply ZMQmod.patch

# Configure locale
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    echo 'LANG="en_US.UTF-8"'>/etc/default/locale && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=en_US.UTF-8

# Configure dummy git user
RUN git config --global user.email "noreply@example.com" && \
    git config --global user.name  "Chapel user"

ENV PATH $PATH:$CHPL_HOME/bin/linux64:$CHPL_HOME/util
