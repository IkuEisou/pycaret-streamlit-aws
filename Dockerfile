FROM python:3.7

USER root
ENV DEBIAN_FRONTEND noninteractive

# Install dependencies
RUN apt-get update
RUN apt-get install -y libopenblas-dev wget curl musl-dev python3-dev 
#python3-matplotlib apt-utils lsb-release

#install cuda for catboost
#RUN echo "deb http://ftp.jp.debian.org/debian buster main non-free" >> /etc/apt/sources.list
#RUN apt-file update
#RUN apt-file search  nvidia-cuda-toolkit
#RUN apt-get remove -y libgcc-7-dev
#RUN apt install -y nvidia-cuda-toolkit
#RUN ln -s /usr/lib/nvidia-cuda-toolkit/ /usr/local/cuda
#ENV LD_LIBRARY_PATH="/usr/lib/nvidia-cuda-toolkit/libdevice:$LD_LIBRARY_PATH"
#ENV PATH="/usr/lib/nvidia-cuda-toolkit/bin:$PATH"

#Install llvm
#RUN uname -a
#RUN lsb_release -a
RUN echo "deb http://ftp.de.debian.org/debian sid main " >> /etc/apt/sources.list
RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
RUN apt-get install -y apt-file 
RUN apt-file update
#RUN apt-file search  apt-add-repository
RUN apt-file search  llvm-9
RUN apt-get remove -y libgcc-8-dev
RUN apt-get install -y llvm-9 llvm-9-dev
RUN ls -l /usr/bin/llvm-config-9
ENV LLVM_CONFIG /usr/bin/llvm-config-9
RUN apt-get install -y g++ gcc gfortran cmake zlib1g-dev libpng-dev libjpeg-dev libxml2-dev libxslt-dev 

# Import the GPG key
RUN apt-get install -y -V ca-certificates 
RUN wget https://dist.apache.org/repos/dist/dev/arrow/KEYS 
RUN apt-key add < KEYS

# Install libarrow
RUN wget https://apache.bintray.com/arrow/debian/apache-arrow-archive-keyring-latest-buster.deb
RUN apt-get install -y ./apache-arrow-archive-keyring-latest-buster.deb
RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6 git
#RUN wget -P /opt https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-ppc64le.sh && \
#     bash /opt/Anaconda3-2020.11-Linux-ppc64le.sh -b -p /opt/anaconda3 && \
#     rm /opt/Anaconda3-2020.11-Linux-ppc64le.sh && \
#     echo "export PATH=/opt/anaconda3/bin:$PATH" >> ~/.bashrc && \
#     . ~/.bashrc && \
#     conda init
#RUN /opt/anaconda3/bin/conda install arrow-cpp -c conda-forge
RUN apt-get install -y libboost-all-dev libncurses5-dev  libjemalloc-dev libboost-dev libboost-filesystem-dev libboost-system-dev libboost-regex-dev 
RUN apt-get install -y libtool flex bison 

RUN pip install --upgrade pip
#RUN pip install virtualenv
#ENV VIRTUAL_ENV=/venv
#RUN virtualenv venv -p python3
#ENV PATH="VIRTUAL_ENV/bin:$PATH"

#install arrow
RUN git clone https://github.com/apache/arrow.git
WORKDIR /arrow/cpp/
RUN mkdir release
RUN cd release
ENV ARROW_HOME /usr/local
ENV D_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH
RUN pip install numpy --upgrade
RUN pip install six pandas cython pytest psutil 
RUN cmake -DCMAKE_INSTALL_PREFIX=$ARROW_HOME \
    -DCMAKE_INSTALL_LIBDIR=lib \
    -DARROW_WITH_BZ2=ON \
    -DARROW_WITH_ZLIB=ON \
    -DARROW_WITH_ZSTD=ON \
    -DARROW_WITH_LZ4=ON \
    -DARROW_WITH_SNAPPY=ON \
    -DARROW_PARQUET=ON \
    -DARROW_PYTHON=ON \
    -DARROW_BUILD_TESTS=OFF \
    .
RUN make -j4
RUN make install

#Build and install pyarrow
WORKDIR  /arrow/python
RUN python3 setup.py build_ext --build-type=release --with-parquet
RUN python3 setup.py install 

#install latest catboost
#RUN apt-get install -y  libtinfo-dev libncurses5 libc6-dev 
#WORKDIR / 
#RUN git clone https://github.com/catboost/catboost.git
#Build wheel
#WORKDIR /catboost/catboost/python-package/
#RUN python3 mk_wheel.py -DCUDA_ROOT="/usr/lib/nvidia-cuda-toolkit/bin"
#RUN ls 
#RUN pip -V
#RUN pip3.7 install ./catboost-0.24.4-cp37-none-manylinux1_ppc64le.whl
#WORKDIR /catboost/catboost/python-package/
#ENV PYTHONPATH=$PYTHONPATH:$(pwd)
#Build binary
#WORKDIR /catboost/catboost/python-package/catboost
#RUN ../../../ya make -r --target-platform=CLANG7-LINUX-PPC64LE --target-platform-flag=ALLOCATOR=SYSTEM -DLDFLAGS="-Wl,--no-toc-optimize" -DUSE_ARCADIA_PYTHON=no -DOS_SDK=local -DPYTHON_CONFIG=python3-config 

#fix python3-matplotlib runtime error
RUN pip install six cython pytest psutil
RUN pip install numpy --upgrade
RUN apt-get install -y pkg-config libssl-dev automake build-essential autoconf rapidjson-dev liblz4-dev libzstd-dev libre2-dev libfreetype6-dev freetype2-demos libfreetype6
RUN ln -s /usr/include/freetype2/ft2build.h /usr/include/
RUN ls -l /usr/include/
RUN ls /usr/include/freetype2/ 

#install rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
RUN ls /root/.cargo/bin/
ENV PATH="/root/.cargo/bin:$PATH"
RUN echo $PATH


# copying all files over
WORKDIR /app
ADD . /app
#COPY . /app

#install pycaret latest
WORKDIR /app/pycaret
RUN ls
RUN cat requirements.txt
RUN pip install .

WORKDIR /app
RUN pip install numpy --upgrade
RUN pip install -r requirements.txt
RUN pip show numpy
RUN pip show matplotlib
RUN pip show pycaret
# Expose port 
ENV PORT 8501

# cmd to launch app when container is run (automl_api)
CMD streamlit run app.py 

# streamlit-specific commands for config
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
RUN mkdir -p /root/.streamlit
RUN bash -c 'echo -e "\
[general]\n\
email = \"\"\n\
" > /root/.streamlit/credentials.toml'

RUN bash -c 'echo -e "\
[server]\n\
enableCORS = false\n\
" > /root/.streamlit/config.toml'

