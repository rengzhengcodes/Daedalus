# For Timeloop installation
BARVINOK_VER ?= 0.41.7
NTL_VER      ?= 11.5.1

export PATH := $(PATH):/usr/local/lib
export CPATH := $(CPATH):/usr/local/lib
export LIBRARY_PATH := $(LIBRARY_PATH):/usr/local/lib

export TIMELOOP_INCLUDE_PATH := $(shell pwd)/lib/timeloop/include
export TIMELOOP_LIB_PATH := $(shell pwd)/lib/timeloop/lib

## @note Very explicitly ripped from accelergy-timeloop-infrastructure. Thanks Tanner!
# https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure/blob/master/Makefile
install_timeloop:
	mkdir -p /tmp/build-timeloop
	
	apt-get update \
		&& DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata \
		&& apt-get install -y --no-install-recommends \
						locales \
						curl \
						git \
						wget \
						python3-dev \
						python3-pip \
						scons \
						make \
						autotools-dev \
						autoconf \
						automake \
						libtool \
		&& apt-get install -y --no-install-recommends \
						g++ \
						cmake

	apt-get update \
		&& apt-get install -y --no-install-recommends \
						g++ \
						libconfig++-dev \
						libboost-dev \
						libboost-iostreams-dev \
						libboost-serialization-dev \
						libboost-all-dev \
						libyaml-cpp-dev \
						libncurses5-dev \
						libtinfo-dev \
						libgpm-dev \
						libgmp-dev

	cd /tmp/build-timeloop \
		&& wget https://libntl.org/ntl-${NTL_VER}.tar.gz \
		&& tar -xvzf ntl-${NTL_VER}.tar.gz \
		&& cd ntl-${NTL_VER}/src \
		&& ./configure NTL_GMP_LIP=on SHARED=on \
		&& make \
		&& make install

	cd /tmp/build-timeloop \
	    && wget https://barvinok.sourceforge.io/barvinok-${BARVINOK_VER}.tar.gz \
		&& tar -xvzf barvinok-${BARVINOK_VER}.tar.gz \
		&& cd barvinok-${BARVINOK_VER} \
		&& ./configure  --enable-shared-barvinok --with-gmp-prefix=/usr/local/lib --with-ntl-prefix=/usr/local/lib \
		&& make CXXFLAG='-std=ansi'\
		&& make install

	cd lib/timeloop \
		&& cp -r pat-public/src/pat src/pat  \
		&& scons -j4 --with-isl --static --accelergy \


install_pytimeloop:
	echo $(TIMELOOP_INCLUDE_PATH)
	echo $(TIMELOOP_LIB_PATH)
	cd lib/timeloop-python \
		&& pip3 install -e . \
		&& rm -rf build

install_accelergy:
	python3 -m pip install setuptools wheel libconf numpy joblib
	cd lib/accelergy-cacti-plug-in && make
	cd lib/accelergy-neurosim-plug-in && make
	cd lib && pip3 install ./accelergy*