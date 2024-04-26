# For Timeloop installation
BARVINOK_VER ?= 0.41.7
NTL_VER      ?= 11.5.1

export PATH := $(PATH):/usr/local/lib
export CPATH := $(CPATH):/usr/local/lib
export LIBRARY_PATH := $(LIBRARY_PATH):/usr/local/lib

## @note Very explicitly ripped from accelergy-timeloop-infrastructure. Thanks Tanner!
# https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure/blob/master/Makefile
install_timeloop:
	mkdir -p /tmp/build-timeloop
	echo $$PATH
	echo $$CPATH

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
		&& scons -j4 --with-isl --static --accelergy


install_accelergy:
# Goes to the right venv
	source .venv/bin/activate

	python3 -m pip install setuptools wheel libconf numpy joblib
	cd lib && pip3 install ./accelergy*