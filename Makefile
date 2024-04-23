# For Timeloop installation
BARVINOK_VER ?= 0.41.6
NTL_VER      ?= 11.5.1

## @note Very explicitly ripped from accelergy-timeloop-infrastructure. Thanks Tanner!
# https://github.com/Accelergy-Project/accelergy-timeloop-infrastructure/blob/master/Makefile
install_timeloop:
	export PATH="$$PATH:/usr/local/lib:/tmp/build-timeloop"
	mkdir -p /tmp/build-timeloop
	echo "PATH: $$PATH"
	echo "CPATH: $$CPATH"
	echo "LIBRARY_PATH: $$LIBRARY_PATH"

	cd /tmp/build-timeloop \
		&& wget https://libntl.org/ntl-${NTL_VER}.tar.gz \
		&& tar -xvzf ntl-${NTL_VER}.tar.gz \
		&& cd ntl-${NTL_VER}/src \
		&& ./configure NTL_GMP_LIP=on SHARED=on \
		&& make \
		&& sudo make install

	cd /tmp/build-timeloop \
	    && wget https://barvinok.sourceforge.io/barvinok-${BARVINOK_VER}.tar.gz \
		&& tar -xvzf barvinok-${BARVINOK_VER}.tar.gz \
		&& cd barvinok-${BARVINOK_VER} \
		&& ./configure --enable-shared-barvinok \
		&& make \
		&& sudo make install

	cd src/timeloop \
		&& cp -r pat-public/src/pat src/pat  \
		&& scons -j4 --with-isl --static --accelergy
	cp src/timeloop/build/timeloop-mapper  ~/.local/bin/timeloop-mapper
	cp src/timeloop/build/timeloop-metrics ~/.local/bin/timeloop-metrics
	cp src/timeloop/build/timeloop-model ~/.local/bin/timeloop-model


install_accelergy:
	# Goes to the right venv
	source env/bin/activate

	python3 -m pip install setuptools wheel libconf numpy joblib
	cd lib && pip3 install ./accelergy*