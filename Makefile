CUR_PATH = $(shell pwd)
CPATH := $(CPATH):$(CUR_PATH)/lib/barvinok/barvinok
LIBRARY_PATH := $(LIBRARY_PATH):$(CUR_PATH)/lib/barvinok/barvinok
dependencies: lib/timeloop/SConstruct env/bin/activate lib/accelergy/setup.py
	export CPATH=$(CPATH)
	export LIBRARY_PATH=$(LIBRARY_PATH)
	echo $CPATH $LIBRARY_PATH
	. env/bin/activate
	cd lib/timeloop/ && scons -j8 --with-isl --static
	cd lib/accelergy/ && python3 setup.py && pip3 install -e .