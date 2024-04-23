dependencies: lib/timeloop/SConstruct env/bin/activate lib/accelergy/setup.py lib/barvinok/barvinok lib/isl
	LIB_PATH=$$PWD/lib
	export CPATH=$$CPATH:$(LIB_PATH)/barvinok
	export LIBRARY_PATH=$$LIBRARY_PATH:$(LIB_PATH)/barvinok
	cd lib/timeloop && scons -j8
	cd lib/accelergy/ && python3 setup.py && pip3 install -e .