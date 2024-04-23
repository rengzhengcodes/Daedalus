dependencies: timeloop/SConstruct env/bin/activate accelergy/setup.py

	source env/bin/activate && cd timeloop && scons -j8 --with-isl --static
	cd ..
	cd accelergy && python3 setup.py && pip3 install -e .