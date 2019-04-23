all:
	chpl --warn-unstable --devel -o yggdrasil -M src/ main.chpl

test:
	chpl -o test -M src/ test.chpl

release:
	chpl --warn-unstable --fast -o yggdrasil -M src/ main.chpl

python:
	#gcc -o ypy src/python.c -framework Python
	#gcc -o ypy src/python.c -I /usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -I /usr/local/lib/python3.7/site-packages/numpy/core/include 
	#gcc -o ypy.so src/python.c -I /usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -I /usr/local/lib/python3.7/site-packages/numpy/core/include -framework Python -g
	#gcc -o ypy src/python.c -I /usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -I /usr/local/lib/python3.7/site-packages/numpy/core/include/numpy -framework Python -g
	#gcc -o ypy.so src/python.c -I /usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -I /usr/local/lib/python3.7/site-packages/numpy/core/include -g
