NUMPY:=`python3 -c 'import numpy.distutils.misc_util as m; print(m.get_numpy_include_dirs()[0])'`
PYTHONC:=`python3-config --cflags`
PYTHONL:=`python3-config --ldflags`

all:
	make valkyrie
	make yggdrasil

clean:
	rm valkyrie
	rm yggdrasil

valkyrie:
	chpl -I ZMQHelper/ -L /usr/local/lib -I /usr/local/include --warn-unstable --devel -o valkyrie -M src -M python valkyrie.chpl --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks

yggdrasil:
	chpl --warn-unstable --devel -o yggdrasil -M src/ -M python/ main.chpl --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks

spawn:
	env CHPL_DEVELOPER=true chpl -o spawn -L /usr/local/lib -I /usr/local/include -M src/ -M python/ sp.chpl --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks

test:
	chpl -o test -M src/ test.chpl

release:
	chpl --warn-unstable --fast -o yggdrasil -M src/ main.chpl

python:
	## see python3-config
	gcc -o gjallarbru python/gjallarbru.c -I/usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -I/usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/usr/include -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/System/Library/Frameworks/Tk.framework/Versions/8.5/Headers -I /usr/local/lib/python3.7/site-packages/numpy/core/include -L/usr/local/opt/python/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin -lpython3.7m -ldl -framework CoreFoundation

valgrind:
	chpl --warn-unstable --devel -o yggdrasil -M src/ -M python/ main.chpl --ccflags "-fsanitize=address -fno-omit-frame-pointer -w -I $(NUMPY) $(PYTHONC) " --ldflags "-fsanitize=address $(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks
	#chpl --warn-unstable --devel -o yggdrasil -M src/ -M python/ main.chpl `python3-config --cflags` `python3 -c 'import numpy.distutils.misc_util as m; print(m.get_numpy_include_dirs())'`  `python3-config --ldflags` -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks

docker:
	#chpl --warn-unstable --devel -o yggdrasil -M src/ -M python/ main.chpl --ccflags “-w -I /usr/local/lib/python3.5/dist-packages/numpy/core/include -I/usr/include/python3.5m -I/usr/include/python3.5m” --ldflags “-L/usr/lib/python3.5/config-3.5m-x86_64-linux-gnu -L/usr/lib -lpython3.5m -ldl  -lutil -lm”
	chpl --warn-unstable --devel -o yggdrasil -M src/ -M python/ main.chpl --ccflags "-w -I $(NUMPY) $(PYTHONC) " --ldflags "$(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks
