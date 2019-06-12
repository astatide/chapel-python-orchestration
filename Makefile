NUMPY:=`python3 -c 'import numpy.distutils.misc_util as m; print(m.get_numpy_include_dirs()[0])'`
PYTHONC:=`python3-config --cflags`
PYTHONL:=`python3-config --ldflags`
LINCLUDE:=--warn-unstable --fast -M src -M python
MACLUDE:=-L /usr/local/lib -I /usr/local/include
ENVSTATE:=env CHPL_COMM_SUBSTRATE=udp CHPL_COMM=gasnet

all:
	make valkyrie
	make yggdrasil

clean:
	rm valkyrie
	rm yggdrasil

valkyrie:
	chpl -o valkyrie valkyrie.chpl $(LINCLUDE) --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" --comm none --launcher none
yggdrasil:
	chpl -o yggdrasil main.chpl $(LINCLUDE)    --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" --comm gasnet

spawn:
	env CHPL_DEVELOPER=true chpl -o spawn -L /usr/local/lib -I /usr/local/include -M src/ -M python/ sp.chpl --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks
