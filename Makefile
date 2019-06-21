NUMPY:=`python3 -c 'import numpy.distutils.misc_util as m; print(m.get_numpy_include_dirs()[0])'`
PYTHONC:=`python3-config --cflags`
PYTHONL:=`python3-config --ldflags` 
LINCLUDE:=--warn-unstable -M src -M python
ENVSTATE:=env CHPL_COMM_SUBSTRATE=udp CHPL_COMM=gasnet
HOST=$(shell hostname)

ifeq ($(HOST), cicero)
 	COMM:=--comm ugni --launcher slurm-srun
	MACLUDE:=
	DEBUG:=--fast
else
	COMM:=--comm gasnet
	#COMM:=--comm none --launcher none
	MACLUDE:= -L ZMQHelper/ -L /usr/local/lib -I /usr/local/include
	DEBUG:=-g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks --devel
endif

all:
	@echo $(HOST)
	@echo $(COMM)
	make valkyrie
	make yggdrasil

clean:
	rm valkyrie
	rm yggdrasil

valkyrie:
	chpl -o valkyrie valkyrie.chpl $(LINCLUDE) $(MACLUDE) --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" --comm none --launcher none $(DEBUG)

yggdrasil:
	chpl -o yggdrasil main.chpl $(LINCLUDE)    $(MACLUDE) --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" $(COMM) $(DEBUG)

spawn:
	env CHPL_DEVELOPER=true chpl -o spawn -L /usr/local/lib -I /usr/local/include -M src/ -M python/ sp.chpl --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks
