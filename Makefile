NUMPY:=`python3 -c 'import numpy.distutils.misc_util as m; print(m.get_numpy_include_dirs()[0])'`
PYTHONC:=`python3-config --cflags`
PYTHONL:=`python3-config --ldflags` 
#PYTHONC:=`python3-config --includes`
#PYTHONL:=`python3-config --includes` 
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
	DEBUG:=-g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks #--devel
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
	chpl -o valkyrie valkyrieBinary.chpl $(LINCLUDE) $(MACLUDE) --ccflags "-O2 -w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" --comm none --launcher none $(DEBUG)

pythonlib:
	#chpl src/valkyrie.chpl -M src -L ZMQHelper/ --ccflags "-O2 -w -lpthread -I" --ldflags "-lpthread -v" --comm none --launcher none $(DEBUG) --library --library-python
	#export CFLAGS="$(PYTHONC) $(PYTHONL)"
	#chpl src/valkyrie.chpl -M src -L ZMQHelper/ --library-python  $(DEBUG) --comm none --launcher none $(MACLUDE) --ccflags "-O2 -w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" --library-python-name=mist --library
	#chpl src/valkyrie.chpl -M src -L ZMQHelper/ --library-python  --library-python-name=mist --library $(DEBUG) --comm none --launcher none $(MACLUDE) --ccflags "-O2 -w -lpthread -I $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)"
	chpl src/valkyrie.chpl -M src -L ZMQHelper/ --library-python  --library-python-name=mist --library $(DEBUG) --comm none --launcher none $(MACLUDE)

mist:
	chpl src/mist.chpl -M src -L ZMQHelper/ $(DEBUG) --comm none --launcher none $(MACLUDE) --ccflags "-fsanitize=address  -O2 -w -lpthread -I $(PYTHONC)" --ldflags "-fsanitize=address  -lpthread -v $(PYTHONL)" -o mist

yggdrasil:
	chpl -o yggdrasil main.chpl $(LINCLUDE)    $(MACLUDE) --ccflags "-O2 -w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" $(COMM) $(DEBUG)

spawn:
	env CHPL_DEVELOPER=true chpl -o spawn -L /usr/local/lib -I /usr/local/include -M src/ -M python/ sp.chpl --ccflags "-w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" -g --codegen --cpp-lines --savec /Users/apratt/work/yggdrasil/C --bounds-checks --stack-checks --nil-checks

gj:
	chpl -o gj gj.chpl $(LINCLUDE) $(MACLUDE) --ccflags "-O2 -w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" --comm none --launcher none $(DEBUG)
