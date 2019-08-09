NUMPY:=`python3 -c 'import numpy.distutils.misc_util as m; print(m.get_numpy_include_dirs()[0])'`
PYTHONC:=`python3-config --cflags`
PYTHONL:=`python3-config --ldflags` 
#PYTHONC:=`python3-config --includes`
#PYTHONL:=`python3-config --includes` 
LINCLUDE:=--warn-unstable -M ../src -M ../python
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
	#chpl -o vtest valkyrieTest.chpl -M ../src/ -M ../python/
	chpl -o vtest valkyrieTest.chpl $(LINCLUDE) $(MACLUDE) --ccflags "-O2 -w -lpthread -I $(NUMPY) $(PYTHONC)" --ldflags "-lpthread -v $(PYTHONL)" $(COMM) $(DEBUG)
