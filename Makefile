all:
	chpl --warn-unstable -o yggdrasil -M src/ main.chpl

test:
	chpl -o test -M src/ test.chpl
