all:
	chpl --warn-unstable --fast -o yggdrasil -M src/ main.chpl

test:
	chpl -o test -M src/ test.chpl
