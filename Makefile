all:
	chpl --warn-unstable --devel -o yggdrasil -M src/ main.chpl

test:
	chpl -o test -M src/ test.chpl

release:
	chpl --warn-unstable --fast -o yggdrasil -M src/ main.chpl
