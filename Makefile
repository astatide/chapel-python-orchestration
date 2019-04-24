all:
	#chpl --warn-unstable --devel -o yggdrasil -M src/ main.chpl
	chpl --warn-unstable --devel -o yggdrasil -M src/ -M python/ main.chpl --ccflags "-w -I/usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -I/usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/usr/include -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/System/Library/Frameworks/Tk.framework/Versions/8.5/Headers -I /usr/local/lib/python3.7/site-packages/numpy/core/include" --ldflags "-L/usr/local/opt/python/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin -lpython3.7m -ldl -framework CoreFoundation" -g

test:
	chpl -o test -M src/ test.chpl

release:
	chpl --warn-unstable --fast -o yggdrasil -M src/ main.chpl

python:
	## see python3-config
	gcc -o gjallarbru python/gjallarbru.c -I/usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -I/usr/local/Cellar/python/3.7.2/Frameworks/Python.framework/Versions/3.7/include/python3.7m -Wno-unused-result -Wsign-compare -Wunreachable-code -fno-common -dynamic -DNDEBUG -g -fwrapv -O3 -Wall -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/usr/include -I/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk/System/Library/Frameworks/Tk.framework/Versions/8.5/Headers -I /usr/local/lib/python3.7/site-packages/numpy/core/include -L/usr/local/opt/python/Frameworks/Python.framework/Versions/3.7/lib/python3.7/config-3.7m-darwin -lpython3.7m -ldl -framework CoreFoundation
