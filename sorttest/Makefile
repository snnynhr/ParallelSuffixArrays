CC=/opt/openmpi/bin/mpic++ 
CFLAGS=-Wall -Wextra -std=c++11

sorttest: main.cpp ssort.hpp
	$(CC) -o sorttest main.cpp ssort.hpp $(CFLAGS)

clean:
	rm sorttest

cleanlogs:
	rm result.e* result.o*
