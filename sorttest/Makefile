CC=/opt/openmpi/bin/mpic++ 
CFLAGS=-Wall -Wextra

sorttest: main.cpp ssort.hpp
	$(CC) -o sorttest main.cpp ssort.hpp $(CFLAGS)

clean:
	rm sorttest

cleanlogs:
	rm result.e* result.o*
