CC=/opt/openmpi/bin/mpic++ 
CFLAGS=-Wall -Wextra

sorttest: main.cpp sort.hpp
	$(CC) -o sorttest main.cpp sort.hpp $(CFLAGS)

clean:
	rm sorttest

cleanlogs:
	rm result.e* result.o*
