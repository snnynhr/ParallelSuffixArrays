build:
	/opt/openmpi/bin/mpic++ -c fileio.cpp -lm -Wall
	/opt/openmpi/bin/mpic++ -o suffixArray main.cpp fileio.o -lm -Wall

clean:
	rm *.o; rm -f suffixArray
