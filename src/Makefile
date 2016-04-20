
build:
	/opt/openmpi/bin/mpic++ -c fileio.cpp -lm
	/opt/openmpi/bin/mpic++ -o suffixArray main.cpp fileio.o -lm

clean:
	rm *.o; rm -f suffixArray
