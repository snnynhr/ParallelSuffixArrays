build:
	/opt/openmpi/bin/mpic++ -c io/fileio.cpp -lm -Wall -std=c++11
#	/opt/openmpi/bin/mpic++ -c sort/ssort.cpp -lm -Wall -std=c++11
	/opt/openmpi/bin/mpic++ -c suffix_array/suffix_array.cpp -lm -Wall -std=c++11
	/opt/openmpi/bin/mpic++ -o suffixArray main.cpp fileio.o suffix_array.o -lm -Wall -std=c++11

clean:
	rm *.o; rm -f suffixArray
