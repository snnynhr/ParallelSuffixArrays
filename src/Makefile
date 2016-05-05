build:
	/usr/lib64/openmpi/bin/mpic++ -c io/fileio.cpp -lm -Wall -std=c++11 -Wno-literal-suffix
#	/opt/openmpi/bin/mpic++ -c sort/ssort.cpp -lm -Wall -std=c++11
	/usr/lib64/openmpi/bin/mpic++ -c suffix_array/suffix_array.cpp -lm -Wall -std=c++11 -Wno-literal-suffix
	/usr/lib64/openmpi/bin/mpic++ -o suffixArray main.cpp fileio.o suffix_array.o -lm -Wall -std=c++11 -Wno-literal-suffix

clean:
	rm *.o; rm -f suffixArray
