build:
	/usr/lib64/openmpi/bin/mpic++ -o suffixArray main.cpp suffix_array.cpp ../sais/sais.c -lm -Wall -std=c++11 -Wno-literal-suffix -Wextra -D_GLIBCXX_PARALLEL -fopenmp

clean:
	rm *.o; rm -f suffixArray
