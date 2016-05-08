#include <fstream>
#include <string.h>
#include <streambuf>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <stdlib.h>


//#include <ofstream>



int main(int argc, const char* argv[]) {
   
   int N = atoi(argv[1]);

   std::ofstream o(argv[2]);
   
   for(int i = 0; i < N; i++ ) {
     int c = rand() % 52;
     if( c < 26) c+= 65;
     else c+=71;
     char cc = c;
     
     o.write(&cc, 1);
   }
   o.close();
   return 0;
}
