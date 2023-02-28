#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <hdf5.h>
#include <H5Zpublic.h>

#include <cuda.h>

// prototype
void execute_example(char* input_data, const size_t in_bytes);


int main (int argc, char **argv)
{
  printf("Hello, world!\n");

  // Initialize a random array of chars
  const size_t in_bytes = 1000000;
  const size_t chunksize =   1000;

  char* uncompressed_data;

  cudaMallocHost((void**) &uncompressed_data, in_bytes);

  char randval = (char) rand() % CHAR_MAX;
  for (size_t ix = 0; ix < in_bytes; ++ix) {
    if (ix % chunksize == 0)
      randval = (char) rand() % CHAR_MAX;

    uncompressed_data[ix] = randval;
    printf("uncompressed_data[%d]=%d\n", ix, uncompressed_data[ix]);
  }

  execute_example(uncompressed_data, in_bytes);

  return 0;
}
