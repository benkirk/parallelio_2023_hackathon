#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <hdf5.h>
#include <H5Zpublic.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include "trivial_filter.h"


bool tf_is_device_pointer (const void *ptr)
{
    struct cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, ptr);
    return (attributes.devicePointer != NULL);
}


// ref: https://docs.hdfgroup.org/hdf5/develop/_f_i_l_t_e_r.html
size_t trivial_filter( unsigned int flags,
                       size_t cd_nelmts,
                       const unsigned int cd_values[],
                       size_t nbytes,
                       size_t *buf_size, void **buf)
{
  const bool input = flags & H5Z_FLAG_REVERSE;

  fprintf(stderr,
          "trivial_filter called with input=%d, is_device_pointer=%d, buf_size=%d, nbytes=%d, cd_nelmts=%d\n",
          input,
          tf_is_device_pointer(*buf),
          *buf_size,
          nbytes,
          cd_nelmts);

  /* Input */
  if (input)
    {
      return nbytes;
    }

  /* Output */
  else
    {
      return nbytes;
    }

  /*fail*/
  return 0;
}



void register_trivial_filter()
{
  // ref: https://docs.hdfgroup.org/archive/support/HDF5/doc/RM/RM_H5Z.html

  /*
   typedef struct H5Z_class2_t {
            int version;
            H5Z_filter_t id;
            unsigned encoder_present;
            unsigned decoder_present;
            const char  *name;
            H5Z_can_apply_func_t can_apply;
            H5Z_set_local_func_t set_local;
            H5Z_func_t filter;
        } H5Z_class2_t;
  */

  /* registrer the filter */
  H5Z_class_t filter_spec;
  filter_spec.version = H5Z_CLASS_T_VERS;

  filter_spec.id = TRIVIAL_FILTER_IDX;
  filter_spec.name = "trivial filter";
  filter_spec.can_apply = NULL;
  filter_spec.set_local = NULL;
  filter_spec.filter = trivial_filter;


  herr_t status = H5Zregister(&filter_spec);
}



#ifdef COMPILE_MAIN
int main (int argc, char** argv)
{
  register_trivial_filter();

  return 0;
}
#endif
