#pragma
#ifdef OVX
#include <OVX/UtilityOVX.hpp>
#else 
#include <assert.h>
#define NVXIO_SAFE_CALL VX_API_CALL
#define NVXIO_ASSERT assert
#define NVXIO_CHECK_REFERENCE 
#endif