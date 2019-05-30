/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */

#include "utility.hpp"

#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <stdio.h>
#include <sys/time.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  query for hipsparse version and git commit SHA-1. */
void query_version(char* version)
{
    hipsparseHandle_t handle;
    hipsparseCreate(&handle);

    int ver;
    hipsparseGetVersion(handle, &ver);

    char rev[128];
    hipsparseGetGitRevision(handle, rev);

    sprintf(version, "v%d.%d.%d-%s", ver / 100000, ver / 100 % 1000, ver % 100, rev);

    hipsparseDestroy(handle);
}

/* ============================================================================================ */
/*  device query and print out their ID and name; return number of compute-capable devices. */
int query_device_property()
{
    int               device_count;
    hipsparseStatus_t status = (hipsparseStatus_t)hipGetDeviceCount(&device_count);
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        printf("Query device error: cannot get device count\n");
        return -1;
    }
    else
    {
        printf("Query device success: there are %d devices\n", device_count);
    }

    for(int i = 0; i < device_count; i++)
    {
        hipDeviceProp_t   props;
        hipsparseStatus_t status = (hipsparseStatus_t)hipGetDeviceProperties(&props, i);
        if(status != HIPSPARSE_STATUS_SUCCESS)
        {
            printf("Query device error: cannot get device ID %d's property\n", i);
        }
        else
        {
            printf("Device ID %d : %s\n", i, props.name);
            printf("-------------------------------------------------------------------------\n");
            printf("with %ldMB memory, clock rate %dMHz @ computing capability %d.%d \n",
                   props.totalGlobalMem >> 20,
                   (int)(props.clockRate / 1000),
                   props.major,
                   props.minor);
            printf("maxGridDimX %d, sharedMemPerBlock %ldKB, maxThreadsPerBlock %d, warpSize %d\n",
                   props.maxGridSize[0],
                   props.sharedMemPerBlock >> 10,
                   props.maxThreadsPerBlock,
                   props.warpSize);

            printf("-------------------------------------------------------------------------\n");
        }
    }

    return device_count;
}

/*  set current device to device_id */
void set_device(int device_id)
{
    hipsparseStatus_t status = (hipsparseStatus_t)hipSetDevice(device_id);
    if(status != HIPSPARSE_STATUS_SUCCESS)
    {
        printf("Set device error: cannot set device ID %d, there may not be such device ID\n",
               (int)device_id);
    }
}
/* ============================================================================================ */
/*  timing:*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void)
{
    hipDeviceSynchronize();
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream)
{
    hipStreamSynchronize(stream);
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 * 1000) + tv.tv_usec;
};

#ifdef __cplusplus
}
#endif
