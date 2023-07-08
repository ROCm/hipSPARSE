/* ************************************************************************
 * Copyright (C) 2018-2019 Advanced Micro Devices, Inc. All rights Reserved.
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
#include <hipsparse/hipsparse.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

int main(int argc, char* argv[])
{
    // Parse command line
    if(argc < 2)
    {
        fprintf(stderr, "%s <ndim> [<trials> <batch_size>]\n", argv[0]);
        return -1;
    }

    int ndim       = atoi(argv[1]);
    int trials     = 200;
    int batch_size = 1;

    if(argc > 2)
    {
        trials = atoi(argv[2]);
    }
    if(argc > 3)
    {
        batch_size = atoi(argv[3]);
    }

    // hipSPARSE handle
    hipsparseHandle_t handle;
    hipsparseCreate(&handle);

    hipDeviceProp_t devProp;
    int             device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    printf("Device: %s\n", devProp.name);

    // Generate problem
    std::vector<int>    hAptr;
    std::vector<int>    hAcol;
    std::vector<double> hAval;
    int                 m = gen_2d_laplacian(ndim, hAptr, hAcol, hAval, HIPSPARSE_INDEX_BASE_ZERO);
    int                 n = m;
    int                 nnz = hAptr[m];

    // Sample some random data
    srand(12345ULL);

    double halpha = static_cast<double>(rand()) / RAND_MAX;
    double hbeta  = 0.0;

    std::vector<double> hx(n);
    hipsparseInit(hx, 1, n);

    // Matrix descriptor
    hipsparseMatDescr_t descrA;
    hipsparseCreateMatDescr(&descrA);

    // Offload data to device
    int*    dAptr = NULL;
    int*    dAcol = NULL;
    double* dAval = NULL;
    double* dx    = NULL;
    double* dy    = NULL;

    hipMalloc((void**)&dAptr, sizeof(int) * (m + 1));
    hipMalloc((void**)&dAcol, sizeof(int) * nnz);
    hipMalloc((void**)&dAval, sizeof(double) * nnz);
    hipMalloc((void**)&dx, sizeof(double) * n);
    hipMalloc((void**)&dy, sizeof(double) * m);

    hipMemcpy(dAptr, hAptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice);

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call hipsparse csrmv
        hipsparseDcsrmv(handle,
                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        m,
                        n,
                        nnz,
                        &halpha,
                        descrA,
                        dAval,
                        dAptr,
                        dAcol,
                        dx,
                        &hbeta,
                        dy);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // CSR matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int j = 0; j < batch_size; ++j)
        {
            // Call hipsparse csrmv
            hipsparseDcsrmv(handle,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            m,
                            n,
                            nnz,
                            &halpha,
                            descrA,
                            dAval,
                            dAptr,
                            dAcol,
                            dx,
                            &hbeta,
                            dy);
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth
        = static_cast<double>(sizeof(double) * (2 * m + nnz) + sizeof(int) * (m + 1 + nnz)) / time
          / 1e6;
    double gflops = static_cast<double>(2 * nnz) / time / 1e6;
    printf("m\t\tn\t\tnnz\t\talpha\tbeta\tGFlops\tGB/s\tusec\n");
    printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
           m,
           n,
           nnz,
           halpha,
           hbeta,
           gflops,
           bandwidth,
           time);

    // Clear up on device
    hipFree(dAptr);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    hipsparseDestroyMatDescr(descrA);
    hipsparseDestroy(handle);

    return 0;
}
