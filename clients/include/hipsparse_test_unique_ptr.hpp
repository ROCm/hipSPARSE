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

#pragma once
#ifndef GUARD_HIPSPARSE_MANAGE_PTR
#define GUARD_HIPSPARSE_MANAGE_PTR

#include "arg_check.hpp"

#include <hip/hip_runtime_api.h>
#include <hipsparse.h>
#include <memory>

#define PRINT_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                             \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                    \
        {                                                         \
            fprintf(stderr,                                       \
                    "hip error code: %d at %s:%d\n",              \
                    TMP_STATUS_FOR_CHECK,                         \
                    __FILE__,                                     \
                    __LINE__);                                    \
        }                                                         \
    }

namespace hipsparse_test
{

    // device_malloc wraps hipMalloc and provides same API as malloc
    static void* device_malloc(size_t byte_size)
    {
        void* pointer;
        PRINT_IF_HIP_ERROR(hipMalloc(&pointer, byte_size));
        return pointer;
    }

    // device_free wraps hipFree and provides same API as free
    static void device_free(void* ptr)
    {
        PRINT_IF_HIP_ERROR(hipFree(ptr));
    }

    struct handle_struct
    {
        hipsparseHandle_t handle;
        handle_struct()
        {
            hipsparseStatus_t status = hipsparseCreate(&handle);
            verify_hipsparse_status_success(status, "ERROR: handle_struct constructor");
        }

        ~handle_struct()
        {
            hipsparseStatus_t status = hipsparseDestroy(handle);
            verify_hipsparse_status_success(status, "ERROR: handle_struct destructor");
        }
    };

    struct descr_struct
    {
        hipsparseMatDescr_t descr;
        descr_struct()
        {
            hipsparseStatus_t status = hipsparseCreateMatDescr(&descr);
            verify_hipsparse_status_success(status, "ERROR: descr_struct constructor");
        }

        ~descr_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyMatDescr(descr);
            verify_hipsparse_status_success(status, "ERROR: descr_struct destructor");
        }
    };

    struct hyb_struct
    {
        hipsparseHybMat_t hyb;
        hyb_struct()
        {
            hipsparseStatus_t status = hipsparseCreateHybMat(&hyb);
            verify_hipsparse_status_success(status, "ERROR: hyb_struct constructor");
        }

        ~hyb_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyHybMat(hyb);
            verify_hipsparse_status_success(status, "ERROR: hyb_struct destructor");
        }
    };

    struct csrsv2_struct
    {
        csrsv2Info_t info;
        csrsv2_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsrsv2Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csrsv2_struct constructor");
        }

        ~csrsv2_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsrsv2Info(info);
            verify_hipsparse_status_success(status, "ERROR: csrsv2_struct destructor");
        }
    };

    struct csrilu02_struct
    {
        csrilu02Info_t info;
        csrilu02_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsrilu02Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csrilu02_struct constructor");
        }

        ~csrilu02_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsrilu02Info(info);
            verify_hipsparse_status_success(status, "ERROR: csrilu02_struct destructor");
        }
    };

    struct csrgemm2_struct
    {
        csrgemm2Info_t info;
        csrgemm2_struct()
        {
            hipsparseStatus_t status = hipsparseCreateCsrgemm2Info(&info);
            verify_hipsparse_status_success(status, "ERROR: csrgemm2_struct constructor");
        }

        ~csrgemm2_struct()
        {
            hipsparseStatus_t status = hipsparseDestroyCsrgemm2Info(info);
            verify_hipsparse_status_success(status, "ERROR: csrgemm2_struct destructor");
        }
    };

} // namespace hipsparse_test

using hipsparse_unique_ptr = std::unique_ptr<void, void (*)(void*)>;

#endif // GUARD_HIPSPARSE_MANAGE_PTR
