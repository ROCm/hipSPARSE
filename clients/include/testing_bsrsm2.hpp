/* ************************************************************************
 * Copyright (c) 2021 Advanced Micro Devices, Inc.
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
#ifndef TESTING_BSRSV2_HPP
#define TESTING_BSRSV2_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <string>
#include <unistd.h>

using namespace hipsparse;
using namespace hipsparse_test;

void testing_bsrsm2_bad_arg(void)
{
#ifdef __HIP_PLATFORM_AMD__
    int   mb        = 100;
    int   nrhs      = 100;
    int   nnzb      = 100;
    int   block_dim = 100;
    int   safe_size = 100;
    float alpha     = 0.6;
    int   ldb       = 100;
    int   ldx       = 100;

    hipsparseDirection_t   dirA   = HIPSPARSE_DIRECTION_ROW;
    hipsparseOperation_t   transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t   transX = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseSolvePolicy_t policy = HIPSPARSE_SOLVE_POLICY_USE_LEVEL;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<bsrsm2_struct> unique_ptr_info(new bsrsm2_struct);
    bsrsm2Info_t                   info = unique_ptr_info->info;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dB_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dX_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dbuf_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * safe_size), device_free};

    int*   dptr = (int*)dptr_managed.get();
    int*   dcol = (int*)dcol_managed.get();
    float* dval = (float*)dval_managed.get();
    float* dB   = (float*)dB_managed.get();
    float* dX   = (float*)dX_managed.get();
    void*  dbuf = (void*)dbuf_managed.get();

    if(!dval || !dptr || !dcol || !dB || !dX || !dbuf)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing hipsparseXbsrsm2_bufferSize
    int size;

    verify_hipsparse_status_invalid_handle(hipsparseXbsrsm2_bufferSize(nullptr,
                                                                       dirA,
                                                                       transA,
                                                                       transX,
                                                                       mb,
                                                                       nrhs,
                                                                       nnzb,
                                                                       descr,
                                                                       dval,
                                                                       dptr,
                                                                       dcol,
                                                                       block_dim,
                                                                       info,
                                                                       &size));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        nullptr,
                                                                        dval,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        (float*)nullptr,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        nullptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        dptr,
                                                                        nullptr,
                                                                        block_dim,
                                                                        info,
                                                                        &size),
                                            "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        nullptr,
                                                                        &size),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_bufferSize(handle,
                                                                        dirA,
                                                                        transA,
                                                                        transX,
                                                                        mb,
                                                                        nrhs,
                                                                        nnzb,
                                                                        descr,
                                                                        dval,
                                                                        dptr,
                                                                        dcol,
                                                                        block_dim,
                                                                        info,
                                                                        nullptr),
                                            "Error: size is nullptr");

    // testing hipsparseXbsrsm2_analysis
    verify_hipsparse_status_invalid_handle(hipsparseXbsrsm2_analysis(nullptr,
                                                                     dirA,
                                                                     transA,
                                                                     transX,
                                                                     mb,
                                                                     nrhs,
                                                                     nnzb,
                                                                     descr,
                                                                     dval,
                                                                     dptr,
                                                                     dcol,
                                                                     block_dim,
                                                                     info,
                                                                     policy,
                                                                     dbuf));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      nullptr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      (const float*)nullptr,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      nullptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      dptr,
                                                                      nullptr,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      dbuf),
                                            "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      nullptr,
                                                                      policy,
                                                                      dbuf),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_analysis(handle,
                                                                      dirA,
                                                                      transA,
                                                                      transX,
                                                                      mb,
                                                                      nrhs,
                                                                      nnzb,
                                                                      descr,
                                                                      dval,
                                                                      dptr,
                                                                      dcol,
                                                                      block_dim,
                                                                      info,
                                                                      policy,
                                                                      nullptr),
                                            "Error: dbuf is nullptr");

    // testing hipsparseXbsrsm2_solve
    verify_hipsparse_status_invalid_handle(hipsparseXbsrsm2_solve(nullptr,
                                                                  dirA,
                                                                  transA,
                                                                  transX,
                                                                  mb,
                                                                  nrhs,
                                                                  nnzb,
                                                                  &alpha,
                                                                  descr,
                                                                  dval,
                                                                  dptr,
                                                                  dcol,
                                                                  block_dim,
                                                                  info,
                                                                  dB,
                                                                  ldb,
                                                                  dX,
                                                                  ldx,
                                                                  policy,
                                                                  dbuf));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   (const float*)nullptr,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   nullptr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   (const float*)nullptr,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dval is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   nullptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   nullptr,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dcol is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   nullptr,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: info is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   (const float*)nullptr,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dB is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   (float*)nullptr,
                                                                   ldx,
                                                                   policy,
                                                                   dbuf),
                                            "Error: dX is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrsm2_solve(handle,
                                                                   dirA,
                                                                   transA,
                                                                   transX,
                                                                   mb,
                                                                   nrhs,
                                                                   nnzb,
                                                                   &alpha,
                                                                   descr,
                                                                   dval,
                                                                   dptr,
                                                                   dcol,
                                                                   block_dim,
                                                                   info,
                                                                   dB,
                                                                   ldb,
                                                                   dX,
                                                                   ldx,
                                                                   policy,
                                                                   nullptr),
                                            "Error: dbuf is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_bsrsm2(void)
{
    T   h_alpha = make_DataType<T>(2.0);
    int nrhs    = 15;

    // Determine absolute path of test matrix

    // Get current executables absolute path
    char    path_exe[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", path_exe, sizeof(path_exe) - 1);
    if(len < 14)
        path_exe[0] = '\0';
    else
        path_exe[len - 14] = '\0';

    // Matrices are stored at the same path in matrices directory
    std::string filename = std::string(path_exe) + "../matrices/nos3.bin";

    // hipSPARSE handle and opaque structs
    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t              handle = test_handle->handle;

    std::unique_ptr<descr_struct> test_descr(new descr_struct);
    hipsparseMatDescr_t           descr = test_descr->descr;

    std::unique_ptr<bsrsm2_struct> unique_ptr_bsrsm2_info(new bsrsm2_struct);
    bsrsm2Info_t                   info = unique_ptr_bsrsm2_info->info;

    // Host structures
    std::vector<int> hcsr_row_ptr;
    std::vector<int> hcsr_col_ind;
    std::vector<T>   hcsr_val;

    // Initial Data on CPU
    srand(12345ULL);

    int m;
    int n;
    int nnz;

    if(read_bin_matrix(filename.c_str(),
                       m,
                       n,
                       nnz,
                       hcsr_row_ptr,
                       hcsr_col_ind,
                       hcsr_val,
                       HIPSPARSE_INDEX_BASE_ZERO)
       != 0)
    {
        fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }

    int block_dim = 3;
    int mb        = (m + block_dim - 1) / block_dim;

    int ldb = m;
    int ldx = m;

    std::vector<T> hB(mb * block_dim * nrhs);
    std::vector<T> hX_1(mb * block_dim * nrhs);
    std::vector<T> hX_2(mb * block_dim * nrhs);
    std::vector<T> hX_gold(mb * block_dim * nrhs);

    hipsparseInit<T>(hB, m, nrhs);

    // Allocate memory on device
    auto dcsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (m + 1)), device_free};
    auto dcsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dcsr_val_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dB_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim * nrhs), device_free};
    auto dX_1_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim * nrhs), device_free};
    auto dX_2_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * mb * block_dim * nrhs), device_free};
    auto dalpha_managed = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto dpos_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};

    int* dcsr_row_ptr = (int*)dcsr_row_ptr_managed.get();
    int* dcsr_col_ind = (int*)dcsr_col_ind_managed.get();
    T*   dcsr_val     = (T*)dcsr_val_managed.get();
    int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
    T*   dB           = (T*)dB_managed.get();
    T*   dX_1         = (T*)dX_1_managed.get();
    T*   dX_2         = (T*)dX_2_managed.get();
    T*   dalpha       = (T*)dalpha_managed.get();
    int* dposition    = (int*)dpos_managed.get();

    if(!dcsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dbsr_row_ptr || !dB || !dX_1 || !dX_2
       || !dalpha || !dposition)
    {
        verify_hipsparse_status_success(
            HIPSPARSE_STATUS_ALLOC_FAILED,
            "!dcsr_val || !dcsr_row_ptr || !dcsr_col_ind || !dbsr_row_ptr || !dB || !dX_1 || !dX_2 "
            "|| !dalpha || !dposition");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_row_ptr, hcsr_row_ptr.data(), sizeof(int) * (m + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dcsr_col_ind, hcsr_col_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dcsr_val, hcsr_val.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dB, hB.data(), sizeof(T) * m * nrhs, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dalpha, &h_alpha, sizeof(T), hipMemcpyHostToDevice));

    // Convert to BSR
    int nnzb;
    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsrNnz(handle,
                                               HIPSPARSE_DIRECTION_ROW,
                                               m,
                                               m,
                                               descr,
                                               dcsr_row_ptr,
                                               dcsr_col_ind,
                                               block_dim,
                                               descr,
                                               dbsr_row_ptr,
                                               &nnzb));

    auto dbsr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
    auto dbsr_val_managed = hipsparse_unique_ptr{
        device_malloc(sizeof(T) * nnzb * block_dim * block_dim), device_free};

    int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
    T*   dbsr_val     = (T*)dbsr_val_managed.get();

    if(!dbsr_val || !dbsr_col_ind)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dbsr_val || !dbsr_col_ind");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    CHECK_HIPSPARSE_ERROR(hipsparseXcsr2bsr(handle,
                                            HIPSPARSE_DIRECTION_ROW,
                                            m,
                                            m,
                                            descr,
                                            dcsr_val,
                                            dcsr_row_ptr,
                                            dcsr_col_ind,
                                            block_dim,
                                            descr,
                                            dbsr_val,
                                            dbsr_row_ptr,
                                            dbsr_col_ind));

    // Obtain bsrsm2 buffer size
    int size;

    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_bufferSize(handle,
                                                      HIPSPARSE_DIRECTION_ROW,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                      mb,
                                                      nrhs,
                                                      nnzb,
                                                      descr,
                                                      dbsr_val,
                                                      dbsr_row_ptr,
                                                      dbsr_col_ind,
                                                      block_dim,
                                                      info,
                                                      &size));

    // Allocate buffer on the device
    auto dbuffer_managed = hipsparse_unique_ptr{device_malloc(sizeof(char) * size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dbuffer");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // bsrsm2 analysis
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_analysis(handle,
                                                    HIPSPARSE_DIRECTION_ROW,
                                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                    HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                    mb,
                                                    nrhs,
                                                    nnzb,
                                                    descr,
                                                    dbsr_val,
                                                    dbsr_row_ptr,
                                                    dbsr_col_ind,
                                                    block_dim,
                                                    info,
                                                    HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                    dbuffer));

    int               pos_analysis;
    hipsparseStatus_t pivot_analysis = hipsparseXbsrsm2_zeroPivot(handle, info, &pos_analysis);

    // HIPSPARSE pointer mode host
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_solve(handle,
                                                 HIPSPARSE_DIRECTION_ROW,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 mb,
                                                 nrhs,
                                                 nnzb,
                                                 &h_alpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 block_dim,
                                                 info,
                                                 dB,
                                                 ldb,
                                                 dX_1,
                                                 ldx,
                                                 HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 dbuffer));

    int               hposition_1;
    hipsparseStatus_t pivot_status_1 = hipsparseXbsrsm2_zeroPivot(handle, info, &hposition_1);

    // HIPSPARSE pointer mode device
    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseXbsrsm2_solve(handle,
                                                 HIPSPARSE_DIRECTION_ROW,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                                 mb,
                                                 nrhs,
                                                 nnzb,
                                                 dalpha,
                                                 descr,
                                                 dbsr_val,
                                                 dbsr_row_ptr,
                                                 dbsr_col_ind,
                                                 block_dim,
                                                 info,
                                                 dB,
                                                 ldb,
                                                 dX_2,
                                                 ldx,
                                                 HIPSPARSE_SOLVE_POLICY_USE_LEVEL,
                                                 dbuffer));

    hipsparseStatus_t pivot_status_2 = hipsparseXbsrsm2_zeroPivot(handle, info, dposition);

    // Copy output from device to CPU
    int hposition_2;
    CHECK_HIP_ERROR(hipMemcpy(hX_1.data(), dX_1, sizeof(T) * m * nrhs, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hX_2.data(), dX_2, sizeof(T) * m * nrhs, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(&hposition_2, dposition, sizeof(int), hipMemcpyDeviceToHost));

    // Host bsrsm2
    std::vector<int> hbsr_row_ptr(mb + 1);
    std::vector<int> hbsr_col_ind(nnzb);
    std::vector<T>   hbsr_val(nnzb * block_dim * block_dim);

    CHECK_HIP_ERROR(hipMemcpy(
        hbsr_row_ptr.data(), dbsr_row_ptr, sizeof(int) * (mb + 1), hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(
        hipMemcpy(hbsr_col_ind.data(), dbsr_col_ind, sizeof(int) * nnzb, hipMemcpyDeviceToHost));
    CHECK_HIP_ERROR(hipMemcpy(hbsr_val.data(),
                              dbsr_val,
                              sizeof(T) * nnzb * block_dim * block_dim,
                              hipMemcpyDeviceToHost));

    int struct_position_gold;
    int numeric_position_gold;

    bsrsm(mb,
          nrhs,
          nnzb,
          HIPSPARSE_DIRECTION_ROW,
          HIPSPARSE_OPERATION_NON_TRANSPOSE,
          HIPSPARSE_OPERATION_NON_TRANSPOSE,
          h_alpha,
          hbsr_row_ptr.data(),
          hbsr_col_ind.data(),
          hbsr_val.data(),
          block_dim,
          hB.data(),
          ldb,
          hX_gold.data(),
          ldx,
          HIPSPARSE_DIAG_TYPE_NON_UNIT,
          HIPSPARSE_FILL_MODE_LOWER,
          HIPSPARSE_INDEX_BASE_ZERO,
          &struct_position_gold,
          &numeric_position_gold);

    unit_check_general(1, 1, 1, &struct_position_gold, &pos_analysis);
    unit_check_general(1, 1, 1, &numeric_position_gold, &hposition_1);
    unit_check_general(1, 1, 1, &numeric_position_gold, &hposition_2);

    if(hposition_1 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_1, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

    if(hposition_2 != -1)
    {
        verify_hipsparse_status_zero_pivot(pivot_status_2, "expected HIPSPARSE_STATUS_ZERO_PIVOT");
        return HIPSPARSE_STATUS_SUCCESS;
    }

    unit_check_near(m, nrhs, ldx, hX_gold.data(), hX_1.data());
    unit_check_near(m, nrhs, ldx, hX_gold.data(), hX_2.data());

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSRSV2_HPP
