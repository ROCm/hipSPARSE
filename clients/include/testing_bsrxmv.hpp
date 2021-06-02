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
#ifndef TESTING_BSRXMV_HPP
#define TESTING_BSRXMV_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <cmath>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;


void testing_bsrxmv_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif

    int                  safe_size = 100;
    int                  safe_dim  = 2;
    float                    alpha     = 0.6;
    float                    beta      = 0.2;
    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseDirection_t dirA      = HIPSPARSE_DIRECTION_COLUMN;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    auto dptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dend_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dmask_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dcol_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dval_managed = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dx_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};
    auto dy_managed   = hipsparse_unique_ptr{device_malloc(sizeof(float) * safe_size), device_free};

    int* dptr = (int*)dptr_managed.get();
    int* dend_ptr = (int*)dend_ptr_managed.get();
    int* dmask_ptr = (int*)dmask_ptr_managed.get();
    int* dcol = (int*)dcol_managed.get();
    float*   dval = (float*)dval_managed.get();
    float*   dx   = (float*)dx_managed.get();
    float*   dy   = (float*)dy_managed.get();

    if(!dval || !dptr || !dcol || !dx || !dy || !dend_ptr)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Test hipsparseXbsrxmv
    verify_hipsparse_status_invalid_handle(hipsparseXbsrxmv(nullptr,
                                                           dirA,
                                                           transA,
                                                           safe_size,
                                                           safe_size,
                                                           safe_size,
                                                           safe_size,
                                                           &alpha,
                                                           descr,
                                                           dval,
							    dmask_ptr,
                                                           dptr,
							    dend_ptr,
                                                           dcol,
                                                           safe_dim,
                                                           dx,
                                                           &beta,
                                                           dy));
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
							     safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            (float*)nullptr,
                                                            descr,
                                                            dval,
							     dmask_ptr,
							     dptr,
							     dend_ptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: alpha is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
							     safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            nullptr,
                                                            dval,
							     dmask_ptr,
							     dptr,
							     dend_ptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: descr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
							     safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
							     (float*)nullptr,
							     dmask_ptr,
							     dptr,
							     dend_ptr,

							     dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_val is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
							     safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
							     nullptr,
                                                            dptr,
							    dend_ptr,
							     dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_mask_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
							     safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
							     dmask_ptr,
                                                            nullptr,
							     dend_ptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_row_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
							     safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
							     dmask_ptr,
                                                            dptr,
							     nullptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_end_ptr is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                            safe_size,
							     safe_size,
                                                            &alpha,
							     descr,
                                                            dval,
							     dmask_ptr,
							     dptr,
							     dend_ptr,
							     nullptr,
							     safe_dim,
                                                            dx,
                                                            &beta,
                                                            dy),
                                            "Error: bsr_col_ind is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                           safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
							     dmask_ptr,
							     dptr,
							     dend_ptr,
                                                            dcol,
                                                            safe_dim,
                                                            (float*)nullptr,
                                                            &beta,
                                                            dy),
                                            "Error: xy is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                           safe_size,
                                                            safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
							     dmask_ptr,
							     dptr,
							     dend_ptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            (float*)nullptr,
                                                            dy),
                                            "Error: beta is nullptr");
    verify_hipsparse_status_invalid_pointer(hipsparseXbsrxmv(handle,
                                                            dirA,
                                                            transA,
                                                            safe_size,
                                                            safe_size,
                                                           safe_size,
                                                            safe_size,
                                                            &alpha,
                                                            descr,
                                                            dval,
							     dmask_ptr,
							     dptr,
							     dend_ptr,
                                                            dcol,
                                                            safe_dim,
                                                            dx,
                                                            &beta,
                                                            (float*)nullptr),
                                            "Error: y is nullptr");
    verify_hipsparse_status_invalid_size(hipsparseXbsrxmv(handle,
                                                         dirA,
                                                         transA,
                                                         -1,
							 safe_size,
                                                         safe_size,
                                                         safe_size,
                                                         &alpha,
                                                         descr,
                                                         dval,
							  dmask_ptr,
							  dptr,
							  dend_ptr,							  
                                                         dcol,
                                                         safe_dim,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: sizeOfMask is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrxmv(handle,
                                                         dirA,
                                                         transA,
							  safe_size,
                                                         -1,
                                                         safe_size,
                                                         safe_size,
                                                         &alpha,
                                                         descr,
							  dval,
							  dmask_ptr,
							  dptr,
							  dend_ptr,							  
                                                         dcol,
                                                         safe_dim,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: mb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrxmv(handle,
                                                         dirA,
                                                         transA,
							  safe_size,
                                                         safe_size,
                                                         -1,
                                                         safe_size,
                                                         &alpha,
                                                         descr,
                                                         dval,
							  dmask_ptr,
							  dptr,
							  dend_ptr,
                                                         dcol,
                                                         safe_dim,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: nb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrxmv(handle,
                                                         dirA,
                                                         transA,
                                                           safe_size,
                                                         safe_size,
                                                         safe_size,
                                                         -1,
                                                         &alpha,
                                                         descr,
							  dval,
							  dmask_ptr,
							  dptr,
							  dend_ptr,
                                                         dcol,
                                                         safe_dim,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: nnzb is invalid");
    verify_hipsparse_status_invalid_size(hipsparseXbsrxmv(handle,
                                                         dirA,
                                                         transA,
                                                           safe_size,
                                                         safe_size,
                                                         safe_size,
                                                         safe_size,
                                                         &alpha,
                                                         descr,
							  dval,
							  dmask_ptr,
							  dptr,
							  dend_ptr,
                                                         dcol,
                                                         -1,
                                                         dx,
                                                         &beta,
                                                         dy),
                                         "Error: block_dim is invalid");
}

template <typename T>
hipsparseStatus_t testing_bsrxmv()
{
  hipsparseDirection_t dir   = HIPSPARSE_DIRECTION_COLUMN;
  hipsparseOperation_t trans = HIPSPARSE_OPERATION_NON_TRANSPOSE;
  static constexpr int size_of_mask = 1;
  static constexpr int mb = 2;
  static constexpr int nb = 3;
  static constexpr int nnzb = 5;
  static constexpr int block_dim = 2;

  T h_alpha  = make_DataType<T>(2.0);
  T h_beta = make_DataType<T>(1.0);  
  
  std::vector<T>   hbsr_val(block_dim * block_dim * nnzb);
  std::vector<int> hbsr_mask_ptr(size_of_mask);
  std::vector<int> hbsr_row_ptr(mb);
  std::vector<int> hbsr_end_ptr(mb);
  std::vector<int> hbsr_col_ind(nnzb);
  std::vector<T> hx(nb * block_dim);
  std::vector<T> hy(mb * block_dim);
  std::vector<T> hyref(mb * block_dim);

  auto dbsr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * block_dim * block_dim * nnzb), device_free};
  auto dbsr_mask_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * size_of_mask), device_free};
  auto dbsr_row_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * mb), device_free};
  auto dbsr_end_ptr_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * mb), device_free};
  auto dbsr_col_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnzb), device_free};
  auto dx_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * block_dim * nb), device_free};
  auto dy_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * block_dim * mb), device_free};

  T* dbsr_val = (T*)dbsr_val_managed.get();
  int* dbsr_mask_ptr = (int*)dbsr_mask_ptr_managed.get();
  int* dbsr_row_ptr = (int*)dbsr_row_ptr_managed.get();
  int* dbsr_end_ptr = (int*)dbsr_end_ptr_managed.get();
  int* dbsr_col_ind = (int*)dbsr_col_ind_managed.get();
  T* dx = (T*)dx_managed.get();
  T* dy = (T*)dy_managed.get();


  hbsr_mask_ptr[0] = 2;
  
  hbsr_row_ptr[0] = 1;
  hbsr_row_ptr[1] = 4;

  hbsr_end_ptr[0] = 1;
  hbsr_end_ptr[1] = 5;

  hbsr_col_ind[0] = 1;
  hbsr_col_ind[1] = 2;
  hbsr_col_ind[2] = 1;
  hbsr_col_ind[3] = 2;
  hbsr_col_ind[4] = 3;

  hx[0] = make_DataType<T>(1.0);
  hx[1] = make_DataType<T>(1.0);
  hx[2] = make_DataType<T>(1.0);
  hx[3] = make_DataType<T>(1.0);
  hx[4] = make_DataType<T>(1.0);
  hx[5] = make_DataType<T>(1.0);

  hy[0] = make_DataType<T>(2.0);
  hy[1] = make_DataType<T>(2.0);
  hy[2] = make_DataType<T>(2.0);
  hy[3] = make_DataType<T>(2.0);

  hyref[0] = make_DataType<T>(2.0);
  hyref[1] = make_DataType<T>(2.0);
  hyref[2] = make_DataType<T>(58.0);
  hyref[3] = make_DataType<T>(62.0);

  hbsr_val[0] = make_DataType<T>(1.0);
  hbsr_val[1] = make_DataType<T>(2.0);
  hbsr_val[2] = make_DataType<T>(3.0);
  hbsr_val[3] = make_DataType<T>(4.0);

  hbsr_val[4] = make_DataType<T>(5.0);
  hbsr_val[5] = make_DataType<T>(6.0);
  hbsr_val[6] = make_DataType<T>(7.0);
  hbsr_val[7] = make_DataType<T>(8.0);

  hbsr_val[8] = make_DataType<T>(9.0);
  hbsr_val[9] = make_DataType<T>(10.0);
  hbsr_val[10] = make_DataType<T>(11.0);
  hbsr_val[11] = make_DataType<T>(12.0);
  
  hbsr_val[12] = make_DataType<T>(13.0);
  hbsr_val[13] = make_DataType<T>(14.0);
  hbsr_val[14] = make_DataType<T>(15.0);
  hbsr_val[15] = make_DataType<T>(16.0);

  hbsr_val[16] = make_DataType<T>(17.0);
  hbsr_val[17] = make_DataType<T>(18.0);
  hbsr_val[18] = make_DataType<T>(19.0);
  hbsr_val[19] = make_DataType<T>(20.0);

  CHECK_HIP_ERROR(hipMemcpy(dbsr_val, hbsr_val.data(), sizeof(T) * nnzb * block_dim * block_dim, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(dbsr_mask_ptr, hbsr_mask_ptr.data(), sizeof(int) * size_of_mask, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(dbsr_row_ptr, hbsr_row_ptr.data(), sizeof(int) * mb, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(dbsr_end_ptr, hbsr_end_ptr.data(), sizeof(int) * mb, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(dbsr_col_ind, hbsr_col_ind.data(), sizeof(int) * nnzb, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(dx, hx.data(), sizeof(T) * nb * block_dim, hipMemcpyHostToDevice));
  CHECK_HIP_ERROR(hipMemcpy(dy, hy.data(), sizeof(T) * mb * block_dim, hipMemcpyHostToDevice));



  std::unique_ptr<handle_struct> test_handle(new handle_struct);
  hipsparseHandle_t              handle = test_handle->handle;
  
  std::unique_ptr<descr_struct> test_descr(new descr_struct);
  hipsparseMatDescr_t           descr = test_descr->descr;
  
  hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ONE;
  CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));
  
  CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
  hipsparseStatus_t  status = hipsparseXbsrxmv(handle,
					       dir,
					       trans,
					       size_of_mask,
					       mb,
					       nb,
					       nnzb,
					       &h_alpha,
					       descr,
					       dbsr_val,
					       dbsr_mask_ptr,
					       dbsr_row_ptr,
					       dbsr_end_ptr,
					       dbsr_col_ind,
					       block_dim,
					       dx,
					       &h_beta,
					       dy);
  verify_hipsparse_status_success(status, "bsrxmv failed.");

  CHECK_HIP_ERROR(hipMemcpy(hy.data(), dy, sizeof(T) * mb * block_dim, hipMemcpyDeviceToHost));
  
  unit_check_near(1, mb * block_dim, 1, hyref.data(), hy.data());
  return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_BSRXMV_HPP
