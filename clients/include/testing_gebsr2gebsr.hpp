/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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
#ifndef TESTING_GEBSR2GEBSR_HPP
#define TESTING_GEBSR2GEBSR_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

#include <iostream>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_gebsr2gebsr_bad_arg(void)
{
#ifdef __HIP_PLATFORM_NVCC__
    // do not test for bad args
    return;
#endif
    int                  mb              = 100;
    int                  nb              = 100;
    int                  nnzb            = 100;
    int                  safe_size       = 100;
    int                  row_block_dim_A = 2;
    int                  col_block_dim_A = 2;
    int                  row_block_dim_C = 2;
    int                  col_block_dim_C = 2;
    hipsparseIndexBase_t idx_base_A      = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t idx_base_C      = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseDirection_t dir             = HIPSPARSE_DIRECTION_ROW;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t            descr_A = unique_ptr_descr_A->descr;
    std::unique_ptr<descr_struct>  unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t            descr_C = unique_ptr_descr_C->descr;

    hipsparseSetMatIndexBase(descr_A, idx_base_A);
    hipsparseSetMatIndexBase(descr_C, idx_base_C);

    auto bsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto bsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_col_ind_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto bsr_val_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto temp_buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* bsr_row_ptr_A = (int*)bsr_row_ptr_A_managed.get();
    int* bsr_col_ind_A = (int*)bsr_col_ind_A_managed.get();
    T*   bsr_val_A     = (T*)bsr_val_A_managed.get();
    int* bsr_row_ptr_C = (int*)bsr_row_ptr_C_managed.get();
    int* bsr_col_ind_C = (int*)bsr_col_ind_C_managed.get();
    T*   bsr_val_C     = (T*)bsr_val_C_managed.get();
    T*   temp_buffer   = (T*)temp_buffer_managed.get();

    if(!bsr_row_ptr_A || !bsr_col_ind_A || !bsr_val_A || !bsr_row_ptr_C || !bsr_col_ind_C
       || !bsr_val_C)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // Testing hipsparseXgebsr2gebsr_bufferSize()

    int buffer_size;

    // Test invalid handle
    status = hipsparseXgebsr2gebsr_bufferSize(nullptr,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              nullptr,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              (const T*)nullptr,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              nullptr,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              nullptr,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_A is nullptr");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer_size is nullptr");

    // Test invalid sizes
    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              -1,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              -1,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              -1,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              -1,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              -1,
                                              row_block_dim_C,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              -1,
                                              col_block_dim_C,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_C is invalid");

    status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                              dir,
                                              mb,
                                              nb,
                                              nnzb,
                                              descr_A,
                                              bsr_val_A,
                                              bsr_row_ptr_A,
                                              bsr_col_ind_A,
                                              row_block_dim_A,
                                              col_block_dim_A,
                                              row_block_dim_C,
                                              -1,
                                              &buffer_size);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_C is invalid");

    // Testing hipsparseXgebsr2gebsrNnz()

    int nnz_total_dev_host_ptr;

    // Test invalid handle
    status = hipsparseXgebsr2gebsrNnz(nullptr,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      nullptr,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      nullptr,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_A is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      nullptr,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_A is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      nullptr,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      nullptr,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_C is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      nullptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: nnz_total_dev_host_ptr is nullptr");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer is nullptr");

    // Test invalid sizes
    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      -1,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      -1,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      -1,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      -1,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      -1,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      -1,
                                      col_block_dim_C,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_C is invalid");

    status = hipsparseXgebsr2gebsrNnz(handle,
                                      dir,
                                      mb,
                                      nb,
                                      nnzb,
                                      descr_A,
                                      bsr_row_ptr_A,
                                      bsr_col_ind_A,
                                      row_block_dim_A,
                                      col_block_dim_A,
                                      descr_C,
                                      bsr_row_ptr_C,
                                      row_block_dim_C,
                                      -1,
                                      &nnz_total_dev_host_ptr,
                                      temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_C is invalid");

    // Test hipsparseXgebsr2gebsr()

    // Test invalid handle
    status = hipsparseXgebsr2gebsr(nullptr,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    // Test invalid pointers
    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   nullptr,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   (const T*)nullptr,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   nullptr,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   nullptr,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_A is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   nullptr,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr_C is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   (T*)nullptr,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_val_C is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   nullptr,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_row_ptr_C is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   nullptr,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: bsr_col_ind_C is nullptr");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: temp_buffer is nullptr");

    // Test invalid sizes
    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   -1,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: mb is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   -1,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nb is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   -1,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: nnzb is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   -1,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   -1,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_A is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   -1,
                                   col_block_dim_C,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: row_block_dim_C is invalid");

    status = hipsparseXgebsr2gebsr(handle,
                                   dir,
                                   mb,
                                   nb,
                                   nnzb,
                                   descr_A,
                                   bsr_val_A,
                                   bsr_row_ptr_A,
                                   bsr_col_ind_A,
                                   row_block_dim_A,
                                   col_block_dim_A,
                                   descr_C,
                                   bsr_val_C,
                                   bsr_row_ptr_C,
                                   bsr_col_ind_C,
                                   row_block_dim_C,
                                   -1,
                                   temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: col_block_dim_C is invalid");
}

template <typename T>
hipsparseStatus_t testing_gebsr2gebsr(Arguments argus)
{
    int                  m               = argus.M;
    int                  n               = argus.N;
    int                  row_block_dim_A = argus.block_dim;
    int                  col_block_dim_A = argus.block_dim;
    int                  row_block_dim_C = argus.block_dim;
    int                  col_block_dim_C = argus.block_dim;
    hipsparseIndexBase_t idx_base_A      = argus.idx_base;
    hipsparseIndexBase_t idx_base_C      = argus.idx_base2;
    hipsparseDirection_t dir             = argus.dirA;
    std::string          binfile         = "";
    std::string          filename        = "";
    hipsparseStatus_t    status;

    // When in testing mode, M == N == -99 indicates that we are testing with a real
    // matrix from cise.ufl.edu
    int safe_size = 100;
    if(m == -99 && n == -99 && argus.timing == 0)
    {
        binfile = argus.filename;
        m = n = safe_size;
    }

    if(argus.timing == 1)
    {
        filename = argus.filename;
    }

    int mb = -1;
    int nb = -1;
    if(row_block_dim_A > 0 && col_block_dim_A > 0)
    {
        mb = (m + row_block_dim_A - 1) / row_block_dim_A;
        nb = (n + col_block_dim_A - 1) / col_block_dim_A;
    }

    int mb_C = -1;
    int nb_C = -1;
    if(row_block_dim_C > 0 && col_block_dim_C > 0)
    {
        mb_C = (m + row_block_dim_C - 1) / row_block_dim_C;
        nb_C = (n + col_block_dim_C - 1) / col_block_dim_C;
    }

    std::cout << "mb: " << mb << " nb: " << nb << " mb_C: " << mb_C << " nb_C: " << nb_C
              << std::endl;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;
    std::unique_ptr<descr_struct>  unique_ptr_descr_A(new descr_struct);
    hipsparseMatDescr_t            descr_A = unique_ptr_descr_A->descr;
    std::unique_ptr<descr_struct>  unique_ptr_descr_C(new descr_struct);
    hipsparseMatDescr_t            descr_C = unique_ptr_descr_C->descr;

    hipsparseSetMatIndexBase(descr_A, idx_base_A);
    hipsparseSetMatIndexBase(descr_C, idx_base_C);

    // Argument sanity check before allocating invalid memory
    if(mb <= 0 || nb <= 0 || row_block_dim_A <= 0 || col_block_dim_A <= 0 || row_block_dim_C <= 0
       || col_block_dim_C <= 0)
    {
#ifdef __HIP_PLATFORM_NVCC__
        // Do not test args in cusparse
        return HIPSPARSE_STATUS_SUCCESS;
#endif
        auto dbsr_row_ptr_A_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dbsr_col_ind_A_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dbsr_val_A_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dbsr_row_ptr_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dbsr_col_ind_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dbsr_val_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dtemp_buffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* dbsr_row_ptr_A = (int*)dbsr_row_ptr_A_managed.get();
        int* dbsr_col_ind_A = (int*)dbsr_col_ind_A_managed.get();
        T*   dbsr_val_A     = (T*)dbsr_val_A_managed.get();
        int* dbsr_row_ptr_C = (int*)dbsr_row_ptr_C_managed.get();
        int* dbsr_col_ind_C = (int*)dbsr_col_ind_C_managed.get();
        T*   dbsr_val_C     = (T*)dbsr_val_C_managed.get();
        T*   dtemp_buffer   = (T*)dtemp_buffer_managed.get();

        if(!dbsr_row_ptr_A || !dbsr_col_ind_A || !dbsr_val_A || !dbsr_row_ptr_C || !dbsr_col_ind_C
           || !dbsr_val_C || !dtemp_buffer)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dbsr_row_ptr_A || !dbsr_col_ind_A || !dbsr_val_A || "
                                            "!dbsr_row_ptr_C || !dbsr_col_ind_C || !dbsr_val_C || "
                                            "!dtemp_buffer");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        int buffer_size;
        status = hipsparseXgebsr2gebsr_bufferSize(handle,
                                                  dir,
                                                  mb,
                                                  nb,
                                                  safe_size,
                                                  descr_A,
                                                  dbsr_val_A,
                                                  dbsr_row_ptr_A,
                                                  dbsr_col_ind_A,
                                                  row_block_dim_A,
                                                  col_block_dim_A,
                                                  row_block_dim_C,
                                                  col_block_dim_C,
                                                  &buffer_size);

        if(mb < 0 || nb < 0 || row_block_dim_A < 0 || col_block_dim_A < 0 || row_block_dim_C < 0
           || col_block_dim_C < 0)
        {
            verify_hipsparse_status_invalid_size(
                status,
                "Error: mb < 0 || nb < 0 || row_block_dim_A < 0 || col_block_dim_A < 0 || "
                "row_block_dim_C < 0 || col_block_dim_C < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status,
                "mb >= 0 && nb >= 0 && row_block_dim_A > 0 && col_block_dim_A > 0 && "
                "row_block_dim_C > 0 && col_block_dim_C > 0");
        }

        int nnz_total_dev_host_ptr;
        status = hipsparseXgebsr2gebsrNnz(handle,
                                          dir,
                                          mb,
                                          nb,
                                          safe_size,
                                          descr_A,
                                          dbsr_row_ptr_A,
                                          dbsr_col_ind_A,
                                          row_block_dim_A,
                                          col_block_dim_A,
                                          descr_C,
                                          dbsr_row_ptr_C,
                                          row_block_dim_C,
                                          col_block_dim_C,
                                          &nnz_total_dev_host_ptr,
                                          dtemp_buffer);

        if(mb < 0 || nb < 0 || row_block_dim_A < 0 || col_block_dim_A < 0 || row_block_dim_C < 0
           || col_block_dim_C < 0)
        {
            verify_hipsparse_status_invalid_size(
                status,
                "Error: mb < 0 || nb < 0 || row_block_dim_A < 0 || col_block_dim_A < 0 || "
                "row_block_dim_C < 0 || col_block_dim_C < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status,
                "mb >= 0 && nb >= 0 && row_block_dim_A > 0 && col_block_dim_A > 0 && "
                "row_block_dim_C > 0 && col_block_dim_C > 0");
        }

        status = hipsparseXgebsr2gebsr(handle,
                                       dir,
                                       mb,
                                       nb,
                                       safe_size,
                                       descr_A,
                                       dbsr_val_A,
                                       dbsr_row_ptr_A,
                                       dbsr_col_ind_A,
                                       row_block_dim_A,
                                       col_block_dim_A,
                                       descr_C,
                                       dbsr_val_C,
                                       dbsr_row_ptr_C,
                                       dbsr_col_ind_C,
                                       row_block_dim_C,
                                       col_block_dim_C,
                                       dtemp_buffer);

        if(mb < 0 || nb < 0 || row_block_dim_A < 0 || col_block_dim_A < 0 || row_block_dim_C < 0
           || col_block_dim_C < 0)
        {
            verify_hipsparse_status_invalid_size(
                status,
                "Error: mb < 0 || nb < 0 || row_block_dim_A < 0 || col_block_dim_A < 0 || "
                "row_block_dim_C < 0 || col_block_dim_C < 0");
        }
        else
        {
            verify_hipsparse_status_success(
                status,
                "mb >= 0 && nb >= 0 && row_block_dim_A > 0 && col_block_dim_A > 0 && "
                "row_block_dim_C > 0 && col_block_dim_C > 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Generate a CSR matrix and use it to then generate a GEBSR matrix
    std::vector<int> csr_row_ptr;
    std::vector<int> csr_col_ind;
    std::vector<T>   csr_val;
    int              nnzb;
    srand(12345ULL);
    if(binfile != "")
    {
        if(read_bin_matrix(
               binfile.c_str(), mb, nb, nnzb, csr_row_ptr, csr_col_ind, csr_val, idx_base_A)
           != 0)
        {
            fprintf(stderr, "Cannot open [read] %s\n", binfile.c_str());
            return HIPSPARSE_STATUS_INTERNAL_ERROR;
        }
    }
    else if(argus.laplacian)
    {
        mb = nb = gen_2d_laplacian(argus.laplacian, csr_row_ptr, csr_col_ind, csr_val, idx_base_A);
        nnzb    = csr_row_ptr[mb];
    }
    else
    {
        std::vector<int> coo_row_ind;

        if(filename != "")
        {
            if(read_mtx_matrix(
                   filename.c_str(), mb, nb, nnzb, coo_row_ind, csr_col_ind, csr_val, idx_base_A)
               != 0)
            {
                fprintf(stderr, "Cannot open [read] %s\n", filename.c_str());
                return HIPSPARSE_STATUS_INTERNAL_ERROR;
            }
        }
        else
        {
            double scale = 0.02;
            if(mb > 1000 || nb > 1000)
            {
                scale = 2.0 / std::max(mb, nb);
            }
            nnzb = mb * scale * nb;
            gen_matrix_coo(mb, nb, nnzb, coo_row_ind, csr_col_ind, csr_val, idx_base_A);
        }

        // Convert COO to CSR
        csr_row_ptr.resize(mb + 1, 0);
        for(int i = 0; i < nnzb; ++i)
        {
            ++csr_row_ptr[coo_row_ind[i] + 1 - idx_base_A];
        }

        csr_row_ptr[0] = idx_base_A;
        for(int i = 0; i < mb; ++i)
        {
            csr_row_ptr[i + 1] += csr_row_ptr[i];
        }
    }

    // mb and nb can be modified if reading from a file
    m       = mb * row_block_dim_A;
    n       = nb * col_block_dim_A;
    mb_C    = (m + row_block_dim_C - 1) / row_block_dim_C;
    nb_C    = (n + col_block_dim_C - 1) / col_block_dim_C;
    int nnz = nnzb * row_block_dim_A * col_block_dim_A;

    // Now use the csr matrix as the symbolic for the gebsr matrix.
    std::vector<int> hbsr_row_ptr_A = csr_row_ptr;
    std::vector<int> hbsr_col_ind_A = csr_col_ind;
    std::vector<T>   hbsr_val_A     = csr_val;

    hbsr_val_A.resize(nnz);

    int idx = 0;
    switch(dir)
    {
    case HIPSPARSE_DIRECTION_COLUMN:
    {
        for(int i = 0; i < mb; ++i)
        {
            for(int r = 0; r < row_block_dim_A; ++r)
            {
                for(int k = hbsr_row_ptr_A[i] - idx_base_A; k < hbsr_row_ptr_A[i + 1] - idx_base_A;
                    ++k)
                {
                    int j = hbsr_col_ind_A[k] - idx_base_A;
                    for(int c = 0; c < col_block_dim_A; ++c)
                    {
                        hbsr_val_A[k * row_block_dim_A * col_block_dim_A + c * row_block_dim_A + r]
                            = static_cast<T>(++idx);
                    }
                }
            }
        }
        break;
    }

    case HIPSPARSE_DIRECTION_ROW:
    {
        for(int i = 0; i < mb; ++i)
        {
            for(int r = 0; r < row_block_dim_A; ++r)
            {
                for(int k = hbsr_row_ptr_A[i] - idx_base_A; k < hbsr_row_ptr_A[i + 1] - idx_base_A;
                    ++k)
                {
                    int j = hbsr_col_ind_A[k] - idx_base_A;
                    for(int c = 0; c < col_block_dim_A; ++c)
                    {
                        hbsr_val_A[k * row_block_dim_A * col_block_dim_A + r * col_block_dim_A + c]
                            = static_cast<T>(++idx);
                    }
                }
            }
        }
        break;
    }
    }

    std::cout << "mb: " << mb << " nb: " << nb << " mb_C: " << mb_C << " nb_C: " << nb_C
              << " nnz: " << nnz << std::endl;

    // Allocate memory on the device
    auto dbsr_row_ptr_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb + 1)), device_free};
    auto dbsr_col_ind_A_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dbsr_val_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dbsr_row_ptr_C_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (mb_C + 1)), device_free};

    int* dbsr_row_ptr_A = (int*)dbsr_row_ptr_A_managed.get();
    int* dbsr_col_ind_A = (int*)dbsr_col_ind_A_managed.get();
    T*   dbsr_val_A     = (T*)dbsr_val_A_managed.get();
    int* dbsr_row_ptr_C = (int*)dbsr_row_ptr_C_managed.get();

    if(!dbsr_row_ptr_A || !dbsr_col_ind_A || !dbsr_val_A || !dbsr_row_ptr_C)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!dbsr_row_ptr_A || !dbsr_col_ind_A || !dbsr_val_A || "
                                        "!dbsr_row_ptr_C");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Copy data from host to device
    CHECK_HIP_ERROR(hipMemcpy(
        dbsr_row_ptr_A, hbsr_row_ptr_A.data(), sizeof(int) * (mb + 1), hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_col_ind_A, hbsr_col_ind_A.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(
        hipMemcpy(dbsr_val_A, hbsr_val_A.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));

    int buffer_size = 0;
    CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsr_bufferSize(handle,
                                                           dir,
                                                           mb,
                                                           nb,
                                                           nnzb,
                                                           descr_A,
                                                           dbsr_val_A,
                                                           dbsr_row_ptr_A,
                                                           dbsr_col_ind_A,
                                                           row_block_dim_A,
                                                           col_block_dim_A,
                                                           row_block_dim_C,
                                                           col_block_dim_C,
                                                           &buffer_size));

    std::cout << "buffer_size: " << buffer_size << std::endl;

    // Allocate buffer on the device
    auto dbuffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(char) * buffer_size), device_free};

    void* dbuffer = (void*)dbuffer_managed.get();

    if(!dbuffer)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!dbuffer");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    if(argus.unit_check)
    {
        // Obtain BSR nnzb first on the host and then using the device and ensure they give the same results
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        int hnnzb_C;
        CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsrNnz(handle,
                                                       dir,
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       descr_A,
                                                       dbsr_row_ptr_A,
                                                       dbsr_col_ind_A,
                                                       row_block_dim_A,
                                                       col_block_dim_A,
                                                       descr_C,
                                                       dbsr_row_ptr_C,
                                                       row_block_dim_C,
                                                       col_block_dim_C,
                                                       &hnnzb_C,
                                                       dbuffer));

        std::cout << "AAAA" << std::endl;

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));

        auto dnnzb_C_managed = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
        int* dnnzb_C         = (int*)dnnzb_C_managed.get();
        CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsrNnz(handle,
                                                       dir,
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       descr_A,
                                                       dbsr_row_ptr_A,
                                                       dbsr_col_ind_A,
                                                       row_block_dim_A,
                                                       col_block_dim_A,
                                                       descr_C,
                                                       dbsr_row_ptr_C,
                                                       row_block_dim_C,
                                                       col_block_dim_C,
                                                       dnnzb_C,
                                                       dbuffer));

        int hnnzb_C_copied_from_device;
        CHECK_HIP_ERROR(
            hipMemcpy(&hnnzb_C_copied_from_device, dnnzb_C, sizeof(int), hipMemcpyDeviceToHost));

        std::cout << "hnnzb_C: " << hnnzb_C
                  << " hnnzb_C_copied_from_device: " << hnnzb_C_copied_from_device << std::endl;

        // Check that using host and device pointer mode gives the same result
        unit_check_general(1, 1, 1, &hnnzb_C_copied_from_device, &hnnzb_C);

        // Allocate memory on the device
        auto dbsr_col_ind_C_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * hnnzb_C), device_free};
        auto dbsr_val_C_managed = hipsparse_unique_ptr{
            device_malloc(sizeof(T) * hnnzb_C * row_block_dim_C * col_block_dim_C), device_free};

        int* dbsr_col_ind_C = (int*)dbsr_col_ind_C_managed.get();
        T*   dbsr_val_C     = (T*)dbsr_val_C_managed.get();

        if(!dbsr_col_ind_C || !dbsr_val_C)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!bsr_col_ind_C || !bsr_val_C");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        CHECK_HIPSPARSE_ERROR(hipsparseXgebsr2gebsr(handle,
                                                    dir,
                                                    mb,
                                                    nb,
                                                    nnzb,
                                                    descr_A,
                                                    dbsr_val_A,
                                                    dbsr_row_ptr_A,
                                                    dbsr_col_ind_A,
                                                    row_block_dim_A,
                                                    col_block_dim_A,
                                                    descr_C,
                                                    dbsr_val_C,
                                                    dbsr_row_ptr_C,
                                                    dbsr_col_ind_C,
                                                    row_block_dim_C,
                                                    col_block_dim_C,
                                                    dbuffer));

        // Copy output from device to host
        std::vector<int> hbsr_row_ptr_C(mb_C + 1);
        std::vector<int> hbsr_col_ind_C(hnnzb_C);
        std::vector<T>   hbsr_val_C(hnnzb_C * row_block_dim_C * col_block_dim_C);

        CHECK_HIP_ERROR(hipMemcpy(hbsr_row_ptr_C.data(),
                                  dbsr_row_ptr_C,
                                  sizeof(int) * (mb_C + 1),
                                  hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(
            hbsr_col_ind_C.data(), dbsr_col_ind_C, sizeof(int) * hnnzb_C, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hbsr_val_C.data(),
                                  dbsr_val_C,
                                  sizeof(T) * hnnzb_C * row_block_dim_C * col_block_dim_C,
                                  hipMemcpyDeviceToHost));

        // Host csr2bsr conversion
        std::vector<int> hbsr_row_ptr_gold(mb_C + 1);
        std::vector<int> hbsr_col_ind_gold(hnnzb_C, 0);
        std::vector<T>   hbsr_val_gold(hnnzb_C * row_block_dim_C * col_block_dim_C);

        // call host csr2bsr here
        // int nnzb_C_gold;
        // host_csr_to_bsr<T>(dir,
        //                    m,
        //                    n,
        //                    block_dim,
        //                    bsr_nnzb_gold,
        //                    csr_idx_base,
        //                    hcsr_row_ptr,
        //                    hcsr_col_ind,
        //                    hcsr_val,
        //                    bsr_idx_base,
        //                    hbsr_row_ptr_gold,
        //                    hbsr_col_ind_gold,
        //                    hbsr_val_gold);

        // // Unit check
        // unit_check_general(1, 1, 1, &bsr_nnzb_gold, &hbsr_nnzb);
        // unit_check_general(1, mb + 1, 1, hbsr_row_ptr_gold.data(), hbsr_row_ptr.data());
        // unit_check_general(1, hbsr_nnzb, 1, hbsr_col_ind_gold.data(), hbsr_col_ind.data());
        // unit_check_general(
        //     1, hbsr_nnzb * block_dim * block_dim, 1, hbsr_val_gold.data(), hbsr_val.data());
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_GEBSR2GEBSR_HPP
