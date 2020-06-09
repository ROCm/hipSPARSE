/* ************************************************************************
 * Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include "hipsparse.h"
#include <algorithm>
#include <assert.h>
#include <complex>
#include <hip/hip_runtime_api.h>
#include <math.h>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

/*!\file
 * \brief provide data initialization and timing utilities.
 */

#define CHECK_HIP_ERROR(error)                \
    if(error != hipSuccess)                   \
    {                                         \
        fprintf(stderr,                       \
                "error: '%s'(%d) at %s:%d\n", \
                hipGetErrorString(error),     \
                error,                        \
                __FILE__,                     \
                __LINE__);                    \
        exit(EXIT_FAILURE);                   \
    }

#define CHECK_HIPSPARSE_ERROR(error)                             \
    if(error != HIPSPARSE_STATUS_SUCCESS)                        \
    {                                                            \
        fprintf(stderr, "hipSPARSE error: ");                    \
        if(error == HIPSPARSE_STATUS_NOT_INITIALIZED)            \
        {                                                        \
            fprintf(stderr, "HIPSPARSE_STATUS_NOT_INITIALIZED"); \
        }                                                        \
        else if(error == HIPSPARSE_STATUS_INTERNAL_ERROR)        \
        {                                                        \
            fprintf(stderr, " HIPSPARSE_STATUS_INTERNAL_ERROR"); \
        }                                                        \
        else if(error == HIPSPARSE_STATUS_INVALID_VALUE)         \
        {                                                        \
            fprintf(stderr, "HIPSPARSE_STATUS_INVALID_VALUE");   \
        }                                                        \
        else if(error == HIPSPARSE_STATUS_ALLOC_FAILED)          \
        {                                                        \
            fprintf(stderr, "HIPSPARSE_STATUS_ALLOC_FAILED");    \
        }                                                        \
        else                                                     \
        {                                                        \
            fprintf(stderr, "HIPSPARSE_STATUS ERROR");           \
        }                                                        \
        fprintf(stderr, "\n");                                   \
        return error;                                            \
    }

#ifdef __HIP_PLATFORM_NVCC__
static inline hipComplex operator-(const hipComplex& op)
{
    hipComplex ret;
    ret.x = -op.x;
    ret.y = -op.y;
    return ret;
}
static inline hipDoubleComplex operator-(const hipDoubleComplex& op)
{
    hipDoubleComplex ret;
    ret.x = -op.x;
    ret.y = -op.y;
    return ret;
}

static inline bool operator==(const hipComplex& lhs, const hipComplex& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}
static inline bool operator==(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    return lhs.x == rhs.x && lhs.y == rhs.y;
}

static inline bool operator!=(const hipComplex& lhs, const hipComplex& rhs)
{
    return !(lhs == rhs);
}
static inline bool operator!=(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    return !(lhs == rhs);
}

static inline hipComplex operator+(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = lhs.x + rhs.x;
    ret.y = lhs.y + rhs.y;
    return ret;
}
static inline hipDoubleComplex operator+(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = lhs.x + rhs.x;
    ret.y = lhs.y + rhs.y;
    return ret;
}

static inline hipComplex operator-(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = lhs.x - rhs.x;
    ret.y = lhs.y - rhs.y;
    return ret;
}
static inline hipDoubleComplex operator-(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = lhs.x - rhs.x;
    ret.y = lhs.y - rhs.y;
    return ret;
}

static inline hipComplex operator*(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = lhs.x * rhs.x - lhs.y * rhs.y;
    ret.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return ret;
}
static inline hipDoubleComplex operator*(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = lhs.x * rhs.x - lhs.y * rhs.y;
    ret.y = lhs.x * rhs.y + lhs.y * rhs.x;
    return ret;
}

static inline hipComplex operator/(const hipComplex& lhs, const hipComplex& rhs)
{
    hipComplex ret;
    ret.x = (lhs.x * rhs.x + lhs.y * rhs.y);
    ret.y = (rhs.x * lhs.y - lhs.x * rhs.y);
    ret.x = ret.x / (rhs.x * rhs.x + rhs.y * rhs.y);
    ret.y = ret.y / (rhs.x * rhs.x + rhs.y * rhs.y);
    return ret;
}
static inline hipDoubleComplex operator/(const hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    hipDoubleComplex ret;
    ret.x = (lhs.x * rhs.x + lhs.y * rhs.y);
    ret.y = (rhs.x * lhs.y - lhs.x * rhs.y);
    ret.x = ret.x / (rhs.x * rhs.x + rhs.y * rhs.y);
    ret.y = ret.y / (rhs.x * rhs.x + rhs.y * rhs.y);
    return ret;
}

static inline hipComplex operator+=(hipComplex& lhs, const hipComplex& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
static inline hipDoubleComplex operator+=(hipDoubleComplex& lhs, const hipDoubleComplex& rhs)
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}
#endif

/* ============================================================================================ */
/*! \brief Make data type */
template <typename T>
inline T make_DataType2(double real, double imag)
{
    return static_cast<T>(real);
}

template <>
inline hipComplex make_DataType2(double real, double imag)
{
    return make_hipFloatComplex(static_cast<float>(real), static_cast<float>(imag));
}

template <>
inline hipDoubleComplex make_DataType2(double real, double imag)
{
    return make_hipDoubleComplex(real, imag);
}

template <typename T>
inline T make_DataType(double real, double imag = 0.0)
{
    return make_DataType2<T>(real, imag);
}

/* ============================================================================================ */
/*! \brief fma */
template <typename T>
inline T testing_fma(T p, T q, T r)
{
    return std::fma(p, q, r);
}

template <>
inline hipComplex testing_fma(hipComplex p, hipComplex q, hipComplex r)
{
    return hipCfmaf(p, q, r);
}

template <>
inline hipDoubleComplex testing_fma(hipDoubleComplex p, hipDoubleComplex q, hipDoubleComplex r)
{
    return hipCfma(p, q, r);
}

/* ============================================================================================ */
/*! \brief abs */
static inline float testing_abs(float x)
{
    return std::abs(x);
}

static inline double testing_abs(double x)
{
    return std::abs(x);
}

static inline float testing_abs(hipComplex x)
{
    return hipCabsf(x);
}

static inline double testing_abs(hipDoubleComplex x)
{
    return hipCabs(x);
}

/* ============================================================================================ */
/*! \brief real */
static inline float testing_real(float x)
{
    return std::real(x);
}

static inline double testing_real(double x)
{
    return std::real(x);
}

static inline float testing_real(hipComplex x)
{
    return hipCrealf(x);
}

static inline double testing_real(hipDoubleComplex x)
{
    return hipCreal(x);
}

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number between [0, 0.999...] . */
template <typename T>
inline T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return make_DataType<T>(rand() % 10 + 1,
                            rand() % 10 + 1); // generate a integer number between [1, 10]
};

/* ============================================================================================ */
/*! \brief  matrix/vector initialization: */
// for vector x (M=1, N=lengthX);
// for complex number, the real/imag part would be initialized with the same value
template <typename T>
void hipsparseInit(std::vector<T>& A, int M, int N)
{
    for(int i = 0; i < M; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            A[i + j] = random_generator<T>();
        }
    }
};

/* ============================================================================================ */
/*! \brief  vector initialization: */
// initialize sparse index vector with nnz entries ranging from start to end
template <typename I>
void hipsparseInitIndex(I* x, int nnz, int start, int end)
{
    std::vector<bool> check(end - start, false);
    int               num = 0;
    while(num < nnz)
    {
        int val = start + rand() % (end - start);
        if(!check[val - start])
        {
            x[num]             = val;
            check[val - start] = true;
            ++num;
        }
    }
    std::sort(x, x + nnz);
};

/* ============================================================================================ */
/*! \brief  csr matrix initialization */
template <typename T>
void hipsparseInitCSR(
    std::vector<int>& ptr, std::vector<int>& col, std::vector<T>& val, int nrow, int ncol, int nnz)
{
    // Row offsets
    ptr[0]    = 0;
    ptr[nrow] = nnz;

    for(int i = 1; i < nrow; ++i)
    {
        ptr[i] = rand() % (nnz - 1) + 1;
    }
    std::sort(ptr.begin(), ptr.end());

    // Column indices
    for(int i = 0; i < nrow; ++i)
    {
        hipsparseInitIndex(&col[ptr[i]], ptr[i + 1] - ptr[i], 0, ncol - 1);
        std::sort(&col[ptr[i]], &col[ptr[i + 1]]);
    }

    // Random values
    for(int i = 0; i < nnz; ++i)
    {
        val[i] = random_generator<T>();
    }
}

/* ============================================================================================ */
/*! \brief  Generate 2D laplacian on unit square in CSR format */
template <typename T>
int gen_2d_laplacian(int                  ndim,
                     std::vector<int>&    rowptr,
                     std::vector<int>&    col,
                     std::vector<T>&      val,
                     hipsparseIndexBase_t idx_base)
{
    if(ndim == 0)
    {
        return 0;
    }

    int n       = ndim * ndim;
    int nnz_mat = n * 5 - ndim * 4;

    rowptr.resize(n + 1);
    col.resize(nnz_mat);
    val.resize(nnz_mat);

    int nnz = 0;

    // Fill local arrays
    for(int i = 0; i < ndim; ++i)
    {
        for(int j = 0; j < ndim; ++j)
        {
            int idx     = i * ndim + j;
            rowptr[idx] = nnz + idx_base;
            // if no upper boundary element, connect with upper neighbor
            if(i != 0)
            {
                col[nnz] = idx - ndim + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if(j != 0)
            {
                col[nnz] = idx - 1 + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
            // element itself
            col[nnz] = idx + idx_base;
            val[nnz] = make_DataType<T>(4.0);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if(j != ndim - 1)
            {
                col[nnz] = idx + 1 + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if(i != ndim - 1)
            {
                col[nnz] = idx + ndim + idx_base;
                val[nnz] = make_DataType<T>(-1.0);
                ++nnz;
            }
        }
    }
    rowptr[n] = nnz + idx_base;

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate a random sparsity pattern with a dense format, generated floating point values of type T are positive and normalized. */
template <typename T>
void gen_dense_random_sparsity_pattern(int m, int n, T* A, int lda, float sparsity_ratio = 0.3)
{
    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            const float d  = ((float)rand()) / ((float)RAND_MAX);
            A[j * lda + i] = (d < sparsity_ratio)
                                 ? make_DataType<T>(rand()) / make_DataType<T>(RAND_MAX)
                                 : make_DataType<T>(0);
        }
    }
}

/* ============================================================================================ */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
void gen_matrix_coo(int                  m,
                    int                  n,
                    int                  nnz,
                    std::vector<int>&    row_ind,
                    std::vector<int>&    col_ind,
                    std::vector<T>&      val,
                    hipsparseIndexBase_t idx_base)
{
    if((int)row_ind.size() != nnz)
    {
        row_ind.resize(nnz);
    }
    if((int)col_ind.size() != nnz)
    {
        col_ind.resize(nnz);
    }
    if((int)val.size() != nnz)
    {
        val.resize(nnz);
    }

    // Uniform distributed row indices
    for(int i = 0; i < nnz; ++i)
    {
        row_ind[i] = rand() % m;
    }

    // Sort row indices
    std::sort(row_ind.begin(), row_ind.end());

    // Sample column indices
    std::vector<bool> check(nnz, false);

    int i = 0;
    while(i < nnz)
    {
        int begin = i;
        while(row_ind[i] == row_ind[begin])
        {
            ++i;
            if(i >= nnz)
            {
                break;
            }
        }

        // Sample i disjunct column indices
        int idx = begin;
        while(idx < i)
        {
            // Normal distribution around the diagonal
            int rng = (i - begin) * sqrt(-2.0 * log((double)rand() / RAND_MAX))
                      * cos(2.0 * M_PI * (double)rand() / RAND_MAX);

            if(m <= n)
            {
                rng += row_ind[begin];
            }

            // Repeat if running out of bounds
            if(rng < 0 || rng > n - 1)
            {
                continue;
            }

            // Check for disjunct column index in current row
            if(!check[rng])
            {
                check[rng]   = true;
                col_ind[idx] = rng;
                ++idx;
            }
        }

        // Reset disjunct check array
        for(int j = begin; j < i; ++j)
        {
            check[col_ind[j]] = false;
        }

        // Partially sort column indices
        std::sort(&col_ind[begin], &col_ind[i]);
    }

    // Correct index base accordingly
    if(idx_base == HIPSPARSE_INDEX_BASE_ONE)
    {
        for(int i = 0; i < nnz; ++i)
        {
            ++row_ind[i];
            ++col_ind[i];
        }
    }

    // Sample random values
    for(int i = 0; i < nnz; ++i)
    {
        val[i] = random_generator<T>(); //(double) rand() / RAND_MAX;
    }
}

/* ============================================================================================ */
/*! \brief  Read matrix from mtx file in COO format */
static inline void read_mtx_value(std::istringstream& is, int& row, int& col, float& val)
{
    is >> row >> col >> val;
}

static inline void read_mtx_value(std::istringstream& is, int& row, int& col, double& val)
{
    is >> row >> col >> val;
}

static inline void read_mtx_value(std::istringstream& is, int& row, int& col, hipComplex& val)
{
    float real;
    float imag;

    is >> row >> col >> real >> imag;

    val = make_DataType<hipComplex>(real, imag);
}

static inline void read_mtx_value(std::istringstream& is, int& row, int& col, hipDoubleComplex& val)
{
    double real;
    double imag;

    is >> row >> col >> real >> imag;

    val = make_DataType<hipDoubleComplex>(real, imag);
}

template <typename T>
int read_mtx_matrix(const char*          filename,
                    int&                 nrow,
                    int&                 ncol,
                    int&                 nnz,
                    std::vector<int>&    row,
                    std::vector<int>&    col,
                    std::vector<T>&      val,
                    hipsparseIndexBase_t idx_base)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("Reading matrix %s...", filename);
        fflush(stdout);
    }

    FILE* f = fopen(filename, "r");
    if(!f)
    {
        return -1;
    }

    char line[1024];

    // Check for banner
    if(!fgets(line, 1024, f))
    {
        return -1;
    }

    char banner[16];
    char array[16];
    char coord[16];
    char data[16];
    char type[16];

    // Extract banner
    if(sscanf(line, "%s %s %s %s %s", banner, array, coord, data, type) != 5)
    {
        return -1;
    }

    // Convert to lower case
    for(char* p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char* p = type; *p != '\0'; *p = tolower(*p), p++)
        ;

    // Check banner
    if(strncmp(line, "%%MatrixMarket", 14) != 0)
    {
        return -1;
    }

    // Check array type
    if(strcmp(array, "matrix") != 0)
    {
        return -1;
    }

    // Check coord
    if(strcmp(coord, "coordinate") != 0)
    {
        return -1;
    }

    // Check data
    if(strcmp(data, "real") != 0 && strcmp(data, "integer") != 0 && strcmp(data, "pattern") != 0)
    {
        return -1;
    }

    // Check type
    if(strcmp(type, "general") != 0 && strcmp(type, "symmetric") != 0)
    {
        return -1;
    }

    // Symmetric flag
    int symm = !strcmp(type, "symmetric");

    // Skip comments
    while(fgets(line, 1024, f))
    {
        if(line[0] != '%')
        {
            break;
        }
    }

    // Read dimensions
    int snnz;

    sscanf(line, "%d %d %d", &nrow, &ncol, &snnz);
    nnz = symm ? (snnz - nrow) * 2 + nrow : snnz;

    std::vector<int> unsorted_row(nnz);
    std::vector<int> unsorted_col(nnz);
    std::vector<T>   unsorted_val(nnz);

    // Read entries
    int idx = 0;
    while(fgets(line, 1024, f))
    {
        if(idx >= nnz)
        {
            return true;
        }

        int irow;
        int icol;
        T   ival;

        std::istringstream ss(line);

        if(!strcmp(data, "pattern"))
        {
            ss >> irow >> icol;
            ival = make_DataType<T>(1.0);
        }
        else
        {
            read_mtx_value(ss, irow, icol, ival);
        }

        if(idx_base == HIPSPARSE_INDEX_BASE_ZERO)
        {
            --irow;
            --icol;
        }

        unsorted_row[idx] = irow;
        unsorted_col[idx] = icol;
        unsorted_val[idx] = ival;

        ++idx;

        if(symm && irow != icol)
        {
            if(idx >= nnz)
            {
                return true;
            }

            unsorted_row[idx] = icol;
            unsorted_col[idx] = irow;
            unsorted_val[idx] = ival;
            ++idx;
        }
    }
    fclose(f);

    row.resize(nnz);
    col.resize(nnz);
    val.resize(nnz);

    // Sort by row and column index
    std::vector<int> perm(nnz);
    for(int i = 0; i < nnz; ++i)
    {
        perm[i] = i;
    }

    std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
        if(unsorted_row[a] < unsorted_row[b])
        {
            return true;
        }
        else if(unsorted_row[a] == unsorted_row[b])
        {
            return (unsorted_col[a] < unsorted_col[b]);
        }
        else
        {
            return false;
        }
    });

    for(int i = 0; i < nnz; ++i)
    {
        row[i] = unsorted_row[perm[i]];
        col[i] = unsorted_col[perm[i]];
        val[i] = unsorted_val[perm[i]];
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("done.\n");
        fflush(stdout);
    }

    return 0;
}

/* ============================================================================================ */
/*! \brief  Read matrix from binary file in CSR format */
template <typename T>
int read_bin_matrix(const char*          filename,
                    int&                 nrow,
                    int&                 ncol,
                    int&                 nnz,
                    std::vector<int>&    ptr,
                    std::vector<int>&    col,
                    std::vector<T>&      val,
                    hipsparseIndexBase_t idx_base)
{
    const char* env = getenv("GTEST_LISTENER");
    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("Reading matrix %s...", filename);
        fflush(stdout);
    }

    FILE* f = fopen(filename, "rb");
    if(!f)
    {
        return -1;
    }

    int err;

    err = fread(&nrow, sizeof(int), 1, f);
    err |= fread(&ncol, sizeof(int), 1, f);
    err |= fread(&nnz, sizeof(int), 1, f);

    // Allocate memory
    ptr.resize(nrow + 1);
    col.resize(nnz);
    val.resize(nnz);
    std::vector<double> tmp(nnz);

    err |= fread(ptr.data(), sizeof(int), nrow + 1, f);
    err |= fread(col.data(), sizeof(int), nnz, f);
    err |= fread(tmp.data(), sizeof(double), nnz, f);

    fclose(f);

    for(int i = 0; i < nnz; ++i)
    {
        val[i] = make_DataType<T>(tmp[i]);
    }

    if(idx_base == HIPSPARSE_INDEX_BASE_ONE)
    {
        for(int i = 0; i < nrow + 1; ++i)
        {
            ++ptr[i];
        }

        for(int i = 0; i < nnz; ++i)
        {
            ++col[i];
        }
    }

    if(!env || strcmp(env, "NO_PASS_LINE_IN_LOG"))
    {
        printf("done.\n");
        fflush(stdout);
    }

    return 0;
}

/* ============================================================================================ */
/*! \brief  Compute incomplete LU factorization without fill-ins and no pivoting using CSR
 *  matrix storage format.
 */
static inline float testing_neg(float val)
{
    return -val;
}

static inline double testing_neg(double val)
{
    return -val;
}

static inline hipComplex testing_neg(hipComplex val)
{
    hipComplex ret;
    ret.x = -val.x;
    ret.y = -val.y;
    return ret;
}

static inline hipDoubleComplex testing_neg(hipDoubleComplex val)
{
    hipDoubleComplex ret;
    ret.x = -val.x;
    ret.y = -val.y;
    return ret;
}

template <typename T>
void host_nnz(hipsparseDirection_t      dirA,
              int                       m,
              int                       n,
              const hipsparseMatDescr_t descrA,
              const T*                  A,
              int                       lda,
              int*                      nnzPerRowColumn,
              int*                      nnzTotalDevHostPtr)
{
    int mn = (dirA == HIPSPARSE_DIRECTION_ROW) ? m : n;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int j = 0; j < mn; ++j)
    {
        nnzPerRowColumn[j] = 0;
    }

    for(int j = 0; j < n; ++j)
    {
        for(int i = 0; i < m; ++i)
        {
            if(A[j * lda + i] != make_DataType<T>(0))
            {
                if(dirA == HIPSPARSE_DIRECTION_ROW)
                {
                    nnzPerRowColumn[i] += 1;
                }
                else
                {
                    nnzPerRowColumn[j] += 1;
                }
            }
        }
    }

    int sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : sum)
#endif
    for(int j = 0; j < mn; ++j)
    {
        sum = sum + nnzPerRowColumn[j];
    }
    nnzTotalDevHostPtr[0] = sum;
}

template <hipsparseDirection_t DIRA, typename T>
void host_dense2csx(int                  m,
                    int                  n,
                    hipsparseIndexBase_t base,
                    const T*             A,
                    int                  ld,
                    const int*           nnz_per_row_columns,
                    T*                   csx_val,
                    int*                 csx_row_col_ptr,
                    int*                 csx_col_row_ind)
{
    static constexpr T s_zero = {};
    int                len    = (HIPSPARSE_DIRECTION_ROW == DIRA) ? m : n;
    *csx_row_col_ptr          = base;
    for(int i = 0; i < len; ++i)
    {
        csx_row_col_ptr[i + 1] = nnz_per_row_columns[i] + csx_row_col_ptr[i];
    }

    switch(DIRA)
    {
    case HIPSPARSE_DIRECTION_COLUMN:
    {
        for(int j = 0; j < n; ++j)
        {
            for(int i = 0; i < m; ++i)
            {
                if(A[j * ld + i] != s_zero)
                {
                    *csx_val++         = A[j * ld + i];
                    *csx_col_row_ind++ = i + base;
                }
            }
        }
        break;
    }

    case HIPSPARSE_DIRECTION_ROW:
    {
        //
        // Does not matter having an orthogonal traversal ... testing only.
        // Otherwise, we would use csxRowPtrA to store the shifts.
        // and once the job is done a simple memory move would reinitialize the csxRowPtrA to its initial state)
        //
        for(int i = 0; i < m; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                if(A[j * ld + i] != s_zero)
                {
                    *csx_val++         = A[j * ld + i];
                    *csx_col_row_ind++ = j + base;
                }
            }
        }
        break;
    }
    }
}

template <hipsparseDirection_t DIRA, typename T>
void host_csx2dense(int                  m,
                    int                  n,
                    hipsparseIndexBase_t base,
                    const T*             csx_val,
                    const int*           csx_row_col_ptr,
                    const int*           csx_col_row_ind,
                    T*                   A,
                    int                  ld)
{
    static constexpr T s_zero = {};
    switch(DIRA)
    {
    case HIPSPARSE_DIRECTION_COLUMN:
    {
        static constexpr T s_zero = {};
        for(int col = 0; col < n; ++col)
        {
            for(int row = 0; row < m; ++row)
            {
                A[row + ld * col] = s_zero;
            }
            const int bound = csx_row_col_ptr[col + 1] - base;
            for(int at = csx_row_col_ptr[col] - base; at < bound; ++at)
            {
                A[(csx_col_row_ind[at] - base) + ld * col] = csx_val[at];
            }
        }
        break;
    }

    case HIPSPARSE_DIRECTION_ROW:
    {
        static constexpr T s_zero = {};
        for(int row = 0; row < m; ++row)
        {
            for(int col = 0; col < n; ++col)
            {
                A[col * ld + row] = s_zero;
            }

            const int bound = csx_row_col_ptr[row + 1] - base;
            for(int at = csx_row_col_ptr[row] - base; at < bound; ++at)
            {
                A[(csx_col_row_ind[at] - base) * ld + row] = csx_val[at];
            }
        }
        break;
    }
    }
}

template <typename T>
inline void host_csr_to_csr_compress(int                     M,
                                     int                     N,
                                     const std::vector<int>& csr_row_ptr_A,
                                     const std::vector<int>& csr_col_ind_A,
                                     const std::vector<T>&   csr_val_A,
                                     std::vector<int>&       csr_row_ptr_C,
                                     std::vector<int>&       csr_col_ind_C,
                                     std::vector<T>&         csr_val_C,
                                     hipsparseIndexBase_t    base,
                                     T                       tol)
{
    if(M <= 0 || N <= 0)
    {
        return;
    }

    // find how many entries will be in each compressed CSR matrix row
    std::vector<int> nnz_per_row(M);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; i++)
    {
        int start = csr_row_ptr_A[i] - base;
        int end   = csr_row_ptr_A[i + 1] - base;
        int count = 0;

        for(int j = start; j < end; j++)
        {
            if(testing_abs(csr_val_A[j]) > testing_real(tol)
               && testing_abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                count++;
            }
        }

        nnz_per_row[i] = count;
    }

    // add up total number of entries
    int nnz_C = 0;
    for(int i = 0; i < M; i++)
    {
        nnz_C += nnz_per_row[i];
    }

    //column indices and value arrays for compressed CSR matrix
    csr_col_ind_C.resize(nnz_C);
    csr_val_C.resize(nnz_C);

    // fill in row pointer array for compressed CSR matrix
    csr_row_ptr_C.resize(M + 1);

    csr_row_ptr_C[0] = base;
    for(int i = 0; i < M; i++)
    {
        csr_row_ptr_C[i + 1] = csr_row_ptr_C[i] + nnz_per_row[i];
    }

    // fill in column indices and value arrays for compressed CSR matrix
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; i++)
    {
        int start = csr_row_ptr_A[i] - base;
        int end   = csr_row_ptr_A[i + 1] - base;
        int index = csr_row_ptr_C[i] - base;

        for(int j = start; j < end; j++)
        {
            if(testing_abs(csr_val_A[j]) > testing_real(tol)
               && testing_abs(csr_val_A[j]) > std::numeric_limits<float>::min())
            {
                csr_col_ind_C[index] = csr_col_ind_A[j];
                csr_val_C[index]     = csr_val_A[j];
                index++;
            }
        }
    }
}

template <typename T>
inline void host_csr_to_csc(int                     M,
                            int                     N,
                            int                     nnz,
                            const std::vector<int>& csr_row_ptr,
                            const std::vector<int>& csr_col_ind,
                            const std::vector<T>&   csr_val,
                            std::vector<int>&       csc_row_ind,
                            std::vector<int>&       csc_col_ptr,
                            std::vector<T>&         csc_val,
                            hipsparseAction_t       action,
                            hipsparseIndexBase_t    base)
{
    csc_row_ind.resize(nnz);
    csc_col_ptr.resize(N + 1, 0);
    csc_val.resize(nnz);

    // Determine nnz per column
    for(int i = 0; i < nnz; ++i)
    {
        ++csc_col_ptr[csr_col_ind[i] + 1 - base];
    }

    // Scan
    for(int i = 0; i < N; ++i)
    {
        csc_col_ptr[i + 1] += csc_col_ptr[i];
    }

    // Fill row indices and values
    for(int i = 0; i < M; ++i)
    {
        int row_begin = csr_row_ptr[i] - base;
        int row_end   = csr_row_ptr[i + 1] - base;

        for(int j = row_begin; j < row_end; ++j)
        {
            int col = csr_col_ind[j] - base;
            int idx = csc_col_ptr[col];

            csc_row_ind[idx] = i + base;
            csc_val[idx]     = csr_val[j];

            ++csc_col_ptr[col];
        }
    }

    // Shift column pointer array
    for(int i = N; i > 0; --i)
    {
        csc_col_ptr[i] = csc_col_ptr[i - 1] + base;
    }

    csc_col_ptr[0] = base;
}

template <typename T>
inline void host_csr_to_bsr(hipsparseDirection_t    direction,
                            int                     M,
                            int                     N,
                            int                     block_dim,
                            int&                    nnzb,
                            hipsparseIndexBase_t    csr_base,
                            const std::vector<int>& csr_row_ptr,
                            const std::vector<int>& csr_col_ind,
                            const std::vector<T>&   csr_val,
                            hipsparseIndexBase_t    bsr_base,
                            std::vector<int>&       bsr_row_ptr,
                            std::vector<int>&       bsr_col_ind,
                            std::vector<T>&         bsr_val)
{
    int mb = (M + block_dim - 1) / block_dim;
    int nb = (N + block_dim - 1) / block_dim;

    // quick return if block_dim == 1
    if(block_dim == 1)
    {
        bsr_row_ptr.resize(mb + 1, 0);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_row_ptr.size(); i++)
        {
            bsr_row_ptr[i] = (csr_row_ptr[i] - csr_base) + bsr_base;
        }

        nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

        bsr_col_ind.resize(nnzb, 0);
        bsr_val.resize(nnzb * block_dim * block_dim, make_DataType<T>(0));
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_col_ind.size(); i++)
        {
            bsr_col_ind[i] = (csr_col_ind[i] - csr_base) + bsr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < csr_val.size(); i++)
        {
            bsr_val[i] = csr_val[i];
        }

        return;
    }

    // determine number of non-zero block columns for each block row of the bsr matrix
    bsr_row_ptr.resize(mb + 1, 0);

    bsr_row_ptr[0] = bsr_base;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < mb; i++)
    {
        int start = csr_row_ptr[i * block_dim] - csr_base;
        int end   = csr_row_ptr[std::min(M, block_dim * i + block_dim)] - csr_base;

        std::vector<int> temp(nb, 0);
        for(int j = start; j < end; j++)
        {
            int blockCol   = (csr_col_ind[j] - csr_base) / block_dim;
            temp[blockCol] = 1;
        }

        int sum = 0;
        for(int j = 0; j < temp.size(); j++)
        {
            sum += temp[j];
        }

        bsr_row_ptr[i + 1] = sum;
    }

    for(int i = 0; i < mb; i++)
    {
        bsr_row_ptr[i + 1] += bsr_row_ptr[i];
    }

    nnzb = bsr_row_ptr[mb] - bsr_row_ptr[0];

    // find bsr col indices array
    bsr_col_ind.resize(nnzb, 0);
    bsr_val.resize(nnzb * block_dim * block_dim, make_DataType<T>(0));

    int colIndex = 0;

    for(int i = 0; i < mb; i++)
    {
        int start = csr_row_ptr[i * block_dim] - csr_base;
        int end   = csr_row_ptr[std::min(M, block_dim * i + block_dim)] - csr_base;

        std::vector<int> temp(nb, 0);

        for(int j = start; j < end; j++)
        {
            int blockCol   = (csr_col_ind[j] - csr_base) / block_dim;
            temp[blockCol] = 1;
        }

        for(int j = 0; j < nb; j++)
        {
            if(temp[j] == 1)
            {
                bsr_col_ind[colIndex] = j + bsr_base;
                colIndex++;
            }
        }
    }

    // find bsr values array
    for(int i = 0; i < M; i++)
    {
        int blockRow = i / block_dim;

        int start = csr_row_ptr[i] - csr_base;
        int end   = csr_row_ptr[i + 1] - csr_base;

        for(int j = start; j < end; j++)
        {
            int blockCol = (csr_col_ind[j] - csr_base) / block_dim;

            colIndex = -1;
            for(int k = bsr_row_ptr[blockRow] - bsr_base; k < bsr_row_ptr[blockRow + 1] - bsr_base;
                k++)
            {
                if(bsr_col_ind[k] - bsr_base == blockCol)
                {
                    colIndex = k - (bsr_row_ptr[blockRow] - bsr_base);
                    break;
                }
            }

            assert(colIndex != -1);

            int blockIndex = 0;
            if(direction == HIPSPARSE_DIRECTION_ROW)
            {
                blockIndex = (csr_col_ind[j] - csr_base) % block_dim + (i % block_dim) * block_dim;
            }
            else
            {
                blockIndex
                    = ((csr_col_ind[j] - csr_base) % block_dim) * block_dim + (i % block_dim);
            }

            int index = (bsr_row_ptr[blockRow] - bsr_base) * block_dim * block_dim
                        + colIndex * block_dim * block_dim + blockIndex;

            bsr_val[index] = csr_val[j];
        }
    }
}

template <typename T>
inline void host_bsr_to_csr(hipsparseDirection_t    direction,
                            int                     Mb,
                            int                     Nb,
                            int                     block_dim,
                            hipsparseIndexBase_t    bsr_base,
                            const std::vector<int>& bsr_row_ptr,
                            const std::vector<int>& bsr_col_ind,
                            const std::vector<T>&   bsr_val,
                            hipsparseIndexBase_t    csr_base,
                            std::vector<int>&       csr_row_ptr,
                            std::vector<int>&       csr_col_ind,
                            std::vector<T>&         csr_val)
{
    int m    = Mb * block_dim;
    int n    = Nb * block_dim;
    int nnzb = bsr_row_ptr[Mb] - bsr_row_ptr[0];

    csr_row_ptr.resize(m + 1, 0);
    csr_col_ind.resize(nnzb * block_dim * block_dim, 0);
    csr_val.resize(nnzb * block_dim * block_dim, make_DataType<T>(0));

    // quick return if block_dim == 1
    if(block_dim == 1)
    {
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_row_ptr.size(); i++)
        {
            csr_row_ptr[i] = (bsr_row_ptr[i] - bsr_base) + csr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_col_ind.size(); i++)
        {
            csr_col_ind[i] = (bsr_col_ind[i] - bsr_base) + csr_base;
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(size_t i = 0; i < bsr_val.size(); i++)
        {
            csr_val[i] = bsr_val[i];
        }

        return;
    }

    csr_row_ptr[0] = csr_base;

    // find csr row ptr array
    for(int i = 0; i < Mb; i++)
    {
        int entries_in_row = block_dim * (bsr_row_ptr[i + 1] - bsr_row_ptr[i]);

        for(int j = 0; j < block_dim; j++)
        {
            csr_row_ptr[i * block_dim + j + 1] = csr_row_ptr[i * block_dim + j] + entries_in_row;
        }
    }

    int entries_in_block = block_dim * block_dim;

    // find csr col indices and values arrays
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < Mb; i++)
    {
        int entries_in_Row     = (bsr_row_ptr[i + 1] - bsr_row_ptr[i]) * block_dim;
        int entries_in_row_sum = (bsr_row_ptr[i] - bsr_base) * entries_in_block;

        for(int j = bsr_row_ptr[i] - bsr_base; j < bsr_row_ptr[i + 1] - bsr_base; j++)
        {
            int col    = bsr_col_ind[j] - bsr_base;
            int offset = entries_in_row_sum + block_dim * (j - (bsr_row_ptr[i] - bsr_base));

            for(int k = 0; k < block_dim; k++)
            {
                for(int l = 0; l < block_dim; l++)
                {
                    csr_col_ind[offset + k * entries_in_Row + l] = block_dim * col + l + csr_base;
                    if(direction == HIPSPARSE_DIRECTION_ROW)
                    {
                        csr_val[offset + k * entries_in_Row + l]
                            = bsr_val[j * entries_in_block + k * block_dim + l];
                    }
                    else
                    {
                        csr_val[offset + k * entries_in_Row + l]
                            = bsr_val[j * entries_in_block + k + block_dim * l];
                    }
                }
            }
        }
    }
}

template <typename T>
inline void host_bsrmv(hipsparseDirection_t dir,
                       hipsparseOperation_t trans,
                       int                  mb,
                       int                  nb,
                       int                  nnzb,
                       T                    alpha,
                       const int*           bsr_row_ptr,
                       const int*           bsr_col_ind,
                       const T*             bsr_val,
                       int                  bsr_dim,
                       const T*             x,
                       T                    beta,
                       T*                   y,
                       hipsparseIndexBase_t base)
{
    // Quick return
    if(alpha == make_DataType<T>(0))
    {
        if(beta != make_DataType<T>(1))
        {
            for(int i = 0; i < mb * bsr_dim; ++i)
            {
                y[i] = beta * y[i];
            }
        }

        return;
    }

    int WFSIZE;

    if(bsr_dim == 2)
    {
        int blocks_per_row = nnzb / mb;

        if(blocks_per_row < 8)
        {
            WFSIZE = 4;
        }
        else if(blocks_per_row < 16)
        {
            WFSIZE = 8;
        }
        else if(blocks_per_row < 32)
        {
            WFSIZE = 16;
        }
        else if(blocks_per_row < 64)
        {
            WFSIZE = 32;
        }
        else
        {
            WFSIZE = 64;
        }
    }
    else if(bsr_dim <= 8)
    {
        WFSIZE = 8;
    }
    else if(bsr_dim <= 16)
    {
        WFSIZE = 16;
    }
    else
    {
        WFSIZE = 32;
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int row = 0; row < mb; ++row)
    {
        int row_begin = bsr_row_ptr[row] - base;
        int row_end   = bsr_row_ptr[row + 1] - base;

        if(bsr_dim == 2)
        {
            std::vector<T> sum0(WFSIZE, make_DataType<T>(0));
            std::vector<T> sum1(WFSIZE, make_DataType<T>(0));

            for(int j = row_begin; j < row_end; j += WFSIZE)
            {
                for(int k = 0; k < WFSIZE; ++k)
                {
                    if(j + k < row_end)
                    {
                        int col = bsr_col_ind[j + k] - base;

                        if(dir == HIPSPARSE_DIRECTION_COLUMN)
                        {
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                                  x[col * bsr_dim + 0],
                                                  sum0[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                                  x[col * bsr_dim + 0],
                                                  sum1[k]);
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                                  x[col * bsr_dim + 1],
                                                  sum0[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                                  x[col * bsr_dim + 1],
                                                  sum1[k]);
                        }
                        else
                        {
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 0],
                                                  x[col * bsr_dim + 0],
                                                  sum0[k]);
                            sum0[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 1],
                                                  x[col * bsr_dim + 1],
                                                  sum0[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 2],
                                                  x[col * bsr_dim + 0],
                                                  sum1[k]);
                            sum1[k] = testing_fma(bsr_val[bsr_dim * bsr_dim * (j + k) + 3],
                                                  x[col * bsr_dim + 1],
                                                  sum1[k]);
                        }
                    }
                }
            }

            for(unsigned int j = 1; j < WFSIZE; j <<= 1)
            {
                for(unsigned int k = 0; k < WFSIZE - j; ++k)
                {
                    sum0[k] = sum0[k] + sum0[k + j];
                    sum1[k] = sum1[k] + sum1[k + j];
                }
            }

            if(beta != make_DataType<T>(0))
            {
                y[row * bsr_dim + 0] = testing_fma(beta, y[row * bsr_dim + 0], alpha * sum0[0]);
                y[row * bsr_dim + 1] = testing_fma(beta, y[row * bsr_dim + 1], alpha * sum1[0]);
            }
            else
            {
                y[row * bsr_dim + 0] = alpha * sum0[0];
                y[row * bsr_dim + 1] = alpha * sum1[0];
            }
        }
        else
        {
            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                std::vector<T> sum(WFSIZE, make_DataType<T>(0));

                for(int j = row_begin; j < row_end; ++j)
                {
                    int col = bsr_col_ind[j] - base;

                    for(int bj = 0; bj < bsr_dim; bj += WFSIZE)
                    {
                        for(unsigned int k = 0; k < WFSIZE; ++k)
                        {
                            if(bj + k < bsr_dim)
                            {
                                if(dir == HIPSPARSE_DIRECTION_COLUMN)
                                {
                                    sum[k] = testing_fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * (bj + k) + bi],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                                else
                                {
                                    sum[k] = testing_fma(
                                        bsr_val[bsr_dim * bsr_dim * j + bsr_dim * bi + (bj + k)],
                                        x[bsr_dim * col + (bj + k)],
                                        sum[k]);
                                }
                            }
                        }
                    }
                }

                for(unsigned int j = 1; j < WFSIZE; j <<= 1)
                {
                    for(unsigned int k = 0; k < WFSIZE - j; ++k)
                    {
                        sum[k] = sum[k] + sum[k + j];
                    }
                }

                if(beta != make_DataType<T>(0))
                {
                    y[row * bsr_dim + bi]
                        = testing_fma(beta, y[row * bsr_dim + bi], alpha * sum[0]);
                }
                else
                {
                    y[row * bsr_dim + bi] = alpha * sum[0];
                }
            }
        }
    }
}

template <typename T>
inline void host_bsrmm(int                     Mb,
                       int                     N,
                       int                     Kb,
                       int                     block_dim,
                       hipsparseDirection_t               dir,
                       hipsparseOperation_t               transA,
                       hipsparseOperation_t               transB,
                       T                                 alpha,
                       const std::vector<int>& bsr_row_ptr_A,
                       const std::vector<int>& bsr_col_ind_A,
                       const std::vector<T>&             bsr_val_A,
                       const std::vector<T>&             B,
                       int                     ldb,
                       T                                 beta,
                       std::vector<T>&                   C,
                       int                     ldc,
                       hipsparseIndexBase_t              base)
{
    if(transA != HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        return;
    }

    if(transB != HIPSPARSE_OPERATION_NON_TRANSPOSE && transB != HIPSPARSE_OPERATION_TRANSPOSE)
    {
        return;
    }

    int M = Mb * block_dim;
    int K = Kb * block_dim;

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; i++)
    {
        int local_row = i % block_dim;

        int row_begin = bsr_row_ptr_A[i / block_dim] - base;
        int row_end   = bsr_row_ptr_A[i / block_dim + 1] - base;

        for(int j = 0; j < N; j++)
        {
            int idx_C = i + j * ldc;

            T sum = static_cast<T>(0);

            for(int s = row_begin; s < row_end; s++)
            {
                for(int t = 0; t < block_dim; t++)
                {
                    int idx_A
                        = (dir == HIPSPARSE_DIRECTION_ROW)
                              ? block_dim * block_dim * s + block_dim * local_row + t
                              : block_dim * block_dim * s + block_dim * t + local_row;
                    int idx_B
                        = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE)
                              ? j * ldb + block_dim * (bsr_col_ind_A[s] - base) + t
                              : (block_dim * (bsr_col_ind_A[s] - base) + t) * ldb + j;

                    sum = sum + alpha * bsr_val_A[idx_A] * B[idx_B];
                }
            }

            if(beta == static_cast<T>(0))
            {
                C[idx_C] = sum;
            }
            else
            {
                C[idx_C] = sum + beta * C[idx_C]; 
            }
        }
    }
}

template <typename T>
int csrilu0(int m, const int* ptr, const int* col, T* val, hipsparseIndexBase_t idx_base)
{
    // pointer of upper part of each row
    std::vector<int> diag_offset(m);
    std::vector<int> nnz_entries(m, 0);

    // ai = 0 to N loop over all rows
    for(int ai = 0; ai < m; ++ai)
    {
        // ai-th row entries
        int row_start = ptr[ai] - idx_base;
        int row_end   = ptr[ai + 1] - idx_base;
        int j;

        // nnz position of ai-th row in val array
        for(j = row_start; j < row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = j;
        }

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_start; j < row_end; ++j)
        {
            // if nnz entry is in lower matrix
            if(col[j] - idx_base < ai)
            {

                int col_j  = col[j] - idx_base;
                int diag_j = diag_offset[col_j];

                if(val[diag_j] != make_DataType<T>(0.0))
                {
                    // multiplication factor
                    val[j] = val[j] / val[diag_j];

                    // loop over upper offset pointer and do linear combination for nnz entry
                    for(int k = diag_j + 1; k < ptr[col_j + 1] - idx_base; ++k)
                    {
                        // if nnz at this position do linear combination
                        if(nnz_entries[col[k] - idx_base] != 0)
                        {
                            int idx  = nnz_entries[col[k] - idx_base];
                            val[idx] = testing_fma(testing_neg(val[j]), val[k], val[idx]);
                        }
                    }
                }
                else
                {
                    // Numerical zero diagonal
                    return col_j + idx_base;
                }
            }
            else if(col[j] - idx_base == ai)
            {
                has_diag = true;
                break;
            }
            else
            {
                break;
            }
        }

        if(!has_diag)
        {
            // Structural zero digonal
            return ai + idx_base;
        }

        // set diagonal pointer to diagonal element
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_start; j < row_end; ++j)
        {
            nnz_entries[col[j] - idx_base] = 0;
        }
    }

    return -1;
}

template <typename T>
void csric0(int                  M,
            const int*           csr_row_ptr,
            const int*           csr_col_ind,
            T*                   csr_val,
            hipsparseIndexBase_t idx_base,
            int&                 struct_pivot,
            int&                 numeric_pivot)
{
    // Initialize pivot
    struct_pivot  = -1;
    numeric_pivot = -1;

    // pointer of upper part of each row
    std::vector<int> diag_offset(M);
    std::vector<int> nnz_entries(M, 0);

    // ai = 0 to N loop over all rows
    for(int ai = 0; ai < M; ++ai)
    {
        // ai-th row entries
        int row_begin = csr_row_ptr[ai] - idx_base;
        int row_end   = csr_row_ptr[ai + 1] - idx_base;
        int j;

        // nnz position of ai-th row in val array
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - idx_base] = j;
        }

        T sum = make_DataType<T>(0.0);

        bool has_diag = false;

        // loop over ai-th row nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            int col_j = csr_col_ind[j] - idx_base;
            T   val_j = csr_val[j];

            // Mark diagonal and skip row
            if(col_j == ai)
            {
                has_diag = true;
                break;
            }

            // Skip upper triangular
            if(col_j > ai)
            {
                break;
            }

            int row_begin_j = csr_row_ptr[col_j] - idx_base;
            int row_diag_j  = diag_offset[col_j];

            T local_sum = make_DataType<T>(0.0);
            T inv_diag  = csr_val[row_diag_j];

            // Check for numeric zero
            if(inv_diag == make_DataType<T>(0.0))
            {
                // Numerical zero diagonal
                numeric_pivot = col_j + idx_base;
                return;
            }

            inv_diag = make_DataType<T>(1.0) / inv_diag;

            // loop over upper offset pointer and do linear combination for nnz entry
            for(int k = row_begin_j; k < row_diag_j; ++k)
            {
                int col_k = csr_col_ind[k] - idx_base;

                // if nnz at this position do linear combination
                if(nnz_entries[col_k] != 0)
                {
                    int idx   = nnz_entries[col_k];
                    local_sum = testing_fma(csr_val[k], csr_val[idx], local_sum);
                }
            }

            val_j = (val_j - local_sum) * inv_diag;
            sum   = testing_fma(val_j, val_j, sum);

            csr_val[j] = val_j;
        }

        if(!has_diag)
        {
            // Structural (and numerical) zero diagonal
            struct_pivot  = ai + idx_base;
            numeric_pivot = ai + idx_base;
            return;
        }

        // Process diagonal entry
        T diag_entry = make_DataType<T>(std::sqrt(testing_abs(csr_val[j] - sum)));
        csr_val[j]   = diag_entry;

        // Store diagonal offset
        diag_offset[ai] = j;

        // clear nnz entries
        for(j = row_begin; j < row_end; ++j)
        {
            nnz_entries[csr_col_ind[j] - idx_base] = 0;
        }
    }
}

/* ============================================================================================ */
/*! \brief  Sparse triangular system solve using CSR storage format. */
template <typename T>
static inline void host_lssolve(int                     M,
                                int                     nrhs,
                                hipsparseOperation_t    transB,
                                T                       alpha,
                                const std::vector<int>& csr_row_ptr,
                                const std::vector<int>& csr_col_ind,
                                const std::vector<T>&   csr_val,
                                std::vector<T>&         B,
                                int                     ldb,
                                hipsparseDiagType_t     diag_type,
                                hipsparseIndexBase_t    base,
                                int&                    struct_pivot,
                                int&                    numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < nrhs; ++i)
    {
        std::vector<T> temp(prop.warpSize);

        // Process lower triangular part
        for(int row = 0; row < M; ++row)
        {
            temp.assign(prop.warpSize, make_DataType<T>(0.0));

            int idx_B
                = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldb + row : row * ldb + i;
            temp[0] = alpha * B[idx_B];

            int diag      = -1;
            int row_begin = csr_row_ptr[row] - base;
            int row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = make_DataType<T>(0.0);

            for(int l = row_begin; l < row_end; l += prop.warpSize)
            {
                for(unsigned int k = 0; k < prop.warpSize; ++k)
                {
                    int j = l + k;

                    // Do not run out of bounds
                    if(j >= row_end)
                    {
                        break;
                    }

                    int local_col = csr_col_ind[j] - base;
                    T   local_val = csr_val[j];

                    if(local_val == make_DataType<T>(0.0) && local_col == row
                       && diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        // Numerical zero pivot found, avoid division by 0 and store
                        // index for later use
                        numeric_pivot = std::min(numeric_pivot, row + base);
                        local_val     = make_DataType<T>(1.0);
                    }

                    // Ignore all entries that are above the diagonal
                    if(local_col > row)
                    {
                        break;
                    }

                    // Diagonal entry
                    if(local_col == row)
                    {
                        // If diagonal type is non unit, do division by diagonal entry
                        // This is not required for unit diagonal for obvious reasons
                        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                        {
                            diag     = j;
                            diag_val = make_DataType<T>(1.0) / local_val;
                        }

                        break;
                    }

                    // Lower triangular part
                    int idx = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldb + local_col
                                                                            : local_col * ldb + i;
                    T neg_val = make_DataType<T>(-1.0) * local_val;
                    temp[k]   = testing_fma(neg_val, B[idx], temp[k]);
                }
            }

            for(unsigned int j = 1; j < prop.warpSize; j <<= 1)
            {
                for(unsigned int k = 0; k < prop.warpSize - j; ++k)
                {
                    temp[k] = temp[k] + temp[k + j];
                }
            }

            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
            {
                if(diag == -1)
                {
                    struct_pivot = std::min(struct_pivot, row + base);
                }

                B[idx_B] = temp[0] * diag_val;
            }
            else
            {
                B[idx_B] = temp[0];
            }
        }
    }
}

template <typename T>
static inline void host_ussolve(int                     M,
                                int                     nrhs,
                                hipsparseOperation_t    transB,
                                T                       alpha,
                                const std::vector<int>& csr_row_ptr,
                                const std::vector<int>& csr_col_ind,
                                const std::vector<T>&   csr_val,
                                std::vector<T>&         B,
                                int                     ldb,
                                hipsparseDiagType_t     diag_type,
                                hipsparseIndexBase_t    base,
                                int&                    struct_pivot,
                                int&                    numeric_pivot)
{
    // Get device properties
    int             dev;
    hipDeviceProp_t prop;

    hipGetDevice(&dev);
    hipGetDeviceProperties(&prop, dev);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < nrhs; ++i)
    {
        std::vector<T> temp(prop.warpSize);

        // Process upper triangular part
        for(int row = M - 1; row >= 0; --row)
        {
            temp.assign(prop.warpSize, make_DataType<T>(0.0));

            int idx_B
                = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldb + row : row * ldb + i;
            temp[0] = alpha * B[idx_B];

            int diag      = -1;
            int row_begin = csr_row_ptr[row] - base;
            int row_end   = csr_row_ptr[row + 1] - base;

            T diag_val = make_DataType<T>(0.0);

            for(int l = row_end - 1; l >= row_begin; l -= prop.warpSize)
            {
                for(unsigned int k = 0; k < prop.warpSize; ++k)
                {
                    int j = l - k;

                    // Do not run out of bounds
                    if(j < row_begin)
                    {
                        break;
                    }

                    int local_col = csr_col_ind[j] - base;
                    T   local_val = csr_val[j];

                    // Ignore all entries that are below the diagonal
                    if(local_col < row)
                    {
                        continue;
                    }

                    // Diagonal entry
                    if(local_col == row)
                    {
                        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                        {
                            // Check for numerical zero
                            if(local_val == make_DataType<T>(0.0))
                            {
                                numeric_pivot = std::min(numeric_pivot, row + base);
                                local_val     = make_DataType<T>(1.0);
                            }

                            diag     = j;
                            diag_val = make_DataType<T>(1.0) / local_val;
                        }

                        continue;
                    }

                    // Upper triangular part
                    int idx = (transB == HIPSPARSE_OPERATION_NON_TRANSPOSE) ? i * ldb + local_col
                                                                            : local_col * ldb + i;
                    T neg_val = make_DataType<T>(-1.0) * local_val;
                    temp[k]   = testing_fma(neg_val, B[idx], temp[k]);
                }
            }

            for(unsigned int j = 1; j < prop.warpSize; j <<= 1)
            {
                for(unsigned int k = 0; k < prop.warpSize - j; ++k)
                {
                    temp[k] = temp[k] + temp[k + j];
                }
            }

            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
            {
                if(diag == -1)
                {
                    struct_pivot = std::min(struct_pivot, row + base);
                }

                B[idx_B] = temp[0] * diag_val;
            }
            else
            {
                B[idx_B] = temp[0];
            }
        }
    }
}

template <typename T>
void csrsm(int                     M,
           int                     nrhs,
           int                     nnz,
           hipsparseOperation_t    transA,
           hipsparseOperation_t    transB,
           T                       alpha,
           const std::vector<int>& csr_row_ptr,
           const std::vector<int>& csr_col_ind,
           const std::vector<T>&   csr_val,
           std::vector<T>&         B,
           int                     ldb,
           hipsparseDiagType_t     diag_type,
           hipsparseFillMode_t     fill_mode,
           hipsparseIndexBase_t    base,
           int&                    struct_pivot,
           int&                    numeric_pivot)
{
    // Initialize pivot
    struct_pivot  = M + 1;
    numeric_pivot = M + 1;

    if(transA == HIPSPARSE_OPERATION_NON_TRANSPOSE)
    {
        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            host_lssolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
        else
        {
            host_ussolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csr_row_ptr,
                         csr_col_ind,
                         csr_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
    }
    else if(transA == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        // Transpose matrix
        std::vector<int> csrt_row_ptr(M + 1);
        std::vector<int> csrt_col_ind(nnz);
        std::vector<T>   csrt_val(nnz);

        host_csr_to_csc(M,
                        M,
                        nnz,
                        csr_row_ptr,
                        csr_col_ind,
                        csr_val,
                        csrt_col_ind,
                        csrt_row_ptr,
                        csrt_val,
                        HIPSPARSE_ACTION_NUMERIC,
                        base);

        if(fill_mode == HIPSPARSE_FILL_MODE_LOWER)
        {
            host_ussolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csrt_row_ptr,
                         csrt_col_ind,
                         csrt_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
        else
        {
            host_lssolve(M,
                         nrhs,
                         transB,
                         alpha,
                         csrt_row_ptr,
                         csrt_col_ind,
                         csrt_val,
                         B,
                         ldb,
                         diag_type,
                         base,
                         struct_pivot,
                         numeric_pivot);
        }
    }

    numeric_pivot = std::min(numeric_pivot, struct_pivot);

    struct_pivot  = (struct_pivot == M + 1) ? -1 : struct_pivot;
    numeric_pivot = (numeric_pivot == M + 1) ? -1 : numeric_pivot;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular lower solve using BSR storage format. */
template <typename T>
int bsr_lsolve(hipsparseDirection_t dir,
               hipsparseOperation_t trans,
               int                  mb,
               const int*           ptr,
               const int*           col,
               const T*             val,
               int                  bsr_dim,
               T                    alpha,
               const T*             x,
               T*                   y,
               hipsparseIndexBase_t base,
               hipsparseDiagType_t  diag_type)
{
    const int* bsr_row_ptr = ptr;
    const int* bsr_col_ind = col;
    const T*   bsr_val     = val;

    std::vector<int> vptr;
    std::vector<int> vcol;
    std::vector<T>   vval;

    if(trans == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        int nnzb = ptr[mb] - base;

        vptr.resize(mb + 1);
        vcol.resize(nnzb);
        vval.resize(nnzb * bsr_dim * bsr_dim);

        // Transpose
        transpose_bsr(mb,
                      mb,
                      nnzb,
                      bsr_dim,
                      ptr,
                      col,
                      val,
                      vptr.data(),
                      vcol.data(),
                      vval.data(),
                      base,
                      base);

        bsr_row_ptr = vptr.data();
        bsr_col_ind = vcol.data();
        bsr_val     = vval.data();
    }

    int pivot = std::numeric_limits<int>::max();

    // Process lower triangular part
    for(int bsr_row = 0; bsr_row < mb; ++bsr_row)
    {
        int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
        int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

        for(int bi = 0; bi < bsr_dim; ++bi)
        {
            int local_row = bsr_row * bsr_dim + bi;

            T sum = alpha * x[local_row];

            int diag     = -1;
            T   diag_val = make_DataType<T>(0);

            for(int j = bsr_row_begin; j < bsr_row_end; ++j)
            {
                int bsr_col = bsr_col_ind[j] - base;

                for(int bj = 0; bj < bsr_dim; ++bj)
                {
                    int local_col = bsr_col * bsr_dim + bj;
                    T   local_val = dir == HIPSPARSE_DIRECTION_ROW
                                      ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                      : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                    // Ignore all entries that are above the diagonal
                    if(local_col > local_row)
                    {
                        break;
                    }

                    // Diagonal
                    if(local_col == local_row)
                    {
                        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                        {
                            // Check for numerical zero
                            if(local_val == make_DataType<T>(0))
                            {
                                pivot     = std::min(pivot, bsr_row + base);
                                local_val = make_DataType<T>(1);
                            }

                            diag     = j;
                            diag_val = make_DataType<T>(1) / local_val;
                        }

                        break;
                    }

                    // Lower triangular part
                    sum = testing_fma(-local_val, y[local_col], sum);
                }
            }

            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
            {
                if(diag == -1)
                {
                    pivot = std::min(pivot, bsr_row + base);
                }

                y[local_row] = sum * diag_val;
            }
            else
            {
                y[local_row] = sum;
            }
        }
    }

    if(pivot != std::numeric_limits<int>::max())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular upper solve using BSR storage format. */
template <typename T>
int bsr_usolve(hipsparseDirection_t dir,
               hipsparseOperation_t trans,
               int                  mb,
               const int*           ptr,
               const int*           col,
               const T*             val,
               int                  bsr_dim,
               T                    alpha,
               const T*             x,
               T*                   y,
               hipsparseIndexBase_t base,
               hipsparseDiagType_t  diag_type)
{
    const int* bsr_row_ptr = ptr;
    const int* bsr_col_ind = col;
    const T*   bsr_val     = val;

    std::vector<int> vptr;
    std::vector<int> vcol;
    std::vector<T>   vval;

    if(trans == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        int nnzb = ptr[mb] - base;

        vptr.resize(mb + 1);
        vcol.resize(nnzb);
        vval.resize(nnzb * bsr_dim * bsr_dim);

        // Transpose
        transpose_bsr(mb,
                      mb,
                      nnzb,
                      bsr_dim,
                      ptr,
                      col,
                      val,
                      vptr.data(),
                      vcol.data(),
                      vval.data(),
                      base,
                      base);

        bsr_row_ptr = vptr.data();
        bsr_col_ind = vcol.data();
        bsr_val     = vval.data();
    }

    int pivot = std::numeric_limits<int>::max();

    // Process upper triangular part
    for(int bsr_row = mb - 1; bsr_row >= 0; --bsr_row)
    {
        int bsr_row_begin = bsr_row_ptr[bsr_row] - base;
        int bsr_row_end   = bsr_row_ptr[bsr_row + 1] - base;

        for(int bi = bsr_dim - 1; bi >= 0; --bi)
        {
            int local_row = bsr_row * bsr_dim + bi;

            T sum = alpha * x[local_row];

            int diag     = -1;
            T   diag_val = make_DataType<T>(0);

            for(int j = bsr_row_end - 1; j >= bsr_row_begin; --j)
            {
                int bsr_col = bsr_col_ind[j] - base;

                for(int bj = bsr_dim - 1; bj >= 0; --bj)
                {
                    int local_col = bsr_col * bsr_dim + bj;
                    T   local_val = dir == HIPSPARSE_DIRECTION_ROW
                                      ? bsr_val[bsr_dim * bsr_dim * j + bi * bsr_dim + bj]
                                      : bsr_val[bsr_dim * bsr_dim * j + bi + bj * bsr_dim];

                    // Ignore all entries that are below the diagonal
                    if(local_col < local_row)
                    {
                        continue;
                    }

                    // Diagonal
                    if(local_col == local_row)
                    {
                        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                        {
                            // Check for numerical zero
                            if(local_val == make_DataType<T>(0))
                            {
                                pivot     = std::min(pivot, bsr_row + base);
                                local_val = make_DataType<T>(1);
                            }

                            diag     = j;
                            diag_val = make_DataType<T>(1) / local_val;
                        }

                        continue;
                    }

                    // Upper triangular part
                    sum = testing_fma(-local_val, y[local_col], sum);
                }
            }

            if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
            {
                if(diag == -1)
                {
                    pivot = std::min(pivot, bsr_row + base);
                }

                y[local_row] = sum * diag_val;
            }
            else
            {
                y[local_row] = sum;
            }
        }
    }

    if(pivot != std::numeric_limits<int>::max())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular lower solve using CSR storage format. */
template <typename T>
int csr_lsolve(hipsparseOperation_t trans,
               int                  m,
               const int*           ptr,
               const int*           col,
               const T*             val,
               T                    alpha,
               const T*             x,
               T*                   y,
               hipsparseIndexBase_t idx_base,
               hipsparseDiagType_t  diag_type,
               unsigned int         wf_size)
{
    const int* csr_row_ptr = ptr;
    const int* csr_col_ind = col;
    const T*   csr_val     = val;

    std::vector<int> vptr;
    std::vector<int> vcol;
    std::vector<T>   vval;

    if(trans == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        int nnz = ptr[m] - idx_base;

        vptr.resize(m + 1);
        vcol.resize(nnz);
        vval.resize(nnz);

        // Transpose
        transpose_csr(
            m, m, nnz, ptr, col, val, vptr.data(), vcol.data(), vval.data(), idx_base, idx_base);

        csr_row_ptr = vptr.data();
        csr_col_ind = vcol.data();
        csr_val     = vval.data();
    }

    int            pivot = std::numeric_limits<int>::max();
    std::vector<T> temp(wf_size);

    for(int i = 0; i < m; ++i)
    {
        temp.assign(wf_size, make_DataType<T>(0.0));
        temp[0] = alpha * x[i];

        int diag      = -1;
        int row_begin = csr_row_ptr[i] - idx_base;
        int row_end   = csr_row_ptr[i + 1] - idx_base;

        T diag_val = make_DataType<T>(0.0);

        for(unsigned int l = row_begin; l < row_end; l += wf_size)
        {
            for(unsigned int k = 0; k < wf_size; ++k)
            {
                int j = l + k;

                // Do not run out of bounds
                if(j >= row_end)
                {
                    break;
                }

                int col_j = csr_col_ind[j] - idx_base;
                T   val_j = csr_val[j];

                if(col_j < i)
                {
                    // Lower part
                    temp[k] = testing_fma(-csr_val[j], y[col_j], temp[k]);
                }
                else if(col_j == i)
                {
                    // Diagonal
                    if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        // Check for numerical zero
                        if(val_j == make_DataType<T>(0.0))
                        {
                            pivot = std::min(pivot, i + idx_base);
                            val_j = make_DataType<T>(1.0);
                        }

                        diag     = j;
                        diag_val = make_DataType<T>(1.0) / val_j;
                    }

                    break;
                }
                else
                {
                    // Upper part
                    break;
                }
            }
        }

        for(unsigned int j = 1; j < wf_size; j <<= 1)
        {
            for(unsigned int k = 0; k < wf_size - j; ++k)
            {
                temp[k] = temp[k] + temp[k + j];
            }
        }

        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
        {
            if(diag == -1)
            {
                pivot = std::min(pivot, i + idx_base);
            }

            y[i] = temp[0] * diag_val;
        }
        else
        {
            y[i] = temp[0];
        }
    }

    if(pivot != std::numeric_limits<int>::max())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Sparse triangular upper solve using CSR storage format. */
template <typename T>
int csr_usolve(hipsparseOperation_t trans,
               int                  m,
               const int*           ptr,
               const int*           col,
               const T*             val,
               T                    alpha,
               const T*             x,
               T*                   y,
               hipsparseIndexBase_t idx_base,
               hipsparseDiagType_t  diag_type,
               unsigned int         wf_size)
{
    const int* csr_row_ptr = ptr;
    const int* csr_col_ind = col;
    const T*   csr_val     = val;

    std::vector<int> vptr;
    std::vector<int> vcol;
    std::vector<T>   vval;

    if(trans == HIPSPARSE_OPERATION_TRANSPOSE)
    {
        int nnz = ptr[m] - idx_base;

        vptr.resize(m + 1);
        vcol.resize(nnz);
        vval.resize(nnz);

        // Transpose
        transpose_csr(
            m, m, nnz, ptr, col, val, vptr.data(), vcol.data(), vval.data(), idx_base, idx_base);

        csr_row_ptr = vptr.data();
        csr_col_ind = vcol.data();
        csr_val     = vval.data();
    }

    int            pivot = std::numeric_limits<int>::max();
    std::vector<T> temp(wf_size);

    for(int i = m - 1; i >= 0; --i)
    {
        temp.assign(wf_size, make_DataType<T>(0.0));
        temp[0] = alpha * x[i];

        int diag      = -1;
        int row_begin = csr_row_ptr[i] - idx_base;
        int row_end   = csr_row_ptr[i + 1] - idx_base;

        T diag_val = make_DataType<T>(0.0);

        for(int l = row_end - 1; l >= row_begin; l -= wf_size)
        {
            for(unsigned int k = 0; k < wf_size; ++k)
            {
                int j = l - k;

                // Do not run out of bounds
                if(j < row_begin)
                {
                    break;
                }

                int col_j = csr_col_ind[j] - idx_base;
                T   val_j = csr_val[j];

                if(col_j < i)
                {
                    // Lower part
                    continue;
                }
                else if(col_j == i)
                {
                    // Diagonal
                    if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
                    {
                        // Check for numerical zero
                        if(val_j == make_DataType<T>(0.0))
                        {
                            pivot = std::min(pivot, i + idx_base);
                            val_j = make_DataType<T>(1.0);
                        }

                        diag     = j;
                        diag_val = make_DataType<T>(1.0) / val_j;
                    }

                    continue;
                }
                else
                {
                    // Upper part
                    temp[k] = testing_fma(-csr_val[j], y[col_j], temp[k]);
                }
            }
        }

        for(unsigned int j = 1; j < wf_size; j <<= 1)
        {
            for(unsigned int k = 0; k < wf_size - j; ++k)
            {
                temp[k] = temp[k] + temp[k + j];
            }
        }

        if(diag_type == HIPSPARSE_DIAG_TYPE_NON_UNIT)
        {
            if(diag == -1)
            {
                pivot = std::min(pivot, i + idx_base);
            }

            y[i] = temp[0] * diag_val;
        }
        else
        {
            y[i] = temp[0];
        }
    }

    if(pivot != std::numeric_limits<int>::max())
    {
        return pivot;
    }

    return -1;
}

/* ============================================================================================ */
/*! \brief  Transpose sparse matrix using CSR storage format. */
template <typename T>
void transpose_csr(int                  m,
                   int                  n,
                   int                  nnz,
                   const int*           csr_row_ptr_A,
                   const int*           csr_col_ind_A,
                   const T*             csr_val_A,
                   int*                 csr_row_ptr_B,
                   int*                 csr_col_ind_B,
                   T*                   csr_val_B,
                   hipsparseIndexBase_t idx_base_A,
                   hipsparseIndexBase_t idx_base_B)
{
    memset(csr_row_ptr_B, 0, sizeof(int) * (n + 1));

    // Determine nnz per column
    for(int i = 0; i < nnz; ++i)
    {
        ++csr_row_ptr_B[csr_col_ind_A[i] + 1 - idx_base_A];
    }

    // Scan
    for(int i = 0; i < n; ++i)
    {
        csr_row_ptr_B[i + 1] += csr_row_ptr_B[i];
    }

    // Fill row indices and values
    for(int i = 0; i < m; ++i)
    {
        int row_begin = csr_row_ptr_A[i] - idx_base_A;
        int row_end   = csr_row_ptr_A[i + 1] - idx_base_A;

        for(int j = row_begin; j < row_end; ++j)
        {
            int col = csr_col_ind_A[j] - idx_base_A;
            int idx = csr_row_ptr_B[col];

            csr_col_ind_B[idx] = i + idx_base_B;
            csr_val_B[idx]     = csr_val_A[j];

            ++csr_row_ptr_B[col];
        }
    }

    // Shift column pointer array
    for(int i = n; i > 0; --i)
    {
        csr_row_ptr_B[i] = csr_row_ptr_B[i - 1] + idx_base_B;
    }

    csr_row_ptr_B[0] = idx_base_B;
}

/* ============================================================================================ */
/*! \brief  Transpose sparse matrix using CSR storage format. */
template <typename T>
void transpose_bsr(int                  mb,
                   int                  nb,
                   int                  nnzb,
                   int                  bsr_dim,
                   const int*           bsr_row_ptr_A,
                   const int*           bsr_col_ind_A,
                   const T*             bsr_val_A,
                   int*                 bsr_row_ptr_B,
                   int*                 bsr_col_ind_B,
                   T*                   bsr_val_B,
                   hipsparseIndexBase_t idx_base_A,
                   hipsparseIndexBase_t idx_base_B)
{
    memset(bsr_row_ptr_B, 0, sizeof(int) * (nb + 1));

    // Determine nnz per column
    for(int i = 0; i < nnzb; ++i)
    {
        ++bsr_row_ptr_B[bsr_col_ind_A[i] + 1 - idx_base_A];
    }

    // Scan
    for(int i = 0; i < nb; ++i)
    {
        bsr_row_ptr_B[i + 1] += bsr_row_ptr_B[i];
    }

    // Fill row indices and values
    for(int i = 0; i < mb; ++i)
    {
        int row_begin = bsr_row_ptr_A[i] - idx_base_A;
        int row_end   = bsr_row_ptr_A[i + 1] - idx_base_A;

        for(int j = row_begin; j < row_end; ++j)
        {
            int col = bsr_col_ind_A[j] - idx_base_A;
            int idx = bsr_row_ptr_B[col];

            bsr_col_ind_B[idx] = i + idx_base_B;

            for(int bi = 0; bi < bsr_dim; ++bi)
            {
                for(int bj = 0; bj < bsr_dim; ++bj)
                {
                    bsr_val_B[bsr_dim * bsr_dim * idx + bi + bj * bsr_dim]
                        = bsr_val_A[bsr_dim * bsr_dim * j + bi * bsr_dim + bj];
                }
            }

            ++bsr_row_ptr_B[col];
        }
    }

    // Shift column pointer array
    for(int i = nb; i > 0; --i)
    {
        bsr_row_ptr_B[i] = bsr_row_ptr_B[i - 1] + idx_base_B;
    }

    bsr_row_ptr_B[0] = idx_base_B;
}

/* ============================================================================================ */
/*! \brief  Compute sparse matrix sparse matrix addition. */
template <typename T>
static int host_csrgeam_nnz(int                  M,
                            int                  N,
                            T                    alpha,
                            const int*           csr_row_ptr_A,
                            const int*           csr_col_ind_A,
                            T                    beta,
                            const int*           csr_row_ptr_B,
                            const int*           csr_col_ind_B,
                            int*                 csr_row_ptr_C,
                            hipsparseIndexBase_t base_A,
                            hipsparseIndexBase_t base_B,
                            hipsparseIndexBase_t base_C)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<int> nnz(N, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        int rows_per_thread = (M + nthreads - 1) / nthreads;
        int chunk_begin     = rows_per_thread * tid;
        int chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Index base
        csr_row_ptr_C[0] = base_C;

        // Loop over rows
        for(int i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            int row_begin_A = csr_row_ptr_A[i] - base_A;
            int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Loop over columns of A
            for(int j = row_begin_A; j < row_end_A; ++j)
            {
                int col_A = csr_col_ind_A[j] - base_A;

                nnz[col_A] = i;
                ++csr_row_ptr_C[i + 1];
            }

            int row_begin_B = csr_row_ptr_B[i] - base_B;
            int row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(int j = row_begin_B; j < row_end_B; ++j)
            {
                int col_B = csr_col_ind_B[j] - base_B;

                // Check if a new nnz is generated
                if(nnz[col_B] != i)
                {
                    nnz[col_B] = i;
                    ++csr_row_ptr_C[i + 1];
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(int i = 0; i < M; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    return csr_row_ptr_C[M] - base_C;
}

template <typename T>
static void host_csrgeam(int                  M,
                         int                  N,
                         T                    alpha,
                         const int*           csr_row_ptr_A,
                         const int*           csr_col_ind_A,
                         const T*             csr_val_A,
                         T                    beta,
                         const int*           csr_row_ptr_B,
                         const int*           csr_col_ind_B,
                         const T*             csr_val_B,
                         const int*           csr_row_ptr_C,
                         int*                 csr_col_ind_C,
                         T*                   csr_val_C,
                         hipsparseIndexBase_t base_A,
                         hipsparseIndexBase_t base_B,
                         hipsparseIndexBase_t base_C)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<int> nnz(N, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        int rows_per_thread = (M + nthreads - 1) / nthreads;
        int chunk_begin     = rows_per_thread * tid;
        int chunk_end       = std::min(chunk_begin + rows_per_thread, M);

        // Loop over rows
        for(int i = chunk_begin; i < chunk_end; ++i)
        {
            int row_begin_C = csr_row_ptr_C[i] - base_C;
            int row_end_C   = row_begin_C;

            int row_begin_A = csr_row_ptr_A[i] - base_A;
            int row_end_A   = csr_row_ptr_A[i + 1] - base_A;

            // Copy A into C
            for(int j = row_begin_A; j < row_end_A; ++j)
            {
                // Current column of A
                int col_A = csr_col_ind_A[j] - base_A;

                // Current value of A
                T val_A = alpha * csr_val_A[j];

                nnz[col_A] = row_end_C;

                csr_col_ind_C[row_end_C] = col_A + base_C;
                csr_val_C[row_end_C]     = val_A;
                ++row_end_C;
            }

            int row_begin_B = csr_row_ptr_B[i] - base_B;
            int row_end_B   = csr_row_ptr_B[i + 1] - base_B;

            // Loop over columns of B
            for(int j = row_begin_B; j < row_end_B; ++j)
            {
                // Current column of B
                int col_B = csr_col_ind_B[j] - base_B;

                // Current value of B
                T val_B = beta * csr_val_B[j];

                // Check if a new nnz is generated or if the value is added
                if(nnz[col_B] < row_begin_C)
                {
                    nnz[col_B] = row_end_C;

                    csr_col_ind_C[row_end_C] = col_B + base_C;
                    csr_val_C[row_end_C]     = val_B;
                    ++row_end_C;
                }
                else
                {
                    csr_val_C[nnz[col_B]] = csr_val_C[nnz[col_B]] + val_B;
                }
            }
        }
    }

    int nnz = csr_row_ptr_C[M] - base_C;

    std::vector<int> col(nnz);
    std::vector<T>   val(nnz);

    for(int i = 0; i < nnz; ++i)
    {
        col[i] = csr_col_ind_C[i];
        val[i] = csr_val_C[i];
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
    for(int i = 0; i < M; ++i)
    {
        int row_begin = csr_row_ptr_C[i] - base_C;
        int row_end   = csr_row_ptr_C[i + 1] - base_C;
        int row_nnz   = row_end - row_begin;

        std::vector<int> perm(row_nnz);
        for(int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        int* col_entry = &col[row_begin];
        T*   val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
            return col_entry[a] <= col_entry[b];
        });

        for(int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

/* ============================================================================================ */
/*! \brief  Compute sparse matrix sparse matrix multiplication. */
template <typename T>
static int csrgemm2_nnz(int                  m,
                        int                  n,
                        int                  k,
                        const T*             alpha,
                        const int*           csr_row_ptr_A,
                        const int*           csr_col_ind_A,
                        const int*           csr_row_ptr_B,
                        const int*           csr_col_ind_B,
                        const T*             beta,
                        const int*           csr_row_ptr_D,
                        const int*           csr_col_ind_D,
                        int*                 csr_row_ptr_C,
                        hipsparseIndexBase_t idx_base_A,
                        hipsparseIndexBase_t idx_base_B,
                        hipsparseIndexBase_t idx_base_C,
                        hipsparseIndexBase_t idx_base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<int> nnz(n, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        int rows_per_thread = (m + nthreads - 1) / nthreads;
        int chunk_begin     = rows_per_thread * tid;
        int chunk_end       = std::min(chunk_begin + rows_per_thread, m);

        // Index base
        csr_row_ptr_C[0] = idx_base_C;

        // Loop over rows of A
        for(int i = chunk_begin; i < chunk_end; ++i)
        {
            // Initialize csr row pointer with previous row offset
            csr_row_ptr_C[i + 1] = 0;

            if(alpha)
            {
                int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
                int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

                // Loop over columns of A
                for(int j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    int col_A = csr_col_ind_A[j] - idx_base_A;

                    int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                    int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Loop over columns of B in row col_A
                    for(int k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        int col_B = csr_col_ind_B[k] - idx_base_B;

                        // Check if a new nnz is generated
                        if(nnz[col_B] != i)
                        {
                            nnz[col_B] = i;
                            ++csr_row_ptr_C[i + 1];
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                int row_begin_D = csr_row_ptr_D[i] - idx_base_D;
                int row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

                // Loop over columns of D
                for(int j = row_begin_D; j < row_end_D; ++j)
                {
                    int col_D = csr_col_ind_D[j] - idx_base_D;

                    // Check if a new nnz is generated
                    if(nnz[col_D] != i)
                    {
                        nnz[col_D] = i;
                        ++csr_row_ptr_C[i + 1];
                    }
                }
            }
        }
    }

    // Scan to obtain row offsets
    for(int i = 0; i < m; ++i)
    {
        csr_row_ptr_C[i + 1] += csr_row_ptr_C[i];
    }

    return csr_row_ptr_C[m] - idx_base_C;
}

template <typename T>
static void csrgemm2(int                  m,
                     int                  n,
                     int                  k,
                     const T*             alpha,
                     const int*           csr_row_ptr_A,
                     const int*           csr_col_ind_A,
                     const T*             csr_val_A,
                     const int*           csr_row_ptr_B,
                     const int*           csr_col_ind_B,
                     const T*             csr_val_B,
                     const T*             beta,
                     const int*           csr_row_ptr_D,
                     const int*           csr_col_ind_D,
                     const T*             csr_val_D,
                     const int*           csr_row_ptr_C,
                     int*                 csr_col_ind_C,
                     T*                   csr_val_C,
                     hipsparseIndexBase_t idx_base_A,
                     hipsparseIndexBase_t idx_base_B,
                     hipsparseIndexBase_t idx_base_C,
                     hipsparseIndexBase_t idx_base_D)
{
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        std::vector<int> nnz(n, -1);

#ifdef _OPENMP
        int nthreads = omp_get_num_threads();
        int tid      = omp_get_thread_num();
#else
        int nthreads = 1;
        int tid      = 0;
#endif

        int rows_per_thread = (m + nthreads - 1) / nthreads;
        int chunk_begin     = rows_per_thread * tid;
        int chunk_end       = std::min(chunk_begin + rows_per_thread, m);

        // Loop over rows of A
        for(int i = chunk_begin; i < chunk_end; ++i)
        {
            int row_begin_C = csr_row_ptr_C[i] - idx_base_C;
            int row_end_C   = row_begin_C;

            if(alpha)
            {
                int row_begin_A = csr_row_ptr_A[i] - idx_base_A;
                int row_end_A   = csr_row_ptr_A[i + 1] - idx_base_A;

                // Loop over columns of A
                for(int j = row_begin_A; j < row_end_A; ++j)
                {
                    // Current column of A
                    int col_A = csr_col_ind_A[j] - idx_base_A;
                    // Current value of A
                    T val_A = *alpha * csr_val_A[j];

                    int row_begin_B = csr_row_ptr_B[col_A] - idx_base_B;
                    int row_end_B   = csr_row_ptr_B[col_A + 1] - idx_base_B;

                    // Loop over columns of B in row col_A
                    for(int k = row_begin_B; k < row_end_B; ++k)
                    {
                        // Current column of B
                        int col_B = csr_col_ind_B[k] - idx_base_B;
                        // Current value of B
                        T val_B = csr_val_B[k];

                        // Check if a new nnz is generated or if the product is appended
                        if(nnz[col_B] < row_begin_C)
                        {
                            nnz[col_B]               = row_end_C;
                            csr_col_ind_C[row_end_C] = col_B + idx_base_C;
                            csr_val_C[row_end_C]     = val_A * val_B;
                            ++row_end_C;
                        }
                        else
                        {
                            csr_val_C[nnz[col_B]] = csr_val_C[nnz[col_B]] + val_A * val_B;
                        }
                    }
                }
            }

            // Add nnz of D if beta != 0
            if(beta)
            {
                int row_begin_D = csr_row_ptr_D[i] - idx_base_D;
                int row_end_D   = csr_row_ptr_D[i + 1] - idx_base_D;

                // Loop over columns of D
                for(int j = row_begin_D; j < row_end_D; ++j)
                {
                    // Current column of D
                    int col_D = csr_col_ind_D[j] - idx_base_D;
                    // Current value of D
                    T val_D = *beta * csr_val_D[j];

                    // Check if a new nnz is generated or if the value is added
                    if(nnz[col_D] < row_begin_C)
                    {
                        nnz[col_D] = row_end_C;

                        csr_col_ind_C[row_end_C] = col_D + idx_base_C;
                        csr_val_C[row_end_C]     = val_D;
                        ++row_end_C;
                    }
                    else
                    {
                        csr_val_C[nnz[col_D]] = csr_val_C[nnz[col_D]] + val_D;
                    }
                }
            }
        }
    }

    int nnz = csr_row_ptr_C[m] - idx_base_C;

    std::vector<int> col(nnz);
    std::vector<T>   val(nnz);

    memcpy(col.data(), csr_col_ind_C, sizeof(int) * nnz);
    memcpy(val.data(), csr_val_C, sizeof(T) * nnz);

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i = 0; i < m; ++i)
    {
        int row_begin = csr_row_ptr_C[i] - idx_base_C;
        int row_end   = csr_row_ptr_C[i + 1] - idx_base_C;
        int row_nnz   = row_end - row_begin;

        std::vector<int> perm(row_nnz);
        for(int j = 0; j < row_nnz; ++j)
        {
            perm[j] = j;
        }

        int* col_entry = &col[row_begin];
        T*   val_entry = &val[row_begin];

        std::sort(perm.begin(), perm.end(), [&](const int& a, const int& b) {
            return col_entry[a] <= col_entry[b];
        });

        for(int j = 0; j < row_nnz; ++j)
        {
            csr_col_ind_C[row_begin + j] = col_entry[perm[j]];
            csr_val_C[row_begin + j]     = val_entry[perm[j]];
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================================ */
/*  query for hipsparse version and git commit SHA-1. */
void query_version(char* version);

/* ============================================================================================ */
/*  device query and print out their ID and name */
int query_device_property();

/*  set current device to device_id */
void set_device(int device_id);

/* ============================================================================================ */
/*  timing: HIP only provides very limited timers function clock() and not general;
            hipsparse sync CPU and device and use more accurate CPU timer*/

/*! \brief  CPU Timer(in microsecond): synchronize with the default device and return wall time */
double get_time_us(void);

/*! \brief  CPU Timer(in microsecond): synchronize with given queue/stream and return wall time */
double get_time_us_sync(hipStream_t stream);

#ifdef __cplusplus
}
#endif

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this hipsparse library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments
{
public:
    int M         = 128;
    int N         = 128;
    int K         = 128;
    int nnz       = 32;
    int block_dim = 1;

    int lda;
    int ldb;
    int ldc;

    double alpha  = 1.0;
    double alphai = 0.0;
    double beta   = 0.0;
    double betai  = 0.0;

    hipsparseOperation_t    transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t    transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t    idx_base  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t    idx_base2 = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t    idx_base3 = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t    idx_base4 = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseAction_t       action    = HIPSPARSE_ACTION_NUMERIC;
    hipsparseHybPartition_t part      = HIPSPARSE_HYB_PARTITION_AUTO;
    hipsparseDiagType_t     diag_type = HIPSPARSE_DIAG_TYPE_NON_UNIT;
    hipsparseFillMode_t     fill_mode = HIPSPARSE_FILL_MODE_LOWER;
    hipsparseDirection_t    dirA      = HIPSPARSE_DIRECTION_ROW;

    int norm_check = 0;
    int unit_check = 1;
    int timing     = 0;

    int iters     = 10;
    int laplacian = 0;
    int ell_width = 0;
    int temp      = 0;

    std::string filename = "";

    Arguments& operator=(const Arguments& rhs)
    {
        this->M         = rhs.M;
        this->N         = rhs.N;
        this->K         = rhs.K;
        this->nnz       = rhs.nnz;
        this->block_dim = rhs.block_dim;

        this->lda = rhs.lda;
        this->ldb = rhs.ldb;
        this->ldc = rhs.ldc;

        this->alpha = rhs.alpha;
        this->beta  = rhs.beta;

        this->transA    = rhs.transA;
        this->transB    = rhs.transB;
        this->idx_base  = rhs.idx_base;
        this->idx_base2 = rhs.idx_base2;
        this->idx_base3 = rhs.idx_base3;
        this->idx_base4 = rhs.idx_base4;
        this->action    = rhs.action;
        this->part      = rhs.part;
        this->diag_type = rhs.diag_type;
        this->fill_mode = rhs.fill_mode;
        this->dirA      = rhs.dirA;

        this->norm_check = rhs.norm_check;
        this->unit_check = rhs.unit_check;
        this->timing     = rhs.timing;

        this->iters     = rhs.iters;
        this->laplacian = rhs.laplacian;
        this->ell_width = rhs.ell_width;
        this->temp      = rhs.temp;

        this->filename = rhs.filename;

        return *this;
    }
};

#endif // TESTING_UTILITY_HPP
