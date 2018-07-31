/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <hipsparse.h>
#include <hip/hip_runtime_api.h>

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

#define CHECK_HIPSPARSE_ERROR(error)                              \
    if(error != HIPSPARSE_STATUS_SUCCESS)                         \
    {                                                             \
        fprintf(stderr, "hipSPARSE error: ");                     \
        if(error == HIPSPARSE_STATUS_NOT_INITIALIZED)             \
        {                                                         \
            fprintf(stderr, "HIPSPARSE_STATUS_NOT_INITIALIZED");  \
        }                                                         \
        else if(error == HIPSPARSE_STATUS_INTERNAL_ERROR)         \
        {                                                         \
            fprintf(stderr, " HIPSPARSE_STATUS_INTERNAL_ERROR");  \
        }                                                         \
        else if(error == HIPSPARSE_STATUS_INVALID_VALUE)          \
        {                                                         \
            fprintf(stderr, "HIPSPARSE_STATUS_INVALID_VALUE");    \
        }                                                         \
        else if(error == HIPSPARSE_STATUS_ALLOC_FAILED)           \
        {                                                         \
            fprintf(stderr, "HIPSPARSE_STATUS_ALLOC_FAILED");     \
        }                                                         \
        else                                                      \
        {                                                         \
            fprintf(stderr, "HIPSPARSE_STATUS ERROR");            \
        }                                                         \
        fprintf(stderr, "\n");                                    \
        return error;                                             \
    }

/* ============================================================================================ */
/* generate random number :*/

/*! \brief  generate a random number between [0, 0.999...] . */
template <typename T>
T random_generator()
{
    // return rand()/( (T)RAND_MAX + 1);
    return (T)(rand() % 10 + 1); // generate a integer number between [1, 10]
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
    int num = 0;
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
void hipsparseInitCSR(std::vector<int>& ptr,
                      std::vector<int>& col,
                      std::vector<T>& val,
                      int nrow,
                      int ncol,
                      int nnz)
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
int gen_2d_laplacian(int ndim,
                     std::vector<int>& rowptr,
                     std::vector<int>& col,
                     std::vector<T>& val,
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
            int idx = i * ndim + j;
            rowptr[idx]       = nnz + idx_base;
            // if no upper boundary element, connect with upper neighbor
            if(i != 0)
            {
                col[nnz] = idx - ndim + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if(j != 0)
            {
                col[nnz] = idx - 1 + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // element itself
            col[nnz] = idx + idx_base;
            val[nnz] = static_cast<T>(4);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if(j != ndim - 1)
            {
                col[nnz] = idx + 1 + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if(i != ndim - 1)
            {
                col[nnz] = idx + ndim + idx_base;
                val[nnz] = static_cast<T>(-1);
                ++nnz;
            }
        }
    }
    rowptr[n] = nnz + idx_base;

    return n;
}

/* ============================================================================================ */
/*! \brief  Generate a random sparse matrix in COO format */
template <typename T>
void gen_matrix_coo(int m,
                    int n,
                    int nnz,
                    std::vector<int>& row_ind,
                    std::vector<int>& col_ind,
                    std::vector<T>& val,
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
            int rng = (i - begin) * sqrt(-2.0 * log((double)rand() / RAND_MAX)) *
                                cos(2.0 * M_PI * (double)rand() / RAND_MAX);

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
template <typename T>
int read_mtx_matrix(const char* filename,
                    int& nrow,
                    int& ncol,
                    int& nnz,
                    std::vector<int>& row,
                    std::vector<int>& col,
                    std::vector<T>& val,
                    hipsparseIndexBase_t idx_base)
{
    printf("Reading matrix %s...", filename);
    fflush(stdout);

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
    for(char *p = array; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char *p = coord; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char *p = data; *p != '\0'; *p = tolower(*p), p++)
        ;
    for(char *p = type; *p != '\0'; *p = tolower(*p), p++)
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
    std::vector<T> unsorted_val(nnz);

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
        T ival;

        std::istringstream ss(line);

        if(!strcmp(data, "pattern"))
        {
            ss >> irow >> icol;
            ival = static_cast<T>(1);
        }
        else
        {
            ss >> irow >> icol >> ival;
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

    printf("done.\n");
    fflush(stdout);

    return 0;
}

/* ============================================================================================ */
/*! \brief  Read matrix from binary file in CSR format */
template <typename T>
int read_bin_matrix(const char* filename,
                    int& nrow,
                    int& ncol,
                    int& nnz,
                    std::vector<int>& ptr,
                    std::vector<int>& col,
                    std::vector<T>& val,
                    hipsparseIndexBase_t idx_base)
{
    printf("Reading matrix %s...", filename);
    fflush(stdout);

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

    for(int i = 0; i < nnz; ++i)
    {
        val[i] = static_cast<T>(tmp[i]);
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

    printf("done.\n");
    fflush(stdout);

    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif

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
    int M   = 128;
    int N   = 128;
    int K   = 128;
    int nnz = 32;

    int ldb;
    int ldc;

    double alpha = 1.0;
    double beta  = 0.0;

    hipsparseOperation_t transA    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseOperation_t transB    = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    hipsparseIndexBase_t idx_base  = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseIndexBase_t idx_base2 = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseAction_t action       = HIPSPARSE_ACTION_NUMERIC;
    hipsparseHybPartition_t part   = HIPSPARSE_HYB_PARTITION_AUTO;

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
        this->M   = rhs.M;
        this->N   = rhs.N;
        this->K   = rhs.K;
        this->nnz = rhs.nnz;

        this->ldb = rhs.ldb;
        this->ldc = rhs.ldc;

        this->alpha = rhs.alpha;
        this->beta  = rhs.beta;

        this->transA    = rhs.transA;
        this->transB    = rhs.transB;
        this->idx_base  = rhs.idx_base;
        this->idx_base2 = rhs.idx_base2;
        this->action    = rhs.action;
        this->part      = rhs.part;

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
