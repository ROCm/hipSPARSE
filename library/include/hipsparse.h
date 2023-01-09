/* ************************************************************************
* Copyright (C) 2018-2023 Advanced Micro Devices, Inc. All rights Reserved.
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

/*! \file
*  \brief hipsparse.h provides Sparse Linear Algebra Subprograms
*  of Level 1, 2 and 3, using HIP optimized for AMD GPU hardware.
*/

// HIP = Heterogeneous-compute Interface for Portability
//
// Define a extremely thin runtime layer that allows source code to be compiled
// unmodified through either AMD HCC or NVCC. Key features tend to be in the spirit
// and terminology of CUDA, but with a portable path to other accelerators as well.
//
// This is the master include file for hipSPARSE, wrapping around rocSPARSE and
// cuSPARSE "version 2".

#pragma once
#ifndef _HIPSPARSE_H_
#define _HIPSPARSE_H_

#include "hipsparse-export.h"
#include "hipsparse-version.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>

/// \cond DO_NOT_DOCUMENT
#define DEPRECATED_CUDA_11000(warning)
#define DEPRECATED_CUDA_10000(warning)
#define DEPRECATED_CUDA_9000(warning)

#ifdef __cplusplus
#ifndef __has_cpp_attribute
#define __has_cpp_attribute(X) 0
#endif
#define HIPSPARSE_HAS_DEPRECATED_MSG __has_cpp_attribute(deprecated) >= 201309L
#else
#ifndef __has_c_attribute
#define __has_c_attribute(X) 0
#endif
#define HIPSPARSE_HAS_DEPRECATED_MSG __has_c_attribute(deprecated) >= 201904L
#endif

#if HIPSPARSE_HAS_DEPRECATED_MSG
#define HIPSPARSE_DEPRECATED_MSG(MSG) [[deprecated(MSG)]]
#else
#define HIPSPARSE_DEPRECATED_MSG(MSG) HIPSPARSE_DEPRECATED // defined in hipsparse-export.h
#endif
/// \endcond

#if defined(CUDART_VERSION)
#if CUDART_VERSION < 10000
#undef DEPRECATED_CUDA_9000
#define DEPRECATED_CUDA_9000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 11000
#undef DEPRECATED_CUDA_10000
#define DEPRECATED_CUDA_10000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#elif CUDART_VERSION < 12000
#undef DEPRECATED_CUDA_11000
#define DEPRECATED_CUDA_11000(warning) HIPSPARSE_DEPRECATED_MSG(warning)
#endif
#endif

/*! \ingroup types_module
 *  \brief Handle to the hipSPARSE library context queue.
 *
 *  \details
 *  The hipSPARSE handle is a structure holding the hipSPARSE library context. It must
 *  be initialized using hipsparseCreate() and the returned handle must be passed to all
 *  subsequent library function calls. It should be destroyed at the end using
 *  hipsparseDestroy().
 */
typedef void* hipsparseHandle_t;

/*! \ingroup types_module
 *  \brief Descriptor of the matrix.
 *
 *  \details
 *  The hipSPARSE matrix descriptor is a structure holding all properties of a matrix.
 *  It must be initialized using hipsparseCreateMatDescr() and the returned descriptor
 *  must be passed to all subsequent library calls that involve the matrix. It should be
 *  destroyed at the end using hipsparseDestroyMatDescr().
 */
typedef void* hipsparseMatDescr_t;

/*! \ingroup types_module
 *  \brief HYB matrix storage format.
 *
 *  \details
 *  The hipSPARSE HYB matrix structure holds the HYB matrix. It must be initialized using
 *  hipsparseCreateHybMat() and the returned HYB matrix must be passed to all subsequent
 *  library calls that involve the matrix. It should be destroyed at the end using
 *  hipsparseDestroyHybMat().
 */
typedef void* hipsparseHybMat_t;

/*! \ingroup types_module
 *  \brief Coloring info.
 *
 *  \details
 *  The hipSPARSE ColorInfo structure holds the coloring information. It must be
 *  initialized using hipsparseCreateColorInfo() and the returned structure must be
 *  passed to all subsequent library calls that involve the coloring. It should be
 *  destroyed at the end using hipsparseDestroyColorInfo().
 */
typedef void* hipsparseColorInfo_t;

/*! \ingroup types_module
 *  \brief bsrsv2 info to hold collected meta information.
 */
struct bsrsv2Info;
typedef struct bsrsv2Info* bsrsv2Info_t;
/*! \ingroup types_module
 *  \brief bsrsn2 info to hold collected meta information.
 */
struct bsrsm2Info;
typedef struct bsrsm2Info* bsrsm2Info_t;
/*! \ingroup types_module
 *  \brief bsrilu02 info to hold collected meta information.
 */
struct bsrilu02Info;
typedef struct bsrilu02Info* bsrilu02Info_t;
/*! \ingroup types_module
 *  \brief bsric02 info to hold collected meta information.
 */
struct bsric02Info;
typedef struct bsric02Info* bsric02Info_t;
/*! \ingroup types_module
 *  \brief csrsv2 info to hold collected meta information.
 */
struct csrsv2Info;
typedef struct csrsv2Info* csrsv2Info_t;
/*! \ingroup types_module
 *  \brief csrsm2 info to hold collected meta information.
 */
struct csrsm2Info;
typedef struct csrsm2Info* csrsm2Info_t;
/*! \ingroup types_module
 *  \brief csrilu02 info to hold collected meta information.
 */
struct csrilu02Info;
typedef struct csrilu02Info* csrilu02Info_t;
/*! \ingroup types_module
 *  \brief csric02 info to hold collected meta information.
 */
struct csric02Info;
typedef struct csric02Info* csric02Info_t;
/*! \ingroup types_module
 *  \brief csrgemm2 info to hold collected meta information.
 */
struct csrgemm2Info;
typedef struct csrgemm2Info* csrgemm2Info_t;
/*! \ingroup types_module
 *  \brief prune info to hold collected meta information.
 */
struct pruneInfo;
typedef struct pruneInfo* pruneInfo_t;
/*! \ingroup types_module
 *  \brief csru2csr info to hold collected meta information.
 */
struct csru2csrInfo;
typedef struct csru2csrInfo* csru2csrInfo_t;

// clang-format off

/*! \ingroup types_module
 *  \brief List of hipsparse status codes definition.
 *
 *  \details
 *  This is a list of the \ref hipsparseStatus_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum {
    HIPSPARSE_STATUS_SUCCESS                   = 0, /**< Function succeeds */
    HIPSPARSE_STATUS_NOT_INITIALIZED           = 1, /**< hipSPARSE was not initialized */
    HIPSPARSE_STATUS_ALLOC_FAILED              = 2, /**< Resource allocation failed */
    HIPSPARSE_STATUS_INVALID_VALUE             = 3, /**< Unsupported value was passed to the function */
    HIPSPARSE_STATUS_ARCH_MISMATCH             = 4, /**< Device architecture not supported */
    HIPSPARSE_STATUS_MAPPING_ERROR             = 5, /**< Access to GPU memory space failed */
    HIPSPARSE_STATUS_EXECUTION_FAILED          = 6, /**< GPU program failed to execute */
    HIPSPARSE_STATUS_INTERNAL_ERROR            = 7, /**< An internal hipSPARSE operation failed */
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8, /**< Matrix type not supported */
    HIPSPARSE_STATUS_ZERO_PIVOT                = 9, /**< Zero pivot was computed */
    HIPSPARSE_STATUS_NOT_SUPPORTED             = 10, /**< Operation is not supported */
    HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11 /**< Resources are insufficient */
} hipsparseStatus_t;
#else
#if(CUDART_VERSION >= 11003)
typedef enum {
    HIPSPARSE_STATUS_SUCCESS                   = 0, /**< Function succeeds */
    HIPSPARSE_STATUS_NOT_INITIALIZED           = 1, /**< hipSPARSE was not initialized */
    HIPSPARSE_STATUS_ALLOC_FAILED              = 2, /**< Resource allocation failed */
    HIPSPARSE_STATUS_INVALID_VALUE             = 3, /**< Unsupported value was passed to the function */
    HIPSPARSE_STATUS_ARCH_MISMATCH             = 4, /**< Device architecture not supported */
    HIPSPARSE_STATUS_MAPPING_ERROR             = 5, /**< Access to GPU memory space failed */
    HIPSPARSE_STATUS_EXECUTION_FAILED          = 6, /**< GPU program failed to execute */
    HIPSPARSE_STATUS_INTERNAL_ERROR            = 7, /**< An internal hipSPARSE operation failed */
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8, /**< Matrix type not supported */
    HIPSPARSE_STATUS_ZERO_PIVOT                = 9, /**< Zero pivot was computed */
    HIPSPARSE_STATUS_NOT_SUPPORTED             = 10, /**< Operation is not supported */
    HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11 /**< Resources are insufficient */
} hipsparseStatus_t;
#elif(CUDART_VERSION >= 10010)
typedef enum {
    HIPSPARSE_STATUS_SUCCESS                   = 0, /**< Function succeeds */
    HIPSPARSE_STATUS_NOT_INITIALIZED           = 1, /**< hipSPARSE was not initialized */
    HIPSPARSE_STATUS_ALLOC_FAILED              = 2, /**< Resource allocation failed */
    HIPSPARSE_STATUS_INVALID_VALUE             = 3, /**< Unsupported value was passed to the function */
    HIPSPARSE_STATUS_ARCH_MISMATCH             = 4, /**< Device architecture not supported */
    HIPSPARSE_STATUS_MAPPING_ERROR             = 5, /**< Access to GPU memory space failed */
    HIPSPARSE_STATUS_EXECUTION_FAILED          = 6, /**< GPU program failed to execute */
    HIPSPARSE_STATUS_INTERNAL_ERROR            = 7, /**< An internal hipSPARSE operation failed */
    HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8, /**< Matrix type not supported */
    HIPSPARSE_STATUS_ZERO_PIVOT                = 9, /**< Zero pivot was computed */
    HIPSPARSE_STATUS_NOT_SUPPORTED             = 10 /**< Operation is not supported */
} hipsparseStatus_t;
#endif
#endif

/*! \ingroup types_module
 *  \brief Indicates if the pointer is device pointer or host pointer.
 *
 *  \details
 *  The \ref hipsparsePointerMode_t indicates whether scalar values are passed by
 *  reference on the host or device. The \ref hipsparsePointerMode_t can be changed by
 *  hipsparseSetPointerMode(). The currently used pointer mode can be obtained by
 *  hipsparseGetPointerMode().
 */
typedef enum {
    HIPSPARSE_POINTER_MODE_HOST   = 0, /**< Scalar pointers are in host memory */
    HIPSPARSE_POINTER_MODE_DEVICE = 1 /**< Scalar pointers are in device memory */
} hipsparsePointerMode_t;

/*! \ingroup types_module
 *  \brief Specify where the operation is performed on.
 *
 *  \details
 *  The \ref hipsparseAction_t indicates whether the operation is performed on the full
 *  matrix, or only on the sparsity pattern of the matrix.
 */
typedef enum {
    HIPSPARSE_ACTION_SYMBOLIC = 0, /**< Operate only on indices */
    HIPSPARSE_ACTION_NUMERIC  = 1 /**< Operate on data and indices */
} hipsparseAction_t;

/*! \ingroup types_module
 *  \brief Specify the matrix type.
 *
 *  \details
 *  The \ref hipsparseMatrixType_t indices the type of a matrix. For a given
 *  \ref hipsparseMatDescr_t, the \ref hipsparseMatrixType_t can be set using
 *  hipsparseSetMatType(). The current \ref hipsparseMatrixType_t of a matrix can be
 *  obtained by hipsparseGetMatType().
 */
typedef enum {
    HIPSPARSE_MATRIX_TYPE_GENERAL    = 0, /**< General matrix type */
    HIPSPARSE_MATRIX_TYPE_SYMMETRIC  = 1, /**< Symmetric matrix type */
    HIPSPARSE_MATRIX_TYPE_HERMITIAN  = 2, /**< Hermitian matrix type */
    HIPSPARSE_MATRIX_TYPE_TRIANGULAR = 3 /**< Triangular matrix type */
} hipsparseMatrixType_t;

/*! \ingroup types_module
 *  \brief Specify the matrix fill mode.
 *
 *  \details
 *  The \ref hipsparseFillMode_t indicates whether the lower or the upper part is stored
 *  in a sparse triangular matrix. For a given \ref hipsparseMatDescr_t, the
 *  \ref hipsparseFillMode_t can be set using hipsparseSetMatFillMode(). The current
 *  \ref hipsparseFillMode_t of a matrix can be obtained by hipsparseGetMatFillMode().
 */
typedef enum {
    HIPSPARSE_FILL_MODE_LOWER = 0, /**< Lower triangular part is stored */
    HIPSPARSE_FILL_MODE_UPPER = 1 /**< Upper triangular part is stored */
} hipsparseFillMode_t;

/*! \ingroup types_module
 *  \brief Indicates if the diagonal entries are unity.
 *
 *  \details
 *  The \ref hipsparseDiagType_t indicates whether the diagonal entries of a matrix are
 *  unity or not. If \ref HIPSPARSE_DIAG_TYPE_UNIT is specified, all present diagonal
 *  values will be ignored. For a given \ref hipsparseMatDescr_t, the
 *  \ref hipsparseDiagType_t can be set using hipsparseSetMatDiagType(). The current
 *  \ref hipsparseDiagType_t of a matrix can be obtained by hipsparseGetMatDiagType().
 */
typedef enum {
    HIPSPARSE_DIAG_TYPE_NON_UNIT = 0, /**< Diagonal entries are non-unity */
    HIPSPARSE_DIAG_TYPE_UNIT     = 1  /**< Diagonal entries are unity */
} hipsparseDiagType_t;

/*! \ingroup types_module
 *  \brief Specify the matrix index base.
 *
 *  \details
 *  The \ref hipsparseIndexBase_t indicates the index base of the indices. For a
 *  given \ref hipsparseMatDescr_t, the \ref hipsparseIndexBase_t can be set using
 *  hipsparseSetMatIndexBase(). The current \ref hipsparseIndexBase_t of a matrix
 *  can be obtained by hipsparseGetMatIndexBase().
 */
typedef enum {
    HIPSPARSE_INDEX_BASE_ZERO = 0, /**< Zero based indexing */
    HIPSPARSE_INDEX_BASE_ONE  = 1  /**< One based indexing */
} hipsparseIndexBase_t;

/*! \ingroup types_module
 *  \brief Specify whether the matrix is to be transposed or not.
 *
 *  \details
 *  The \ref hipsparseOperation_t indicates the operation performed with the given matrix.
 */
typedef enum {
    HIPSPARSE_OPERATION_NON_TRANSPOSE       = 0, /**< Operate with matrix */
    HIPSPARSE_OPERATION_TRANSPOSE           = 1, /**< Operate with transpose */
    HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2  /**< Operate with conj. transpose */
} hipsparseOperation_t;

/*! \ingroup types_module
 *  \brief HYB matrix partitioning type.
 *
 *  \details
 *  The \ref hipsparseHybPartition_t type indicates how the hybrid format partitioning
 *  between COO and ELL storage formats is performed.
 */
typedef enum {
    HIPSPARSE_HYB_PARTITION_AUTO = 0, /**< Automatically decide on ELL nnz per row */
    HIPSPARSE_HYB_PARTITION_USER = 1, /**< User given ELL nnz per row */
    HIPSPARSE_HYB_PARTITION_MAX  = 2  /**< Max ELL nnz per row, no COO part */
} hipsparseHybPartition_t;

/*! \ingroup types_module
 *  \brief Specify policy in triangular solvers and factorizations.
 *
 *  \details
 *  The \ref hipsparseSolvePolicy_t type indicates the solve policy for the triangular
 *  solve.
 */
typedef enum {
    HIPSPARSE_SOLVE_POLICY_NO_LEVEL  = 0, /**< No level information generated */
    HIPSPARSE_SOLVE_POLICY_USE_LEVEL = 1  /**< Generate level information */
} hipsparseSolvePolicy_t;

typedef enum {
    HIPSPARSE_SIDE_LEFT  = 0,
    HIPSPARSE_SIDE_RIGHT = 1
} hipsparseSideMode_t;

/*! \ingroup types_module
 *  \brief Specify the matrix direction.
 *
 *  \details
 *  The \ref hipsparseDirection_t indicates whether a dense matrix should be parsed by
 *  rows or by columns, assuming column-major storage.
 */
typedef enum {
    HIPSPARSE_DIRECTION_ROW = 0, /**< Parse the matrix by rows */
    HIPSPARSE_DIRECTION_COLUMN = 1 /**< Parse the matrix by columns */
} hipsparseDirection_t;

// clang-format on

#ifdef __cplusplus
extern "C" {
#endif

/*! \ingroup aux_module
 *  \brief Create a hipsparse handle
 *
 *  \details
 *  \p hipsparseCreate creates the hipSPARSE library context. It must be
 *  initialized before any other hipSPARSE API function is invoked and must be passed to
 *  all subsequent library function calls. The handle should be destroyed at the end
 *  using hipsparseDestroy().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle);

/*! \ingroup aux_module
 *  \brief Destroy a hipsparse handle
 *
 *  \details
 *  \p hipsparseDestroy destroys the hipSPARSE library context and releases all
 *  resources used by the hipSPARSE library.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle);

/*! \ingroup aux_module
 *  \brief Get hipSPARSE version
 *
 *  \details
 *  \p hipsparseGetVersion gets the hipSPARSE library version number.
 *  - patch = version % 100
 *  - minor = version / 100 % 1000
 *  - major = version / 100000
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version);

/*! \ingroup aux_module
 *  \brief Get hipSPARSE git revision
 *
 *  \details
 *  \p hipsparseGetGitRevision gets the hipSPARSE library git commit revision (SHA-1).
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetGitRevision(hipsparseHandle_t handle, char* rev);

/*! \ingroup aux_module
 *  \brief Specify user defined HIP stream
 *
 *  \details
 *  \p hipsparseSetStream specifies the stream to be used by the hipSPARSE library
 *  context and all subsequent function calls.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId);

/*! \ingroup aux_module
 *  \brief Get current stream from library context
 *
 *  \details
 *  \p hipsparseGetStream gets the hipSPARSE library context stream which is currently
 *  used for all subsequent function calls.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId);

/*! \ingroup aux_module
 *  \brief Specify pointer mode
 *
 *  \details
 *  \p hipsparseSetPointerMode specifies the pointer mode to be used by the hipSPARSE
 *  library context and all subsequent function calls. By default, all values are passed
 *  by reference on the host. Valid pointer modes are \ref HIPSPARSE_POINTER_MODE_HOST
 *  or \ref HIPSPARSE_POINTER_MODE_DEVICE.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode);

/*! \ingroup aux_module
 *  \brief Get current pointer mode from library context
 *
 *  \details
 *  \p hipsparseGetPointerMode gets the hipSPARSE library context pointer mode which
 *  is currently used for all subsequent function calls.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode);

/*! \ingroup aux_module
 *  \brief Create a matrix descriptor
 *  \details
 *  \p hipsparseCreateMatDescr creates a matrix descriptor. It initializes
 *  \ref hipsparseMatrixType_t to \ref HIPSPARSE_MATRIX_TYPE_GENERAL and
 *  \ref hipsparseIndexBase_t to \ref HIPSPARSE_INDEX_BASE_ZERO. It should be destroyed
 *  at the end using hipsparseDestroyMatDescr().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA);

/*! \ingroup aux_module
 *  \brief Destroy a matrix descriptor
 *
 *  \details
 *  \p hipsparseDestroyMatDescr destroys a matrix descriptor and releases all
 *  resources used by the descriptor.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Copy a matrix descriptor
 *  \details
 *  \p hipsparseCopyMatDescr copies a matrix descriptor. Both, source and destination
 *  matrix descriptors must be initialized prior to calling \p hipsparseCopyMatDescr.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src);

/*! \ingroup aux_module
 *  \brief Specify the matrix type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatType sets the matrix type of a matrix descriptor. Valid
 *  matrix types are \ref HIPSPARSE_MATRIX_TYPE_GENERAL,
 *  \ref HIPSPARSE_MATRIX_TYPE_SYMMETRIC, \ref HIPSPARSE_MATRIX_TYPE_HERMITIAN or
 *  \ref HIPSPARSE_MATRIX_TYPE_TRIANGULAR.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type);

/*! \ingroup aux_module
 *  \brief Get the matrix type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatType returns the matrix type of a matrix descriptor.
 */
HIPSPARSE_EXPORT
hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Specify the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatFillMode sets the matrix fill mode of a matrix descriptor.
 *  Valid fill modes are \ref HIPSPARSE_FILL_MODE_LOWER or
 *  \ref HIPSPARSE_FILL_MODE_UPPER.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA, hipsparseFillMode_t fillMode);

/*! \ingroup aux_module
 *  \brief Get the matrix fill mode of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatFillMode returns the matrix fill mode of a matrix descriptor.
 */
HIPSPARSE_EXPORT
hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Specify the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatDiagType sets the matrix diagonal type of a matrix
 *  descriptor. Valid diagonal types are \ref HIPSPARSE_DIAG_TYPE_UNIT or
 *  \ref HIPSPARSE_DIAG_TYPE_NON_UNIT.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA, hipsparseDiagType_t diagType);

/*! \ingroup aux_module
 *  \brief Get the matrix diagonal type of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatDiagType returns the matrix diagonal type of a matrix
 *  descriptor.
 */
HIPSPARSE_EXPORT
hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA);

/*! \ingroup aux_module
 *  \brief Specify the index base of a matrix descriptor
 *
 *  \details
 *  \p hipsparseSetMatIndexBase sets the index base of a matrix descriptor. Valid
 *  options are \ref HIPSPARSE_INDEX_BASE_ZERO or \ref HIPSPARSE_INDEX_BASE_ONE.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base);

/*! \ingroup aux_module
 *  \brief Get the index base of a matrix descriptor
 *
 *  \details
 *  \p hipsparseGetMatIndexBase returns the index base of a matrix descriptor.
 */
HIPSPARSE_EXPORT
hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup aux_module
 *  \brief Create a \p HYB matrix structure
 *
 *  \details
 *  \p hipsparseCreateHybMat creates a structure that holds the matrix in \p HYB
 *  storage format. It should be destroyed at the end using hipsparseDestroyHybMat().
 */
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA);

/*! \ingroup aux_module
 *  \brief Destroy a \p HYB matrix structure
 *
 *  \details
 *  \p hipsparseDestroyHybMat destroys a \p HYB structure.
 */
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA);
#endif

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsrsv2 info structure
 *
 *  \details
 *  \p hipsparseCreateBsrsv2Info creates a structure that holds the bsrsv2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsrsv2Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a bsrsv2 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsrsv2Info destroys a bsrsv2 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsrsm2 info structure
 *
 *  \details
 *  \p hipsparseCreateBsrsm2Info creates a structure that holds the bsrsm2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsrsm2Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a bsrsm2 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsrsm2Info destroys a bsrsm2 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsrilu02 info structure
 *
 *  \details
 *  \p hipsparseCreateBsrilu02Info creates a structure that holds the bsrilu02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsrilu02Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a bsrilu02 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsrilu02Info destroys a bsrilu02 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a bsric02 info structure
 *
 *  \details
 *  \p hipsparseCreateBsric02Info creates a structure that holds the bsric02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyBsric02Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a bsric02 info structure
 *
 *  \details
 *  \p hipsparseDestroyBsric02Info destroys a bsric02 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrsv2 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrsv2Info creates a structure that holds the csrsv2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrsv2Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csrsv2 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrsv2Info destroys a csrsv2 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrsm2 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrsm2Info creates a structure that holds the csrsm2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrsm2Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csrsm2 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrsm2Info destroys a csrsm2 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrilu02 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrilu02Info creates a structure that holds the csrilu02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrilu02Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csrilu02 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrilu02Info destroys a csrilu02 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csric02 info structure
 *
 *  \details
 *  \p hipsparseCreateCsric02Info creates a structure that holds the csric02 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsric02Info().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csric02 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsric02Info destroys a csric02 info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csru2csr info structure
 *
 *  \details
 *  \p hipsparseCreateCsru2csrInfo creates a structure that holds the csru2csr info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsru2csrInfo().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csru2csr info structure
 *
 *  \details
 *  \p hipsparseDestroyCsru2csrInfo destroys a csru2csr info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info);

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a color info structure
 *
 *  \details
 *  \p hipsparseCreateColorInfo creates a structure that holds the color info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyColorInfo().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateColorInfo(hipsparseColorInfo_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a color info structure
 *
 *  \details
 *  \p hipsparseDestroyColorInfo destroys a color info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyColorInfo(hipsparseColorInfo_t info);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a csrgemm2 info structure
 *
 *  \details
 *  \p hipsparseCreateCsrgemm2Info creates a structure that holds the csrgemm2 info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyCsrgemm2Info().
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a csrgemm2 info structure
 *
 *  \details
 *  \p hipsparseDestroyCsrgemm2Info destroys a csrgemm2 info structure.
 */
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info);
#endif

/* Info structures */
/*! \ingroup aux_module
 *  \brief Create a prune info structure
 *
 *  \details
 *  \p hipsparseCreatePruneInfo creates a structure that holds the prune info data
 *  that is gathered during the analysis routines available. It should be destroyed
 *  at the end using hipsparseDestroyPruneInfo().
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info);

/*! \ingroup aux_module
 *  \brief Destroy a prune info structure
 *
 *  \details
 *  \p hipsparseDestroyPruneInfo destroys a prune info structure.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info);

/*
* ===========================================================================
*    level 1 SPARSE
* ===========================================================================
*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Scale a sparse vector and add it to a dense vector.
*
*  \details
*  \p hipsparseXaxpyi multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
*  adds the result to the dense vector \f$y\f$, such that
*
*  \f[
*      y := y + \alpha \cdot x
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] = y[x_ind[i]] + alpha * x_val[i];
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const float*         alpha,
                                  const float*         xVal,
                                  const int*           xInd,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const double*        alpha,
                                  const double*        xVal,
                                  const int*           xInd,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    alpha,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZaxpyi(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level1_module
*  \brief Compute the dot product of a sparse vector with a dense vector.
*
*  \details
*  \p hipsparseXdoti computes the dot product of the sparse vector \f$x\f$ with the
*  dense vector \f$y\f$, such that
*  \f[
*    \text{result} := y^T x
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          result += x_val[i] * y[x_ind[i]];
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 const float*         y,
                                 float*               result,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 const double*        y,
                                 double*              result,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 const hipComplex*    y,
                                 hipComplex*          result,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdoti(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       result,
                                 hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level1_module
*  \brief Compute the dot product of a complex conjugate sparse vector with a dense
*  vector.
*
*  \details
*  \p hipsparseXdotci computes the dot product of the complex conjugate sparse vector
*  \f$x\f$ with the dense vector \f$y\f$, such that
*  \f[
*    \text{result} := \bar{x}^H y
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          result += conj(x_val[i]) * y[x_ind[i]];
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdotci(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  const hipComplex*    y,
                                  hipComplex*          result,
                                  hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdotci(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  const hipDoubleComplex* y,
                                  hipDoubleComplex*       result,
                                  hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Gather elements from a dense vector and store them into a sparse vector.
*
*  \details
*  \p hipsparseXgthr gathers the elements that are listed in \p x_ind from the dense
*  vector \f$y\f$ and stores them in the sparse vector \f$x\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_val[i] = y[x_ind[i]];
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         y,
                                 float*               xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        y,
                                 double*              xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    y,
                                 hipComplex*          xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgthr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       xVal,
                                 const int*              xInd,
                                 hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Gather and zero out elements from a dense vector and store them into a sparse
*  vector.
*
*  \details
*  \p hipsparseXgthrz gathers the elements that are listed in \p x_ind from the dense
*  vector \f$y\f$ and stores them in the sparse vector \f$x\f$. The gathered elements
*  in \f$y\f$ are replaced by zero.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_val[i]    = y[x_ind[i]];
*          y[x_ind[i]] = 0;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  float*               y,
                                  float*               xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  double*              y,
                                  double*              xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  hipComplex*          y,
                                  hipComplex*          xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  hipDoubleComplex*    y,
                                  hipDoubleComplex*    xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \p hipsparseXroti applies the Givens rotation matrix \f$G\f$ to the sparse vector
*  \f$x\f$ and the dense vector \f$y\f$, where
*  \f[
*    G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_tmp = x_val[i];
*          y_tmp = y[x_ind[i]];
*
*          x_val[i]    = c * x_tmp + s * y_tmp;
*          y[x_ind[i]] = c * y_tmp - s * x_tmp;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 float*               xVal,
                                 const int*           xInd,
                                 float*               y,
                                 const float*         c,
                                 const float*         s,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 double*              xVal,
                                 const int*           xInd,
                                 double*              y,
                                 const double*        c,
                                 const double*        s,
                                 hipsparseIndexBase_t idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level1_module
*  \brief Scatter elements from a dense vector across a sparse vector.
*
*  \details
*  \p hipsparseXsctr scatters the elements that are listed in \p x_ind from the sparse
*  vector \f$x\f$ into the dense vector \f$y\f$. Indices of \f$y\f$ that are not listed
*  in \p x_ind remain unchanged.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] = x_val[i];
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 float*               y,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 double*              y,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 hipComplex*          y,
                                 hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZsctr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 hipDoubleComplex*       y,
                                 hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

/*
* ===========================================================================
*    level 2 SPARSE
* ===========================================================================
*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in CSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      for(i = 0; i < m; ++i)
*      {
*          y[i] = beta * y[i];
*
*          for(j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; ++j)
*          {
*              y[i] = y[i] + alpha * csr_val[j] * x[csr_col_ind[j]];
*          }
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y);
/**@}*/
#endif

/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() or hipsparseZcsrsv2_solve()
*  computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position,
*  using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsrsv2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position);

/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_bufferSize returns the size of the temporary storage buffer that
*  is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
*  hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_bufferSizeExt returns the size of the temporary storage buffer that
*  is required by hipsparseScsrsv2_analysis(), hipsparseDcsrsv2_analysis(),
*  hipsparseCcsrsv2_analysis(), hipsparseZcsrsv2_analysis(), hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_analysis performs the analysis step for hipsparseScsrsv2_solve(),
*  hipsparseDcsrsv2_solve(), hipsparseCcsrsv2_solve() and hipsparseZcsrsv2_solve().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsv2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \p hipsparseXcsrsv2_solve requires a user allocated temporary buffer. Its size is
*  returned by hipsparseXcsrsv2_bufferSize() or hipsparseXcsrsv2_bufferSizeExt().
*  Furthermore, analysis meta data is required. It can be obtained by
*  hipsparseXcsrsv2_analysis(). \p hipsparseXcsrsv2_solve reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be checked calling
*  hipsparseXcsrsv2_zeroPivot(). If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
*  \p trans == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using HYB storage format
*
*  \details
*  \p hipsparseXhybmv multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times n\f$
*  matrix, defined in HYB storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseShybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseChybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y);
/**@}*/
#endif

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication using BSR storage format
*
*  \details
*  \p hipsparseXbsrmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
*  matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse matrix vector multiplication with mask operation using BSR storage format
*
*  \details
*  \p hipsparseXbsrxmv multiplies the scalar \f$\alpha\f$ with a sparse
*  \f$(mb \cdot \text{block_dim}) \times (nb \cdot \text{block_dim})\f$
*  modified matrix, defined in BSR storage format, and the dense vector \f$x\f$ and adds the
*  result to the dense vector \f$y\f$ that is multiplied by the scalar \f$\beta\f$,
*  such that
*  \f[
*    y := \left( \alpha \cdot op(A) \cdot x + \beta \cdot y \right)\left( \text{mask} \right),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  The \f$\text{mask}\f$ is defined as an array of block row indices.
*  The input sparse matrix is defined with a modified BSR storage format where the beginning and the end of each row
*  is defined with two arrays, \p bsr_row_ptr and \p bsr_end_ptr (both of size \p mb), rather the usual \p bsr_row_ptr of size \p mb + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  Currently, \p block_dim == 1 is not supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const float*              alpha,
                                   const hipsparseMatDescr_t descr,
                                   const float*              bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const float*              x,
                                   const float*              beta,
                                   float*                    y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const double*             alpha,
                                   const hipsparseMatDescr_t descr,
                                   const double*             bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const double*             x,
                                   const double*             beta,
                                   double*                   y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const hipComplex*         alpha,
                                   const hipsparseMatDescr_t descr,
                                   const hipComplex*         bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const hipComplex*         x,
                                   const hipComplex*         beta,
                                   hipComplex*               y);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const hipDoubleComplex*   alpha,
                                   const hipsparseMatDescr_t descr,
                                   const hipDoubleComplex*   bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const hipDoubleComplex*   x,
                                   const hipDoubleComplex*   beta,
                                   hipDoubleComplex*         y);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseXbsrsv2_analysis() or
*  hipsparseXbsrsv2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
*  is stored in \p position, using same index base as the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXbsrsv2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position);

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_bufferSize returns the size of the temporary storage buffer that
*  is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_bufferSizeExt returns the size of the temporary storage buffer that
*  is required by hipsparseXbsrsv2_analysis() and hipsparseXbsrsv2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dirA,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_analysis performs the analysis step for hipsparseXbsrsv2_solve().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/

/*! \ingroup level2_module
*  \brief Sparse triangular solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsv2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \p hipsparseXbsrsv2_solve requires a user allocated temporary buffer. Its size is
*  returned by hipsparseXbsrsv2_bufferSize() or hipsparseXbsrsv2_bufferSizeExt().
*  Furthermore, analysis meta data is required. It can be obtained by
*  hipsparseXbsrsv2_analysis(). \p hipsparseXbsrsv2_solve reports the first zero pivot
*  (either numerical or structural zero). The zero pivot status can be checked calling
*  hipsparseXbsrsv2_zeroPivot(). If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE and
*  \p trans == \ref HIPSPARSE_OPERATION_TRANSPOSE is supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/

/*! \ingroup level2_module
 *  \brief Dense matrix sparse vector multiplication
 *
 *  \details
 *  \p hipsparseXgemvi_bufferSize returns the size of the temporary storage buffer
 *  required by hipsparseXgemvi(). The temporary storage buffer must be allocated by the
 *  user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize);
/**@}*/

/*! \ingroup level2_module
 *  \brief Dense matrix sparse vector multiplication
 *
 *  \details
 *  \p hipsparseXgemvi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times n\f$
 *  matrix \f$A\f$ and the sparse vector \f$x\f$ and adds the result to the dense vector
 *  \f$y\f$ that is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    y := \alpha \cdot op(A) \cdot x + \beta \cdot y,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
 *        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
 *        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \p hipsparseXgemvi requires a user allocated temporary buffer. Its size is returned
 *  by hipsparseXgemvi_bufferSize().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const float*         alpha,
                                  const float*         A,
                                  int                  lda,
                                  int                  nnz,
                                  const float*         x,
                                  const int*           xInd,
                                  const float*         beta,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const double*        alpha,
                                  const double*        A,
                                  int                  lda,
                                  int                  nnz,
                                  const double*        x,
                                  const int*           xInd,
                                  const double*        beta,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const hipComplex*    alpha,
                                  const hipComplex*    A,
                                  int                  lda,
                                  int                  nnz,
                                  const hipComplex*    x,
                                  const int*           xInd,
                                  const hipComplex*    beta,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemvi(hipsparseHandle_t       handle,
                                  hipsparseOperation_t    transA,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  int                     nnz,
                                  const hipDoubleComplex* x,
                                  const int*              xInd,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase,
                                  void*                   pBuffer);
/**@}*/

/*
* ===========================================================================
*    level 3 SPARSE
* ===========================================================================
*/

/*! \ingroup level3_module
 *  \brief Sparse matrix dense matrix multiplication using BSR storage format
 *
 *  \details
 *  \p hipsparseXbsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$mb \times kb\f$
 *  matrix \f$A\f$, defined in BSR storage format, and the dense \f$k \times n\f$
 *  matrix \f$B\f$ (where \f$k = block\_dim \times kb\f$) and adds the result to the dense
 *  \f$m \times n\f$ matrix \f$C\f$ (where \f$m = block\_dim \times mb\f$) that
 *  is multiplied by the scalar \f$\beta\f$, such that
 *  \f[
 *    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
 *  \f]
 *  with
 *  \f[
 *    op(A) = \left\{
 *    \begin{array}{ll}
 *        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
 *    \end{array}
 *    \right.
 *  \f]
 *  and
 *  \f[
 *    op(B) = \left\{
 *    \begin{array}{ll}
 *        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
 *        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
 *    \end{array}
 *    \right.
 *  \f]
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 *
 *  \note
 *  Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level3_module
*  \brief Sparse matrix dense matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrmm multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
*  matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
*  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot B + \beta \cdot C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      for(i = 0; i < ldc; ++i)
*      {
*          for(j = 0; j < n; ++j)
*          {
*              C[i][j] = beta * C[i][j];
*
*              for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
*              {
*                  C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
*              }
*          }
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup level3_module
*  \brief Sparse matrix dense matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrmm2 multiplies the scalar \f$\alpha\f$ with a sparse \f$m \times k\f$
*  matrix \f$A\f$, defined in CSR storage format, and the dense \f$k \times n\f$
*  matrix \f$B\f$ and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \code{.c}
*      for(i = 0; i < ldc; ++i)
*      {
*          for(j = 0; j < n; ++j)
*          {
*              C[i][j] = beta * C[i][j];
*
*              for(k = csr_row_ptr[i]; k < csr_row_ptr[i + 1]; ++k)
*              {
*                  C[i][j] += alpha * csr_val[k] * B[csr_col_ind[k]][j];
*              }
*          }
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const float*              alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const float*              csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const float*              B,
                                   int                       ldb,
                                   const float*              beta,
                                   float*                    C,
                                   int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const double*             alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const double*             csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const double*             B,
                                   int                       ldb,
                                   const double*             beta,
                                   double*                   C,
                                   int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipComplex*         alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipComplex*         csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipComplex*         B,
                                   int                       ldb,
                                   const hipComplex*         beta,
                                   hipComplex*               C,
                                   int                       ldc);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipDoubleComplex*   alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipDoubleComplex*   csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipDoubleComplex*   B,
                                   int                       ldb,
                                   const hipDoubleComplex*   beta,
                                   hipDoubleComplex*         C,
                                   int                       ldc);
/**@}*/
#endif

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseXbsrsm2_analysis() or
*  hipsparseXbsrsm2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
*  is stored in \p position, using same index base as the BSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXbsrsm2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsrsm2_zeroPivot(hipsparseHandle_t handle, bsrsm2Info_t info, int* position);

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsm2_buffer_size returns the size of the temporary storage buffer that
*  is required by hipsparseXbsrsm2_analysis() and hipsparseXbsrsm2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsm2_analysis performs the analysis step for hipsparseXbsrsm2_solve().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using BSR storage format
*
*  \details
*  \p hipsparseXbsrsm2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in BSR storage format, a dense solution matrix
*  \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot op(X) = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  ,
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if trans_X == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        X^T, & \text{if trans_X == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        X^H, & \text{if trans_X == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \p hipsparseXbsrsm2_solve requires a user allocated temporary buffer. Its size is
*  returned by hipsparseXbsrsm2_bufferSize(). Furthermore, analysis meta data is
*  required. It can be obtained by hipsparseXbsrsm2_analysis(). \p hipsparseXbsrsm2_solve
*  reports the first zero pivot (either numerical or structural zero). The zero pivot
*  status can be checked calling hipsparseXbsrsm2_zeroPivot(). If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse BSR matrix has to be sorted.
*
*  \note
*  Operation type of B and X must match, if \f$op(B)=B, op(X)=X\f$.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans_A != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
*  \p trans_X != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const float*              B,
                                         int                       ldb,
                                         float*                    X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const double*             B,
                                         int                       ldb,
                                         double*                   X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipComplex*         B,
                                         int                       ldb,
                                         hipComplex*               X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipDoubleComplex*   B,
                                         int                       ldb,
                                         hipDoubleComplex*         X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseXcsrsm2_analysis() or
*  hipsparseXcsrsm2_solve() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
*  is stored in \p position, using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsrsm2_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsrsm2_zeroPivot(hipsparseHandle_t handle, csrsm2Info_t info, int* position);

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_bufferSizeExt returns the size of the temporary storage buffer
*  that is required by hipsparseXcsrsm2_analysis() and hipsparseXcsrsm2_solve(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const float*              alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const float*              csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const float*              B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const double*             alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const double*             csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const double*             B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipComplex*         alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipComplex*         B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipDoubleComplex*   alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipDoubleComplex*   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipDoubleComplex*   B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_analysis performs the analysis step for hipsparseXcsrsm2_solve().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const float*              alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const float*              B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const double*             alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const double*             B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipComplex*         alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipComplex*         B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipDoubleComplex*   alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipDoubleComplex*   B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer);
/**@}*/

/*! \ingroup level3_module
*  \brief Sparse triangular system solve using CSR storage format
*
*  \details
*  \p hipsparseXcsrsm2_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR storage format, a dense solution matrix
*  \f$X\f$ and the right-hand side matrix \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot op(X) = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  ,
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(X) = \left\{
*    \begin{array}{ll}
*        X,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        X^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        X^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \p hipsparseXcsrsm2_solve requires a user allocated temporary buffer. Its size is
*  returned by hipsparseXcsrsm2_bufferSizeExt(). Furthermore, analysis meta data is
*  required. It can be obtained by hipsparseXcsrsm2_analysis().
*  \p hipsparseXcsrsm2_solve reports the first zero pivot (either numerical or structural
*  zero). The zero pivot status can be checked calling hipsparseXcsrsm2_zeroPivot(). If
*  \ref hipsparseDiagType_t == \ref HIPSPARSE_DIAG_TYPE_UNIT, no zero pivot will be
*  reported, even if \f$A_{j,j} = 0\f$ for some \f$j\f$.
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Currently, only \p trans_A != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE and
*  \p trans_B != \ref HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE is supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         float*                    B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         double*                   B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipComplex*               B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipDoubleComplex*         B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup level3_module
*  \brief Dense matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXgemmi multiplies the scalar \f$\alpha\f$ with a dense \f$m \times k\f$
*  matrix \f$A\f$ and the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format and adds the result to the dense \f$m \times n\f$ matrix \f$C\f$ that
*  is multiplied by the scalar \f$\beta\f$, such that
*  \f[
*    C := \alpha \cdot op(A) \cdot op(B) + \beta \cdot C
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const float*      alpha,
                                  const float*      A,
                                  int               lda,
                                  const float*      cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const float*      beta,
                                  float*            C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const double*     alpha,
                                  const double*     A,
                                  int               lda,
                                  const double*     cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const double*     beta,
                                  double*           C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const hipComplex* alpha,
                                  const hipComplex* A,
                                  int               lda,
                                  const hipComplex* cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const hipComplex* beta,
                                  hipComplex*       C,
                                  int               ldc);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgemmi(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  const hipDoubleComplex* cscValB,
                                  const int*              cscColPtrB,
                                  const int*              cscRowIndB,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       C,
                                  int                     ldc);
/**@}*/
#endif

/*
* ===========================================================================
*    extra SPARSE
* ===========================================================================
*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeamNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
*  size \p m + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t         handle,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeamNnz().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const float*              alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const float*              beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const double*             alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const double*             beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipComplex*         alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipComplex*         beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipDoubleComplex*   alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipDoubleComplex*   beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2_bufferSizeExt returns the size of the temporary storage buffer
*  that is required by hipsparseXcsrgeam2Nnz() and hipsparseXcsrgeam2(). The temporary
*  storage buffer must be allocated by the user.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const float*              csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const float*              csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const float*              csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const double*             csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const double*             csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const double*             csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipComplex*         csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipComplex*         csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipDoubleComplex*   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipDoubleComplex*   csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipDoubleComplex*   csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting matrix C. It is assumed that \p csr_row_ptr_C has been allocated with
*  size \p m + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgeam2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        void*                     workspace);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix addition using CSR storage format
*
*  \details
*  \p hipsparseXcsrgeam2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times n\f$ matrix \f$A\f$, defined in CSR storage format, multiplies the
*  scalar \f$\beta\f$ with the sparse \f$m \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and adds both resulting matrices to obtain the sparse
*  \f$m \times n\f$ matrix \f$C\f$, defined in CSR storage format, such that
*  \f[
*    C := \alpha \cdot A + \beta \cdot B.
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgeam2Nnz().
*
*  \note Both scalars \f$\alpha\f$ and \f$beta\f$ have to be valid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemmNnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
*  with size \p m + 1.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \p trans_A == \p trans_B == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is
*  supported.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm multiplies the sparse \f$m \times k\f$ matrix \f$A\f$, defined in
*  CSR storage format with the sparse \f$k \times n\f$ matrix \f$B\f$, defined in CSR
*  storage format, and stores the result in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  defined in CSR storage format, such that
*  \f[
*    C := op(A) \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans_A == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans_A == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans_A == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemmNnz().
*
*  \note Currently, only \p trans_A == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \p trans_B == \ref HIPSPARSE_OPERATION_NON_TRANSPOSE is supported.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 12000)
/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2_bufferSizeExt returns the size of the temporary storage buffer
*  that is required by hipsparseXcsrgemm2Nnz() and hipsparseXcsrgemm2(). The temporary
*  storage buffer must be allocated by the user.
*
*  \note
*  Please note, that for matrix products with more than 4096 non-zero entries per row,
*  additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes);
/**@}*/

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2Nnz computes the total CSR non-zero elements and the CSR row
*  offsets, that point to the start of every row of the sparse CSR matrix, of the
*  resulting multiplied matrix C. It is assumed that \p csr_row_ptr_C has been allocated
*  with size \p m + 1.
*  The required buffer size can be obtained by hipsparseXcsrgemm2_bufferSizeExt().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*
*  \note
*  Please note, that for matrix products with more than 8192 intermediate products per
*  row, additional temporary storage buffer is allocated by the algorithm.
*
*  \note
*  Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrgemm2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrD,
                                        int                       nnzD,
                                        const int*                csrRowPtrD,
                                        const int*                csrColIndD,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        const csrgemm2Info_t      info,
                                        void*                     pBuffer);

/*! \ingroup extra_module
*  \brief Sparse matrix sparse matrix multiplication using CSR storage format
*
*  \details
*  \p hipsparseXcsrgemm2 multiplies the scalar \f$\alpha\f$ with the sparse
*  \f$m \times k\f$ matrix \f$A\f$, defined in CSR storage format, and the sparse
*  \f$k \times n\f$ matrix \f$B\f$, defined in CSR storage format, and adds the result
*  to the sparse \f$m \times n\f$ matrix \f$D\f$ that is multiplied by \f$\beta\f$. The
*  final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$, defined in CSR
*  storage format, such
*  that
*  \f[
*    C := \alpha \cdot A \cdot B + \beta \cdot D
*  \f]
*
*  It is assumed that \p csr_row_ptr_C has already been filled and that \p csr_val_C and
*  \p csr_col_ind_C are allocated by the user. \p csr_row_ptr_C and allocation size of
*  \p csr_col_ind_C and \p csr_val_C is defined by the number of non-zero elements of
*  the sparse CSR matrix C. Both can be obtained by hipsparseXcsrgemm2Nnz(). The
*  required buffer size for the computation can be obtained by
*  hipsparseXcsrgemm2_bufferSizeExt().
*
*  \note If \f$\alpha == 0\f$, then \f$C = \beta \cdot D\f$ will be computed.
*  \note If \f$\beta == 0\f$, then \f$C = \alpha \cdot A \cdot B\f$ will be computed.
*  \note \f$\alpha == beta == 0\f$ is invalid.
*  \note Currently, only \ref HIPSPARSE_MATRIX_TYPE_GENERAL is supported.
*  \note This function is non blocking and executed asynchronously with respect to the
*        host. It may return before the actual computation has finished.
*  \note Please note, that for matrix products with more than 4096 non-zero entries per
*  row, additional temporary storage buffer is allocated by the algorithm.
*/
/**@{*/
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const float*              csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const double*             csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipComplex*         csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipDoubleComplex*   csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer);
/**@}*/
#endif

/*
* ===========================================================================
*    preconditioner SPARSE
* ===========================================================================
*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
 *  structural or numerical zero has been found during hipsparseXbsrilu02_analysis() or
 *  hipsparseXbsrilu02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is
 *  stored in \p position, using same index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position \f$=j\f$ means that either the diagonal block
 *  \f$A_{j,j}\f$ is missing (structural zero) or the diagonal block \f$A_{j,j}\f$ is not
 *  invertible (numerical zero).
 *
 *  \note \p hipsparseXbsrilu02_zeroPivot is a blocking function. It might influence
 *  performance negatively.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle, bsrilu02Info_t info, int* position);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_numericBoost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  double*           boost_val);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_bufferSize returns the size of the temporary storage buffer
 *  that is required by hipsparseXbsrilu02_analysis() and hipsparseXbsrilu02_solve().
 *  The temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         bsrSortedValA,
                                                const int*                bsrSortedRowPtrA,
                                                const int*                bsrSortedColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02_analysis performs the analysis step for hipsparseXbsrilu02().
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using BSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXbsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
 *  pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LU
 *  \f]
 *
 *  \p hipsparseXbsrilu02 requires a user allocated temporary buffer. Its size is
 *  returned by hipsparseXbsrilu02_bufferSize(). Furthermore, analysis meta data is
 *  required. It can be obtained by hipsparseXbsrilu02_analysis(). \p hipsparseXbsrilu02
 *  reports the first zero pivot (either numerical or structural zero). The zero pivot
 *  status can be obtained by calling hipsparseXbsrilu02_zeroPivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         bsrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             bsrSortedRowPtrA,
                                     const int*             bsrSortedColIndA,
                                     int                    blockDim,
                                     bsrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseXcsrilu02() computation.
*  The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is stored in \p position, using same
*  index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsrilu02_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position);

/*! \ingroup precond_module
 *  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR storage
 *  format
 *
 *  \details
 *  \p hipsparseXcsrilu02_numericBoost enables the user to replace a numerical value in
 *  an incomplete LU factorization. \p tol is used to determine whether a numerical value
 *  is replaced by \p boost_val, such that \f$A_{j,j} = \text{boost_val}\f$ if
 *  \f$\text{tol} \ge \left|A_{j,j}\right|\f$.
 *
 *  \note The boost value is enabled by setting \p enable_boost to 1 or disabled by
 *  setting \p enable_boost to 0.
 *
 *  \note \p tol and \p boost_val can be in host or device memory.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  double*           boost_val);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_bufferSize returns the size of the temporary storage buffer
*  that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_bufferSizeExt returns the size of the temporary storage buffer
*  that is required by hipsparseXcsrilu02_analysis() and hipsparseXcsrilu02_solve(). the
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02_analysis performs the analysis step for hipsparseXcsrilu02().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const float*              csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const double*             csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipDoubleComplex*   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete LU factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsrilu02 computes the incomplete LU factorization with 0 fill-ins and no
*  pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LU
*  \f]
*
*  \p hipsparseXcsrilu02 requires a user allocated temporary buffer. Its size is returned
*  by hipsparseXcsrilu02_bufferSize() or hipsparseXcsrilu02_bufferSizeExt(). Furthermore,
*  analysis meta data is required. It can be obtained by hipsparseXcsrilu02_analysis().
*  \p hipsparseXcsrilu02 reports the first zero pivot (either numerical or structural
*  zero). The zero pivot status can be obtained by calling hipsparseXcsrilu02_zeroPivot().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
 *  structural or numerical zero has been found during hipsparseXbsric02_analysis() or
 *  hipsparseXbsric02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$ is
 *  stored in \p position, using same index base as the BSR matrix.
 *
 *  \p position can be in host or device memory. If no zero pivot has been found,
 *  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
 *
 *  \note
 *  If a zero pivot is found, \p position=j means that either the diagonal block \p A(j,j)
 *  is missing (structural zero) or the diagonal block \p A(j,j) is not positive definite
 *  (numerical zero).
 *
 *  \note \p hipsparseXbsric02_zeroPivot is a blocking function. It might influence
 *  performance negatively.
 */
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position);

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_bufferSize returns the size of the temporary storage buffer
 *  that is required by hipsparseXbsric02_analysis() and hipsparseXbsric02(). The
 *  temporary storage buffer must be allocated by the user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02_analysis performs the analysis step for hipsparseXbsric02().
 *
 *  \note
 *  If the matrix sparsity pattern changes, the gathered information will become invalid.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
/**@}*/

/*! \ingroup precond_module
 *  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using BSR
 *  storage format
 *
 *  \details
 *  \p hipsparseXbsric02 computes the incomplete Cholesky factorization with 0 fill-ins
 *  and no pivoting of a sparse \f$mb \times mb\f$ BSR matrix \f$A\f$, such that
 *  \f[
 *    A \approx LL^T
 *  \f]
 *
 *  \p hipsparseXbsric02 requires a user allocated temporary buffer. Its size is returned
 *  by hipsparseXbsric02_bufferSize(). Furthermore, analysis meta data is required. It
 *  can be obtained by hipsparseXbsric02_analysis(). \p hipsparseXbsric02 reports the
 *  first zero pivot (either numerical or structural zero). The zero pivot status can be
 *  obtained by calling hipsparseXbsric02_zeroPivot().
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_zeroPivot returns \ref HIPSPARSE_STATUS_ZERO_PIVOT, if either a
*  structural or numerical zero has been found during hipsparseXcsric02_analysis() or
*  hipsparseXcsric02() computation. The first zero pivot \f$j\f$ at \f$A_{j,j}\f$
*  is stored in \p position, using same index base as the CSR matrix.
*
*  \p position can be in host or device memory. If no zero pivot has been found,
*  \p position is set to -1 and \ref HIPSPARSE_STATUS_SUCCESS is returned instead.
*
*  \note \p hipsparseXcsric02_zeroPivot is a blocking function. It might influence
*  performance negatively.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseXcsric02_zeroPivot(hipsparseHandle_t handle, csric02Info_t info, int* position);

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_bufferSize returns the size of the temporary storage buffer
*  that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_bufferSizeExt returns the size of the temporary storage buffer
*  that is required by hipsparseXcsric02_analysis() and hipsparseXcsric02().
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02_analysis performs the analysis step for hipsparseXcsric02().
*
*  \note
*  If the matrix sparsity pattern changes, the gathered information will become invalid.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Incomplete Cholesky factorization with 0 fill-ins and no pivoting using CSR
*  storage format
*
*  \details
*  \p hipsparseXcsric02 computes the incomplete Cholesky factorization with 0 fill-ins
*  and no pivoting of a sparse \f$m \times m\f$ CSR matrix \f$A\f$, such that
*  \f[
*    A \approx LL^T
*  \f]
*
*  \p hipsparseXcsric02 requires a user allocated temporary buffer. Its size is returned
*  by hipsparseXcsric02_bufferSize() or hipsparseXcsric02_bufferSizeExt(). Furthermore,
*  analysis meta data is required. It can be obtained by hipsparseXcsric02_analysis().
*  \p hipsparseXcsric02 reports the first zero pivot (either numerical or structural
*  zero). The zero pivot status can be obtained by calling hipsparseXcsric02_zeroPivot().
*
*  \note
*  The sparse CSR matrix has to be sorted. This can be achieved by calling
*  hipsparseXcsrsort().
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver with pivoting
*
*  \details
*  \p hipsparseXgtsv2_bufferSize returns the size of the temporary storage buffer
*  that is required by hipsparseXgtsv2(). The temporary storage buffer must be
*  allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const float*      dl,
                                                const float*      d,
                                                const float*      du,
                                                const float*      B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const double*     dl,
                                                const double*     d,
                                                const double*     du,
                                                const double*     B,
                                                int               db,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const hipComplex* dl,
                                                const hipComplex* d,
                                                const hipComplex* du,
                                                const hipComplex* B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(hipsparseHandle_t       handle,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* dl,
                                                const hipDoubleComplex* d,
                                                const hipDoubleComplex* du,
                                                const hipDoubleComplex* B,
                                                int                     ldb,
                                                size_t*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver with pivoting
*
*  \details
*  \p hipsparseXgtsv2 solves a tridiagonal system for multiple right hand sides using pivoting.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const float*      dl,
                                  const float*      d,
                                  const float*      du,
                                  float*            B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const double*     dl,
                                  const double*     d,
                                  const double*     du,
                                  double*           B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const hipComplex* dl,
                                  const hipComplex* d,
                                  const hipComplex* du,
                                  hipComplex*       B,
                                  int               ldb,
                                  void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* dl,
                                  const hipDoubleComplex* d,
                                  const hipDoubleComplex* du,
                                  hipDoubleComplex*       B,
                                  int                     ldb,
                                  void*                   pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2_nopivot_bufferSizeExt returns the size of the temporary storage
*  buffer that is required by hipsparseXgtsv2_nopivot(). The temporary storage buffer
*  must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const float*      dl,
                                                        const float*      d,
                                                        const float*      du,
                                                        const float*      B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const double*     dl,
                                                        const double*     d,
                                                        const double*     du,
                                                        const double*     B,
                                                        int               db,
                                                        size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const hipComplex* dl,
                                                        const hipComplex* d,
                                                        const hipComplex* du,
                                                        const hipComplex* B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t       handle,
                                                        int                     m,
                                                        int                     n,
                                                        const hipDoubleComplex* dl,
                                                        const hipDoubleComplex* d,
                                                        const hipDoubleComplex* du,
                                                        const hipDoubleComplex* B,
                                                        int                     ldb,
                                                        size_t*                 pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2_nopivot solves a tridiagonal linear system for multiple right-hand sides
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const float*      dl,
                                          const float*      d,
                                          const float*      du,
                                          float*            B,
                                          int               ldb,
                                          void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const double*     dl,
                                          const double*     d,
                                          const double*     du,
                                          double*           B,
                                          int               ldb,
                                          void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const hipComplex* dl,
                                          const hipComplex* d,
                                          const hipComplex* du,
                                          hipComplex*       B,
                                          int               ldb,
                                          void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2_nopivot(hipsparseHandle_t       handle,
                                          int                     m,
                                          int                     n,
                                          const hipDoubleComplex* dl,
                                          const hipDoubleComplex* d,
                                          const hipDoubleComplex* du,
                                          hipDoubleComplex*       B,
                                          int                     ldb,
                                          void*                   pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2StridedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer that is required by hipsparseXgtsv2StridedBatch(). The temporary storage buffer
*  must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* x,
                                                            int                     batchCount,
                                                            int                     batchStride,
                                                            size_t* pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Strided Batch tridiagonal solver (no pivoting)
*
*  \details
*  \p hipsparseXgtsv2StridedBatch solves a batched tridiagonal linear system
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t       handle,
                                              int                     m,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       x,
                                              int                     batchCount,
                                              int                     batchStride,
                                              void*                   pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch tridiagonal solver
*
*  \details
*  \p hipsparseXgtsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer that is required by hipsparseXgtsvInterleavedBatch(). The temporary storage buffer
*  must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch tridiagonal solver
*
*  \details
*  \p hipsparseXgtsvInterleavedBatch solves a batched tridiagonal linear system
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgtsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer);
/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch pentadiagonal solver
*
*  \details
*  \p hipsparseXgpsvInterleavedBatch_bufferSizeExt returns the size of the temporary storage
*  buffer that is required by hipsparseXgpsvInterleavedBatch(). The temporary storage buffer
*  must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const float*      ds,
                                                               const float*      dl,
                                                               const float*      d,
                                                               const float*      du,
                                                               const float*      dw,
                                                               const float*      x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const double*     ds,
                                                               const double*     dl,
                                                               const double*     d,
                                                               const double*     du,
                                                               const double*     dw,
                                                               const double*     x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                               int               algo,
                                                               int               m,
                                                               const hipComplex* ds,
                                                               const hipComplex* dl,
                                                               const hipComplex* d,
                                                               const hipComplex* du,
                                                               const hipComplex* dw,
                                                               const hipComplex* x,
                                                               int               batchCount,
                                                               size_t* pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgpsvInterleavedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                               int                     algo,
                                                               int                     m,
                                                               const hipDoubleComplex* ds,
                                                               const hipDoubleComplex* dl,
                                                               const hipDoubleComplex* d,
                                                               const hipDoubleComplex* du,
                                                               const hipDoubleComplex* dw,
                                                               const hipDoubleComplex* x,
                                                               int                     batchCount,
                                                               size_t* pBufferSizeInBytes);

/**@}*/

/*! \ingroup precond_module
*  \brief Interleaved Batch pentadiagonal solver
*
*  \details
*  \p hipsparseXgpsvInterleavedBatch solves a batched pentadiagonal linear system
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 float*            ds,
                                                 float*            dl,
                                                 float*            d,
                                                 float*            du,
                                                 float*            dw,
                                                 float*            x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 double*           ds,
                                                 double*           dl,
                                                 double*           d,
                                                 double*           du,
                                                 double*           dw,
                                                 double*           x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipComplex*       ds,
                                                 hipComplex*       dl,
                                                 hipComplex*       d,
                                                 hipComplex*       du,
                                                 hipComplex*       dw,
                                                 hipComplex*       x,
                                                 int               batchCount,
                                                 void*             pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgpsvInterleavedBatch(hipsparseHandle_t handle,
                                                 int               algo,
                                                 int               m,
                                                 hipDoubleComplex* ds,
                                                 hipDoubleComplex* dl,
                                                 hipDoubleComplex* d,
                                                 hipDoubleComplex* du,
                                                 hipDoubleComplex* dw,
                                                 hipDoubleComplex* x,
                                                 int               batchCount,
                                                 void*             pBuffer);

/**@}*/

/*
* ===========================================================================
*    Sparse Format Conversions
* ===========================================================================
*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row or column and the total
*  number of nonzero elements in a dense matrix.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const float*              A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const double*             A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipComplex*         A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipDoubleComplex*   A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix A in dense format into a sparse matrix in CSR format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays
*  are filled in based on nnz_per_row, which can be pre-computed with hipsparseXnnz().
*  It is executed asynchronously with respect to the host and may return control to the
*  application on the host before the entire result is ready.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnz_per_rows,
                                      float*                    csr_val,
                                      int*                      csr_row_ptr,
                                      int*                      csr_col_ind);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnz_per_rows,
                                      double*                   csr_val,
                                      int*                      csr_row_ptr,
                                      int*                      csr_col_ind);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnz_per_rows,
                                      hipComplex*               csr_val,
                                      int*                      csr_row_ptr,
                                      int*                      csr_col_ind);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnz_per_rows,
                                      hipDoubleComplex*         csr_val,
                                      int*                      csr_row_ptr,
                                      int*                      csr_col_ind);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the the size of the user allocated temporary storage buffer used when converting and pruning
*  a dense matrix to a CSR matrix.
*
*  \details
*  \p hipsparseXpruneDense2csr_bufferSizeExt returns the size of the temporary storage buffer
*  that is required by hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr(). The
*  temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const float*              A,
                                                      int                       lda,
                                                      const float*              threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const float*              csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const double*             A,
                                                      int                       lda,
                                                      const double*             threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const double*             csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const float*              A,
                                                         int                       lda,
                                                         const float*              threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const float*              csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t*                   bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const double*             A,
                                                         int                       lda,
                                                         const double*             threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const double*             csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t*                   bufferSize);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix once elements less than the threshold are pruned
*  from the matrix.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const float*              A,
                                              int                       lda,
                                              const float*              threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const double*             A,
                                              int                       lda,
                                              const double*             threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the matrix A in dense format into a sparse matrix in CSR format
*  while pruning values that are less than the threshold. All the parameters are assumed
*  to have been pre-allocated by the user.
*
*  \details
*  The user first allocates \p csrRowPtr to have \p m+1 elements and then calls
*  hipsparseXpruneDense2csrNnz() which fills in the \p csrRowPtr array and stores the
*  number of elements that are larger than the pruning threshold in \p nnzTotalDevHostPtr.
*  The user then allocates \p csrColInd and \p csrVal to have size \p nnzTotalDevHostPtr
*  and completes the conversion by calling hipsparseXpruneDense2csr(). A temporary storage
*  buffer is used by both hipsparseXpruneDense2csrNnz() and hipsparseXpruneDense2csr() and
*  must be allocated by the user and whose size is determined by
*  hipsparseXpruneDense2csr_bufferSizeExt(). The routine hipsparseXpruneDense2csr() is
*  executed asynchronously with respect to the host and may return control to the
*  application on the host before the entire result is ready.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const float*              A,
                                           int                       lda,
                                           const float*              threshold,
                                           const hipsparseMatDescr_t descr,
                                           float*                    csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const double*             A,
                                           int                       lda,
                                           const double*             threshold,
                                           const hipsparseMatDescr_t descr,
                                           double*                   csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the size of the user allocated temporary storage buffer used
*  when converting and pruning by percentage a dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
*  and may return control to the application on the host before the entire result is
*  ready.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const float*      A,
                                                                  int               lda,
                                                                  float             percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const float*              csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const double*     A,
                                                                  int               lda,
                                                                  double            percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const double*             csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     bufferSize);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the size of the user allocated temporary storage buffer used
*  when converting and pruning by percentage a dense matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSizeExt which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. It is executed asynchronously with respect to the host
*  and may return control to the application on the host before the entire result is
*  ready.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseSpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const float*              csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t
    hipsparseDpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const double*             csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   bufferSize);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
*  pointer mode is set to device.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const float*              A,
                                                          int                       lda,
                                                          float                     percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const double*             A,
                                                          int                       lda,
                                                          double                    percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero elements per row and the total number of
*  nonzero elements in a dense matrix when converting and pruning by percentage a dense
*  matrix to a CSR matrix.
*
*  \details
*  When converting and pruning a dense matrix A to a CSR matrix by percentage the
*  following steps are performed. First the user calls
*  \p hipsparseXpruneDense2csrByPercentage_bufferSize which determines the size of the
*  temporary storage buffer. Once determined, this buffer must be allocated by the user.
*  Next the user allocates the csr_row_ptr array to have \p m+1 elements and calls
*  \p hipsparseXpruneDense2csrNnzByPercentage. Finally the user finishes the conversion
*  by allocating the csr_col_ind and csr_val arrays (whos size is determined by the value
*  at nnz_total_dev_host_ptr) and calling \p hipsparseXpruneDense2csrByPercentage.
*
*  The pruning by percentage works by first sorting the absolute values of the dense
*  matrix \p A. We then determine a position in this sorted array by
*  \f[
*    pos = ceil(m*n*(percentage/100)) - 1
*    pos = min(pos, m*n-1)
*    pos = max(pos, 0)
*    threshold = sorted_A[pos]
*  \f]
*  Once we have this threshold we prune values in the dense matrix \p A as in
*  \p hipsparseXpruneDense2csr. The routine does support asynchronous execution if the
*  pointer mode is set to device.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       float*                    csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       double*                   csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*
*  This function converts the matrix A in dense format into a sparse matrix in CSC format.
*  All the parameters are assumed to have been pre-allocated by the user and the arrays are filled in based on nnz_per_columns, which can be pre-computed with hipsparseXnnz().
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      float*                    csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      double*                   csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      hipComplex*               csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      hipDoubleComplex*         csc_val,
                                      int*                      csc_row_ind,
                                      int*                      csc_col_ptr);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in CSR format into a dense matrix.
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      float*                    A,
                                      int                       ld);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      double*                   A,
                                      int                       ld);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      hipComplex*               A,
                                      int                       ld);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      hipDoubleComplex*         A,
                                      int                       ld);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts the sparse matrix in CSC format into a dense matrix.
*  It is executed asynchronously with respect to the host and may return control to the application on the host before the entire result is ready.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              csc_val,
                                      const int*                csc_row_ind,
                                      const int*                csc_col_ptr,
                                      float*                    A,
                                      int                       ld);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             csc_val,
                                      const int*                csc_row_ind,
                                      const int*                csc_col_ptr,
                                      double*                   A,
                                      int                       ld);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         csc_val,
                                      const int*                csc_row_ind,
                                      const int*                csc_col_ptr,
                                      hipComplex*               A,
                                      int                       ld);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   csc_val,
                                      const int*                csc_row_ind,
                                      const int*                csc_col_ptr,
                                      hipDoubleComplex*         A,
                                      int                       ld);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  BSR matrix given a sparse CSR matrix as input.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2bsrNnz(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dirA,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       int                       blockDim,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      bsrRowPtrC,
                                       int*                      bsrNnzb);

/*! \ingroup conv_module
*  Given a sparse CSR matrix and a non-negative tolerance, this function computes how many entries would be left
*  in each row of the matrix if elements less than the tolerance were removed. It also computes the total number
*  of remaining elements in the matrix.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         float                     tol);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         double                    tol);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipComplex                tol);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipDoubleComplex          tol);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse COO matrix
*
*  \details
*  \p hipsparseXcsr2coo converts the CSR array containing the row offsets, that point
*  to the start of every row, into a COO array of row indices.
*
*  \note
*  It can also be used to convert a CSC array containing the column offsets into a COO
*  array of column indices.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t    handle,
                                    const int*           csrRowPtr,
                                    int                  nnz,
                                    int                  m,
                                    int*                 cooRowInd,
                                    hipsparseIndexBase_t idxBase);

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p hipsparseXcsr2csc converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2csc
*  can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
*  whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const float*         csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    float*               cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const double*        csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    double*              cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const hipComplex*    csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    hipComplex*          cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csc(hipsparseHandle_t       handle,
                                    int                     m,
                                    int                     n,
                                    int                     nnz,
                                    const hipDoubleComplex* csrSortedVal,
                                    const int*              csrSortedRowPtr,
                                    const int*              csrSortedColInd,
                                    hipDoubleComplex*       cscSortedVal,
                                    int*                    cscSortedRowInd,
                                    int*                    cscSortedColPtr,
                                    hipsparseAction_t       copyValues,
                                    hipsparseIndexBase_t    idxBase);
/**@}*/
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_CSR2CSC_ALG1 = 1,
    HIPSPARSE_CSR2CSC_ALG2 = 2
} hipsparseCsr2CscAlg_t;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
/*! \ingroup conv_module
*  \brief This function computes the size of the user allocated temporary storage buffer used
*  when converting a sparse CSR matrix into a sparse CSC matrix.
*
*  \details
*  \p hipsparseXcsr2cscEx2_bufferSize calculates the required user allocated temporary buffer needed 
*  by \p hipsparseXcsr2cscEx2 to convert a CSR matrix into a CSC matrix. \p hipsparseXcsr2cscEx2
*  can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
*  whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsr2cscEx2_bufferSize(hipsparseHandle_t     handle,
                                                 int                   m,
                                                 int                   n,
                                                 int                   nnz,
                                                 const void*           csrVal,
                                                 const int*            csrRowPtr,
                                                 const int*            csrColInd,
                                                 void*                 cscVal,
                                                 int*                  cscColPtr,
                                                 int*                  cscRowInd,
                                                 hipDataType           valType,
                                                 hipsparseAction_t     copyValues,
                                                 hipsparseIndexBase_t  idxBase,
                                                 hipsparseCsr2CscAlg_t alg,
                                                 size_t*               bufferSize);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse CSC matrix
*
*  \details
*  \p hipsparseXcsr2cscEx2 converts a CSR matrix into a CSC matrix. \p hipsparseXcsr2cscEx2
*  can also be used to convert a CSC matrix into a CSR matrix. \p copy_values decides
*  whether \p csc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsr2cscEx2(hipsparseHandle_t     handle,
                                      int                   m,
                                      int                   n,
                                      int                   nnz,
                                      const void*           csrVal,
                                      const int*            csrRowPtr,
                                      const int*            csrColInd,
                                      void*                 cscVal,
                                      int*                  cscColPtr,
                                      int*                  cscRowInd,
                                      hipDataType           valType,
                                      hipsparseAction_t     copyValues,
                                      hipsparseIndexBase_t  idxBase,
                                      hipsparseCsr2CscAlg_t alg,
                                      void*                 buffer);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse HYB matrix
*
*  \details
*  \p hipsparseXcsr2hyb converts a CSR matrix into a HYB matrix. It is assumed
*  that \p hyb has been initialized with hipsparseCreateHybMat().
*
*  \note
*  This function requires a significant amount of storage for the HYB matrix,
*  depending on the matrix structure.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType);
/**@}*/
#endif

/*! \ingroup conv_module
*  \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc_bufferSize returns the size of the temporary storage buffer
*  required by hipsparseXgebsr2gebsc().
*  The temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const float*      bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const double*     bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const hipComplex* bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                   int                     mb,
                                                   int                     nb,
                                                   int                     nnzb,
                                                   const hipDoubleComplex* bsr_val,
                                                   const int*              bsr_row_ptr,
                                                   const int*              bsr_col_ind,
                                                   int                     row_block_dim,
                                                   int                     col_block_dim,
                                                   size_t*                 p_buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse GEneral BSR matrix into a sparse GEneral BSC matrix
*
*  \details
*  \p hipsparseXgebsr2gebsc converts a GEneral BSR matrix into a GEneral BSC matrix. \p hipsparseXgebsr2gebsc
*  can also be used to convert a GEneral BSC matrix into a GEneral BSR matrix. \p copy_values decides
*  whether \p bsc_val is being filled during conversion (\ref HIPSPARSE_ACTION_NUMERIC)
*  or not (\ref HIPSPARSE_ACTION_SYMBOLIC).
*
*  \p hipsparseXgebsr2gebsc requires extra temporary storage buffer that has to be allocated
*  by the user. Storage buffer size can be determined by hipsparseXgebsr2gebsc_bufferSize().
*
*  \note
*  The resulting matrix can also be seen as the transpose of the input matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const float*         bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        float*               bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const double*        bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        double*              bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const hipComplex*    bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        hipComplex*          bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsc(hipsparseHandle_t       handle,
                                        int                     mb,
                                        int                     nb,
                                        int                     nnzb,
                                        const hipDoubleComplex* bsr_val,
                                        const int*              bsr_row_ptr,
                                        const int*              bsr_col_ind,
                                        int                     row_block_dim,
                                        int                     col_block_dim,
                                        hipDoubleComplex*       bsc_val,
                                        int*                    bsc_row_ind,
                                        int*                    bsc_col_ptr,
                                        hipsparseAction_t       copy_values,
                                        hipsparseIndexBase_t    idx_base,
                                        void*                   temp_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
 *  \details
 *  \p hipsparseXcsr2gebsr_bufferSize returns the size of the temporary buffer that
 *  is required by \p hipsparseXcsr2gebcsrNnz and \p hipsparseXcsr2gebcsr.
 *  The temporary storage buffer must be allocated by the user.
 *
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  GEneral BSR matrix given a sparse CSR matrix as input.
*
*  \details
*  The routine does support asynchronous execution if the pointer mode is set to device.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const float*              csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const double*             csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipComplex*         csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipDoubleComplex*   csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function computes the number of nonzero block columns per row and the total number of nonzero blocks in a sparse
*  GEneral BSR matrix given a sparse CSR matrix as input.
*
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsr2gebsrNnz(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         int                       m,
                                         int                       n,
                                         const hipsparseMatDescr_t csr_descr,
                                         const int*                csr_row_ptr,
                                         const int*                csr_col_ind,
                                         const hipsparseMatDescr_t bsr_descr,
                                         int*                      bsr_row_ptr,
                                         int                       row_block_dim,
                                         int                       col_block_dim,
                                         int*                      bsr_nnz_devhost,
                                         void*                     p_buffer);

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse GEneral BSR matrix
*
*  \details
*  \p hipsparseXcsr2gebsr converts a CSR matrix into a GEneral BSR matrix. It is assumed,
*  that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
*  for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
*  the GEneral BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
*  \p csr2gebsr_nnz() which also fills in \p bsr_row_ptr.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const float*              csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      float*                    bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const double*             csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      double*                   bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipComplex*         csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipComplex*               bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipDoubleComplex*   csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipDoubleComplex*         bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse CSR matrix into a sparse BSR matrix
*
*  \details
*  \p hipsparseXcsr2bsr converts a CSR matrix into a BSR matrix. It is assumed,
*  that \p bsr_val, \p bsr_col_ind and \p bsr_row_ptr are allocated. Allocation size
*  for \p bsr_row_ptr is computed as \p mb+1 where \p mb is the number of block rows in
*  the BSR matrix. Allocation size for \p bsr_val and \p bsr_col_ind is computed using
*  \p csr2bsr_nnz() which also fills in \p bsr_row_ptr.
*
*  \p hipsparseXcsr2bsr requires extra temporary storage that is allocated internally if
*  \p block_dim>16
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse BSR matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXbsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
*  for \p csr_row_ptr is computed by the number of block rows multiplied by the block
*  dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
*  the number of blocks in the BSR matrix multiplied by the block dimension squared.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC);
/**@}*/

/*! \ingroup conv_module
*  \brief Convert a sparse general BSR matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXgebsr2csr converts a BSR matrix into a CSR matrix. It is assumed,
*  that \p csr_val, \p csr_col_ind and \p csr_row_ptr are allocated. Allocation size
*  for \p csr_row_ptr is computed by the number of block rows multiplied by the block
*  dimension plus one. Allocation for \p csr_val and \p csr_col_ind is computed by the
*  the number of blocks in the BSR matrix multiplied by the product of the block dimensions.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const float*              bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      float*                    csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const double*             bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      double*                   csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipComplex*         bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipComplex*               csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipDoubleComplex*   bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipDoubleComplex*         csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert a sparse CSR matrix into a compressed sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXcsr2csr_compress converts a CSR matrix into a compressed CSR matrix by
 *  removing entries in the input CSR matrix that are below a non-negative threshold \p tol
 *
 *  \note
 *  In the case of complex matrices only the magnitude of the real part of \p tol is used.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             float*                    csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             float                     tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             double*                   csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             double                    tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipComplex*               csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipComplex                tol);
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipDoubleComplex*         csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipDoubleComplex          tol);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSize returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
 *  temporary storage buffer must be allocated by the user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const float*              csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const float*              threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const float*              csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const double*             csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const double*             threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const double*             csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   bufferSize);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csr_bufferSizeExt returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnz and hipsparseXpruneCsr2csr. The
 *  temporary storage buffer must be allocated by the user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const float*              csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const float*              threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const float*              csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t*                   bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const double*             csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const double*             threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const double*             csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t*                   bufferSize);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrNnz computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const float*              threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const double*             threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
 *  by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
 *  allocates \p csr_row_ptr_C to have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills
 *  in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
 *  in \p nnz_total_dev_host_ptr. The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
 *  is executed asynchronously with respect to the host and may return control to the application on the host
 *  before the entire result is ready.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const float*              threshold,
                                         const hipsparseMatDescr_t descrC,
                                         float*                    csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const double*             threshold,
                                         const hipsparseMatDescr_t descrC,
                                         double*                   csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrByPercentage_bufferSize returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnzByPercentage.
 *  The temporary storage buffer must be allocated by the user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const float*              csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                float      percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const float*              csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const double*             csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                double     percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const double*             csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     bufferSize);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrByPercentage_bufferSizeExt returns the size of the temporary buffer that
 *  is required by \p hipsparseXpruneCsr2csrNnzByPercentage.
 *  The temporary storage buffer must be allocated by the user.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const float* csrValA,
                                                                   const int*   csrRowPtrA,
                                                                   const int*   csrColIndA,
                                                                   float        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const float* csrValC,
                                                                   const int*   csrRowPtrC,
                                                                   const int*   csrColIndC,
                                                                   pruneInfo_t  info,
                                                                   size_t*      bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const double* csrValA,
                                                                   const int*    csrRowPtrA,
                                                                   const int*    csrColIndA,
                                                                   double        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const double* csrValC,
                                                                   const int*    csrRowPtrC,
                                                                   const int*    csrColIndC,
                                                                   pruneInfo_t   info,
                                                                   size_t*       bufferSize);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXpruneCsr2csrNnzByPercentage computes the number of nonzero elements per row and the total
 *  number of nonzero elements in a sparse CSR matrix once elements less than the threshold are
 *  pruned from the matrix.
 *
 *  \note The routine does support asynchronous execution if the pointer mode is set to device.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const float*              csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        float                     percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const double*             csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        double                    percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer);
/**@}*/

/*! \ingroup conv_module
 *  \brief Convert and prune by percentage a sparse CSR matrix into a sparse CSR matrix
 *
 *  \details
 *  This function converts the sparse CSR matrix A into a sparse CSR matrix C by pruning values in A
 *  that are less than the threshold. All the parameters are assumed to have been pre-allocated by the user.
 *  The user first calls hipsparseXpruneCsr2csr_bufferSize() to determine the size of the buffer used
 *  by hipsparseXpruneCsr2csrNnz() and hipsparseXpruneCsr2csr() which the user then allocates. The user then
 *  allocates \p csr_row_ptr_C to have \p m+1 elements and then calls hipsparseXpruneCsr2csrNnz() which fills
 *  in the \p csr_row_ptr_C array stores then number of elements that are larger than the pruning threshold
 *  in \p nnz_total_dev_host_ptr. The user then calls hipsparseXpruneCsr2csr() to complete the conversion. It
 *  is executed asynchronously with respect to the host and may return control to the application on the host
 *  before the entire result is ready.
 */
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const float*              csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     float                     percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     float*                    csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const double*             csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     double                    percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     double*                   csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer);
/**@}*/

#if(!defined(CUDART_VERSION) || CUDART_VERSION < 11000)
/*! \ingroup conv_module
*  \brief Convert a sparse HYB matrix into a sparse CSR matrix
*
*  \details
*  \p hipsparseXhyb2csr converts a HYB matrix into a CSR matrix.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
/**@{*/
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseShyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    float*                    csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    double*                   csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseChyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipComplex*               csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
DEPRECATED_CUDA_10000("The routine will be removed in CUDA 11")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipDoubleComplex*         csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA);
/**@}*/
#endif

/*! \ingroup conv_module
 *  \brief Convert a sparse COO matrix into a sparse CSR matrix
 *
 *  \details
 *  \p hipsparseXcoo2csr converts the COO array containing the row indices into a
 *  CSR array of row offsets, that point to the start of every row.
 *  It is assumed that the COO row index array is sorted.
 *
 *  \note It can also be used, to convert a COO array containing the column indices into
 *  a CSC array of column offsets, that point to the start of every column. Then, it is
 *  assumed that the COO column index array is sorted, instead.
 *
 *  \note
 *  This function is non blocking and executed asynchronously with respect to the host.
 *  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t    handle,
                                    const int*           cooRowInd,
                                    int                  nnz,
                                    int                  m,
                                    int*                 csrRowPtr,
                                    hipsparseIndexBase_t idxBase);

/*! \ingroup conv_module
*  \brief Create the identity map
*
*  \details
*  \p hipsparseCreateIdentityPermutation stores the identity map in \p p, such that
*  \f$p = 0:1:(n-1)\f$.
*
*  \code{.c}
*      for(i = 0; i < n; ++i)
*      {
*          p[i] = i;
*      }
*  \endcode
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateIdentityPermutation(hipsparseHandle_t handle, int n, int* p);

/*! \ingroup conv_module
*  \brief Sort a sparse CSR matrix
*
*  \details
*  \p hipsparseXcsrsort_bufferSizeExt returns the size of the temporary storage buffer
*  required by hipsparseXcsrsort(). The temporary storage buffer must be allocated by
*  the user.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        csrRowPtr,
                                                  const int*        csrColInd,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse CSR matrix
*
*  \details
*  \p hipsparseXcsrsort sorts a matrix in CSR format. The sorted permutation vector
*  \p perm can be used to obtain sorted \p csr_val array. In this case, \p perm must be
*  initialized as the identity permutation, see hipsparseCreateIdentityPermutation().
*
*  \p hipsparseXcsrsort requires extra temporary storage buffer that has to be allocated by
*  the user. Storage buffer size can be determined by hipsparseXcsrsort_bufferSizeExt().
*
*  \note
*  \p perm can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcsrsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                csrRowPtr,
                                    int*                      csrColInd,
                                    int*                      P,
                                    void*                     pBuffer);

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort_bufferSizeExt returns the size of the temporary storage buffer
*  required by hipsparseXcscsort(). The temporary storage buffer must be allocated by
*  the user.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cscColPtr,
                                                  const int*        cscRowInd,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse CSC matrix
*
*  \details
*  \p hipsparseXcscsort sorts a matrix in CSC format. The sorted permutation vector
*  \p perm can be used to obtain sorted \p csc_val array. In this case, \p perm must be
*  initialized as the identity permutation, see hipsparseCreateIdentityPermutation().
*
*  \p hipsparseXcscsort requires extra temporary storage buffer that has to be allocated by
*  the user. Storage buffer size can be determined by hipsparseXcscsort_bufferSizeExt().
*
*  \note
*  \p perm can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcscsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                cscColPtr,
                                    int*                      cscRowInd,
                                    int*                      P,
                                    void*                     pBuffer);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix
*
*  \details
*  \p hipsparseXcoosort_bufferSizeExt returns the size of the temporary storage buffer
*  required by hipsparseXcoosort(). The temporary storage buffer must be allocated by
*  the user.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cooRows,
                                                  const int*        cooCols,
                                                  size_t*           pBufferSizeInBytes);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by row
*
*  \details
*  \p hipsparseXcoosortByRow sorts a matrix in COO format by row. The sorted
*  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
*  case, \p perm must be initialized as the identity permutation, see
*  hipsparseCreateIdentityPermutation().
*
*  \p hipsparseXcoosortByRow requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  hipsparseXcoosort_bufferSizeExt().
*
*  \note
*  \p perm can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosortByRow(hipsparseHandle_t handle,
                                         int               m,
                                         int               n,
                                         int               nnz,
                                         int*              cooRows,
                                         int*              cooCols,
                                         int*              P,
                                         void*             pBuffer);

/*! \ingroup conv_module
*  \brief Sort a sparse COO matrix by column
*
*  \details
*  \p hipsparseXcoosortByColumn sorts a matrix in COO format by column. The sorted
*  permutation vector \p perm can be used to obtain sorted \p coo_val array. In this
*  case, \p perm must be initialized as the identity permutation, see
*  hipsparseCreateIdentityPermutation().
*
*  \p hipsparseXcoosortByColumn requires extra temporary storage buffer that has to be
*  allocated by the user. Storage buffer size can be determined by
*  hipsparseXcoosort_bufferSizeExt().
*
*  \note
*  \p perm can be \p NULL if a sorted permutation vector is not required.
*
*  \note
*  This function is non blocking and executed asynchronously with respect to the host.
*  It may return before the actual computation has finished.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXcoosortByColumn(hipsparseHandle_t handle,
                                            int               m,
                                            int               n,
                                            int               nnz,
                                            int*              cooRows,
                                            int*              cooCols,
                                            int*              P,
                                            void*             pBuffer);

/*! \ingroup conv_module
*  \brief
*  This function computes the the size of the user allocated temporary storage buffer used when converting a sparse
*  general BSR matrix to another sparse general BSR matrix.
*
*  \details
*  \p hipsparseXgebsr2gebsr_bufferSize returns the size of the temporary storage buffer
*  that is required by hipsparseXgebsr2gebsrNnz() and hipsparseXgebsr2gebsr().
*  The temporary storage buffer must be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const float*              bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const double*             bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipComplex*         bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipDoubleComplex*   bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize);
/**@}*/

/*! \ingroup conv_module
*  \brief This function is used when converting a general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
*  Specifically, this function determines the number of non-zero blocks that will exist in \p C (stored using either a host
*  or device pointer), and computes the row pointer array for \p C.
*
*  \details
*  The routine does support asynchronous execution.
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseXgebsr2gebsrNnz(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                       mb,
                                           int                       nb,
                                           int                       nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const int*                bsrRowPtrA,
                                           const int*                bsrColIndA,
                                           int                       rowBlockDimA,
                                           int                       colBlockDimA,
                                           const hipsparseMatDescr_t descrC,
                                           int*                      bsrRowPtrC,
                                           int                       rowBlockDimC,
                                           int                       colBlockDimC,
                                           int*                      nnzTotalDevHostPtr,
                                           void*                     buffer);

/*! \ingroup conv_module
*  \brief
*  This function converts the general BSR sparse matrix \p A to another general BSR sparse matrix \p C.
*
*  \details
*  The conversion uses three steps. First, the user calls hipsparseXgebsr2gebsr_bufferSize() to determine the size of
*  the required temporary storage buffer. The user then allocates this buffer. Secondly, the user then allocates \p mb_C+1
*  integers for the row pointer array for \p C where \p mb_C=(m+row_block_dim_C-1)/row_block_dim_C. The user then calls
*  hipsparseXgebsr2gebsrNnz() to fill in the row pointer array for \p C ( \p bsr_row_ptr_C ) and determine the number of
*  non-zero blocks that will exist in \p C. Finally, the user allocates space for the colimn indices array of \p C to have
*  \p nnzb_C elements and space for the values array of \p C to have \p nnzb_C*roc_block_dim_C*col_block_dim_C and then calls
*  hipsparseXgebsr2gebsr() to complete the conversion.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const float*              bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const double*             bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipComplex*         bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipDoubleComplex*   bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function calculates the amount of temporary storage required for
*  hipsparseXcsru2csr() and hipsparseXcsr2csru().
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   float*            csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   double*           csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipComplex*       csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipDoubleComplex* csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts unsorted CSR format to sorted CSR format. The required
*  temporary storage has to be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);
/**@}*/

/*! \ingroup conv_module
*  \brief
*  This function converts sorted CSR format to unsorted CSR format. The required
*  temporary storage has to be allocated by the user.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer);
/**@}*/

/*
* ===========================================================================
*    reordering SPARSE
* ===========================================================================
*/

/*! \ingroup reordering_module
*  \brief Coloring of the adjacency graph of the matrix \f$A\f$ stored in the CSR format.
*
*  \details
*  \p hipsparseXcsrcolor performs the coloring of the undirected graph represented by the (symmetric) sparsity pattern of the matrix \f$A\f$ stored in CSR format. Graph coloring is a way of coloring the nodes of a graph such that no two adjacent nodes are of the same color. The \p fraction_to_color is a parameter to only color a given percentage of the graph nodes, the remaining uncolored nodes receive distinct new colors. The optional \p reordering array is a permutation array such that unknowns of the same color are grouped. The matrix \f$A\f$ must be stored as a general matrix with a symmetric sparsity pattern, and if the matrix \f$A\f$ is non-symmetric then the user is responsible to provide the symmetric part \f$\frac{A+A^T}{2}\f$.
*/
/**@{*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info);
/**@}*/

/*
* ===========================================================================
*    generic SPARSE
* ===========================================================================
*/

/* Generic API opaque structures holding information */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
typedef void* hipsparseSpVecDescr_t;
typedef void* hipsparseSpMatDescr_t;
typedef void* hipsparseDnVecDescr_t;
typedef void* hipsparseDnMatDescr_t;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
struct hipsparseSpGEMMDescr;
typedef struct hipsparseSpGEMMDescr* hipsparseSpGEMMDescr_t;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
struct hipsparseSpSVDescr;
typedef struct hipsparseSpSVDescr* hipsparseSpSVDescr_t;
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
struct hipsparseSpSMDescr;
typedef struct hipsparseSpSMDescr* hipsparseSpSMDescr_t;
#endif

/* Generic API types */

/*! \ingroup generic_module
 *  \brief List of hipsparse sparse matrix formats.
 *
 *  \details
 *  This is a list of the \ref hipsparseFormat_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_FORMAT_CSR         = 1, /* Compressed Sparse Row */
    HIPSPARSE_FORMAT_CSC         = 2, /* Compressed Sparse Column */
    HIPSPARSE_FORMAT_COO         = 3, /* Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS     = 4, /* Coordinate - Array of Structures */
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /* Blocked ELL */
} hipsparseFormat_t;
#else
#if(CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_FORMAT_CSR     = 1, /* Compressed Sparse Row */
#if(CUDART_VERSION >= 11021)
    HIPSPARSE_FORMAT_CSC     = 2, /* Compressed Sparse Column */
#endif
    HIPSPARSE_FORMAT_COO     = 3, /* Coordinate - Structure of Arrays */
    HIPSPARSE_FORMAT_COO_AOS = 4 /* Coordinate - Array of Structures */
#if(CUDART_VERSION >= 11021)
    ,
    HIPSPARSE_FORMAT_BLOCKED_ELL = 5 /* Blocked ELL */
#endif
} hipsparseFormat_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse dense matrix memory layout ordering.
 *
 *  \details
 *  This is a list of the \ref hipsparseOrder_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_ORDER_ROW = 0, /**< Row major */
    HIPSPARSE_ORDER_COLUMN HIPSPARSE_DEPRECATED_MSG("Please use HIPSPARSE_ORDER_COL instead")
    = 1, /**< Column major */
    HIPSPARSE_ORDER_COL = 1 /**< Column major */
} hipsparseOrder_t;
#else
#if(CUDART_VERSION >= 11000)
typedef enum
{
    HIPSPARSE_ORDER_ROW = 0, /**< Row major */
    HIPSPARSE_ORDER_COLUMN HIPSPARSE_DEPRECATED_MSG("Please use HIPSPARSE_ORDER_COL instead")
    = 1, /**< Column major */
    HIPSPARSE_ORDER_COL = 1 /**< Column major */
} hipsparseOrder_t;
#elif(CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_ORDER_COLUMN HIPSPARSE_DEPRECATED_MSG("Please use HIPSPARSE_ORDER_COL instead")
    = 1, /**< Column major */
    HIPSPARSE_ORDER_COL = 1 /**< Column major */
} hipsparseOrder_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse index type used by sparse matrix indices.
 *
 *  \details
 *  This is a list of the \ref hipsparseIndexType_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_INDEX_16U = 1, /**< 16 bit unsigned integer indices */
    HIPSPARSE_INDEX_32I = 2, /**< 32 bit signed integer indices */
    HIPSPARSE_INDEX_64I = 3 /**< 64 bit signed integer indices */
} hipsparseIndexType_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpMV algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpMVAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_MV_ALG_DEFAULT   = 0,
    HIPSPARSE_COOMV_ALG        = 1,
    HIPSPARSE_CSRMV_ALG1       = 2,
    HIPSPARSE_CSRMV_ALG2       = 3,
    HIPSPARSE_SPMV_ALG_DEFAULT = 4,
    HIPSPARSE_SPMV_COO_ALG1    = 5,
    HIPSPARSE_SPMV_COO_ALG2    = 6,
    HIPSPARSE_SPMV_CSR_ALG1    = 7,
    HIPSPARSE_SPMV_CSR_ALG2    = 8
} hipsparseSpMVAlg_t;
#else
#if(CUDART_VERSION >= 11021)
typedef enum
{
    HIPSPARSE_MV_ALG_DEFAULT   = 0,
    HIPSPARSE_COOMV_ALG        = 1,
    HIPSPARSE_CSRMV_ALG1       = 2,
    HIPSPARSE_CSRMV_ALG2       = 3,
    HIPSPARSE_SPMV_ALG_DEFAULT = 4,
    HIPSPARSE_SPMV_COO_ALG1    = 5,
    HIPSPARSE_SPMV_COO_ALG2    = 6,
    HIPSPARSE_SPMV_CSR_ALG1    = 7,
    HIPSPARSE_SPMV_CSR_ALG2    = 8
} hipsparseSpMVAlg_t;
#elif(CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_MV_ALG_DEFAULT = 0,
    HIPSPARSE_COOMV_ALG      = 1,
    HIPSPARSE_CSRMV_ALG1     = 2,
    HIPSPARSE_CSRMV_ALG2     = 3
} hipsparseSpMVAlg_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpMM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpMMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION))
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0,
    HIPSPARSE_COOMM_ALG1            = 1,
    HIPSPARSE_COOMM_ALG2            = 2,
    HIPSPARSE_COOMM_ALG3            = 3,
    HIPSPARSE_CSRMM_ALG1            = 4,
    HIPSPARSE_SPMM_ALG_DEFAULT      = 5,
    HIPSPARSE_SPMM_COO_ALG1         = 6,
    HIPSPARSE_SPMM_COO_ALG2         = 7,
    HIPSPARSE_SPMM_COO_ALG3         = 8,
    HIPSPARSE_SPMM_COO_ALG4         = 9,
    HIPSPARSE_SPMM_CSR_ALG1         = 10,
    HIPSPARSE_SPMM_CSR_ALG2         = 11,
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 12,
    HIPSPARSE_SPMM_CSR_ALG3         = 13
} hipsparseSpMMAlg_t;
#else
#if(CUDART_VERSION >= 11021)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0,
    HIPSPARSE_COOMM_ALG1            = 1,
    HIPSPARSE_COOMM_ALG2            = 2,
    HIPSPARSE_COOMM_ALG3            = 3,
    HIPSPARSE_CSRMM_ALG1            = 4,
    HIPSPARSE_SPMM_ALG_DEFAULT      = 5,
    HIPSPARSE_SPMM_COO_ALG1         = 6,
    HIPSPARSE_SPMM_COO_ALG2         = 7,
    HIPSPARSE_SPMM_COO_ALG3         = 8,
    HIPSPARSE_SPMM_COO_ALG4         = 9,
    HIPSPARSE_SPMM_CSR_ALG1         = 10,
    HIPSPARSE_SPMM_CSR_ALG2         = 11,
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 12,
    HIPSPARSE_SPMM_CSR_ALG3         = 13
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 11003)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT        = 0,
    HIPSPARSE_COOMM_ALG1            = 1,
    HIPSPARSE_COOMM_ALG2            = 2,
    HIPSPARSE_COOMM_ALG3            = 3,
    HIPSPARSE_CSRMM_ALG1            = 4,
    HIPSPARSE_SPMM_ALG_DEFAULT      = 5,
    HIPSPARSE_SPMM_COO_ALG1         = 6,
    HIPSPARSE_SPMM_COO_ALG2         = 7,
    HIPSPARSE_SPMM_COO_ALG3         = 8,
    HIPSPARSE_SPMM_COO_ALG4         = 9,
    HIPSPARSE_SPMM_CSR_ALG1         = 10,
    HIPSPARSE_SPMM_CSR_ALG2         = 11,
    HIPSPARSE_SPMM_BLOCKED_ELL_ALG1 = 12
} hipsparseSpMMAlg_t;
#elif(CUDART_VERSION >= 10010)
typedef enum
{
    HIPSPARSE_MM_ALG_DEFAULT = 0,
    HIPSPARSE_COOMM_ALG1     = 1,
    HIPSPARSE_COOMM_ALG2     = 2,
    HIPSPARSE_COOMM_ALG3     = 3,
    HIPSPARSE_CSRMM_ALG1     = 4
} hipsparseSpMMAlg_t;
#endif
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SparseToDense algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSparseToDenseAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
typedef enum
{
    HIPSPARSE_SPARSETODENSE_ALG_DEFAULT = 0,
} hipsparseSparseToDenseAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse DenseToSparse algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseDenseToSparseAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
typedef enum
{
    HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT = 0,
} hipsparseDenseToSparseAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SDDMM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSDDMMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
typedef enum
{
    HIPSPARSE_SDDMM_ALG_DEFAULT = 0
} hipsparseSDDMMAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpSV algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpSVAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
typedef enum
{
    HIPSPARSE_SPSV_ALG_DEFAULT = 0
} hipsparseSpSVAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpSM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpSMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
typedef enum
{
    HIPSPARSE_SPSM_ALG_DEFAULT = 0
} hipsparseSpSMAlg_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse attributes.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpMatAttribute_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
typedef enum
{
    HIPSPARSE_SPMAT_FILL_MODE = 0, /**< Fill mode attribute */
    HIPSPARSE_SPMAT_DIAG_TYPE = 1 /**< Diag type attribute */
} hipsparseSpMatAttribute_t;
#endif

/*! \ingroup generic_module
 *  \brief List of hipsparse SpGEMM algorithms.
 *
 *  \details
 *  This is a list of the \ref hipsparseSpGEMMAlg_t types that are used by the hipSPARSE
 *  library.
 */
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
typedef enum
{
    HIPSPARSE_SPGEMM_DEFAULT = 0,
    HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC,
    HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC
} hipsparseSpGEMMAlg_t;
#elif(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
typedef enum
{
    HIPSPARSE_SPGEMM_DEFAULT = 0
} hipsparseSpGEMMAlg_t;
#endif

/* Sparse vector API */

/*! \ingroup generic_module
*  \brief Description: Create a sparse vector.
*
*  \details
*  \p hipsparseCreateSpVec creates a sparse vector descriptor. It should be
*  destroyed at the end using hipsparseDestroyMatDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr,
                                       int64_t                size,
                                       int64_t                nnz,
                                       void*                  indices,
                                       void*                  values,
                                       hipsparseIndexType_t   idxType,
                                       hipsparseIndexBase_t   idxBase,
                                       hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy a sparse vector.
*
*  \details
*  \p hipsparseDestroySpVec destroys a sparse vector descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the fields of the sparse vector descriptor.
*
*  \details
*  \p hipsparseSpVecGet gets the fields of the sparse vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr,
                                    int64_t*                    size,
                                    int64_t*                    nnz,
                                    void**                      indices,
                                    void**                      values,
                                    hipsparseIndexType_t*       idxType,
                                    hipsparseIndexBase_t*       idxBase,
                                    hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Get index base of a sparse vector.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*       idxBase);
#endif

/*! \ingroup generic_module
*  \brief Description: Get pointer to a sparse vector data array.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecGetValues(const hipsparseSpVecDescr_t spVecDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Description: Set pointer of a sparse vector data array.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr, void* values);
#endif

/* Sparse matrix API */

/*! \ingroup generic_module
*  \brief Description: Create a sparse COO matrix descriptor
*  \details
*  \p hipsparseCreateCoo creates a sparse COO matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCoo(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cooRowInd,
                                     void*                  cooColInd,
                                     void*                  cooValues,
                                     hipsparseIndexType_t   cooIdxType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Create a sparse COO (AoS) matrix descriptor
*  \details
*  \p hipsparseCreateCooAoS creates a sparse COO (AoS) matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCooAoS(hipsparseSpMatDescr_t* spMatDescr,
                                        int64_t                rows,
                                        int64_t                cols,
                                        int64_t                nnz,
                                        void*                  cooInd,
                                        void*                  cooValues,
                                        hipsparseIndexType_t   cooIdxType,
                                        hipsparseIndexBase_t   idxBase,
                                        hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Create a sparse CSR matrix descriptor
*  \details
*  \p hipsparseCreateCsr creates a sparse CSR matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsr(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  csrRowOffsets,
                                     void*                  csrColInd,
                                     void*                  csrValues,
                                     hipsparseIndexType_t   csrRowOffsetsType,
                                     hipsparseIndexType_t   csrColIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Create a sparse CSC matrix descriptor
*  \details
*  \p hipsparseCreateCsr creates a sparse CSC matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateCsc(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cscColOffsets,
                                     void*                  cscRowInd,
                                     void*                  cscValues,
                                     hipsparseIndexType_t   cscColOffsetsType,
                                     hipsparseIndexType_t   cscRowIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Create a sparse Blocked ELL matrix descriptor
*  \details
*  \p hipsparseCreateCsr creates a sparse Blocked ELL matrix descriptor. It should be
*  destroyed at the end using \p hipsparseDestroySpMat.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateBlockedEll(hipsparseSpMatDescr_t* spMatDescr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                ellBlockSize,
                                            int64_t                ellCols,
                                            void*                  ellColInd,
                                            void*                  ellValue,
                                            hipsparseIndexType_t   ellIdxType,
                                            hipsparseIndexBase_t   idxBase,
                                            hipDataType            valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy a sparse matrix descriptor
*  \details
*  \p hipsparseDestroySpMat destroys a sparse matrix descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr);
#endif

/*! \ingroup generic_module
*  \brief Description: Get pointers of a sparse COO matrix
*  \details
*  \p hipsparseCooGet gets the fields of the sparse COO matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      cooRowInd,
                                  void**                      cooColInd,
                                  void**                      cooValues,
                                  hipsparseIndexType_t*       idxType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Get pointers of a sparse COO (AoS) matrix
*  \details
*  \p hipsparseCooAoSGet gets the fields of the sparse COO (AoS) matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooAoSGet(const hipsparseSpMatDescr_t spMatDescr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    nnz,
                                     void**                      cooInd,
                                     void**                      cooValues,
                                     hipsparseIndexType_t*       idxType,
                                     hipsparseIndexBase_t*       idxBase,
                                     hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Get pointers of a sparse CSR matrix
*  \details
*  \p hipsparseCsrGet gets the fields of the sparse CSR matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsrGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      csrRowOffsets,
                                  void**                      csrColInd,
                                  void**                      csrValues,
                                  hipsparseIndexType_t*       csrRowOffsetsType,
                                  hipsparseIndexType_t*       csrColIndType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Get pointers of a sparse blocked ELL matrix
*  \details
*  \p hipsparseBlockedEllGet gets the fields of the sparse blocked ELL matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseBlockedEllGet(const hipsparseSpMatDescr_t spMatDescr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    ellBlockSize,
                                         int64_t*                    ellCols,
                                         void**                      ellColInd,
                                         void**                      ellValue,
                                         hipsparseIndexType_t*       ellIdxType,
                                         hipsparseIndexBase_t*       idxBase,
                                         hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Set pointers of a sparse CSR matrix
*  \details
*  \p hipsparseCsrSetPointers sets the fields of the sparse CSR matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 csrRowOffsets,
                                          void*                 csrColInd,
                                          void*                 csrValues);
#endif

/*! \ingroup generic_module
*  \brief Description: Set pointers of a sparse CSC matrix
*  \details
*  \p hipsparseCscSetPointers sets the fields of the sparse CSC matrix descriptor
*/
#if(!defined(CUDART_VERSION))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCscSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cscColOffsets,
                                          void*                 cscRowInd,
                                          void*                 cscValues);
#endif

/*! \ingroup generic_module
*  \brief Description: Set pointers of a sparse COO matrix
*  \details
*  \p hipsparseCooSetPointers sets the fields of the sparse COO matrix descriptor
*/
#if(!defined(CUDART_VERSION))
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cooRowInd,
                                          void*                 cooColInd,
                                          void*                 cooValues);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the sizes of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr,
                                        int64_t*              rows,
                                        int64_t*              cols,
                                        int64_t*              nnz);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the format of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetFormat(const hipsparseSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*          format);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the index base of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetIndexBase(const hipsparseSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*       idxBase);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the pointer of the values array of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Description: Set the pointer of the values array of a sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr, void* values);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the batch count of the sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int* batchCount);
#endif

/*! \ingroup generic_module
*  \brief Description: Set the batch count of the sparse matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
DEPRECATED_CUDA_11000("The routine will be removed in CUDA 12")
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatSetStridedBatch(hipsparseSpMatDescr_t spMatDescr, int batchCount);
#endif

/*! \ingroup generic_module
*  \brief Description: Set the batch count and stride of the sparse COO matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCooSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               batchStride);
#endif

/*! \ingroup generic_module
*  \brief Description: Set the batch count and stride of the sparse CSR matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCsrSetStridedBatch(hipsparseSpMatDescr_t spMatDescr,
                                              int                   batchCount,
                                              int64_t               offsetsBatchStride,
                                              int64_t               columnsValuesBatchStride);
#endif

/*! \ingroup generic_module
*  \brief Description: Get attribute from sparse matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             void*                     data,
                                             size_t                    dataSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Set attribute in sparse matrix descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatSetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             const void*               data,
                                             size_t                    dataSize);
#endif

/* Dense vector API */

/*! \ingroup generic_module
*  \brief Description: Create dense vector
*  \details
*  \p hipsparseCreateDnVec creates a dense vector descriptor. It should be
*  destroyed at the end using hipsparseDestroyDnVec().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
                                       int64_t                size,
                                       void*                  values,
                                       hipDataType            valueType);
#endif

/* Description: Destroy dense vector */

/*! \ingroup generic_module
*  \brief Description: Destroy dense vector
*  \details
*  \p hipsparseDestroyDnVec destroys a dense vector descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the fields from a dense vector
*  \details
*  \p hipsparseDnVecGet gets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnVecGet(const hipsparseDnVecDescr_t dnVecDescr,
                                    int64_t*                    size,
                                    void**                      values,
                                    hipDataType*                valueType);
#endif

/*! \ingroup generic_module
*  \brief Description: Get value pointer from a dense vector
*  \details
*  \p hipsparseDnVecGetValues gets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnVecGetValues(const hipsparseDnVecDescr_t dnVecDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Description: Set value pointer of a dense vector
*  \details
*  \p hipsparseDnVecSetValues sets the fields of the dense vector descriptor
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr, void* values);
#endif

/* Dense matrix API */

/* Description: Create dense matrix */

/*! \ingroup generic_module
*  \brief Description: Create dense matrix
*  \details
*  \p hipsparseCreateDnMat creates a dense matrix descriptor. It should be
*  destroyed at the end using hipsparseDestroyDnMat().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
                                       int64_t                rows,
                                       int64_t                cols,
                                       int64_t                ld,
                                       void*                  values,
                                       hipDataType            valueType,
                                       hipsparseOrder_t       order);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy dense matrix
*  \details
*  \p hipsparseDestroyDnMat destroys a dense matrix descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr);
#endif

/*! \ingroup generic_module
*  \brief Description: Get fields from a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    ld,
                                    void**                      values,
                                    hipDataType*                valueType,
                                    hipsparseOrder_t*           order);
#endif

/*! \ingroup generic_module
*  \brief Description: Get value pointer from a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatGetValues(const hipsparseDnMatDescr_t dnMatDescr, void** values);
#endif

/*! \ingroup generic_module
*  \brief Description: Set value pointer of a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr, void* values);
#endif

/*! \ingroup generic_module
*  \brief Description: Get the batch count and batch stride of the dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatGetStridedBatch(hipsparseDnMatDescr_t dnMatDescr,
                                                int*                  batchCount,
                                                int64_t*              batchStride);
#endif

/*! \ingroup generic_module
*  \brief Description: Set the batch count and batch stride of the dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDnMatSetStridedBatch(hipsparseDnMatDescr_t dnMatDescr,
                                                int                   batchCount,
                                                int64_t               batchStride);
#endif

/* Generic API functions */

/*! \ingroup generic_module
*  \brief Description: Scale a sparse vector and add it to a scaled dense vector.
*
*  \details
*  \ref hipsparseAxpby multiplies the sparse vector \f$x\f$ with scalar \f$\alpha\f$ and
*  adds the result to the dense vector \f$y\f$ that is multiplied with scalar
*  \f$\beta\f$, such that
*
*  \f[
*      y := \alpha \cdot x + \beta \cdot y
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] = alpha * x_val[i] + beta * y[x_ind[i]]
*      }
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t     handle,
                                 const void*           alpha,
                                 hipsparseSpVecDescr_t vecX,
                                 const void*           beta,
                                 hipsparseDnVecDescr_t vecY);
#endif

/*! \ingroup generic_module
*  \brief Description: Gather elements from a dense vector and store them into a sparse vector.
*
*  \details
*  \ref hipsparseGather gathers the elements from the dense vector \f$y\f$ and stores
*  them in the sparse vector \f$x\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_val[i] = y[x_ind[i]];
*      }
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseGather(hipsparseHandle_t     handle,
                                  hipsparseDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t vecX);
#endif

/*! \ingroup generic_module
*  \brief Description: Scatter elements from a sparse vector into a dense vector.
*
*  \details
*  \ref hipsparseScatter scatters the elements from the sparse vector \f$x\f$ in the dense
*  vector \f$y\f$.
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          y[x_ind[i]] = x_val[i];
*      }
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScatter(hipsparseHandle_t     handle,
                                   hipsparseSpVecDescr_t vecX,
                                   hipsparseDnVecDescr_t vecY);
#endif

/*! \ingroup generic_module
*  \brief Description: Apply Givens rotation to a dense and a sparse vector.
*
*  \details
*  \ref hipsparseRot applies the Givens rotation matrix \f$G\f$ to the sparse vector
*  \f$x\f$ and the dense vector \f$y\f$, where
*  \f[
*    G = \begin{pmatrix} c & s \\ -s & c \end{pmatrix}
*  \f]
*
*  \code{.c}
*      for(i = 0; i < nnz; ++i)
*      {
*          x_tmp = x_val[i];
*          y_tmp = y[x_ind[i]];
*
*          x_val[i]    = c * x_tmp + s * y_tmp;
*          y[x_ind[i]] = c * y_tmp - s * x_tmp;
*      }
*  \endcode
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseRot(hipsparseHandle_t     handle,
                               const void*           c_coeff,
                               const void*           s_coeff,
                               hipsparseSpVecDescr_t vecX,
                               hipsparseDnVecDescr_t vecY);
#endif

/*! \ingroup generic_module
*  \brief Description: Sparse matrix to dense matrix conversion
*
*  \details
*  \p hipsparseSparseToDense_bufferSize computes the required user allocated buffer size needed when converting 
*  a sparse matrix to a dense matrix.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseSpMatDescr_t       matA,
                                                    hipsparseDnMatDescr_t       matB,
                                                    hipsparseSparseToDenseAlg_t alg,
                                                    size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Sparse matrix to dense matrix conversion
*
*  \details
*  \p hipsparseSparseToDense converts a sparse matrix to a dense matrix. This routine takes a user allocated buffer 
*  whose size must first be computed by calling \p hipsparseSparseToDense_bufferSize
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                         hipsparseSpMatDescr_t       matA,
                                         hipsparseDnMatDescr_t       matB,
                                         hipsparseSparseToDenseAlg_t alg,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Dense matrix to sparse matrix conversion
*
*  \details
*  \p hipsparseDenseToSparse_bufferSize computes the required user allocated buffer size needed when converting 
*  a dense matrix to a sparse matrix.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseDnMatDescr_t       matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Dense matrix to sparse matrix conversion
*
*  \details
*  \p hipsparseDenseToSparse_analysis performs analysis that is later used in \p hipsparseDenseToSparse_convert when 
*  converting a dense matrix to sparse matrix. This routine takes a user allocated buffer whose size must first be computed 
*  using \p hipsparseDenseToSparse_bufferSize.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseDnMatDescr_t       matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Dense matrix to sparse matrix conversion
*
*  \details
*  \p hipsparseDenseToSparse_convert converts a dense matrix to a sparse matrix. This routine requires a user allocated buffer
*  whose size must be determined by first calling \p hipsparseDenseToSparse_bufferSize.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseDnMatDescr_t       matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV_bufferSize computes the required user allocated buffer size needed when computing the 
*  inner dot product of a sparse vector with a dense vector
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute the inner dot product of a sparse vector with a dense vector
*
*  \details
*  \p hipsparseSpVV computes the inner dot product of a sparse vector with a dense vector. This routine takes a user 
*  allocated buffer whose size must first be computed by calling \p hipsparseSpVV_bufferSize
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Buffer size step of the sparse matrix multiplication with a dense vector
*
*  \details
*  \p hipsparseSpMV_bufferSize computes the required user allocated buffer size needed when computing the 
*  sparse matrix multiplication with a dense vector
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Preprocess step of the sparse matrix multiplication with a dense vector (optional)
*
*  \details
*  \p hipsparseSpMV_preprocess performs the optional preprocess used when computing the 
*  sparse matrix multiplication with a dense vector
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute the sparse matrix multiplication with a dense vector
*
*  \details
*  \p hipsparseSpMV computes sparse matrix multiplication with a dense vector
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnVecDescr_t vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Calculate the buffer size required for the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM_bufferSize computes the required user allocated buffer size needed when computing the 
*  sparse matrix multiplication with a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Preprocess step of the sparse matrix multiplication with a dense matrix.
*
*  \details
*  \p hipsparseSpMM_preprocess performs the required preprocessing used when computing the 
*  sparse matrix multiplication with a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11021)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute the sparse matrix multiplication with a dense matrix
*
*  \details
*  \p hipsparseSpMM computes sparse matrix multiplication with a dense matrix
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 10010)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnMatDescr_t matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Create sparse matrix sparse matrix product descriptor
*  \details
*  \p hipsparseSpGEMM_createDescr creates a sparse matrix sparse matrix product descriptor. It should be
*  destroyed at the end using hipsparseSpGEMM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy sparse matrix sparse matrix product descriptor
*  \details
*  \p hipsparseSpGEMM_destroyDescr destroys a sparse matrix sparse matrix product descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Work estimation step of the sparse matrix sparse matrix product.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t      handle,
                                                 hipsparseOperation_t   opA,
                                                 hipsparseOperation_t   opB,
                                                 const void*            alpha,
                                                 hipsparseSpMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t  matB,
                                                 const void*            beta,
                                                 hipsparseSpMatDescr_t  matC,
                                                 hipDataType            computeType,
                                                 hipsparseSpGEMMAlg_t   alg,
                                                 hipsparseSpGEMMDescr_t spgemmDescr,
                                                 size_t*                bufferSize1,
                                                 void*                  externalBuffer1);
#endif

/*! \ingroup generic_module
*  \brief Description: Compute step of the sparse matrix sparse matrix product.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t      handle,
                                          hipsparseOperation_t   opA,
                                          hipsparseOperation_t   opB,
                                          const void*            alpha,
                                          hipsparseSpMatDescr_t  matA,
                                          hipsparseSpMatDescr_t  matB,
                                          const void*            beta,
                                          hipsparseSpMatDescr_t  matC,
                                          hipDataType            computeType,
                                          hipsparseSpGEMMAlg_t   alg,
                                          hipsparseSpGEMMDescr_t spgemmDescr,
                                          size_t*                bufferSize2,
                                          void*                  externalBuffer2);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11000)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t      handle,
                                       hipsparseOperation_t   opA,
                                       hipsparseOperation_t   opB,
                                       const void*            alpha,
                                       hipsparseSpMatDescr_t  matA,
                                       hipsparseSpMatDescr_t  matB,
                                       const void*            beta,
                                       hipsparseSpMatDescr_t  matC,
                                       hipDataType            computeType,
                                       hipsparseSpGEMMAlg_t   alg,
                                       hipsparseSpGEMMDescr_t spgemmDescr);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_workEstimation(hipsparseHandle_t      handle,
                                                      hipsparseOperation_t   opA,
                                                      hipsparseOperation_t   opB,
                                                      hipsparseSpMatDescr_t  matA,
                                                      hipsparseSpMatDescr_t  matB,
                                                      hipsparseSpMatDescr_t  matC,
                                                      hipsparseSpGEMMAlg_t   alg,
                                                      hipsparseSpGEMMDescr_t spgemmDescr,
                                                      size_t*                bufferSize1,
                                                      void*                  externalBuffer1);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_nnz(hipsparseHandle_t      handle,
                                           hipsparseOperation_t   opA,
                                           hipsparseOperation_t   opB,
                                           hipsparseSpMatDescr_t  matA,
                                           hipsparseSpMatDescr_t  matB,
                                           hipsparseSpMatDescr_t  matC,
                                           hipsparseSpGEMMAlg_t   alg,
                                           hipsparseSpGEMMDescr_t spgemmDescr,
                                           size_t*                bufferSize2,
                                           void*                  externalBuffer2,
                                           size_t*                bufferSize3,
                                           void*                  externalBuffer3,
                                           size_t*                bufferSize4,
                                           void*                  externalBuffer4);

#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_compute(hipsparseHandle_t      handle,
                                               hipsparseOperation_t   opA,
                                               hipsparseOperation_t   opB,
                                               const void*            alpha,
                                               hipsparseSpMatDescr_t  matA,
                                               hipsparseSpMatDescr_t  matB,
                                               const void*            beta,
                                               hipsparseSpMatDescr_t  matC,
                                               hipDataType            computeType,
                                               hipsparseSpGEMMAlg_t   alg,
                                               hipsparseSpGEMMDescr_t spgemmDescr);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpGEMMreuse_copy(hipsparseHandle_t      handle,
                                            hipsparseOperation_t   opA,
                                            hipsparseOperation_t   opB,
                                            hipsparseSpMatDescr_t  matA,
                                            hipsparseSpMatDescr_t  matB,
                                            hipsparseSpMatDescr_t  matC,
                                            hipsparseSpGEMMAlg_t   alg,
                                            hipsparseSpGEMMDescr_t spgemmDescr,
                                            size_t*                bufferSize5,
                                            void*                  externalBuffer5);
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
/*! \ingroup generic_module
*  \brief Description: Calculate the buffer size required for the sampled dense dense matrix multiplication
*
*  \details
*  \p hipsparseSDDMM_bufferSize computes the required user allocated buffer size needed when computing the 
*  sampled dense dense matrix multiplication
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            size_t*                     bufferSize);

/*! \ingroup generic_module
*  \brief Description: Preprocess step of the sampled dense dense matrix multiplication.
*
*  \details
*  \p hipsparseSDDMM_preprocess performs the required preprocessing used when computing the 
*  sampled dense dense matrix multiplication
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t A,
                                            const hipsparseDnMatDescr_t B,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       C,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            void*                       tempBuffer);

/*! \ingroup generic_module
*  \brief  Description: Sampled Dense-Dense Matrix Multiplication.
*
*  \details
*  \ref hipsparseSDDMM multiplies the scalar \f$\alpha\f$ with the dense
*  \f$m \times k\f$ matrix \f$A\f$, the dense \f$k \times n\f$ matrix \f$B\f$, filtered by the sparsity pattern of the \f$m \times n\f$ sparse matrix \f$C\f$ and
*  adds the result to \f$C\f$ scaled by
*  \f$\beta\f$. The final result is stored in the sparse \f$m \times n\f$ matrix \f$C\f$,
*  such that
*  \f[
*    C := \alpha ( opA(A) \cdot opB(B) ) \cdot spy(C) + \beta C,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if opA == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T,   & \text{if opA == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f],
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if opB == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T,   & \text{if opB == HIPSPARSE_OPERATION_TRANSPOSE} \\
*    \end{array}
*    \right.
*  \f]
*   and
*  \f[
*    spy(C)_ij = \left\{
*    \begin{array}{ll}
*        1 \text{if i == j},   & 0 \text{if i != j} \\
*    \end{array}
*    \right.
*  \f]
*/
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t           handle,
                                 hipsparseOperation_t        opA,
                                 hipsparseOperation_t        opB,
                                 const void*                 alpha,
                                 const hipsparseDnMatDescr_t A,
                                 const hipsparseDnMatDescr_t B,
                                 const void*                 beta,
                                 hipsparseSpMatDescr_t       C,
                                 hipDataType                 computeType,
                                 hipsparseSDDMMAlg_t         alg,
                                 void*                       tempBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Create sparse matrix triangular solve descriptor
*  \details
*  \p hipsparseSpGEMM_createDescr creates a sparse matrix triangular solve descriptor. It should be
*  destroyed at the end using hipsparseSpSV_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy sparse matrix triangular solve descriptor
*  \details
*  \p hipsparseSpSV_destroyDescr destroys a sparse matrix triangular solve descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Buffer size step of solution of triangular linear system op(A) * Y = alpha * X,
*  where A is a sparse matrix in CSR storage format, x and Y are dense vectors.
*
*  \details
*  \p hipsparseSpSV_bufferSize computes the required user allocated buffer size needed when computing the 
*  solution of triangular linear system op(A) * Y = alpha * X, where A is a sparse matrix in CSR storage 
*  format, x and Y are dense vectors.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Analysis step of solution of triangular linear system op(A) * Y = alpha * X,
*  where A is a sparse matrix in CSR storage format, x and Y are dense vectors.
*
*  \details
*  \p hipsparseSpSV_analysis performs the required analysis used when computing the 
*  solution of triangular linear system op(A) * Y = alpha * X,
*  where A is a sparse matrix in CSR storage format, x and Y are dense vectors.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnVecDescr_t x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Sparse triangular solve
*
*  \details
*  \p hipsparseSpSV_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution vector
*  \f$y\f$ and the right-hand side \f$x\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot y = \alpha \cdot x,
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnVecDescr_t x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr,
                                      void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Create sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_createDescr creates a sparse matrix triangular solve with multiple rhs descriptor. It should be
*  destroyed at the end using hipsparseSpSM_destroyDescr().
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Destroy sparse matrix triangular solve with multiple rhs descriptor
*  \details
*  \p hipsparseSpSM_destroyDescr destroys a sparse matrix triangular solve with multiple rhs descriptor and releases all
*  resources used by the descriptor.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr);
#endif

/*! \ingroup generic_module
*  \brief Description: Buffer size step of solution of triangular linear system op(A) * C = alpha * op(B),
*  where A is a sparse matrix in CSR storage format, B and C are dense matrices.
*
*  \details
*  \p hipsparseSpSV_bufferSize computes the required user allocated buffer size needed when computing the 
*  solution of triangular linear system op(A) * C = alpha * op(B), where A is a sparse matrix in CSR storage 
*  format, B and C are dense matrices.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpSMAlg_t          alg,
                                           hipsparseSpSMDescr_t        spsmDescr,
                                           size_t*                     bufferSize);
#endif

/*! \ingroup generic_module
*  \brief Description: Analysis step of solution of triangular linear system op(A) * C = alpha * op(B),
*  where A is a sparse matrix in CSR storage format, B and C are dense vectors.
*
*  \details
*  \p hipsparseSpSV_analysis performs the required analysis used when computing the 
*  solution of triangular linear system op(A) * C = alpha * op(B),
*  where A is a sparse matrix in CSR storage format, B and C are dense vectors.
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         hipsparseOperation_t        opB,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnMatDescr_t matB,
                                         const hipsparseDnMatDescr_t matC,
                                         hipDataType                 computeType,
                                         hipsparseSpSMAlg_t          alg,
                                         hipsparseSpSMDescr_t        spsmDescr,
                                         void*                       externalBuffer);
#endif

/*! \ingroup generic_module
*  \brief Description: Sparse triangular system solve
*
*  \details
*  \p hipsparseSpSM_solve solves a sparse triangular linear system of a sparse
*  \f$m \times m\f$ matrix, defined in CSR or COO storage format, a dense solution matrix
*  \f$C\f$ and the right-hand side \f$B\f$ that is multiplied by \f$\alpha\f$, such that
*  \f[
*    op(A) \cdot C = \alpha \cdot op(B),
*  \f]
*  with
*  \f[
*    op(A) = \left\{
*    \begin{array}{ll}
*        A,   & \text{if trans == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        A^T, & \text{if trans == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        A^H, & \text{if trans == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*  and
*  \f[
*    op(B) = \left\{
*    \begin{array}{ll}
*        B,   & \text{if trans_B == HIPSPARSE_OPERATION_NON_TRANSPOSE} \\
*        B^T, & \text{if trans_B == HIPSPARSE_OPERATION_TRANSPOSE} \\
*        B^H, & \text{if trans_B == HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE}
*    \end{array}
*    \right.
*  \f]
*/
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpSM_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      hipsparseOperation_t        opB,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnMatDescr_t matB,
                                      const hipsparseDnMatDescr_t matC,
                                      hipDataType                 computeType,
                                      hipsparseSpSMAlg_t          alg,
                                      hipsparseSpSMDescr_t        spsmDescr,
                                      void*                       externalBuffer);
#endif

#ifdef __cplusplus
}
#endif

#endif // _HIPSPARSE_H_
