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

#include "hipsparse.h"

#include <cuda_runtime_api.h>
#include <cusparse_v2.h>
#include <hip/hip_runtime_api.h>
#include <stdio.h>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_CUSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                                   \
        cusparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != CUSPARSE_STATUS_SUCCESS)             \
        {                                                               \
            return hipCUSPARSEStatusToHIPStatus(TMP_STATUS_FOR_CHECK);  \
        }                                                               \
    }

hipsparseStatus_t hipCUSPARSEStatusToHIPStatus(cusparseStatus_t cuStatus)
{
    switch(cuStatus)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return HIPSPARSE_STATUS_SUCCESS;
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    case CUSPARSE_STATUS_ALLOC_FAILED:
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    case CUSPARSE_STATUS_INVALID_VALUE:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return HIPSPARSE_STATUS_ARCH_MISMATCH;
    case CUSPARSE_STATUS_MAPPING_ERROR:
        return HIPSPARSE_STATUS_MAPPING_ERROR;
    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return HIPSPARSE_STATUS_EXECUTION_FAILED;
    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED;
    case CUSPARSE_STATUS_ZERO_PIVOT:
        return HIPSPARSE_STATUS_ZERO_PIVOT;
    default:
        throw "Non existent cusparseStatus_t";
    }
}

cusparsePointerMode_t hipPointerModeToCudaPointerMode(hipsparsePointerMode_t mode)
{
    switch(mode)
    {
    case HIPSPARSE_POINTER_MODE_HOST:
        return CUSPARSE_POINTER_MODE_HOST;
    case HIPSPARSE_POINTER_MODE_DEVICE:
        return CUSPARSE_POINTER_MODE_DEVICE;
    default:
        throw "Non existent hipsparsePointerMode_t";
    }
}

hipsparsePointerMode_t CudaPointerModeToHIPPointerMode(cusparsePointerMode_t mode)
{
    switch(mode)
    {
    case CUSPARSE_POINTER_MODE_HOST:
        return HIPSPARSE_POINTER_MODE_HOST;
    case CUSPARSE_POINTER_MODE_DEVICE:
        return HIPSPARSE_POINTER_MODE_DEVICE;
    default:
        throw "Non existent cusparsePointerMode_t";
    }
}

cusparseAction_t hipActionToCudaAction(hipsparseAction_t action)
{
    switch(action)
    {
    case HIPSPARSE_ACTION_SYMBOLIC:
        return CUSPARSE_ACTION_SYMBOLIC;
    case HIPSPARSE_ACTION_NUMERIC:
        return CUSPARSE_ACTION_NUMERIC;
    default:
        throw "Non existent hipsparseAction_t";
    }
}

hipsparseAction_t CudaActionToHIPAction(cusparseAction_t action)
{
    switch(action)
    {
    case CUSPARSE_ACTION_SYMBOLIC:
        return HIPSPARSE_ACTION_SYMBOLIC;
    case CUSPARSE_ACTION_NUMERIC:
        return HIPSPARSE_ACTION_NUMERIC;
    default:
        throw "Non existent cusparseAction_t";
    }
}

cusparseMatrixType_t hipMatrixTypeToCudaMatrixType(hipsparseMatrixType_t type)
{
    switch(type)
    {
    case HIPSPARSE_MATRIX_TYPE_GENERAL:
        return CUSPARSE_MATRIX_TYPE_GENERAL;
    case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
        return CUSPARSE_MATRIX_TYPE_SYMMETRIC;
    case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
        return CUSPARSE_MATRIX_TYPE_HERMITIAN;
    case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
        return CUSPARSE_MATRIX_TYPE_TRIANGULAR;
    default:
        throw "Non existent hipsparseMatrixType_t";
    }
}

hipsparseMatrixType_t CudaMatrixTypeToHIPMatrixType(cusparseMatrixType_t type)
{
    switch(type)
    {
    case CUSPARSE_MATRIX_TYPE_GENERAL:
        return HIPSPARSE_MATRIX_TYPE_GENERAL;
    case CUSPARSE_MATRIX_TYPE_SYMMETRIC:
        return HIPSPARSE_MATRIX_TYPE_SYMMETRIC;
    case CUSPARSE_MATRIX_TYPE_HERMITIAN:
        return HIPSPARSE_MATRIX_TYPE_HERMITIAN;
    case CUSPARSE_MATRIX_TYPE_TRIANGULAR:
        return HIPSPARSE_MATRIX_TYPE_TRIANGULAR;
    default:
        throw "Non existent cusparseMatrixType_t";
    }
}

cusparseFillMode_t hipFillToCudaFill(hipsparseFillMode_t fill)
{
    switch(fill)
    {
    case HIPSPARSE_FILL_MODE_LOWER:
        return CUSPARSE_FILL_MODE_LOWER;
    case HIPSPARSE_FILL_MODE_UPPER:
        return CUSPARSE_FILL_MODE_UPPER;
    default:
        throw "Non existent hipsparseFillMode_t";
    }
}

hipsparseFillMode_t CudaFillToHIPFill(cusparseFillMode_t fill)
{
    switch(fill)
    {
    case CUSPARSE_FILL_MODE_LOWER:
        return HIPSPARSE_FILL_MODE_LOWER;
    case CUSPARSE_FILL_MODE_UPPER:
        return HIPSPARSE_FILL_MODE_UPPER;
    default:
        throw "Non existent cusparseFillMode_t";
    }
}

cusparseDiagType_t hipDiagonalToCudaDiagonal(hipsparseDiagType_t diagonal)
{
    switch(diagonal)
    {
    case HIPSPARSE_DIAG_TYPE_NON_UNIT:
        return CUSPARSE_DIAG_TYPE_NON_UNIT;
    case HIPSPARSE_DIAG_TYPE_UNIT:
        return CUSPARSE_DIAG_TYPE_UNIT;
    default:
        throw "Non existent hipsparseDiagType_t";
    }
}

hipsparseDiagType_t CudaDiagonalToHIPDiagonal(cusparseDiagType_t diagonal)
{
    switch(diagonal)
    {
    case CUSPARSE_DIAG_TYPE_NON_UNIT:
        return HIPSPARSE_DIAG_TYPE_NON_UNIT;
    case CUSPARSE_DIAG_TYPE_UNIT:
        return HIPSPARSE_DIAG_TYPE_UNIT;
    default:
        throw "Non existent cusparseDiagType_t";
    }
}

cusparseIndexBase_t hipIndexBaseToCudaIndexBase(hipsparseIndexBase_t base)
{
    switch(base)
    {
    case HIPSPARSE_INDEX_BASE_ZERO:
        return CUSPARSE_INDEX_BASE_ZERO;
    case HIPSPARSE_INDEX_BASE_ONE:
        return CUSPARSE_INDEX_BASE_ONE;
    default:
        throw "Non existent hipsparseIndexBase_t";
    }
}

hipsparseIndexBase_t CudaIndexBaseToHIPIndexBase(cusparseIndexBase_t base)
{
    switch(base)
    {
    case CUSPARSE_INDEX_BASE_ZERO:
        return HIPSPARSE_INDEX_BASE_ZERO;
    case CUSPARSE_INDEX_BASE_ONE:
        return HIPSPARSE_INDEX_BASE_ONE;
    default:
        throw "Non existent cusparseIndexBase_t";
    }
}

cusparseOperation_t hipOperationToCudaOperation(hipsparseOperation_t op)
{
    switch(op)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return CUSPARSE_OPERATION_NON_TRANSPOSE;
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return CUSPARSE_OPERATION_TRANSPOSE;
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        throw "Non existent hipsparseOperation_t";
    }
}

hipsparseOperation_t CudaOperationToHIPOperation(cusparseOperation_t op)
{
    switch(op)
    {
    case CUSPARSE_OPERATION_NON_TRANSPOSE:
        return HIPSPARSE_OPERATION_NON_TRANSPOSE;
    case CUSPARSE_OPERATION_TRANSPOSE:
        return HIPSPARSE_OPERATION_TRANSPOSE;
    case CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        throw "Non existent cusparseOperation_t";
    }
}

cusparseHybPartition_t hipHybPartitionToCudaHybPartition(hipsparseHybPartition_t part)
{
    switch(part)
    {
    case HIPSPARSE_HYB_PARTITION_AUTO:
        return CUSPARSE_HYB_PARTITION_AUTO;
    case HIPSPARSE_HYB_PARTITION_USER:
        return CUSPARSE_HYB_PARTITION_USER;
    case HIPSPARSE_HYB_PARTITION_MAX:
        return CUSPARSE_HYB_PARTITION_MAX;
    default:
        throw "Non existent hipsparseHybPartition_t";
    }
}

cusparseSolvePolicy_t hipPolicyToCudaPolicy(hipsparseSolvePolicy_t policy)
{
    switch(policy)
    {
    case HIPSPARSE_SOLVE_POLICY_NO_LEVEL:
        return CUSPARSE_SOLVE_POLICY_NO_LEVEL;
    case HIPSPARSE_SOLVE_POLICY_USE_LEVEL:
        return CUSPARSE_SOLVE_POLICY_USE_LEVEL;
    default:
        throw "Non existent hipsparseSolvePolicy_t";
    }
}

cusparseSideMode_t hipSideToCudaSide(hipsparseSideMode_t side)
{
    switch(side)
    {
    case HIPSPARSE_SIDE_LEFT:
        return CUSPARSE_SIDE_LEFT;
    case HIPSPARSE_SIDE_RIGHT:
        return CUSPARSE_SIDE_RIGHT;
    default:
        throw "Non existent hipsparseSideMode_t";
    }
}

hipsparseSideMode_t CudaSideToHIPSide(cusparseSideMode_t side)
{
    switch(side)
    {
    case CUSPARSE_SIDE_LEFT:
        return HIPSPARSE_SIDE_LEFT;
    case CUSPARSE_SIDE_RIGHT:
        return HIPSPARSE_SIDE_RIGHT;
    default:
        throw "Non existent cusparseSideMode_t";
    }
}

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreate((cusparseHandle_t*)handle));
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroy((cusparseHandle_t)handle));
}

hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version)
{
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    }

    *version = hipsparseVersionMajor * 100000 + hipsparseVersionMinor * 100 + hipsparseVersionPatch;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseGetGitRevision(hipsparseHandle_t handle, char* rev)
{
    // Get hipSPARSE revision
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    }

    if(rev == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(hipsparseVersionTweak);

    char hipsparse_rev[64];
    memcpy(hipsparse_rev, v, sizeof(v));

    // Get cuSPARSE version
    int cusparse_ver;
    RETURN_IF_CUSPARSE_ERROR(cusparseGetVersion((cusparseHandle_t)handle, &cusparse_ver));

    // Combine
    sprintf(rev,
            "%s (cuSPARSE %d.%d.%d)",
            hipsparse_rev,
            cusparse_ver / 100000,
            cusparse_ver / 100 % 1000,
            cusparse_ver % 100);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSetStream((cusparseHandle_t)handle, streamId));
}

hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseGetStream((cusparseHandle_t)handle, streamId));
}

hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSetPointerMode((cusparseHandle_t)handle, hipPointerModeToCudaPointerMode(mode)));
}

hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode)
{
    cusparsePointerMode_t cusparseMode;
    cusparseStatus_t      status = cusparseGetPointerMode((cusparseHandle_t)handle, &cusparseMode);
    *mode                        = CudaPointerModeToHIPPointerMode(cusparseMode);
    return hipCUSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateMatDescr((cusparseMatDescr_t*)descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyMatDescr((cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSetMatType((cusparseMatDescr_t)descrA, hipMatrixTypeToCudaMatrixType(type)));
}

hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA)
{
    return CudaMatrixTypeToHIPMatrixType(cusparseGetMatType((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA, hipsparseFillMode_t fillMode)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSetMatFillMode((cusparseMatDescr_t)descrA, hipFillToCudaFill(fillMode)));
}

hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA)
{
    return CudaFillToHIPFill(cusparseGetMatFillMode((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA, hipsparseDiagType_t diagType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSetMatDiagType((cusparseMatDescr_t)descrA, hipDiagonalToCudaDiagonal(diagType)));
}

hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA)
{
    return CudaDiagonalToHIPDiagonal(cusparseGetMatDiagType((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSetMatIndexBase((cusparseMatDescr_t)descrA, hipIndexBaseToCudaIndexBase(base)));
}

hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA)
{
    return CudaIndexBaseToHIPIndexBase(cusparseGetMatIndexBase((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateHybMat((cusparseHybMat_t*)hybA));
}

hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyHybMat((cusparseHybMat_t)hybA));
}

hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrsv2Info((csrsv2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrsv2Info((csrsv2Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrilu02Info((csrilu02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrilu02Info((csrilu02Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrgemm2Info((csrgemm2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrgemm2Info((csrgemm2Info_t)info));
}

hipsparseStatus_t hipsparseSaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const float*         alpha,
                                  const float*         xVal,
                                  const int*           xInd,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSaxpyi(
        (cusparseHandle_t)handle, nnz, alpha, xVal, xInd, y, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const double*        alpha,
                                  const double*        xVal,
                                  const int*           xInd,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDaxpyi(
        (cusparseHandle_t)handle, nnz, alpha, xVal, xInd, y, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 const float*         y,
                                 float*               result,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSdoti((cusparseHandle_t)handle,
                                                      nnz,
                                                      xVal,
                                                      xInd,
                                                      y,
                                                      result,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 const double*        y,
                                 double*              result,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDdoti((cusparseHandle_t)handle,
                                                      nnz,
                                                      xVal,
                                                      xInd,
                                                      y,
                                                      result,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseSgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         y,
                                 float*               xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgthr(
        (cusparseHandle_t)handle, nnz, y, xVal, xInd, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        y,
                                 double*              xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgthr(
        (cusparseHandle_t)handle, nnz, y, xVal, xInd, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseSgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  float*               y,
                                  float*               xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgthrz(
        (cusparseHandle_t)handle, nnz, y, xVal, xInd, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  double*              y,
                                  double*              xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgthrz(
        (cusparseHandle_t)handle, nnz, y, xVal, xInd, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseSroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 float*               xVal,
                                 const int*           xInd,
                                 float*               y,
                                 const float*         c,
                                 const float*         s,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSroti(
        (cusparseHandle_t)handle, nnz, xVal, xInd, y, c, s, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 double*              xVal,
                                 const int*           xInd,
                                 double*              y,
                                 const double*        c,
                                 const double*        s,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDroti(
        (cusparseHandle_t)handle, nnz, xVal, xInd, y, c, s, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseSsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 float*               y,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSsctr(
        (cusparseHandle_t)handle, nnz, xVal, xInd, y, hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 double*              y,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDsctr(
        (cusparseHandle_t)handle, nnz, xVal, xInd, y, hipIndexBaseToCudaIndexBase(idxBase)));
}

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
                                  float*                    y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       x,
                                                       beta,
                                                       y));
}

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
                                  double*                   y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t
    hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcsrsv2_zeroPivot((cusparseHandle_t)handle, (csrsv2Info_t)info, position));
}

hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsv2_analysis((cusparseHandle_t)handle,
                                 hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (csrsv2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsv2_analysis((cusparseHandle_t)handle,
                                 hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (csrsv2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrsv2_solve((cusparseHandle_t)handle,
                                                              hipOperationToCudaOperation(transA),
                                                              m,
                                                              nnz,
                                                              alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              (csrsv2Info_t)info,
                                                              f,
                                                              x,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrsv2_solve((cusparseHandle_t)handle,
                                                              hipOperationToCudaOperation(transA),
                                                              m,
                                                              nnz,
                                                              alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              (csrsv2Info_t)info,
                                                              f,
                                                              x,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

hipsparseStatus_t hipsparseShybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseShybmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cusparseHybMat_t)hybA,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseDhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDhybmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cusparseHybMat_t)hybA,
                                                       x,
                                                       beta,
                                                       y));
}

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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrmm((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (cusparseMatDescr_t)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrmm((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (cusparseMatDescr_t)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

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
                                   int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrmm2((cusparseHandle_t)handle,
                                                        hipOperationToCudaOperation(transA),
                                                        hipOperationToCudaOperation(transB),
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        alpha,
                                                        (cusparseMatDescr_t)descrA,
                                                        csrSortedValA,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        B,
                                                        ldb,
                                                        beta,
                                                        C,
                                                        ldc));
}

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
                                   int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrmm2((cusparseHandle_t)handle,
                                                        hipOperationToCudaOperation(transA),
                                                        hipOperationToCudaOperation(transB),
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        alpha,
                                                        (cusparseMatDescr_t)descrA,
                                                        csrSortedValA,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        B,
                                                        ldb,
                                                        beta,
                                                        C,
                                                        ldc));
}

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
                                       int*                      nnzTotalDevHostPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsrgemmNnz((cusparseHandle_t)handle,
                                                            hipOperationToCudaOperation(transA),
                                                            hipOperationToCudaOperation(transB),
                                                            m,
                                                            n,
                                                            k,
                                                            (const cusparseMatDescr_t)descrA,
                                                            nnzA,
                                                            csrRowPtrA,
                                                            csrColIndA,
                                                            (const cusparseMatDescr_t)descrB,
                                                            nnzB,
                                                            csrRowPtrB,
                                                            csrColIndB,
                                                            (const cusparseMatDescr_t)descrC,
                                                            csrRowPtrC,
                                                            nnzTotalDevHostPtr));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrgemm((cusparseHandle_t)handle,
                                                         hipOperationToCudaOperation(transA),
                                                         hipOperationToCudaOperation(transB),
                                                         m,
                                                         n,
                                                         k,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const cusparseMatDescr_t)descrB,
                                                         nnzB,
                                                         csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const cusparseMatDescr_t)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrgemm((cusparseHandle_t)handle,
                                                         hipOperationToCudaOperation(transA),
                                                         hipOperationToCudaOperation(transB),
                                                         m,
                                                         n,
                                                         k,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const cusparseMatDescr_t)descrB,
                                                         nnzB,
                                                         csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const cusparseMatDescr_t)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsrgemm2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        csrRowPtrA,
                                        csrColIndA,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        csrRowPtrB,
                                        csrColIndB,
                                        beta,
                                        (const cusparseMatDescr_t)descrD,
                                        nnzD,
                                        csrRowPtrD,
                                        csrColIndD,
                                        (csrgemm2Info_t)info,
                                        pBufferSizeInBytes));
}

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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrgemm2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        k,
                                        alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        csrRowPtrA,
                                        csrColIndA,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        csrRowPtrB,
                                        csrColIndB,
                                        beta,
                                        (const cusparseMatDescr_t)descrD,
                                        nnzD,
                                        csrRowPtrD,
                                        csrColIndD,
                                        (csrgemm2Info_t)info,
                                        pBufferSizeInBytes));
}

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
                                        void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsrgemm2Nnz((cusparseHandle_t)handle,
                                                             m,
                                                             n,
                                                             k,
                                                             (const cusparseMatDescr_t)descrA,
                                                             nnzA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             (const cusparseMatDescr_t)descrB,
                                                             nnzB,
                                                             csrRowPtrB,
                                                             csrColIndB,
                                                             (const cusparseMatDescr_t)descrD,
                                                             nnzD,
                                                             csrRowPtrD,
                                                             csrColIndD,
                                                             (const cusparseMatDescr_t)descrC,
                                                             csrRowPtrC,
                                                             nnzTotalDevHostPtr,
                                                             (const csrgemm2Info_t)info,
                                                             pBuffer));
}

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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrgemm2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          csrValB,
                                                          csrRowPtrB,
                                                          csrColIndB,
                                                          beta,
                                                          (const cusparseMatDescr_t)descrD,
                                                          nnzD,
                                                          csrValD,
                                                          csrRowPtrD,
                                                          csrColIndD,
                                                          (const cusparseMatDescr_t)descrC,
                                                          csrValC,
                                                          csrRowPtrC,
                                                          csrColIndC,
                                                          (const csrgemm2Info_t)info,
                                                          pBuffer));
}

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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrgemm2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          k,
                                                          alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          csrValB,
                                                          csrRowPtrB,
                                                          csrColIndB,
                                                          beta,
                                                          (const cusparseMatDescr_t)descrD,
                                                          nnzD,
                                                          csrValD,
                                                          csrRowPtrD,
                                                          csrColIndD,
                                                          (const cusparseMatDescr_t)descrC,
                                                          csrValC,
                                                          csrRowPtrC,
                                                          csrColIndC,
                                                          (const csrgemm2Info_t)info,
                                                          pBuffer));
}

hipsparseStatus_t
    hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcsrilu02_zeroPivot((cusparseHandle_t)handle, (csrilu02Info_t)info, position));
}

hipsparseStatus_t hipsparseScsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrilu02_bufferSize((cusparseHandle_t)handle,
                                                                     m,
                                                                     nnz,
                                                                     (cusparseMatDescr_t)descrA,
                                                                     csrSortedValA,
                                                                     csrSortedRowPtrA,
                                                                     csrSortedColIndA,
                                                                     (csrilu02Info_t)info,
                                                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrilu02_bufferSize((cusparseHandle_t)handle,
                                                                     m,
                                                                     nnz,
                                                                     (cusparseMatDescr_t)descrA,
                                                                     csrSortedValA,
                                                                     csrSortedRowPtrA,
                                                                     csrSortedColIndA,
                                                                     (csrilu02Info_t)info,
                                                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseScsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const float*              csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrilu02_analysis((cusparseHandle_t)handle,
                                                                   m,
                                                                   nnz,
                                                                   (cusparseMatDescr_t)descrA,
                                                                   csrSortedValA,
                                                                   csrSortedRowPtrA,
                                                                   csrSortedColIndA,
                                                                   (csrilu02Info_t)info,
                                                                   hipPolicyToCudaPolicy(policy),
                                                                   pBuffer));
}

hipsparseStatus_t hipsparseDcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const double*             csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrilu02_analysis((cusparseHandle_t)handle,
                                                                   m,
                                                                   nnz,
                                                                   (cusparseMatDescr_t)descrA,
                                                                   csrSortedValA,
                                                                   csrSortedRowPtrA,
                                                                   csrSortedColIndA,
                                                                   (csrilu02Info_t)info,
                                                                   hipPolicyToCudaPolicy(policy),
                                                                   pBuffer));
}

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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrilu02((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (cusparseMatDescr_t)descrA,
                                                          csrSortedValA_valM,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (csrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrilu02((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (cusparseMatDescr_t)descrA,
                                                          csrSortedValA_valM,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (csrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t    handle,
                                    const int*           csrRowPtr,
                                    int                  nnz,
                                    int                  m,
                                    int*                 cooRowInd,
                                    hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsr2coo((cusparseHandle_t)handle,
                                                         csrRowPtr,
                                                         nnz,
                                                         m,
                                                         cooRowInd,
                                                         hipIndexBaseToCudaIndexBase(idxBase)));
}

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
                                    hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsr2csc((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         csrSortedVal,
                                                         csrSortedRowPtr,
                                                         csrSortedColInd,
                                                         cscSortedVal,
                                                         cscSortedRowInd,
                                                         cscSortedColPtr,
                                                         hipActionToCudaAction(copyValues),
                                                         hipIndexBaseToCudaIndexBase(idxBase)));
}

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
                                    hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsr2csc((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         csrSortedVal,
                                                         csrSortedRowPtr,
                                                         csrSortedColInd,
                                                         cscSortedVal,
                                                         cscSortedRowInd,
                                                         cscSortedColPtr,
                                                         hipActionToCudaAction(copyValues),
                                                         hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipHybPartitionToCudaHybPartition(partitionType)));
}

hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipHybPartitionToCudaHybPartition(partitionType)));
}

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t    handle,
                                    const int*           cooRowInd,
                                    int                  nnz,
                                    int                  m,
                                    int*                 csrRowPtr,
                                    hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcoo2csr((cusparseHandle_t)handle,
                                                         cooRowInd,
                                                         nnz,
                                                         m,
                                                         csrRowPtr,
                                                         hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseCreateIdentityPermutation(hipsparseHandle_t handle, int n, int* p)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCreateIdentityPermutation((cusparseHandle_t)handle, n, p));
}

hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        csrRowPtr,
                                                  const int*        csrColInd,
                                                  size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsrsort_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcsrsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                csrRowPtr,
                                    int*                      csrColInd,
                                    int*                      P,
                                    void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsrsort((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         (const cusparseMatDescr_t)descrA,
                                                         csrRowPtr,
                                                         csrColInd,
                                                         P,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cscColPtr,
                                                  const int*        cscRowInd,
                                                  size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcscsort_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, nnz, cscColPtr, cscRowInd, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcscsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                cscColPtr,
                                    int*                      cscRowInd,
                                    int*                      P,
                                    void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcscsort((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         (const cusparseMatDescr_t)descrA,
                                                         cscColPtr,
                                                         cscRowInd,
                                                         P,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cooRows,
                                                  const int*        cooCols,
                                                  size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcoosort_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, nnz, cooRows, cooCols, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcoosortByRow(hipsparseHandle_t handle,
                                         int               m,
                                         int               n,
                                         int               nnz,
                                         int*              cooRows,
                                         int*              cooCols,
                                         int*              P,
                                         void*             pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcoosortByRow((cusparseHandle_t)handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}

hipsparseStatus_t hipsparseXcoosortByColumn(hipsparseHandle_t handle,
                                            int               m,
                                            int               n,
                                            int               nnz,
                                            int*              cooRows,
                                            int*              cooCols,
                                            int*              P,
                                            void*             pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcoosortByColumn(
        (cusparseHandle_t)handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}

#ifdef __cplusplus
}
#endif
