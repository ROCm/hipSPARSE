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

cusparseDirection_t hipDirectionToCudaDirection(hipsparseDirection_t op)
{
    switch(op)
    {
    case HIPSPARSE_DIRECTION_ROW:
        return CUSPARSE_DIRECTION_ROW;
    case HIPSPARSE_DIRECTION_COLUMN:
        return CUSPARSE_DIRECTION_COLUMN;
    default:
        throw "Non existent hipsparseDirection_t";
    }
}

hipsparseDirection_t CudaDirectionToHIPDirection(cusparseDirection_t op)
{
    switch(op)
    {
    case CUSPARSE_DIRECTION_ROW:
        return HIPSPARSE_DIRECTION_ROW;
    case CUSPARSE_DIRECTION_COLUMN:
        return HIPSPARSE_DIRECTION_COLUMN;
    default:
        throw "Non existent cusparseDirection_t";
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

hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateBsrsv2Info((bsrsv2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsrsv2Info((bsrsv2Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrsv2Info((csrsv2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrsv2Info((csrsv2Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrsm2Info((csrsm2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrsm2Info((csrsm2Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrilu02Info((csrilu02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrilu02Info((csrilu02Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsric02Info((csric02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsric02Info((csric02Info_t)info));
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

hipsparseStatus_t hipsparseCaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    alpha,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCaxpyi((cusparseHandle_t)handle,
                                                       nnz,
                                                       (const cuComplex*)alpha,
                                                       (const cuComplex*)xVal,
                                                       xInd,
                                                       (cuComplex*)y,
                                                       hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZaxpyi(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZaxpyi((cusparseHandle_t)handle,
                                                       nnz,
                                                       (const cuDoubleComplex*)alpha,
                                                       (const cuDoubleComplex*)xVal,
                                                       xInd,
                                                       (cuDoubleComplex*)y,
                                                       hipIndexBaseToCudaIndexBase(idxBase)));
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

hipsparseStatus_t hipsparseCdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 const hipComplex*    y,
                                 hipComplex*          result,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCdoti((cusparseHandle_t)handle,
                                                      nnz,
                                                      (const cuComplex*)xVal,
                                                      xInd,
                                                      (const cuComplex*)y,
                                                      (cuComplex*)result,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZdoti(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       result,
                                 hipsparseIndexBase_t    idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZdoti((cusparseHandle_t)handle,
                                                      nnz,
                                                      (const cuDoubleComplex*)xVal,
                                                      xInd,
                                                      (const cuDoubleComplex*)y,
                                                      (cuDoubleComplex*)result,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseCdotci(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  const hipComplex*    y,
                                  hipComplex*          result,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCdotci((cusparseHandle_t)handle,
                                                       nnz,
                                                       (const cuComplex*)xVal,
                                                       xInd,
                                                       (const cuComplex*)y,
                                                       (cuComplex*)result,
                                                       hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZdotci(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  const hipDoubleComplex* y,
                                  hipDoubleComplex*       result,
                                  hipsparseIndexBase_t    idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZdotci((cusparseHandle_t)handle,
                                                       nnz,
                                                       (const cuDoubleComplex*)xVal,
                                                       xInd,
                                                       (const cuDoubleComplex*)y,
                                                       (cuDoubleComplex*)result,
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

hipsparseStatus_t hipsparseCgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    y,
                                 hipComplex*          xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgthr((cusparseHandle_t)handle,
                                                      nnz,
                                                      (const cuComplex*)y,
                                                      (cuComplex*)xVal,
                                                      xInd,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZgthr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       xVal,
                                 const int*              xInd,
                                 hipsparseIndexBase_t    idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgthr((cusparseHandle_t)handle,
                                                      nnz,
                                                      (const cuDoubleComplex*)y,
                                                      (cuDoubleComplex*)xVal,
                                                      xInd,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
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

hipsparseStatus_t hipsparseCgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  hipComplex*          y,
                                  hipComplex*          xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgthrz((cusparseHandle_t)handle,
                                                       nnz,
                                                       (cuComplex*)y,
                                                       (cuComplex*)xVal,
                                                       xInd,
                                                       hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  hipDoubleComplex*    y,
                                  hipDoubleComplex*    xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgthrz((cusparseHandle_t)handle,
                                                       nnz,
                                                       (cuDoubleComplex*)y,
                                                       (cuDoubleComplex*)xVal,
                                                       xInd,
                                                       hipIndexBaseToCudaIndexBase(idxBase)));
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

hipsparseStatus_t hipsparseCsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 hipComplex*          y,
                                 hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCsctr((cusparseHandle_t)handle,
                                                      nnz,
                                                      (const cuComplex*)xVal,
                                                      xInd,
                                                      (cuComplex*)y,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseZsctr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 hipDoubleComplex*       y,
                                 hipsparseIndexBase_t    idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZsctr((cusparseHandle_t)handle,
                                                      nnz,
                                                      (const cuDoubleComplex*)xVal,
                                                      xInd,
                                                      (cuDoubleComplex*)y,
                                                      hipIndexBaseToCudaIndexBase(idxBase)));
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
                                  hipComplex*               y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       (const cuComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cuComplex*)csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       (const cuComplex*)x,
                                                       (const cuComplex*)beta,
                                                       (cuComplex*)y));
}

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
                                  hipDoubleComplex*         y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       (const cuDoubleComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cuDoubleComplex*)csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       (const cuDoubleComplex*)x,
                                                       (const cuDoubleComplex*)beta,
                                                       (cuDoubleComplex*)y));
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

hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   (cuComplex*)csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (csrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipOperationToCudaOperation(transA),
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)csrSortedValA,
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

hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         csrSortedValA,
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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsv2_analysis((cusparseHandle_t)handle,
                                 hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuComplex*)csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (csrsv2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsv2_analysis((cusparseHandle_t)handle,
                                 hipOperationToCudaOperation(transA),
                                 m,
                                 nnz,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)csrSortedValA,
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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrsv2_solve((cusparseHandle_t)handle,
                                                              hipOperationToCudaOperation(transA),
                                                              m,
                                                              nnz,
                                                              (const cuComplex*)alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              (const cuComplex*)csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              (csrsv2Info_t)info,
                                                              (const cuComplex*)f,
                                                              (cuComplex*)x,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrsv2_solve((cusparseHandle_t)handle,
                                                              hipOperationToCudaOperation(transA),
                                                              m,
                                                              nnz,
                                                              (const cuDoubleComplex*)alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              (const cuDoubleComplex*)csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              (csrsv2Info_t)info,
                                                              (const cuDoubleComplex*)f,
                                                              (cuDoubleComplex*)x,
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

hipsparseStatus_t hipsparseChybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseChybmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       (const cuComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cusparseHybMat_t)hybA,
                                                       (const cuComplex*)x,
                                                       (const cuComplex*)beta,
                                                       (cuComplex*)y));
}

hipsparseStatus_t hipsparseZhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZhybmv((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       (const cuDoubleComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cusparseHybMat_t)hybA,
                                                       (const cuDoubleComplex*)x,
                                                       (const cuDoubleComplex*)beta,
                                                       (cuDoubleComplex*)y));
}

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
                                  float*                    y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsrmv((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       bsrSortedValA,
                                                       bsrSortedRowPtrA,
                                                       bsrSortedColIndA,
                                                       blockDim,
                                                       x,
                                                       beta,
                                                       y));
}

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
                                  double*                   y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsrmv((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       bsrSortedValA,
                                                       bsrSortedRowPtrA,
                                                       bsrSortedColIndA,
                                                       blockDim,
                                                       x,
                                                       beta,
                                                       y));
}

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
                                  hipComplex*               y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsrmv((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       (const cuComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cuComplex*)bsrSortedValA,
                                                       bsrSortedRowPtrA,
                                                       bsrSortedColIndA,
                                                       blockDim,
                                                       (const cuComplex*)x,
                                                       (const cuComplex*)beta,
                                                       (cuComplex*)y));
}

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
                                  hipDoubleComplex*         y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsrmv((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       (const cuDoubleComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cuDoubleComplex*)bsrSortedValA,
                                                       bsrSortedRowPtrA,
                                                       bsrSortedColIndA,
                                                       blockDim,
                                                       (const cuDoubleComplex*)x,
                                                       (const cuDoubleComplex*)beta,
                                                       (cuDoubleComplex*)y));
}

hipsparseStatus_t
    hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXbsrsv2_zeroPivot((cusparseHandle_t)handle, (bsrsv2Info_t)info, position));
}

hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dir),
                                   hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dir),
                                   hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dir),
                                   hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsv2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dir),
                                   hipOperationToCudaOperation(transA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsv2Info_t)info,
                                   pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dir),
                                 hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dir),
                                 hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dir),
                                 hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsv2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dir),
                                 hipOperationToCudaOperation(transA),
                                 mb,
                                 nnzb,
                                 (cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsv2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsrsv2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dir),
                                                              hipOperationToCudaOperation(transA),
                                                              mb,
                                                              nnzb,
                                                              alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsv2Info_t)info,
                                                              f,
                                                              x,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsrsv2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dir),
                                                              hipOperationToCudaOperation(transA),
                                                              mb,
                                                              nnzb,
                                                              alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsv2Info_t)info,
                                                              f,
                                                              x,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsrsv2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dir),
                                                              hipOperationToCudaOperation(transA),
                                                              mb,
                                                              nnzb,
                                                              (const cuComplex*)alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              (const cuComplex*)bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsv2Info_t)info,
                                                              (const cuComplex*)f,
                                                              (cuComplex*)x,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsrsv2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dir),
                                                              hipOperationToCudaOperation(transA),
                                                              mb,
                                                              nnzb,
                                                              (const cuDoubleComplex*)alpha,
                                                              (cusparseMatDescr_t)descrA,
                                                              (const cuDoubleComplex*)bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsv2Info_t)info,
                                                              (const cuDoubleComplex*)f,
                                                              (cuDoubleComplex*)x,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrmm((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       (const cuComplex*)alpha,
                                                       (cusparseMatDescr_t)descrA,
                                                       (const cuComplex*)csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       (const cuComplex*)B,
                                                       ldb,
                                                       (const cuComplex*)beta,
                                                       (cuComplex*)C,
                                                       ldc));
}

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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrmm((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       (const cuDoubleComplex*)alpha,
                                                       (cusparseMatDescr_t)descrA,
                                                       (const cuDoubleComplex*)csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       (const cuDoubleComplex*)B,
                                                       ldb,
                                                       (const cuDoubleComplex*)beta,
                                                       (cuDoubleComplex*)C,
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
                                   int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrmm2((cusparseHandle_t)handle,
                                                        hipOperationToCudaOperation(transA),
                                                        hipOperationToCudaOperation(transB),
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        (const cuComplex*)alpha,
                                                        (cusparseMatDescr_t)descrA,
                                                        (const cuComplex*)csrSortedValA,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        (const cuComplex*)B,
                                                        ldb,
                                                        (const cuComplex*)beta,
                                                        (cuComplex*)C,
                                                        ldc));
}

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
                                   int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrmm2((cusparseHandle_t)handle,
                                                        hipOperationToCudaOperation(transA),
                                                        hipOperationToCudaOperation(transB),
                                                        m,
                                                        n,
                                                        k,
                                                        nnz,
                                                        (const cuDoubleComplex*)alpha,
                                                        (cusparseMatDescr_t)descrA,
                                                        (const cuDoubleComplex*)csrSortedValA,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        (const cuDoubleComplex*)B,
                                                        ldb,
                                                        (const cuDoubleComplex*)beta,
                                                        (cuDoubleComplex*)C,
                                                        ldc));
}

hipsparseStatus_t
    hipsparseXcsrsm2_zeroPivot(hipsparseHandle_t handle, csrsm2Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcsrsm2_zeroPivot((cusparseHandle_t)handle, (csrsm2Info_t)info, position));
}

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
                                                 size_t*                   pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipOperationToCudaOperation(transA),
                                      hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipPolicyToCudaPolicy(policy),
                                      pBufferSize));
}

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
                                                 size_t*                   pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipOperationToCudaOperation(transA),
                                      hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipPolicyToCudaPolicy(policy),
                                      pBufferSize));
}

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
                                                 size_t*                   pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipOperationToCudaOperation(transA),
                                      hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      (const cuComplex*)alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      (const cuComplex*)csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      (const cuComplex*)B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipPolicyToCudaPolicy(policy),
                                      pBufferSize));
}

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
                                                 size_t*                   pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsm2_bufferSizeExt((cusparseHandle_t)handle,
                                      algo,
                                      hipOperationToCudaOperation(transA),
                                      hipOperationToCudaOperation(transB),
                                      m,
                                      nrhs,
                                      nnz,
                                      (const cuDoubleComplex*)alpha,
                                      (const cusparseMatDescr_t)descrA,
                                      (const cuDoubleComplex*)csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      (const cuDoubleComplex*)B,
                                      ldb,
                                      (csrsm2Info_t)info,
                                      hipPolicyToCudaPolicy(policy),
                                      pBufferSize));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 (const cuComplex*)alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuComplex*)csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (const cuComplex*)B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrsm2_analysis((cusparseHandle_t)handle,
                                 algo,
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transB),
                                 m,
                                 nrhs,
                                 nnz,
                                 (const cuDoubleComplex*)alpha,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)csrSortedValA,
                                 csrSortedRowPtrA,
                                 csrSortedColIndA,
                                 (const cuDoubleComplex*)B,
                                 ldb,
                                 (csrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrsm2_solve((cusparseHandle_t)handle,
                                                              algo,
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transB),
                                                              m,
                                                              nrhs,
                                                              nnz,
                                                              alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              B,
                                                              ldb,
                                                              (csrsm2Info_t)info,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrsm2_solve((cusparseHandle_t)handle,
                                                              algo,
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transB),
                                                              m,
                                                              nrhs,
                                                              nnz,
                                                              alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              B,
                                                              ldb,
                                                              (csrsm2Info_t)info,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrsm2_solve((cusparseHandle_t)handle,
                                                              algo,
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transB),
                                                              m,
                                                              nrhs,
                                                              nnz,
                                                              (const cuComplex*)alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              (const cuComplex*)csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              (cuComplex*)B,
                                                              ldb,
                                                              (csrsm2Info_t)info,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrsm2_solve((cusparseHandle_t)handle,
                                                              algo,
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transB),
                                                              m,
                                                              nrhs,
                                                              nnz,
                                                              (const cuDoubleComplex*)alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              (const cuDoubleComplex*)csrSortedValA,
                                                              csrSortedRowPtrA,
                                                              csrSortedColIndA,
                                                              (cuDoubleComplex*)B,
                                                              ldb,
                                                              (csrsm2Info_t)info,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                  int               ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgemmi((cusparseHandle_t)handle,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       cscValB,
                                                       cscColPtrB,
                                                       cscRowIndB,
                                                       beta,
                                                       C,
                                                       ldc));
}

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
                                  int               ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgemmi((cusparseHandle_t)handle,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       cscValB,
                                                       cscColPtrB,
                                                       cscRowIndB,
                                                       beta,
                                                       C,
                                                       ldc));
}

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
                                  int               ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgemmi((cusparseHandle_t)handle,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       (const cuComplex*)alpha,
                                                       (const cuComplex*)A,
                                                       lda,
                                                       (const cuComplex*)cscValB,
                                                       cscColPtrB,
                                                       cscRowIndB,
                                                       (const cuComplex*)beta,
                                                       (cuComplex*)C,
                                                       ldc));
}

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
                                  int                     ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgemmi((cusparseHandle_t)handle,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       (const cuDoubleComplex*)alpha,
                                                       (const cuDoubleComplex*)A,
                                                       lda,
                                                       (const cuDoubleComplex*)cscValB,
                                                       cscColPtrB,
                                                       cscRowIndB,
                                                       (const cuDoubleComplex*)beta,
                                                       (cuDoubleComplex*)C,
                                                       ldc));
}

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
                                       int*                      nnzTotalDevHostPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsrgeamNnz((cusparseHandle_t)handle,
                                                            m,
                                                            n,
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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrgeam((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         alpha,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         beta,
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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrgeam((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         alpha,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         beta,
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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrgeam((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         (const cuComplex*)alpha,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         (const cuComplex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const cuComplex*)beta,
                                                         (const cusparseMatDescr_t)descrB,
                                                         nnzB,
                                                         (const cuComplex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuComplex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrgeam((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         (const cuDoubleComplex*)alpha,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         (const cuDoubleComplex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const cuDoubleComplex*)beta,
                                                         (const cusparseMatDescr_t)descrB,
                                                         nnzB,
                                                         (const cuDoubleComplex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuDoubleComplex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsrgeam2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        csrSortedValA,
                                        csrSortedRowPtrA,
                                        csrSortedColIndA,
                                        beta,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        csrSortedValB,
                                        csrSortedRowPtrB,
                                        csrSortedColIndB,
                                        (const cusparseMatDescr_t)descrC,
                                        csrSortedValC,
                                        csrSortedRowPtrC,
                                        csrSortedColIndC,
                                        pBufferSizeInBytes));
}

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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrgeam2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        csrSortedValA,
                                        csrSortedRowPtrA,
                                        csrSortedColIndA,
                                        beta,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        csrSortedValB,
                                        csrSortedRowPtrB,
                                        csrSortedColIndB,
                                        (const cusparseMatDescr_t)descrC,
                                        csrSortedValC,
                                        csrSortedRowPtrC,
                                        csrSortedColIndC,
                                        pBufferSizeInBytes));
}

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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrgeam2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        (const cuComplex*)alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        (const cuComplex*)csrSortedValA,
                                        csrSortedRowPtrA,
                                        csrSortedColIndA,
                                        (const cuComplex*)beta,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        (const cuComplex*)csrSortedValB,
                                        csrSortedRowPtrB,
                                        csrSortedColIndB,
                                        (const cusparseMatDescr_t)descrC,
                                        (cuComplex*)csrSortedValC,
                                        csrSortedRowPtrC,
                                        csrSortedColIndC,
                                        pBufferSizeInBytes));
}

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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrgeam2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        (const cuDoubleComplex*)alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        (const cuDoubleComplex*)csrSortedValA,
                                        csrSortedRowPtrA,
                                        csrSortedColIndA,
                                        (const cuDoubleComplex*)beta,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        (const cuDoubleComplex*)csrSortedValB,
                                        csrSortedRowPtrB,
                                        csrSortedColIndB,
                                        (const cusparseMatDescr_t)descrC,
                                        (cuDoubleComplex*)csrSortedValC,
                                        csrSortedRowPtrC,
                                        csrSortedColIndC,
                                        pBufferSizeInBytes));
}

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
                                        void*                     workspace)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsrgeam2Nnz((cusparseHandle_t)handle,
                                                             m,
                                                             n,
                                                             (const cusparseMatDescr_t)descrA,
                                                             nnzA,
                                                             csrSortedRowPtrA,
                                                             csrSortedColIndA,
                                                             (const cusparseMatDescr_t)descrB,
                                                             nnzB,
                                                             csrSortedRowPtrB,
                                                             csrSortedColIndB,
                                                             (const cusparseMatDescr_t)descrC,
                                                             csrSortedRowPtrC,
                                                             nnzTotalDevHostPtr,
                                                             workspace));
}

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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrgeam2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          csrSortedValA,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          beta,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          csrSortedValB,
                                                          csrSortedRowPtrB,
                                                          csrSortedColIndB,
                                                          (const cusparseMatDescr_t)descrC,
                                                          csrSortedValC,
                                                          csrSortedRowPtrC,
                                                          csrSortedColIndC,
                                                          pBuffer));
}

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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrgeam2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          csrSortedValA,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          beta,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          csrSortedValB,
                                                          csrSortedRowPtrB,
                                                          csrSortedColIndB,
                                                          (const cusparseMatDescr_t)descrC,
                                                          csrSortedValC,
                                                          csrSortedRowPtrC,
                                                          csrSortedColIndC,
                                                          pBuffer));
}

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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrgeam2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          (const cuComplex*)alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          (const cuComplex*)csrSortedValA,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (const cuComplex*)beta,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          (const cuComplex*)csrSortedValB,
                                                          csrSortedRowPtrB,
                                                          csrSortedColIndB,
                                                          (const cusparseMatDescr_t)descrC,
                                                          (cuComplex*)csrSortedValC,
                                                          csrSortedRowPtrC,
                                                          csrSortedColIndC,
                                                          pBuffer));
}

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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrgeam2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          (const cuDoubleComplex*)alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          (const cuDoubleComplex*)csrSortedValA,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (const cuDoubleComplex*)beta,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          (const cuDoubleComplex*)csrSortedValB,
                                                          csrSortedRowPtrB,
                                                          csrSortedColIndB,
                                                          (const cusparseMatDescr_t)descrC,
                                                          (cuDoubleComplex*)csrSortedValC,
                                                          csrSortedRowPtrC,
                                                          csrSortedColIndC,
                                                          pBuffer));
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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrgemm((cusparseHandle_t)handle,
                                                         hipOperationToCudaOperation(transA),
                                                         hipOperationToCudaOperation(transB),
                                                         m,
                                                         n,
                                                         k,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         (const cuComplex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const cusparseMatDescr_t)descrB,
                                                         nnzB,
                                                         (const cuComplex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuComplex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrgemm((cusparseHandle_t)handle,
                                                         hipOperationToCudaOperation(transA),
                                                         hipOperationToCudaOperation(transB),
                                                         m,
                                                         n,
                                                         k,
                                                         (const cusparseMatDescr_t)descrA,
                                                         nnzA,
                                                         (const cuDoubleComplex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const cusparseMatDescr_t)descrB,
                                                         nnzB,
                                                         (const cuDoubleComplex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuDoubleComplex*)csrValC,
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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCcsrgemm2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        k,
                                        (const cuComplex*)alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        csrRowPtrA,
                                        csrColIndA,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        csrRowPtrB,
                                        csrColIndB,
                                        (const cuComplex*)beta,
                                        (const cusparseMatDescr_t)descrD,
                                        nnzD,
                                        csrRowPtrD,
                                        csrColIndD,
                                        (csrgemm2Info_t)info,
                                        pBufferSizeInBytes));
}

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
                                                   size_t*                   pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrgemm2_bufferSizeExt((cusparseHandle_t)handle,
                                        m,
                                        n,
                                        k,
                                        (const cuDoubleComplex*)alpha,
                                        (const cusparseMatDescr_t)descrA,
                                        nnzA,
                                        csrRowPtrA,
                                        csrColIndA,
                                        (const cusparseMatDescr_t)descrB,
                                        nnzB,
                                        csrRowPtrB,
                                        csrColIndB,
                                        (const cuDoubleComplex*)beta,
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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrgemm2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          k,
                                                          (const cuComplex*)alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          (const cuComplex*)csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          (const cuComplex*)csrValB,
                                                          csrRowPtrB,
                                                          csrColIndB,
                                                          (const cuComplex*)beta,
                                                          (const cusparseMatDescr_t)descrD,
                                                          nnzD,
                                                          (const cuComplex*)csrValD,
                                                          csrRowPtrD,
                                                          csrColIndD,
                                                          (const cusparseMatDescr_t)descrC,
                                                          (cuComplex*)csrValC,
                                                          csrRowPtrC,
                                                          csrColIndC,
                                                          (const csrgemm2Info_t)info,
                                                          pBuffer));
}

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
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrgemm2((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          k,
                                                          (const cuDoubleComplex*)alpha,
                                                          (const cusparseMatDescr_t)descrA,
                                                          nnzA,
                                                          (const cuDoubleComplex*)csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          (const cusparseMatDescr_t)descrB,
                                                          nnzB,
                                                          (const cuDoubleComplex*)csrValB,
                                                          csrRowPtrB,
                                                          csrColIndB,
                                                          (const cuDoubleComplex*)beta,
                                                          (const cusparseMatDescr_t)descrD,
                                                          nnzD,
                                                          (const cuDoubleComplex*)csrValD,
                                                          csrRowPtrD,
                                                          csrColIndD,
                                                          (const cusparseMatDescr_t)descrC,
                                                          (cuDoubleComplex*)csrValC,
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

hipsparseStatus_t hipsparseCcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrilu02_bufferSize((cusparseHandle_t)handle,
                                                                     m,
                                                                     nnz,
                                                                     (cusparseMatDescr_t)descrA,
                                                                     (cuComplex*)csrSortedValA,
                                                                     csrSortedRowPtrA,
                                                                     csrSortedColIndA,
                                                                     (csrilu02Info_t)info,
                                                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrilu02_bufferSize((cusparseHandle_t)handle,
                                     m,
                                     nnz,
                                     (cusparseMatDescr_t)descrA,
                                     (cuDoubleComplex*)csrSortedValA,
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

hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         csrSortedValA,
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

hipsparseStatus_t hipsparseCcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrilu02_analysis((cusparseHandle_t)handle,
                                                                   m,
                                                                   nnz,
                                                                   (cusparseMatDescr_t)descrA,
                                                                   (const cuComplex*)csrSortedValA,
                                                                   csrSortedRowPtrA,
                                                                   csrSortedColIndA,
                                                                   (csrilu02Info_t)info,
                                                                   hipPolicyToCudaPolicy(policy),
                                                                   pBuffer));
}

hipsparseStatus_t hipsparseZcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipDoubleComplex*   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrilu02_analysis((cusparseHandle_t)handle,
                                   m,
                                   nnz,
                                   (cusparseMatDescr_t)descrA,
                                   (const cuDoubleComplex*)csrSortedValA,
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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrilu02((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (cusparseMatDescr_t)descrA,
                                                          (cuComplex*)csrSortedValA_valM,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (csrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrilu02((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (cusparseMatDescr_t)descrA,
                                                          (cuDoubleComplex*)csrSortedValA_valM,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (csrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

hipsparseStatus_t
    hipsparseXcsric02_zeroPivot(hipsparseHandle_t handle, csric02Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcsric02_zeroPivot((cusparseHandle_t)handle, (csric02Info_t)info, position));
}

hipsparseStatus_t hipsparseScsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsric02_bufferSize((cusparseHandle_t)handle,
                                                                    m,
                                                                    nnz,
                                                                    (cusparseMatDescr_t)descrA,
                                                                    csrSortedValA,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (csric02Info_t)info,
                                                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsric02_bufferSize((cusparseHandle_t)handle,
                                                                    m,
                                                                    nnz,
                                                                    (cusparseMatDescr_t)descrA,
                                                                    csrSortedValA,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (csric02Info_t)info,
                                                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsric02_bufferSize((cusparseHandle_t)handle,
                                                                    m,
                                                                    nnz,
                                                                    (cusparseMatDescr_t)descrA,
                                                                    (cuComplex*)csrSortedValA,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (csric02Info_t)info,
                                                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsric02_bufferSize((cusparseHandle_t)handle,
                                                                    m,
                                                                    nnz,
                                                                    (cusparseMatDescr_t)descrA,
                                                                    (cuDoubleComplex*)csrSortedValA,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (csric02Info_t)info,
                                                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseScsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseScsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsric02_analysis((cusparseHandle_t)handle,
                                                                  m,
                                                                  nnz,
                                                                  (cusparseMatDescr_t)descrA,
                                                                  csrSortedValA,
                                                                  csrSortedRowPtrA,
                                                                  csrSortedColIndA,
                                                                  (csric02Info_t)info,
                                                                  hipPolicyToCudaPolicy(policy),
                                                                  pBuffer));
}

hipsparseStatus_t hipsparseDcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsric02_analysis((cusparseHandle_t)handle,
                                                                  m,
                                                                  nnz,
                                                                  (cusparseMatDescr_t)descrA,
                                                                  csrSortedValA,
                                                                  csrSortedRowPtrA,
                                                                  csrSortedColIndA,
                                                                  (csric02Info_t)info,
                                                                  hipPolicyToCudaPolicy(policy),
                                                                  pBuffer));
}

hipsparseStatus_t hipsparseCcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsric02_analysis((cusparseHandle_t)handle,
                                                                  m,
                                                                  nnz,
                                                                  (cusparseMatDescr_t)descrA,
                                                                  (const cuComplex*)csrSortedValA,
                                                                  csrSortedRowPtrA,
                                                                  csrSortedColIndA,
                                                                  (csric02Info_t)info,
                                                                  hipPolicyToCudaPolicy(policy),
                                                                  pBuffer));
}

hipsparseStatus_t hipsparseZcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsric02_analysis((cusparseHandle_t)handle,
                                  m,
                                  nnz,
                                  (cusparseMatDescr_t)descrA,
                                  (const cuDoubleComplex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (csric02Info_t)info,
                                  hipPolicyToCudaPolicy(policy),
                                  pBuffer));
}

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
                                    void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsric02((cusparseHandle_t)handle,
                                                         m,
                                                         nnz,
                                                         (cusparseMatDescr_t)descrA,
                                                         csrSortedValA_valM,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (csric02Info_t)info,
                                                         hipPolicyToCudaPolicy(policy),
                                                         pBuffer));
}

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
                                    void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsric02((cusparseHandle_t)handle,
                                                         m,
                                                         nnz,
                                                         (cusparseMatDescr_t)descrA,
                                                         csrSortedValA_valM,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (csric02Info_t)info,
                                                         hipPolicyToCudaPolicy(policy),
                                                         pBuffer));
}

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
                                    void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsric02((cusparseHandle_t)handle,
                                                         m,
                                                         nnz,
                                                         (cusparseMatDescr_t)descrA,
                                                         (cuComplex*)csrSortedValA_valM,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (csric02Info_t)info,
                                                         hipPolicyToCudaPolicy(policy),
                                                         pBuffer));
}

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
                                    void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsric02((cusparseHandle_t)handle,
                                                         m,
                                                         nnz,
                                                         (cusparseMatDescr_t)descrA,
                                                         (cuDoubleComplex*)csrSortedValA_valM,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (csric02Info_t)info,
                                                         hipPolicyToCudaPolicy(policy),
                                                         pBuffer));
}

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const float*              A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSnnz((cusparseHandle_t)handle,
                                                     hipDirectionToCudaDirection(dirA),
                                                     m,
                                                     n,
                                                     (const cusparseMatDescr_t)descrA,
                                                     A,
                                                     lda,
                                                     nnzPerRowColumn,
                                                     nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const double*             A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDnnz((cusparseHandle_t)handle,
                                                     hipDirectionToCudaDirection(dirA),
                                                     m,
                                                     n,
                                                     (const cusparseMatDescr_t)descrA,
                                                     A,
                                                     lda,
                                                     nnzPerRowColumn,
                                                     nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseCnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipComplex*         A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCnnz((cusparseHandle_t)handle,
                                                     hipDirectionToCudaDirection(dirA),
                                                     m,
                                                     n,
                                                     (const cusparseMatDescr_t)descrA,
                                                     (const cuComplex*)A,
                                                     lda,
                                                     nnzPerRowColumn,
                                                     nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseZnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipDoubleComplex*   A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZnnz((cusparseHandle_t)handle,
                                                     hipDirectionToCudaDirection(dirA),
                                                     m,
                                                     n,
                                                     (const cusparseMatDescr_t)descrA,
                                                     (const cuDoubleComplex*)A,
                                                     lda,
                                                     nnzPerRowColumn,
                                                     nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      float*                    csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSdense2csr((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           A,
                                                           ld,
                                                           nnzPerRow,
                                                           csrVal,
                                                           csrRowPtr,
                                                           csrColInd));
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      double*                   csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDdense2csr((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           A,
                                                           ld,
                                                           nnzPerRow,
                                                           csrVal,
                                                           csrRowPtr,
                                                           csrColInd));
}

hipsparseStatus_t hipsparseCdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipComplex*               csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCdense2csr((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const cuComplex*)A,
                                                           ld,
                                                           nnzPerRow,
                                                           (cuComplex*)csrVal,
                                                           csrRowPtr,
                                                           csrColInd));
}

hipsparseStatus_t hipsparseZdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipDoubleComplex*         csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZdense2csr((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const cuDoubleComplex*)A,
                                                           ld,
                                                           nnzPerRow,
                                                           (cuDoubleComplex*)csrVal,
                                                           csrRowPtr,
                                                           csrColInd));
}

hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      float*                    cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSdense2csc((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           A,
                                                           ld,
                                                           nnzPerColumn,
                                                           cscVal,
                                                           cscRowInd,
                                                           cscColPtr));
}

hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      double*                   cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDdense2csc((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           A,
                                                           ld,
                                                           nnzPerColumn,
                                                           cscVal,
                                                           cscRowInd,
                                                           cscColPtr));
}

hipsparseStatus_t hipsparseCdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      hipComplex*               cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCdense2csc((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const cuComplex*)A,
                                                           ld,
                                                           nnzPerColumn,
                                                           (cuComplex*)cscVal,
                                                           cscRowInd,
                                                           cscColPtr));
}

hipsparseStatus_t hipsparseZdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerColumn,
                                      hipDoubleComplex*         cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZdense2csc((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const cuDoubleComplex*)A,
                                                           ld,
                                                           nnzPerColumn,
                                                           (cuDoubleComplex*)cscVal,
                                                           cscRowInd,
                                                           cscColPtr));
}

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      float*                    A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsr2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           csrVal,
                                                           csrRowPtr,
                                                           csrColInd,
                                                           A,
                                                           ld));
}

hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      double*                   A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsr2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           csrVal,
                                                           csrRowPtr,
                                                           csrColInd,
                                                           A,
                                                           ld));
}

hipsparseStatus_t hipsparseCcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      hipComplex*               A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsr2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const hipComplex*)csrVal,
                                                           csrRowPtr,
                                                           csrColInd,
                                                           (hipComplex*)A,
                                                           ld));
}

hipsparseStatus_t hipsparseZcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      hipDoubleComplex*         A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsr2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const cuDoubleComplex*)csrVal,
                                                           csrRowPtr,
                                                           csrColInd,
                                                           (cuDoubleComplex*)A,
                                                           ld));
}

hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      float*                    A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsc2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           cscVal,
                                                           cscRowInd,
                                                           cscColPtr,
                                                           A,
                                                           ld));
}

hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      double*                   A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsc2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           cscVal,
                                                           cscRowInd,
                                                           cscColPtr,
                                                           A,
                                                           ld));
}

hipsparseStatus_t hipsparseCcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipComplex*               A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsc2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const cuComplex*)cscVal,
                                                           cscRowInd,
                                                           cscColPtr,
                                                           (cuComplex*)A,
                                                           ld));
}

hipsparseStatus_t hipsparseZcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipDoubleComplex*         A,
                                      int                       ld)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsc2dense((cusparseHandle_t)handle,
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)descr,
                                                           (const cuDoubleComplex*)cscVal,
                                                           cscRowInd,
                                                           cscColPtr,
                                                           (cuDoubleComplex*)A,
                                                           ld));
}

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
                                       int*                      bsrNnzb)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXcsr2bsrNnz((cusparseHandle_t)handle,
                                                            hipDirectionToCudaDirection(dirA),
                                                            m,
                                                            n,
                                                            (const cusparseMatDescr_t)descrA,
                                                            csrRowPtrA,
                                                            csrColIndA,
                                                            blockDim,
                                                            (const cusparseMatDescr_t)descrC,
                                                            bsrRowPtrC,
                                                            bsrNnzb));
}

hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         float                     tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSnnz_compress((cusparseHandle_t)handle,
                                                              m,
                                                              (const cusparseMatDescr_t)descrA,
                                                              csrValA,
                                                              csrRowPtrA,
                                                              nnzPerRow,
                                                              nnzC,
                                                              tol));
}

hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         double                    tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDnnz_compress((cusparseHandle_t)handle,
                                                              m,
                                                              (const cusparseMatDescr_t)descrA,
                                                              csrValA,
                                                              csrRowPtrA,
                                                              nnzPerRow,
                                                              nnzC,
                                                              tol));
}

hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipComplex                tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCnnz_compress((cusparseHandle_t)handle,
                                                              m,
                                                              (const cusparseMatDescr_t)descrA,
                                                              (const cuComplex*)csrValA,
                                                              csrRowPtrA,
                                                              nnzPerRow,
                                                              nnzC,
                                                              {cuCrealf(tol), cuCimagf(tol)}));
}

hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipDoubleComplex          tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZnnz_compress((cusparseHandle_t)handle,
                                                              m,
                                                              (const cusparseMatDescr_t)descrA,
                                                              (const cuDoubleComplex*)csrValA,
                                                              csrRowPtrA,
                                                              nnzPerRow,
                                                              nnzC,
                                                              {cuCreal(tol), cuCimag(tol)}));
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
                                    hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsr2csc((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         (const cuComplex*)csrSortedVal,
                                                         csrSortedRowPtr,
                                                         csrSortedColInd,
                                                         (cuComplex*)cscSortedVal,
                                                         cscSortedRowInd,
                                                         cscSortedColPtr,
                                                         hipActionToCudaAction(copyValues),
                                                         hipIndexBaseToCudaIndexBase(idxBase)));
}

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
                                    hipsparseIndexBase_t    idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsr2csc((cusparseHandle_t)handle,
                                                         m,
                                                         n,
                                                         nnz,
                                                         (const cuDoubleComplex*)csrSortedVal,
                                                         csrSortedRowPtr,
                                                         csrSortedColInd,
                                                         (cuDoubleComplex*)cscSortedVal,
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

hipsparseStatus_t hipsparseCcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCcsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         (const cuComplex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipHybPartitionToCudaHybPartition(partitionType)));
}

hipsparseStatus_t hipsparseZcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsr2hyb((cusparseHandle_t)handle,
                         m,
                         n,
                         (const cusparseMatDescr_t)descrA,
                         (const cuDoubleComplex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const cusparseHybMat_t)hybA,
                         userEllWidth,
                         hipHybPartitionToCudaHybPartition(partitionType)));
}

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
                                    int*                      bsrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsr2bsr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         m,
                                                         n,
                                                         (const cusparseMatDescr_t)descrA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         bsrValC,
                                                         bsrRowPtrC,
                                                         bsrColIndC));
}

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
                                    int*                      bsrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsr2bsr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         m,
                                                         n,
                                                         (const cusparseMatDescr_t)descrA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         bsrValC,
                                                         bsrRowPtrC,
                                                         bsrColIndC));
}

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
                                    int*                      bsrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsr2bsr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         m,
                                                         n,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cuComplex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuComplex*)bsrValC,
                                                         bsrRowPtrC,
                                                         bsrColIndC));
}

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
                                    int*                      bsrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsr2bsr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         m,
                                                         n,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cuDoubleComplex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuDoubleComplex*)bsrValC,
                                                         bsrRowPtrC,
                                                         bsrColIndC));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsr2csr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nb,
                                                         (const cusparseMatDescr_t)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsr2csr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nb,
                                                         (const cusparseMatDescr_t)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsr2csr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nb,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cuComplex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuComplex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                    int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsr2csr((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nb,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cuDoubleComplex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (const cusparseMatDescr_t)descrC,
                                                         (cuDoubleComplex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

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
                                             float                     tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsr2csr_compress((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  (const cusparseMatDescr_t)descrA,
                                                                  csrValA,
                                                                  csrColIndA,
                                                                  csrRowPtrA,
                                                                  nnzA,
                                                                  nnzPerRow,
                                                                  csrValC,
                                                                  csrColIndC,
                                                                  csrRowPtrC,
                                                                  tol));
}

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
                                             double                    tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsr2csr_compress((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  (const cusparseMatDescr_t)descrA,
                                                                  csrValA,
                                                                  csrColIndA,
                                                                  csrRowPtrA,
                                                                  nnzA,
                                                                  nnzPerRow,
                                                                  csrValC,
                                                                  csrColIndC,
                                                                  csrRowPtrC,
                                                                  tol));
}

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
                                             hipComplex                tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsr2csr_compress((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  (const cusparseMatDescr_t)descrA,
                                                                  (const cuComplex*)csrValA,
                                                                  csrColIndA,
                                                                  csrRowPtrA,
                                                                  nnzA,
                                                                  nnzPerRow,
                                                                  (cuComplex*)csrValC,
                                                                  csrColIndC,
                                                                  csrRowPtrC,
                                                                  {cuCrealf(tol), cuCimagf(tol)}));
}

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
                                             hipDoubleComplex          tol)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsr2csr_compress((cusparseHandle_t)handle,
                                                                  m,
                                                                  n,
                                                                  (const cusparseMatDescr_t)descrA,
                                                                  (const cuDoubleComplex*)csrValA,
                                                                  csrColIndA,
                                                                  csrRowPtrA,
                                                                  nnzA,
                                                                  nnzPerRow,
                                                                  (cuDoubleComplex*)csrValC,
                                                                  csrColIndC,
                                                                  csrRowPtrC,
                                                                  {cuCreal(tol), cuCimag(tol)}));
}

hipsparseStatus_t hipsparseShyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    float*                    csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseShyb2csr((cusparseHandle_t)handle,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cusparseHybMat_t)hybA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA));
}

hipsparseStatus_t hipsparseDhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    double*                   csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDhyb2csr((cusparseHandle_t)handle,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cusparseHybMat_t)hybA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA));
}

hipsparseStatus_t hipsparseChyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipComplex*               csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseChyb2csr((cusparseHandle_t)handle,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cusparseHybMat_t)hybA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA));
}

hipsparseStatus_t hipsparseZhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipDoubleComplex*         csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZhyb2csr((cusparseHandle_t)handle,
                                                         (const cusparseMatDescr_t)descrA,
                                                         (const cusparseHybMat_t)hybA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA));
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
