/* ************************************************************************
* Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

#if(CUDART_VERSION >= 11003)
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
    case CUSPARSE_STATUS_NOT_SUPPORTED:
        return HIPSPARSE_STATUS_NOT_SUPPORTED;
    case CUSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES;
    default:
        throw "Non existent cusparseStatus_t";
    }
}
#elif(CUDART_VERSION >= 10010)
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
case CUSPARSE_STATUS_NOT_SUPPORTED:
    return HIPSPARSE_STATUS_NOT_SUPPORTED;
default:
    throw "Non existent cusparseStatus_t";
}
#endif

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

#if CUDART_VERSION < 11000
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
#endif

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

#if CUDART_VERSION > 10000
cudaDataType hipDataTypeToCudaDataType(hipDataType datatype)
{
    switch(datatype)
    {
    case HIP_R_32F:
        return CUDA_R_32F;
    case HIP_R_64F:
        return CUDA_R_64F;
    case HIP_C_32F:
        return CUDA_C_32F;
    case HIP_C_64F:
        return CUDA_C_64F;
    default:
        throw "Non existent hipDataType";
    }
}

hipDataType CudaDataTypeToHIPDataType(cudaDataType datatype)
{
    switch(datatype)
    {
    case CUDA_R_32F:
        return HIP_R_32F;
    case CUDA_R_64F:
        return HIP_R_64F;
    case CUDA_C_32F:
        return HIP_C_32F;
    case CUDA_C_64F:
        return HIP_C_64F;
    default:
        throw "Non existent cudaDataType";
    }
}
#endif

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

#if CUDART_VERSION < 11000
hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateHybMat((cusparseHybMat_t*)hybA));
}

hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyHybMat((cusparseHybMat_t)hybA));
}
#endif

hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateBsrsv2Info((bsrsv2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsrsv2Info((bsrsv2Info_t)info));
}

hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateBsrsm2Info((bsrsm2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsrsm2Info((bsrsm2Info_t)info));
}

hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateBsrilu02Info((bsrilu02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsrilu02Info((bsrilu02Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrsv2Info((csrsv2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrsv2Info((csrsv2Info_t)info));
}

hipsparseStatus_t hipsparseCreateColorInfo(hipsparseColorInfo_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateColorInfo((cusparseColorInfo_t*)info));
}

hipsparseStatus_t hipsparseDestroyColorInfo(hipsparseColorInfo_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyColorInfo((cusparseColorInfo_t)info));
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

hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateBsric02Info((bsric02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyBsric02Info((bsric02Info_t)info));
}

hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsric02Info((csric02Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsric02Info((csric02Info_t)info));
}

#if CUDART_VERSION < 12000
hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsrgemm2Info((csrgemm2Info_t*)info));
}

hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsrgemm2Info((csrgemm2Info_t)info));
}
#endif

hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreatePruneInfo((pruneInfo_t*)info));
}

hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyPruneInfo((pruneInfo_t)info));
}

hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCsru2csrInfo(info));
}

hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyCsru2csrInfo(info));
}

#if CUDART_VERSION < 12000
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
#endif

#if CUDART_VERSION < 11000
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
#endif

#if CUDART_VERSION < 11000
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
#endif

#if CUDART_VERSION < 12000
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
#endif

#if CUDART_VERSION < 12000
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
#endif

#if CUDART_VERSION < 12000
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
#endif

#if CUDART_VERSION < 12000
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
#endif

#if CUDART_VERSION < 11000
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
#endif

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

#if CUDART_VERSION < 11000
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
#endif

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
                                   float*                    y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsrxmv((cusparseHandle_t)handle,
                                                        hipDirectionToCudaDirection(dir),
                                                        hipOperationToCudaOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        alpha,
                                                        (const cusparseMatDescr_t)descr,
                                                        bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
                                                        blockDim,
                                                        x,
                                                        beta,
                                                        y));
}

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
                                   double*                   y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsrxmv((cusparseHandle_t)handle,
                                                        hipDirectionToCudaDirection(dir),
                                                        hipOperationToCudaOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        alpha,
                                                        (const cusparseMatDescr_t)descr,
                                                        bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
                                                        blockDim,
                                                        x,
                                                        beta,
                                                        y));
}

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
                                   hipComplex*               y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsrxmv((cusparseHandle_t)handle,
                                                        hipDirectionToCudaDirection(dir),
                                                        hipOperationToCudaOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        (const cuComplex*)alpha,
                                                        (const cusparseMatDescr_t)descr,
                                                        (const cuComplex*)bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
                                                        blockDim,
                                                        (const cuComplex*)x,
                                                        (const cuComplex*)beta,
                                                        (cuComplex*)y));
}

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
                                   hipDoubleComplex*         y)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsrxmv((cusparseHandle_t)handle,
                                                        hipDirectionToCudaDirection(dir),
                                                        hipOperationToCudaOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        (const cuDoubleComplex*)alpha,
                                                        (const cusparseMatDescr_t)descr,
                                                        (const cuDoubleComplex*)bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
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

hipsparseStatus_t hipsparseSgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgemvi_bufferSize(
        (cusparseHandle_t)handle, hipOperationToCudaOperation(transA), m, n, nnz, pBufferSize));
}

hipsparseStatus_t hipsparseDgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgemvi_bufferSize(
        (cusparseHandle_t)handle, hipOperationToCudaOperation(transA), m, n, nnz, pBufferSize));
}

hipsparseStatus_t hipsparseCgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgemvi_bufferSize(
        (cusparseHandle_t)handle, hipOperationToCudaOperation(transA), m, n, nnz, pBufferSize));
}

hipsparseStatus_t hipsparseZgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgemvi_bufferSize(
        (cusparseHandle_t)handle, hipOperationToCudaOperation(transA), m, n, nnz, pBufferSize));
}

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
                                  void*                pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgemvi((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       nnz,
                                                       x,
                                                       xInd,
                                                       beta,
                                                       y,
                                                       hipIndexBaseToCudaIndexBase(idxBase),
                                                       pBuffer));
}

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
                                  void*                pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgemvi((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       nnz,
                                                       x,
                                                       xInd,
                                                       beta,
                                                       y,
                                                       hipIndexBaseToCudaIndexBase(idxBase),
                                                       pBuffer));
}

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
                                  void*                pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgemvi((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       (const cuComplex*)alpha,
                                                       (const cuComplex*)A,
                                                       lda,
                                                       nnz,
                                                       (const cuComplex*)x,
                                                       xInd,
                                                       (const cuComplex*)beta,
                                                       (cuComplex*)y,
                                                       hipIndexBaseToCudaIndexBase(idxBase),
                                                       pBuffer));
}

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
                                  void*                   pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgemvi((cusparseHandle_t)handle,
                                                       hipOperationToCudaOperation(transA),
                                                       m,
                                                       n,
                                                       (const cuDoubleComplex*)alpha,
                                                       (const cuDoubleComplex*)A,
                                                       lda,
                                                       nnz,
                                                       (const cuDoubleComplex*)x,
                                                       xInd,
                                                       (const cuDoubleComplex*)beta,
                                                       (cuDoubleComplex*)y,
                                                       hipIndexBaseToCudaIndexBase(idxBase),
                                                       pBuffer));
}

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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsrmm((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       hipOperationToCudaOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsrmm((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       hipOperationToCudaOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsrmm((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       hipOperationToCudaOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       (const cuComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cuComplex*)bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       (const cuComplex*)B,
                                                       ldb,
                                                       (const cuComplex*)beta,
                                                       (cuComplex*)C,
                                                       ldc));
}

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
                                  int                       ldc)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsrmm((cusparseHandle_t)handle,
                                                       hipDirectionToCudaDirection(dirA),
                                                       hipOperationToCudaOperation(transA),
                                                       hipOperationToCudaOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       (const cuDoubleComplex*)alpha,
                                                       (const cusparseMatDescr_t)descrA,
                                                       (const cuDoubleComplex*)bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       (const cuDoubleComplex*)B,
                                                       ldb,
                                                       (const cuDoubleComplex*)beta,
                                                       (cuDoubleComplex*)C,
                                                       ldc));
}

#if CUDART_VERSION < 11000
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
#endif

#if CUDART_VERSION < 11000
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
#endif

hipsparseStatus_t
    hipsparseXbsrsm2_zeroPivot(hipsparseHandle_t handle, bsrsm2Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXbsrsm2_zeroPivot((cusparseHandle_t)handle, (bsrsm2Info_t)info, position));
}

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
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   hipOperationToCudaOperation(transA),
                                   hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}

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
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   hipOperationToCudaOperation(transA),
                                   hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}

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
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   hipOperationToCudaOperation(transA),
                                   hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   (cuComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}

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
                                              int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsm2_bufferSize((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   hipOperationToCudaOperation(transA),
                                   hipOperationToCudaOperation(transX),
                                   mb,
                                   nrhs,
                                   nnzb,
                                   (const cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrsm2Info_t)info,
                                   pBufferSizeInBytes));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dirA),
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dirA),
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dirA),
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                            void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrsm2_analysis((cusparseHandle_t)handle,
                                 hipDirectionToCudaDirection(dirA),
                                 hipOperationToCudaOperation(transA),
                                 hipOperationToCudaOperation(transX),
                                 mb,
                                 nrhs,
                                 nnzb,
                                 (const cusparseMatDescr_t)descrA,
                                 (const cuDoubleComplex*)bsrSortedValA,
                                 bsrSortedRowPtrA,
                                 bsrSortedColIndA,
                                 blockDim,
                                 (bsrsm2Info_t)info,
                                 hipPolicyToCudaPolicy(policy),
                                 pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsrsm2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dirA),
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transX),
                                                              mb,
                                                              nrhs,
                                                              nnzb,
                                                              alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsm2Info_t)info,
                                                              B,
                                                              ldb,
                                                              X,
                                                              ldx,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsrsm2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dirA),
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transX),
                                                              mb,
                                                              nrhs,
                                                              nnzb,
                                                              alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsm2Info_t)info,
                                                              B,
                                                              ldb,
                                                              X,
                                                              ldx,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsrsm2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dirA),
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transX),
                                                              mb,
                                                              nrhs,
                                                              nnzb,
                                                              (const cuComplex*)alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              (const cuComplex*)bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsm2Info_t)info,
                                                              (const cuComplex*)B,
                                                              ldb,
                                                              (cuComplex*)X,
                                                              ldx,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
}

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
                                         void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsrsm2_solve((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dirA),
                                                              hipOperationToCudaOperation(transA),
                                                              hipOperationToCudaOperation(transX),
                                                              mb,
                                                              nrhs,
                                                              nnzb,
                                                              (const cuDoubleComplex*)alpha,
                                                              (const cusparseMatDescr_t)descrA,
                                                              (const cuDoubleComplex*)bsrSortedValA,
                                                              bsrSortedRowPtrA,
                                                              bsrSortedColIndA,
                                                              blockDim,
                                                              (bsrsm2Info_t)info,
                                                              (const cuDoubleComplex*)B,
                                                              ldb,
                                                              (cuDoubleComplex*)X,
                                                              ldx,
                                                              hipPolicyToCudaPolicy(policy),
                                                              pBuffer));
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

#if CUDART_VERSION < 12000
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
#endif

#if CUDART_VERSION < 11000
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
#endif

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

#if CUDART_VERSION < 11000
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
#endif

#if CUDART_VERSION < 12000
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
#endif

hipsparseStatus_t
    hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle, bsrilu02Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXbsrilu02_zeroPivot((cusparseHandle_t)handle, (bsrilu02Info_t)info, position));
}

hipsparseStatus_t hipsparseSbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsrilu02_numericBoost(
        (cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseDbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsrilu02_numericBoost(
        (cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsrilu02_numericBoost(
        (cusparseHandle_t)handle, (bsrilu02Info_t)info, enable_boost, tol, (cuComplex*)boost_val));
}

hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrilu02_numericBoost((cusparseHandle_t)handle,
                                       (bsrilu02Info_t)info,
                                       enable_boost,
                                       tol,
                                       (cuDoubleComplex*)boost_val));
}

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
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

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
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

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
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     (cuComplex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

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
                                                int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrilu02_bufferSize((cusparseHandle_t)handle,
                                     hipDirectionToCudaDirection(dirA),
                                     mb,
                                     nnzb,
                                     (cusparseMatDescr_t)descrA,
                                     (cuDoubleComplex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (bsrilu02Info_t)info,
                                     pBufferSizeInBytes));
}

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
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

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
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

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
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

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
                                              void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsrilu02_analysis((cusparseHandle_t)handle,
                                   hipDirectionToCudaDirection(dirA),
                                   mb,
                                   nnzb,
                                   (cusparseMatDescr_t)descrA,
                                   (cuDoubleComplex*)bsrSortedValA,
                                   bsrSortedRowPtrA,
                                   bsrSortedColIndA,
                                   blockDim,
                                   (bsrilu02Info_t)info,
                                   hipPolicyToCudaPolicy(policy),
                                   pBuffer));
}

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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsrilu02((cusparseHandle_t)handle,
                                                          hipDirectionToCudaDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (cusparseMatDescr_t)descrA,
                                                          bsrSortedValA_valM,
                                                          bsrSortedRowPtrA,
                                                          bsrSortedColIndA,
                                                          blockDim,
                                                          (bsrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsrilu02((cusparseHandle_t)handle,
                                                          hipDirectionToCudaDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (cusparseMatDescr_t)descrA,
                                                          bsrSortedValA_valM,
                                                          bsrSortedRowPtrA,
                                                          bsrSortedColIndA,
                                                          blockDim,
                                                          (bsrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsrilu02((cusparseHandle_t)handle,
                                                          hipDirectionToCudaDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (cusparseMatDescr_t)descrA,
                                                          (cuComplex*)bsrSortedValA_valM,
                                                          bsrSortedRowPtrA,
                                                          bsrSortedColIndA,
                                                          blockDim,
                                                          (bsrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

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
                                     void*                  pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsrilu02((cusparseHandle_t)handle,
                                                          hipDirectionToCudaDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (cusparseMatDescr_t)descrA,
                                                          (cuDoubleComplex*)bsrSortedValA_valM,
                                                          bsrSortedRowPtrA,
                                                          bsrSortedColIndA,
                                                          blockDim,
                                                          (bsrilu02Info_t)info,
                                                          hipPolicyToCudaPolicy(policy),
                                                          pBuffer));
}

hipsparseStatus_t
    hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcsrilu02_zeroPivot((cusparseHandle_t)handle, (csrilu02Info_t)info, position));
}

hipsparseStatus_t hipsparseScsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrilu02_numericBoost(
        (cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseDcsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrilu02_numericBoost(
        (cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrilu02_numericBoost(
        (cusparseHandle_t)handle, (csrilu02Info_t)info, enable_boost, tol, (cuComplex*)boost_val));
}

hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZcsrilu02_numericBoost((cusparseHandle_t)handle,
                                       (csrilu02Info_t)info,
                                       enable_boost,
                                       tol,
                                       (cuDoubleComplex*)boost_val));
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
    hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXbsric02_zeroPivot((cusparseHandle_t)handle, (bsric02Info_t)info, position));
}

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
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}

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
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}

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
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    (cuComplex*)bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}

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
                                               int*                      pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZbsric02_bufferSize((cusparseHandle_t)handle,
                                    hipDirectionToCudaDirection(dirA),
                                    mb,
                                    nnzb,
                                    (cusparseMatDescr_t)descrA,
                                    (cuDoubleComplex*)bsrValA,
                                    bsrRowPtrA,
                                    bsrColIndA,
                                    blockDim,
                                    (bsric02Info_t)info,
                                    pBufferSizeInBytes));
}

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
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsric02_analysis((cusparseHandle_t)handle,
                                                                  hipDirectionToCudaDirection(dirA),
                                                                  mb,
                                                                  nnzb,
                                                                  (cusparseMatDescr_t)descrA,
                                                                  bsrValA,
                                                                  bsrRowPtrA,
                                                                  bsrColIndA,
                                                                  blockDim,
                                                                  (bsric02Info_t)info,
                                                                  hipPolicyToCudaPolicy(policy),
                                                                  pBuffer));
}

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
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsric02_analysis((cusparseHandle_t)handle,
                                                                  hipDirectionToCudaDirection(dirA),
                                                                  mb,
                                                                  nnzb,
                                                                  (cusparseMatDescr_t)descrA,
                                                                  bsrValA,
                                                                  bsrRowPtrA,
                                                                  bsrColIndA,
                                                                  blockDim,
                                                                  (bsric02Info_t)info,
                                                                  hipPolicyToCudaPolicy(policy),
                                                                  pBuffer));
}

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
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsric02_analysis((cusparseHandle_t)handle,
                                                                  hipDirectionToCudaDirection(dirA),
                                                                  mb,
                                                                  nnzb,
                                                                  (cusparseMatDescr_t)descrA,
                                                                  (const cuComplex*)bsrValA,
                                                                  bsrRowPtrA,
                                                                  bsrColIndA,
                                                                  blockDim,
                                                                  (bsric02Info_t)info,
                                                                  hipPolicyToCudaPolicy(policy),
                                                                  pBuffer));
}

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
                                             void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsric02_analysis((cusparseHandle_t)handle,
                                                                  hipDirectionToCudaDirection(dirA),
                                                                  mb,
                                                                  nnzb,
                                                                  (cusparseMatDescr_t)descrA,
                                                                  (const cuDoubleComplex*)bsrValA,
                                                                  bsrRowPtrA,
                                                                  bsrColIndA,
                                                                  blockDim,
                                                                  (bsric02Info_t)info,
                                                                  hipPolicyToCudaPolicy(policy),
                                                                  pBuffer));
}

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
                                    void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSbsric02((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (cusparseMatDescr_t)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (bsric02Info_t)info,
                                                         hipPolicyToCudaPolicy(policy),
                                                         pBuffer));
}

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
                                    void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDbsric02((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (cusparseMatDescr_t)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (bsric02Info_t)info,
                                                         hipPolicyToCudaPolicy(policy),
                                                         pBuffer));
}

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
                                    void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCbsric02((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (cusparseMatDescr_t)descrA,
                                                         (cuComplex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (bsric02Info_t)info,
                                                         hipPolicyToCudaPolicy(policy),
                                                         pBuffer));
}

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
                                    void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZbsric02((cusparseHandle_t)handle,
                                                         hipDirectionToCudaDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (cusparseMatDescr_t)descrA,
                                                         (cuDoubleComplex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (bsric02Info_t)info,
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
                                                      size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              bufferSize));
}

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
                                                      size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              bufferSize));
}

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
                                                         size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              bufferSize));
}

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
                                                         size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csr_bufferSizeExt((cusparseHandle_t)handle,
                                              m,
                                              n,
                                              A,
                                              lda,
                                              threshold,
                                              (const cusparseMatDescr_t)descr,
                                              csrVal,
                                              csrRowPtr,
                                              csrColInd,
                                              bufferSize));
}

hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const float*              A,
                                              int                       lda,
                                              const float*              threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpruneDense2csrNnz((cusparseHandle_t)handle,
                                                                   m,
                                                                   n,
                                                                   A,
                                                                   lda,
                                                                   threshold,
                                                                   (const cusparseMatDescr_t)descr,
                                                                   csrRowPtr,
                                                                   nnzTotalDevHostPtr,
                                                                   buffer));
}

hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const double*             A,
                                              int                       lda,
                                              const double*             threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDpruneDense2csrNnz((cusparseHandle_t)handle,
                                                                   m,
                                                                   n,
                                                                   A,
                                                                   lda,
                                                                   threshold,
                                                                   (const cusparseMatDescr_t)descr,
                                                                   csrRowPtr,
                                                                   nnzTotalDevHostPtr,
                                                                   buffer));
}

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
                                           void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpruneDense2csr((cusparseHandle_t)handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                threshold,
                                                                (const cusparseMatDescr_t)descr,
                                                                csrVal,
                                                                csrRowPtr,
                                                                csrColInd,
                                                                buffer));
}

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
                                           void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDpruneDense2csr((cusparseHandle_t)handle,
                                                                m,
                                                                n,
                                                                A,
                                                                lda,
                                                                threshold,
                                                                (const cusparseMatDescr_t)descr,
                                                                csrVal,
                                                                csrRowPtr,
                                                                csrColInd,
                                                                buffer));
}

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
                                                                  size_t*     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          bufferSize));
}

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
                                                                  size_t*     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          bufferSize));
}

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
                                                       size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          bufferSize));
}

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
                                                       size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          A,
                                                          lda,
                                                          percentage,
                                                          (const cusparseMatDescr_t)descr,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          (pruneInfo_t)info,
                                                          bufferSize));
}

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
                                                          void*       buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrNnzByPercentage((cusparseHandle_t)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               percentage,
                                               (const cusparseMatDescr_t)descr,
                                               csrRowPtr,
                                               nnzTotalDevHostPtr,
                                               (pruneInfo_t)info,
                                               buffer));
}

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
                                                          void*       buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrNnzByPercentage((cusparseHandle_t)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               percentage,
                                               (const cusparseMatDescr_t)descr,
                                               csrRowPtr,
                                               nnzTotalDevHostPtr,
                                               (pruneInfo_t)info,
                                               buffer));
}

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
                                                       void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneDense2csrByPercentage((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            A,
                                            lda,
                                            percentage,
                                            (const cusparseMatDescr_t)descr,
                                            csrVal,
                                            csrRowPtr,
                                            csrColInd,
                                            (pruneInfo_t)info,
                                            buffer));
}

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
                                                       void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneDense2csrByPercentage((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            A,
                                            lda,
                                            percentage,
                                            (const cusparseMatDescr_t)descr,
                                            csrVal,
                                            csrRowPtr,
                                            csrColInd,
                                            (pruneInfo_t)info,
                                            buffer));
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

hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const float*      bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size)
{
    int               cu_buffer_size;
    hipsparseStatus_t status
        = hipCUSPARSEStatusToHIPStatus(cusparseSgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                                                       mb,
                                                                       nb,
                                                                       nnzb,
                                                                       bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       row_block_dim,
                                                                       col_block_dim,
                                                                       &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const double*     bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size)
{
    int               cu_buffer_size;
    hipsparseStatus_t status
        = hipCUSPARSEStatusToHIPStatus(cusparseDgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                                                       mb,
                                                                       nb,
                                                                       nnzb,
                                                                       bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       row_block_dim,
                                                                       col_block_dim,
                                                                       &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const hipComplex* bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size)
{
    int               cu_buffer_size;
    hipsparseStatus_t status
        = hipCUSPARSEStatusToHIPStatus(cusparseCgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                                                       mb,
                                                                       nb,
                                                                       nnzb,
                                                                       (const cuComplex*)bsr_val,
                                                                       bsr_row_ptr,
                                                                       bsr_col_ind,
                                                                       row_block_dim,
                                                                       col_block_dim,
                                                                       &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                   int                     mb,
                                                   int                     nb,
                                                   int                     nnzb,
                                                   const hipDoubleComplex* bsr_val,
                                                   const int*              bsr_row_ptr,
                                                   const int*              bsr_col_ind,
                                                   int                     row_block_dim,
                                                   int                     col_block_dim,
                                                   size_t*                 p_buffer_size)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipCUSPARSEStatusToHIPStatus(
        cusparseZgebsr2gebsc_bufferSize((cusparseHandle_t)handle,
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cuDoubleComplex*)bsr_val,
                                        bsr_row_ptr,
                                        bsr_col_ind,
                                        row_block_dim,
                                        col_block_dim,
                                        &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

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
                                        void*                temp_buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgebsr2gebsc((cusparseHandle_t)handle,
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             bsc_val,
                                                             bsc_row_ind,
                                                             bsc_col_ptr,
                                                             hipActionToCudaAction(copy_values),
                                                             hipIndexBaseToCudaIndexBase(idx_base),
                                                             temp_buffer));
}

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
                                        void*                temp_buffer)
{

    return hipCUSPARSEStatusToHIPStatus(cusparseDgebsr2gebsc((cusparseHandle_t)handle,
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             bsc_val,
                                                             bsc_row_ind,
                                                             bsc_col_ptr,
                                                             hipActionToCudaAction(copy_values),
                                                             hipIndexBaseToCudaIndexBase(idx_base),
                                                             temp_buffer));
}

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
                                        void*                temp_buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgebsr2gebsc((cusparseHandle_t)handle,
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             (const cuComplex*)bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             (cuComplex*)bsc_val,
                                                             bsc_row_ind,
                                                             bsc_col_ptr,
                                                             hipActionToCudaAction(copy_values),
                                                             hipIndexBaseToCudaIndexBase(idx_base),
                                                             temp_buffer));
}

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
                                        void*                   temp_buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgebsr2gebsc((cusparseHandle_t)handle,
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             (const cuDoubleComplex*)bsr_val,
                                                             bsr_row_ptr,
                                                             bsr_col_ind,
                                                             row_block_dim,
                                                             col_block_dim,
                                                             (cuDoubleComplex*)bsc_val,
                                                             bsc_row_ind,
                                                             bsc_col_ptr,
                                                             hipActionToCudaAction(copy_values),
                                                             hipIndexBaseToCudaIndexBase(idx_base),
                                                             temp_buffer));
}

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
                                                 size_t*                   p_buffer_size)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipCUSPARSEStatusToHIPStatus(
        cusparseScsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                      hipDirectionToCudaDirection(dir),
                                      m,
                                      n,
                                      (const cusparseMatDescr_t)csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      row_block_dim,
                                      col_block_dim,
                                      &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

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
                                                 size_t*                   p_buffer_size)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipCUSPARSEStatusToHIPStatus(
        cusparseDcsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                      hipDirectionToCudaDirection(dir),
                                      m,
                                      n,
                                      (const cusparseMatDescr_t)csr_descr,
                                      csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      row_block_dim,
                                      col_block_dim,
                                      &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

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
                                                 size_t*                   p_buffer_size)
{

    int               cu_buffer_size;
    hipsparseStatus_t status = hipCUSPARSEStatusToHIPStatus(
        cusparseCcsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                      hipDirectionToCudaDirection(dir),
                                      m,
                                      n,
                                      (const cusparseMatDescr_t)csr_descr,
                                      (const cuComplex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      row_block_dim,
                                      col_block_dim,
                                      &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

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
                                                 size_t*                   p_buffer_size)
{
    int               cu_buffer_size;
    hipsparseStatus_t status = hipCUSPARSEStatusToHIPStatus(
        cusparseZcsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                      hipDirectionToCudaDirection(dir),
                                      m,
                                      n,
                                      (const cusparseMatDescr_t)csr_descr,
                                      (const cuDoubleComplex*)csr_val,
                                      csr_row_ptr,
                                      csr_col_ind,
                                      row_block_dim,
                                      col_block_dim,
                                      &cu_buffer_size));
    p_buffer_size[0] = cu_buffer_size;
    return status;
}

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
                                         void*                     p_buffer)
{

    return hipCUSPARSEStatusToHIPStatus(cusparseXcsr2gebsrNnz((cusparseHandle_t)handle,
                                                              hipDirectionToCudaDirection(dir),
                                                              m,
                                                              n,
                                                              (const cusparseMatDescr_t)csr_descr,
                                                              csr_row_ptr,
                                                              csr_col_ind,
                                                              (const cusparseMatDescr_t)bsr_descr,
                                                              bsr_row_ptr,
                                                              row_block_dim,
                                                              col_block_dim,
                                                              bsr_nnz_devhost,
                                                              p_buffer));
}

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
                                      void*                     p_buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsr2gebsr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dir),
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           (const cusparseMatDescr_t)bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           row_block_dim,
                                                           col_block_dim,
                                                           p_buffer));
}

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
                                      void*                     p_buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsr2gebsr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dir),
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)csr_descr,
                                                           csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           (const cusparseMatDescr_t)bsr_descr,
                                                           bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           row_block_dim,
                                                           col_block_dim,
                                                           p_buffer));
}

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
                                      void*                     p_buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsr2gebsr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dir),
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)csr_descr,
                                                           (const cuComplex*)csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           (const cusparseMatDescr_t)bsr_descr,
                                                           (cuComplex*)bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           row_block_dim,
                                                           col_block_dim,
                                                           p_buffer));
}

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
                                      void*                     p_buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsr2gebsr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dir),
                                                           m,
                                                           n,
                                                           (const cusparseMatDescr_t)csr_descr,
                                                           (const cuDoubleComplex*)csr_val,
                                                           csr_row_ptr,
                                                           csr_col_ind,
                                                           (const cusparseMatDescr_t)bsr_descr,
                                                           (cuDoubleComplex*)bsr_val,
                                                           bsr_row_ptr,
                                                           bsr_col_ind,
                                                           row_block_dim,
                                                           col_block_dim,
                                                           p_buffer));
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

#if CUDART_VERSION < 11000
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
#endif

#if CUDART_VERSION < 11000
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
#endif

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
                                      int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgebsr2csr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dirA),
                                                           mb,
                                                           nb,
                                                           (const cusparseMatDescr_t)descrA,
                                                           bsrValA,
                                                           bsrRowPtrA,
                                                           bsrColIndA,
                                                           rowBlockDim,
                                                           colBlockDim,
                                                           (const cusparseMatDescr_t)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC));
}

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
                                      int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgebsr2csr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dirA),
                                                           mb,
                                                           nb,
                                                           (const cusparseMatDescr_t)descrA,
                                                           bsrValA,
                                                           bsrRowPtrA,
                                                           bsrColIndA,
                                                           rowBlockDim,
                                                           colBlockDim,
                                                           (const cusparseMatDescr_t)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC));
}

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
                                      int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgebsr2csr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dirA),
                                                           mb,
                                                           nb,
                                                           (const cusparseMatDescr_t)descrA,
                                                           (const cuComplex*)bsrValA,
                                                           bsrRowPtrA,
                                                           bsrColIndA,
                                                           rowBlockDim,
                                                           colBlockDim,
                                                           (const cusparseMatDescr_t)descrC,
                                                           (cuComplex*)csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC));
}

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
                                      int*                      csrColIndC)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgebsr2csr((cusparseHandle_t)handle,
                                                           hipDirectionToCudaDirection(dirA),
                                                           mb,
                                                           nb,
                                                           (const cusparseMatDescr_t)descrA,
                                                           (const cuDoubleComplex*)bsrValA,
                                                           bsrRowPtrA,
                                                           bsrColIndA,
                                                           rowBlockDim,
                                                           colBlockDim,
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
                                                    size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneCsr2csr_bufferSizeExt((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            nnzA,
                                            (const cusparseMatDescr_t)descrA,
                                            csrValA,
                                            csrRowPtrA,
                                            csrColIndA,
                                            threshold,
                                            (const cusparseMatDescr_t)descrC,
                                            csrValC,
                                            csrRowPtrC,
                                            csrColIndC,
                                            bufferSize));
}

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
                                                    size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneCsr2csr_bufferSizeExt((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            nnzA,
                                            (const cusparseMatDescr_t)descrA,
                                            csrValA,
                                            csrRowPtrA,
                                            csrColIndA,
                                            threshold,
                                            (const cusparseMatDescr_t)descrC,
                                            csrValC,
                                            csrRowPtrC,
                                            csrColIndC,
                                            bufferSize));
}

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
                                                       size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneCsr2csr_bufferSizeExt((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            nnzA,
                                            (const cusparseMatDescr_t)descrA,
                                            csrValA,
                                            csrRowPtrA,
                                            csrColIndA,
                                            threshold,
                                            (const cusparseMatDescr_t)descrC,
                                            csrValC,
                                            csrRowPtrC,
                                            csrColIndC,
                                            bufferSize));
}

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
                                                       size_t*                   bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneCsr2csr_bufferSizeExt((cusparseHandle_t)handle,
                                            m,
                                            n,
                                            nnzA,
                                            (const cusparseMatDescr_t)descrA,
                                            csrValA,
                                            csrRowPtrA,
                                            csrColIndA,
                                            threshold,
                                            (const cusparseMatDescr_t)descrC,
                                            csrValC,
                                            csrRowPtrC,
                                            csrColIndC,
                                            bufferSize));
}

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
                                            void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpruneCsr2csrNnz((cusparseHandle_t)handle,
                                                                 m,
                                                                 n,
                                                                 nnzA,
                                                                 (const cusparseMatDescr_t)descrA,
                                                                 csrValA,
                                                                 csrRowPtrA,
                                                                 csrColIndA,
                                                                 threshold,
                                                                 (const cusparseMatDescr_t)descrC,
                                                                 csrRowPtrC,
                                                                 nnzTotalDevHostPtr,
                                                                 buffer));
}

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
                                            void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDpruneCsr2csrNnz((cusparseHandle_t)handle,
                                                                 m,
                                                                 n,
                                                                 nnzA,
                                                                 (const cusparseMatDescr_t)descrA,
                                                                 csrValA,
                                                                 csrRowPtrA,
                                                                 csrColIndA,
                                                                 threshold,
                                                                 (const cusparseMatDescr_t)descrC,
                                                                 csrRowPtrC,
                                                                 nnzTotalDevHostPtr,
                                                                 buffer));
}

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
                                         void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpruneCsr2csr((cusparseHandle_t)handle,
                                                              m,
                                                              n,
                                                              nnzA,
                                                              (const cusparseMatDescr_t)descrA,
                                                              csrValA,
                                                              csrRowPtrA,
                                                              csrColIndA,
                                                              threshold,
                                                              (const cusparseMatDescr_t)descrC,
                                                              csrValC,
                                                              csrRowPtrC,
                                                              csrColIndC,
                                                              buffer));
}

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
                                         void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDpruneCsr2csr((cusparseHandle_t)handle,
                                                              m,
                                                              n,
                                                              nnzA,
                                                              (const cusparseMatDescr_t)descrA,
                                                              csrValA,
                                                              csrRowPtrA,
                                                              csrColIndA,
                                                              threshold,
                                                              (const cusparseMatDescr_t)descrC,
                                                              csrValC,
                                                              csrRowPtrC,
                                                              csrColIndC,
                                                              buffer));
}

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
                                                                size_t*     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneCsr2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                        m,
                                                        n,
                                                        nnzA,
                                                        (const cusparseMatDescr_t)descrA,
                                                        csrValA,
                                                        csrRowPtrA,
                                                        csrColIndA,
                                                        percentage,
                                                        (const cusparseMatDescr_t)descrC,
                                                        csrValC,
                                                        csrRowPtrC,
                                                        csrColIndC,
                                                        (pruneInfo_t)info,
                                                        bufferSize));
}

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
                                                                size_t*     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneCsr2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                        m,
                                                        n,
                                                        nnzA,
                                                        (const cusparseMatDescr_t)descrA,
                                                        csrValA,
                                                        csrRowPtrA,
                                                        csrColIndA,
                                                        percentage,
                                                        (const cusparseMatDescr_t)descrC,
                                                        csrValC,
                                                        csrRowPtrC,
                                                        csrColIndC,
                                                        (pruneInfo_t)info,
                                                        bufferSize));
}

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
                                                                   size_t*      bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneCsr2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                        m,
                                                        n,
                                                        nnzA,
                                                        (const cusparseMatDescr_t)descrA,
                                                        csrValA,
                                                        csrRowPtrA,
                                                        csrColIndA,
                                                        percentage,
                                                        (const cusparseMatDescr_t)descrC,
                                                        csrValC,
                                                        csrRowPtrC,
                                                        csrColIndC,
                                                        (pruneInfo_t)info,
                                                        bufferSize));
}

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
                                                                   size_t*       bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneCsr2csrByPercentage_bufferSizeExt((cusparseHandle_t)handle,
                                                        m,
                                                        n,
                                                        nnzA,
                                                        (const cusparseMatDescr_t)descrA,
                                                        csrValA,
                                                        csrRowPtrA,
                                                        csrColIndA,
                                                        percentage,
                                                        (const cusparseMatDescr_t)descrC,
                                                        csrValC,
                                                        csrRowPtrC,
                                                        csrColIndC,
                                                        (pruneInfo_t)info,
                                                        bufferSize));
}

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
                                                        void*       buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneCsr2csrNnzByPercentage((cusparseHandle_t)handle,
                                             m,
                                             n,
                                             nnzA,
                                             (const cusparseMatDescr_t)descrA,
                                             csrValA,
                                             csrRowPtrA,
                                             csrColIndA,
                                             percentage,
                                             (const cusparseMatDescr_t)descrC,
                                             csrRowPtrC,
                                             nnzTotalDevHostPtr,
                                             (pruneInfo_t)info,
                                             buffer));
}

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
                                                        void*       buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneCsr2csrNnzByPercentage((cusparseHandle_t)handle,
                                             m,
                                             n,
                                             nnzA,
                                             (const cusparseMatDescr_t)descrA,
                                             csrValA,
                                             csrRowPtrA,
                                             csrColIndA,
                                             percentage,
                                             (const cusparseMatDescr_t)descrC,
                                             csrRowPtrC,
                                             nnzTotalDevHostPtr,
                                             (pruneInfo_t)info,
                                             buffer));
}

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
                                                     void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpruneCsr2csrByPercentage((cusparseHandle_t)handle,
                                          m,
                                          n,
                                          nnzA,
                                          (const cusparseMatDescr_t)descrA,
                                          csrValA,
                                          csrRowPtrA,
                                          csrColIndA,
                                          percentage,
                                          (const cusparseMatDescr_t)descrC,
                                          csrValC,
                                          csrRowPtrC,
                                          csrColIndC,
                                          (pruneInfo_t)info,
                                          buffer));
}

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
                                                     void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDpruneCsr2csrByPercentage((cusparseHandle_t)handle,
                                          m,
                                          n,
                                          nnzA,
                                          (const cusparseMatDescr_t)descrA,
                                          csrValA,
                                          csrRowPtrA,
                                          csrColIndA,
                                          percentage,
                                          (const cusparseMatDescr_t)descrC,
                                          csrValC,
                                          csrRowPtrC,
                                          csrColIndC,
                                          (pruneInfo_t)info,
                                          buffer));
}

#if CUDART_VERSION < 11000
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
#endif

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
                                                   int*                      bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        bufferSize));
}

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
                                                   int*                      bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        bufferSize));
}

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
                                                   int*                      bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        (const cuComplex*)bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        bufferSize));
}

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
                                                   int*                      bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZgebsr2gebsr_bufferSize((cusparseHandle_t)handle,
                                        hipDirectionToCudaDirection(dirA),
                                        mb,
                                        nb,
                                        nnzb,
                                        (const cusparseMatDescr_t)descrA,
                                        (const cuDoubleComplex*)bsrValA,
                                        bsrRowPtrA,
                                        bsrColIndA,
                                        rowBlockDimA,
                                        colBlockDimA,
                                        rowBlockDimC,
                                        colBlockDimC,
                                        bufferSize));
}

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
                                           void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseXgebsr2gebsrNnz((cusparseHandle_t)handle,
                                                                hipDirectionToCudaDirection(dirA),
                                                                mb,
                                                                nb,
                                                                nnzb,
                                                                (const cusparseMatDescr_t)descrA,
                                                                bsrRowPtrA,
                                                                bsrColIndA,
                                                                rowBlockDimA,
                                                                colBlockDimA,
                                                                (cusparseMatDescr_t)descrC,
                                                                bsrRowPtrC,
                                                                rowBlockDimC,
                                                                colBlockDimC,
                                                                nnzTotalDevHostPtr,
                                                                buffer));
}

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
                                        void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgebsr2gebsr((cusparseHandle_t)handle,
                                                             hipDirectionToCudaDirection(dirA),
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             (const cusparseMatDescr_t)descrA,
                                                             bsrValA,
                                                             bsrRowPtrA,
                                                             bsrColIndA,
                                                             rowBlockDimA,
                                                             colBlockDimA,
                                                             (const cusparseMatDescr_t)descrC,
                                                             bsrValC,
                                                             bsrRowPtrC,
                                                             bsrColIndC,
                                                             rowBlockDimC,
                                                             colBlockDimC,
                                                             buffer));
}

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
                                        void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgebsr2gebsr((cusparseHandle_t)handle,
                                                             hipDirectionToCudaDirection(dirA),
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             (const cusparseMatDescr_t)descrA,
                                                             bsrValA,
                                                             bsrRowPtrA,
                                                             bsrColIndA,
                                                             rowBlockDimA,
                                                             colBlockDimA,
                                                             (const cusparseMatDescr_t)descrC,
                                                             bsrValC,
                                                             bsrRowPtrC,
                                                             bsrColIndC,
                                                             rowBlockDimC,
                                                             colBlockDimC,
                                                             buffer));
}

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
                                        void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgebsr2gebsr((cusparseHandle_t)handle,
                                                             hipDirectionToCudaDirection(dirA),
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             (const cusparseMatDescr_t)descrA,
                                                             (const cuComplex*)bsrValA,
                                                             bsrRowPtrA,
                                                             bsrColIndA,
                                                             rowBlockDimA,
                                                             colBlockDimA,
                                                             (const cusparseMatDescr_t)descrC,
                                                             (cuComplex*)bsrValC,
                                                             bsrRowPtrC,
                                                             bsrColIndC,
                                                             rowBlockDimC,
                                                             colBlockDimC,
                                                             buffer));
}

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
                                        void*                     buffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgebsr2gebsr((cusparseHandle_t)handle,
                                                             hipDirectionToCudaDirection(dirA),
                                                             mb,
                                                             nb,
                                                             nnzb,
                                                             (const cusparseMatDescr_t)descrA,
                                                             (const cuDoubleComplex*)bsrValA,
                                                             bsrRowPtrA,
                                                             bsrColIndA,
                                                             rowBlockDimA,
                                                             colBlockDimA,
                                                             (const cusparseMatDescr_t)descrC,
                                                             (cuDoubleComplex*)bsrValC,
                                                             bsrRowPtrC,
                                                             bsrColIndC,
                                                             rowBlockDimC,
                                                             colBlockDimC,
                                                             buffer));
}

hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   float*            csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                                                        m,
                                                                        n,
                                                                        nnz,
                                                                        csrVal,
                                                                        csrRowPtr,
                                                                        csrColInd,
                                                                        info,
                                                                        pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   double*           csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                                                        m,
                                                                        n,
                                                                        nnz,
                                                                        csrVal,
                                                                        csrRowPtr,
                                                                        csrColInd,
                                                                        info,
                                                                        pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipComplex*       csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                                                        m,
                                                                        n,
                                                                        nnz,
                                                                        (cuComplex*)csrVal,
                                                                        csrRowPtr,
                                                                        csrColInd,
                                                                        info,
                                                                        pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipDoubleComplex* csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsru2csr_bufferSizeExt((cusparseHandle_t)handle,
                                                                        m,
                                                                        n,
                                                                        nnz,
                                                                        (cuDoubleComplex*)csrVal,
                                                                        csrRowPtr,
                                                                        csrColInd,
                                                                        info,
                                                                        pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseScsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsru2csr((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

hipsparseStatus_t hipsparseDcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsru2csr((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

hipsparseStatus_t hipsparseCcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsru2csr((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          (cuComplex*)csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

hipsparseStatus_t hipsparseZcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsru2csr((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          (cuDoubleComplex*)csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

hipsparseStatus_t hipsparseScsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsr2csru((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

hipsparseStatus_t hipsparseDcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsr2csru((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

hipsparseStatus_t hipsparseCcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsr2csru((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          (cuComplex*)csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

hipsparseStatus_t hipsparseZcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsr2csru((cusparseHandle_t)handle,
                                                          m,
                                                          n,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          (cuDoubleComplex*)csrVal,
                                                          csrRowPtr,
                                                          csrColInd,
                                                          info,
                                                          pBuffer));
}

/* Generic API */
#if(CUDART_VERSION >= 10010)
cusparseFormat_t hipFormatToCudaFormat(hipsparseFormat_t format)
{
    switch(format)
    {
    case HIPSPARSE_FORMAT_COO:
        return CUSPARSE_FORMAT_COO;
    case HIPSPARSE_FORMAT_COO_AOS:
        return CUSPARSE_FORMAT_COO_AOS;
    case HIPSPARSE_FORMAT_CSR:
        return CUSPARSE_FORMAT_CSR;
    case HIPSPARSE_FORMAT_BLOCKED_ELL:
        return CUSPARSE_FORMAT_BLOCKED_ELL;
    default:
        throw "Non existent hipsparseFormat_t";
    }
}

hipsparseFormat_t CudaFormatToHIPFormat(cusparseFormat_t format)
{
    switch(format)
    {
    case CUSPARSE_FORMAT_COO:
        return HIPSPARSE_FORMAT_COO;
    case CUSPARSE_FORMAT_COO_AOS:
        return HIPSPARSE_FORMAT_COO_AOS;
    case CUSPARSE_FORMAT_CSR:
        return HIPSPARSE_FORMAT_CSR;
    case CUSPARSE_FORMAT_BLOCKED_ELL:
        return HIPSPARSE_FORMAT_BLOCKED_ELL;
    default:
        throw "Non existent cusparseFormat_t";
    }
}
#endif

#if(CUDART_VERSION >= 11000)
cusparseOrder_t hipOrderToCudaOrder(hipsparseOrder_t op)
{
    switch(op)
    {
    case HIPSPARSE_ORDER_ROW:
        return CUSPARSE_ORDER_ROW;
    case HIPSPARSE_ORDER_COLUMN:
        return CUSPARSE_ORDER_COL;
    default:
        throw "Non existent hipsparseOrder_t";
    }
}

hipsparseOrder_t CudaOrderToHIPOrder(cusparseOrder_t op)
{
    switch(op)
    {
    case CUSPARSE_ORDER_ROW:
        return HIPSPARSE_ORDER_ROW;
    case CUSPARSE_ORDER_COL:
        return HIPSPARSE_ORDER_COLUMN;
    default:
        throw "Non existent cusparseOrder_t";
    }
}
#elif(CUDART_VERSION >= 10010)
cusparseOrder_t hipOrderToCudaOrder(hipsparseOrder_t op)
{
    switch(op)
    {
    case HIPSPARSE_ORDER_COLUMN:
        return CUSPARSE_ORDER_COL;
    default:
        throw "Non existent hipsparseOrder_t";
    }
}

hipsparseOrder_t CudaOrderToHIPOrder(cusparseOrder_t op)
{
    switch(op)
    {
    case CUSPARSE_ORDER_COL:
        return HIPSPARSE_ORDER_COLUMN;
    default:
        throw "Non existent cusparseOrder_t";
    }
}
#endif

#if(CUDART_VERSION >= 10010)
cusparseIndexType_t hipIndexTypeToCudaIndexType(hipsparseIndexType_t type)
{
    switch(type)
    {
    case HIPSPARSE_INDEX_16U:
        return CUSPARSE_INDEX_16U;
    case HIPSPARSE_INDEX_32I:
        return CUSPARSE_INDEX_32I;
    case HIPSPARSE_INDEX_64I:
        return CUSPARSE_INDEX_64I;
    default:
        throw "Non existant hipsparseIndexType_t";
    }
}

hipsparseIndexType_t CudaIndexTypeToHIPIndexType(cusparseIndexType_t type)
{
    switch(type)
    {
    case CUSPARSE_INDEX_16U:
        return HIPSPARSE_INDEX_16U;
    case CUSPARSE_INDEX_32I:
        return HIPSPARSE_INDEX_32I;
    case CUSPARSE_INDEX_64I:
        return HIPSPARSE_INDEX_64I;
    default:
        throw "Non existant cusparseIndexType_t";
    }
}
#endif

#if(CUDART_VERSION >= 11021)
cusparseSpMVAlg_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_MV_ALG_DEFAULT:
        return CUSPARSE_MV_ALG_DEFAULT;
    case HIPSPARSE_SPMV_ALG_DEFAULT:
        return CUSPARSE_SPMV_ALG_DEFAULT;
    case HIPSPARSE_COOMV_ALG:
        return CUSPARSE_COOMV_ALG;
    case HIPSPARSE_SPMV_COO_ALG1:
        return CUSPARSE_SPMV_COO_ALG1;
    case HIPSPARSE_SPMV_COO_ALG2:
        return CUSPARSE_SPMV_COO_ALG2;
    case HIPSPARSE_CSRMV_ALG1:
        return CUSPARSE_CSRMV_ALG1;
    case HIPSPARSE_SPMV_CSR_ALG1:
        return CUSPARSE_SPMV_CSR_ALG1;
    case HIPSPARSE_CSRMV_ALG2:
        return CUSPARSE_CSRMV_ALG2;
    case HIPSPARSE_SPMV_CSR_ALG2:
        return CUSPARSE_SPMV_CSR_ALG2;
    default:
        throw "Non existant hipsparseSpMVAlg_t";
    }
}
#elif(CUDART_VERSION >= 10010)
cusparseSpMVAlg_t hipSpMVAlgToCudaSpMVAlg(hipsparseSpMVAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_MV_ALG_DEFAULT:
        return CUSPARSE_MV_ALG_DEFAULT;
    case HIPSPARSE_COOMV_ALG:
        return CUSPARSE_COOMV_ALG;
    case HIPSPARSE_CSRMV_ALG1:
        return CUSPARSE_CSRMV_ALG1;
    case HIPSPARSE_CSRMV_ALG2:
        return CUSPARSE_CSRMV_ALG2;
    default:
        throw "Non existant hipsparseSpMVAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 11021)
cusparseSpMMAlg_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_MM_ALG_DEFAULT:
        return CUSPARSE_MM_ALG_DEFAULT;
    case HIPSPARSE_COOMM_ALG1:
        return CUSPARSE_COOMM_ALG1;
    case HIPSPARSE_COOMM_ALG2:
        return CUSPARSE_COOMM_ALG2;
    case HIPSPARSE_COOMM_ALG3:
        return CUSPARSE_COOMM_ALG3;
    case HIPSPARSE_CSRMM_ALG1:
        return CUSPARSE_CSRMM_ALG1;
    case HIPSPARSE_SPMM_ALG_DEFAULT:
        return CUSPARSE_SPMM_ALG_DEFAULT;
    case HIPSPARSE_SPMM_COO_ALG1:
        return CUSPARSE_SPMM_COO_ALG1;
    case HIPSPARSE_SPMM_COO_ALG2:
        return CUSPARSE_SPMM_COO_ALG2;
    case HIPSPARSE_SPMM_COO_ALG3:
        return CUSPARSE_SPMM_COO_ALG3;
    case HIPSPARSE_SPMM_COO_ALG4:
        return CUSPARSE_SPMM_COO_ALG4;
    case HIPSPARSE_SPMM_CSR_ALG1:
        return CUSPARSE_SPMM_CSR_ALG1;
    case HIPSPARSE_SPMM_CSR_ALG2:
        return CUSPARSE_SPMM_CSR_ALG2;
    case HIPSPARSE_SPMM_CSR_ALG3:
        return CUSPARSE_SPMM_CSR_ALG3;
    case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
        return CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
    default:
        throw "Non existant hipsparseSpMMAlg_t";
    }
}
#elif(CUDART_VERSION >= 11003)
cusparseSpMMAlg_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_MM_ALG_DEFAULT:
        return CUSPARSE_MM_ALG_DEFAULT;
    case HIPSPARSE_COOMM_ALG1:
        return CUSPARSE_COOMM_ALG1;
    case HIPSPARSE_COOMM_ALG2:
        return CUSPARSE_COOMM_ALG2;
    case HIPSPARSE_COOMM_ALG3:
        return CUSPARSE_COOMM_ALG3;
    case HIPSPARSE_CSRMM_ALG1:
        return CUSPARSE_CSRMM_ALG1;
    case HIPSPARSE_SPMM_ALG_DEFAULT:
        return CUSPARSE_SPMM_ALG_DEFAULT;
    case HIPSPARSE_SPMM_COO_ALG1:
        return CUSPARSE_SPMM_COO_ALG1;
    case HIPSPARSE_SPMM_COO_ALG2:
        return CUSPARSE_SPMM_COO_ALG2;
    case HIPSPARSE_SPMM_COO_ALG3:
        return CUSPARSE_SPMM_COO_ALG3;
    case HIPSPARSE_SPMM_COO_ALG4:
        return CUSPARSE_SPMM_COO_ALG4;
    case HIPSPARSE_SPMM_CSR_ALG1:
        return CUSPARSE_SPMM_CSR_ALG1;
    case HIPSPARSE_SPMM_CSR_ALG2:
        return CUSPARSE_SPMM_CSR_ALG2;
    case HIPSPARSE_SPMM_BLOCKED_ELL_ALG1:
        return CUSPARSE_SPMM_BLOCKED_ELL_ALG1;
    default:
        throw "Non existant hipsparseSpMMAlg_t";
    }
}
#elif(CUDART_VERSION >= 10010)
cusparseSpMMAlg_t hipSpMMAlgToCudaSpMMAlg(hipsparseSpMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_MM_ALG_DEFAULT:
        return CUSPARSE_MM_ALG_DEFAULT;
    case HIPSPARSE_COOMM_ALG1:
        return CUSPARSE_COOMM_ALG1;
    case HIPSPARSE_COOMM_ALG2:
        return CUSPARSE_COOMM_ALG2;
    case HIPSPARSE_COOMM_ALG3:
        return CUSPARSE_COOMM_ALG3;
    case HIPSPARSE_CSRMM_ALG1:
        return CUSPARSE_CSRMM_ALG1;
    default:
        throw "Non existant hipsparseSpMMAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 11000)
cusparseSpGEMMAlg_t hipSpGEMMAlgToCudaSpGEMMAlg(hipsparseSpGEMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SPGEMM_DEFAULT:
        return CUSPARSE_SPGEMM_DEFAULT;
    default:
        throw "Non existant cusparseSpGEMMAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 11020)
cusparseSparseToDenseAlg_t hipSpToDnAlgToCudaSpToDnAlg(hipsparseSparseToDenseAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SPARSETODENSE_ALG_DEFAULT:
        return CUSPARSE_SPARSETODENSE_ALG_DEFAULT;
    default:
        throw "Non existent hipsparseSparseToDenseAlg_t";
    }
}

hipsparseSparseToDenseAlg_t CudaSpToDnAlgToHipSpToDnAlg(cusparseSparseToDenseAlg_t alg)
{
    switch(alg)
    {
    case CUSPARSE_SPARSETODENSE_ALG_DEFAULT:
        return HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
    default:
        throw "Non existent cusparseSparseToDenseAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 11020)
cusparseDenseToSparseAlg_t hipDnToSpAlgToCudaDnToSpAlg(hipsparseDenseToSparseAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT:
        return CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    default:
        throw "Non existent hipsparseDenseToSparseAlg_t";
    }
}

hipsparseDenseToSparseAlg_t CudaDnToSpAlgToHipDnToSpAlg(cusparseDenseToSparseAlg_t alg)
{
    switch(alg)
    {
    case CUSPARSE_DENSETOSPARSE_ALG_DEFAULT:
        return HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    default:
        throw "Non existent cusparseDenseToSparseAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 11020)
cusparseSDDMMAlg_t hipSDDMMAlgToCudaSDDMMAlg(hipsparseSDDMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SDDMM_ALG_DEFAULT:
        return CUSPARSE_SDDMM_ALG_DEFAULT;
    default:
        throw "Non existant cusparseSDDMMAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 11030)
cusparseSpSVAlg_t hipSpSVAlgToCudaSpSVAlg(hipsparseSpSVAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SPSV_ALG_DEFAULT:
        return CUSPARSE_SPSV_ALG_DEFAULT;
    default:
        throw "Non existant cusparseSpSVAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 11031)
cusparseSpSMAlg_t hipSpSMAlgToCudaSpSMAlg(hipsparseSpSMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SPSM_ALG_DEFAULT:
        return CUSPARSE_SPSM_ALG_DEFAULT;
    default:
        throw "Non existant cusparseSpSMAlg_t";
    }
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr,
                                       int64_t                size,
                                       int64_t                nnz,
                                       void*                  indices,
                                       void*                  values,
                                       hipsparseIndexType_t   idxType,
                                       hipsparseIndexBase_t   idxBase,
                                       hipDataType            valueType)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateSpVec((cusparseSpVecDescr_t*)spVecDescr,
                                                            size,
                                                            nnz,
                                                            indices,
                                                            values,
                                                            hipIndexTypeToCudaIndexType(idxType),
                                                            hipIndexBaseToCudaIndexBase(idxBase),
                                                            hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroySpVec((cusparseSpVecDescr_t)spVecDescr));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr,
                                    int64_t*                    size,
                                    int64_t*                    nnz,
                                    void**                      indices,
                                    void**                      values,
                                    hipsparseIndexType_t*       idxType,
                                    hipsparseIndexBase_t*       idxBase,
                                    hipDataType*                valueType)
{
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseSpVecGet((const cusparseSpVecDescr_t)spVecDescr,
                                              size,
                                              nnz,
                                              indices,
                                              values,
                                              idxType != nullptr ? &cuda_index_type : nullptr,
                                              idxBase != nullptr ? &cuda_index_base : nullptr,
                                              valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*       idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpVecGetIndexBase(
        (const cusparseSpVecDescr_t)spVecDescr, (cusparseIndexBase_t*)idxBase));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpVecGetValues(const hipsparseSpVecDescr_t spVecDescr, void** values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpVecGetValues((const cusparseSpVecDescr_t)spVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr, void* values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpVecSetValues((const cusparseSpVecDescr_t)spVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCreateCoo(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cooRowInd,
                                     void*                  cooColInd,
                                     void*                  cooValues,
                                     hipsparseIndexType_t   cooIdxType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateCoo((cusparseSpMatDescr_t*)spMatDescr,
                                                          rows,
                                                          cols,
                                                          nnz,
                                                          cooRowInd,
                                                          cooColInd,
                                                          cooValues,
                                                          hipIndexTypeToCudaIndexType(cooIdxType),
                                                          hipIndexBaseToCudaIndexBase(idxBase),
                                                          hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCreateCooAoS(hipsparseSpMatDescr_t* spMatDescr,
                                        int64_t                rows,
                                        int64_t                cols,
                                        int64_t                nnz,
                                        void*                  cooInd,
                                        void*                  cooValues,
                                        hipsparseIndexType_t   cooIdxType,
                                        hipsparseIndexBase_t   idxBase,
                                        hipDataType            valueType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCooAoS((cusparseSpMatDescr_t*)spMatDescr,
                             rows,
                             cols,
                             nnz,
                             cooInd,
                             cooValues,
                             hipIndexTypeToCudaIndexType(cooIdxType),
                             hipIndexBaseToCudaIndexBase(idxBase),
                             hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 10010)
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
                                     hipDataType            valueType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCsr((cusparseSpMatDescr_t*)spMatDescr,
                          rows,
                          cols,
                          nnz,
                          csrRowOffsets,
                          csrColInd,
                          csrValues,
                          hipIndexTypeToCudaIndexType(csrRowOffsetsType),
                          hipIndexTypeToCudaIndexType(csrColIndType),
                          hipIndexBaseToCudaIndexBase(idxBase),
                          hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 11020)
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
                                     hipDataType            valueType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCreateCsc((cusparseSpMatDescr_t*)spMatDescr,
                          rows,
                          cols,
                          nnz,
                          cscColOffsets,
                          cscRowInd,
                          cscValues,
                          hipIndexTypeToCudaIndexType(cscColOffsetsType),
                          hipIndexTypeToCudaIndexType(cscRowIndType),
                          hipIndexBaseToCudaIndexBase(idxBase),
                          hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 11021)
hipsparseStatus_t hipsparseCreateBlockedEll(hipsparseSpMatDescr_t* spMatDescr,
                                            int64_t                rows,
                                            int64_t                cols,
                                            int64_t                ellBlockSize,
                                            int64_t                ellCols,
                                            void*                  ellColInd,
                                            void*                  ellValue,
                                            hipsparseIndexType_t   ellIdxType,
                                            hipsparseIndexBase_t   idxBase,
                                            hipDataType            valueType)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCreateBlockedEll((cusparseSpMatDescr_t*)spMatDescr,
                                 rows,
                                 cols,
                                 ellBlockSize,
                                 ellCols,
                                 ellColInd,
                                 ellValue,
                                 hipIndexTypeToCudaIndexType(ellIdxType),
                                 hipIndexBaseToCudaIndexBase(idxBase),
                                 hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroySpMat((cusparseSpMatDescr_t)spMatDescr));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCooGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      cooRowInd,
                                  void**                      cooColInd,
                                  void**                      cooValues,
                                  hipsparseIndexType_t*       idxType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType)
{
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseCooGet((const cusparseSpMatDescr_t)spMatDescr,
                                            rows,
                                            cols,
                                            nnz,
                                            cooRowInd,
                                            cooColInd,
                                            cooValues,
                                            idxType != nullptr ? &cuda_index_type : nullptr,
                                            idxBase != nullptr ? &cuda_index_base : nullptr,
                                            valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCooAoSGet(const hipsparseSpMatDescr_t spMatDescr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    nnz,
                                     void**                      cooInd,
                                     void**                      cooValues,
                                     hipsparseIndexType_t*       idxType,
                                     hipsparseIndexBase_t*       idxBase,
                                     hipDataType*                valueType)
{
    cusparseIndexType_t cuda_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseCooAoSGet((const cusparseSpMatDescr_t)spMatDescr,
                                               rows,
                                               cols,
                                               nnz,
                                               cooInd,
                                               cooValues,
                                               idxType != nullptr ? &cuda_index_type : nullptr,
                                               idxBase != nullptr ? &cuda_index_base : nullptr,
                                               valueType != nullptr ? &cuda_data_type : nullptr));

    *idxType   = CudaIndexTypeToHIPIndexType(cuda_index_type);
    *idxBase   = CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType = CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
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
                                  hipDataType*                valueType)
{
    cusparseIndexType_t cuda_row_index_type;
    cusparseIndexType_t cuda_col_index_type;
    cusparseIndexBase_t cuda_index_base;
    cudaDataType        cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(
        cusparseCsrGet((const cusparseSpMatDescr_t)spMatDescr,
                       rows,
                       cols,
                       nnz,
                       csrRowOffsets,
                       csrColInd,
                       csrValues,
                       csrRowOffsetsType != nullptr ? &cuda_row_index_type : nullptr,
                       csrColIndType != nullptr ? &cuda_col_index_type : nullptr,
                       idxBase != nullptr ? &cuda_index_base : nullptr,
                       valueType != nullptr ? &cuda_data_type : nullptr));

    *csrRowOffsetsType = CudaIndexTypeToHIPIndexType(cuda_row_index_type);
    *csrColIndType     = CudaIndexTypeToHIPIndexType(cuda_col_index_type);
    *idxBase           = CudaIndexBaseToHIPIndexBase(cuda_index_base);
    *valueType         = CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 11021)
hipsparseStatus_t hipsparseBlockedEllGet(const hipsparseSpMatDescr_t spMatDescr,
                                         int64_t*                    rows,
                                         int64_t*                    cols,
                                         int64_t*                    ellBlockSize,
                                         int64_t*                    ellCols,
                                         void**                      ellColInd,
                                         void**                      ellValue,
                                         hipsparseIndexType_t*       ellIdxType,
                                         hipsparseIndexBase_t*       idxBase,
                                         hipDataType*                valueType)
{
    // As of cusparse 11.4.1, this routine does not actually exist as a symbol in the cusparse
    // library (the documentation indicates that it should exist starting at cusparse 11.2.1).
    // Uncomment once it has been added
    // cusparseIndexType_t cuda_index_type;
    // cusparseIndexBase_t cuda_index_base;
    // cudaDataType        cuda_data_type;

    // RETURN_IF_CUSPARSE_ERROR(
    //     cusparseBlockedEllGet((cusparseSpMatDescr_t)spMatDescr,
    //                           rows,
    //                           cols,
    //                           ellBlockSize,
    //                           ellCols,
    //                           ellColInd,
    //                           ellValue,
    //                           ellIdxType != nullptr ? &cuda_index_type : nullptr,
    //                           idxBase != nullptr ? &cuda_index_base : nullptr,
    //                           valueType != nullptr ? &cuda_data_type : nullptr));

    // *ellIdxType = CudaIndexTypeToHIPIndexType(cuda_index_type);
    // *idxBase    = CudaIndexBaseToHIPIndexBase(cuda_index_base);
    // *valueType  = CudaDataTypeToHIPDataType(cuda_data_type);

    // return HIPSPARSE_STATUS_SUCCESS;
    return HIPSPARSE_STATUS_NOT_SUPPORTED;
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 csrRowOffsets,
                                          void*                 csrColInd,
                                          void*                 csrValues)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCsrSetPointers(
        (cusparseSpMatDescr_t)spMatDescr, csrRowOffsets, csrColInd, csrValues));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr,
                                        int64_t*              rows,
                                        int64_t*              cols,
                                        int64_t*              nnz)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetSize((cusparseSpMatDescr_t)spMatDescr, rows, cols, nnz));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatGetFormat(const hipsparseSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*          format)
{
    cusparseFormat_t cuda_format;

    RETURN_IF_CUSPARSE_ERROR(cusparseSpMatGetFormat((const cusparseSpMatDescr_t)spMatDescr,
                                                    format != nullptr ? &cuda_format : nullptr));

    *format = CudaFormatToHIPFormat(cuda_format);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatGetIndexBase(const hipsparseSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*       idxBase)
{
    cusparseIndexBase_t cuda_index_base;

    RETURN_IF_CUSPARSE_ERROR(cusparseSpMatGetIndexBase(
        (const cusparseSpMatDescr_t)spMatDescr, idxBase != nullptr ? &cuda_index_base : nullptr));

    *idxBase = CudaIndexBaseToHIPIndexBase(cuda_index_base);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr, void** values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatGetValues((cusparseSpMatDescr_t)spMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr, void* values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatSetValues((cusparseSpMatDescr_t)spMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
                                       int64_t                size,
                                       void*                  values,
                                       hipDataType            valueType)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateDnVec(
        (cusparseDnVecDescr_t*)dnVecDescr, size, values, hipDataTypeToCudaDataType(valueType)));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyDnVec((cusparseDnVecDescr_t)dnVecDescr));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnVecGet(const hipsparseDnVecDescr_t dnVecDescr,
                                    int64_t*                    size,
                                    void**                      values,
                                    hipDataType*                valueType)
{
    cudaDataType cuda_data_type;

    RETURN_IF_CUSPARSE_ERROR(cusparseDnVecGet((const cusparseDnVecDescr_t)dnVecDescr,
                                              size,
                                              values,
                                              valueType != nullptr ? &cuda_data_type : nullptr));

    *valueType = CudaDataTypeToHIPDataType(cuda_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnVecGetValues(const hipsparseDnVecDescr_t dnVecDescr, void** values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDnVecGetValues((const cusparseDnVecDescr_t)dnVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr, void* values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDnVecSetValues((cusparseDnVecDescr_t)dnVecDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
                                       int64_t                rows,
                                       int64_t                cols,
                                       int64_t                ld,
                                       void*                  values,
                                       hipDataType            valueType,
                                       hipsparseOrder_t       order)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateDnMat((cusparseDnMatDescr_t*)dnMatDescr,
                                                            rows,
                                                            cols,
                                                            ld,
                                                            values,
                                                            hipDataTypeToCudaDataType(valueType),
                                                            hipOrderToCudaOrder(order)));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyDnMat((cusparseDnMatDescr_t)dnMatDescr));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    ld,
                                    void**                      values,
                                    hipDataType*                valueType,
                                    hipsparseOrder_t*           order)
{
    cudaDataType    cuda_data_type;
    cusparseOrder_t cusparse_order;
    hipCUSPARSEStatusToHIPStatus(cusparseDnMatGet((const cusparseDnMatDescr_t)dnMatDescr,
                                                  rows,
                                                  cols,
                                                  ld,
                                                  values,
                                                  valueType != nullptr ? &cuda_data_type : nullptr,
                                                  order != nullptr ? &cusparse_order : nullptr));

    *valueType = CudaDataTypeToHIPDataType(cuda_data_type);
    *order     = CudaOrderToHIPOrder(cusparse_order);

    return HIPSPARSE_STATUS_SUCCESS;
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatGetValues(const hipsparseDnMatDescr_t dnMatDescr, void** values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDnMatGetValues((const cusparseDnMatDescr_t)dnMatDescr, values));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr, void* values)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDnMatSetValues((cusparseDnMatDescr_t)dnMatDescr, values));
}
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatGetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             void*                     data,
                                             size_t                    dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpMatGetAttribute(
        (cusparseSpMatDescr_t)spMatDescr, (cusparseSpMatAttribute_t)attribute, data, dataSize));
}
#endif

#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseSpMatSetAttribute(hipsparseSpMatDescr_t     spMatDescr,
                                             hipsparseSpMatAttribute_t attribute,
                                             const void*               data,
                                             size_t                    dataSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpMatSetAttribute((cusparseSpMatDescr_t)spMatDescr,
                                  (cusparseSpMatAttribute_t)attribute,
                                  const_cast<void*>(data),
                                  dataSize));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t     handle,
                                 const void*           alpha,
                                 hipsparseSpVecDescr_t vecX,
                                 const void*           beta,
                                 hipsparseDnVecDescr_t vecY)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseAxpby((cusparseHandle_t)handle,
                                                      alpha,
                                                      (cusparseSpVecDescr_t)vecX,
                                                      beta,
                                                      (cusparseDnVecDescr_t)vecY));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseGather(hipsparseHandle_t     handle,
                                  hipsparseDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t vecX)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseGather(
        (cusparseHandle_t)handle, (cusparseDnVecDescr_t)vecY, (cusparseSpVecDescr_t)vecX));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseScatter(hipsparseHandle_t     handle,
                                   hipsparseSpVecDescr_t vecX,
                                   hipsparseDnVecDescr_t vecY)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScatter(
        (cusparseHandle_t)handle, (cusparseSpVecDescr_t)vecX, (cusparseDnVecDescr_t)vecY));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseRot(hipsparseHandle_t     handle,
                               const void*           c_coeff,
                               const void*           s_coeff,
                               hipsparseSpVecDescr_t vecX,
                               hipsparseDnVecDescr_t vecY)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseRot((cusparseHandle_t)handle,
                                                    c_coeff,
                                                    s_coeff,
                                                    (cusparseSpVecDescr_t)vecX,
                                                    (cusparseDnVecDescr_t)vecY));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseSpMatDescr_t       matA,
                                                    hipsparseDnMatDescr_t       matB,
                                                    hipsparseSparseToDenseAlg_t alg,
                                                    size_t*                     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSparseToDense_bufferSize((cusparseHandle_t)handle,
                                         (cusparseSpMatDescr_t)matA,
                                         (cusparseDnMatDescr_t)matB,
                                         hipSpToDnAlgToCudaSpToDnAlg(alg),
                                         bufferSize));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                         hipsparseSpMatDescr_t       matA,
                                         hipsparseDnMatDescr_t       matB,
                                         hipsparseSparseToDenseAlg_t alg,
                                         void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSparseToDense((cusparseHandle_t)handle,
                                                              (cusparseSpMatDescr_t)matA,
                                                              (cusparseDnMatDescr_t)matB,
                                                              hipSpToDnAlgToCudaSpToDnAlg(alg),
                                                              externalBuffer));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseDnMatDescr_t       matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_bufferSize((cusparseHandle_t)handle,
                                         (cusparseDnMatDescr_t)matA,
                                         (cusparseSpMatDescr_t)matB,
                                         hipDnToSpAlgToCudaDnToSpAlg(alg),
                                         bufferSize));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseDnMatDescr_t       matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_analysis((cusparseHandle_t)handle,
                                       (cusparseDnMatDescr_t)matA,
                                       (cusparseSpMatDescr_t)matB,
                                       hipDnToSpAlgToCudaDnToSpAlg(alg),
                                       externalBuffer));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseDnMatDescr_t       matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDenseToSparse_convert((cusparseHandle_t)handle,
                                      (cusparseDnMatDescr_t)matA,
                                      (cusparseSpMatDescr_t)matB,
                                      hipDnToSpAlgToCudaDnToSpAlg(alg),
                                      externalBuffer));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpVV_bufferSize((cusparseHandle_t)handle,
                                hipOperationToCudaOperation(opX),
                                (cusparseSpVecDescr_t)vecX,
                                (cusparseDnVecDescr_t)vecY,
                                result,
                                hipDataTypeToCudaDataType(computeType),
                                bufferSize));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpVV((cusparseHandle_t)handle,
                                                     hipOperationToCudaOperation(opX),
                                                     (cusparseSpVecDescr_t)vecX,
                                                     (cusparseDnVecDescr_t)vecY,
                                                     result,
                                                     hipDataTypeToCudaDataType(computeType),
                                                     externalBuffer));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpMV_bufferSize((cusparseHandle_t)handle,
                                hipOperationToCudaOperation(opA),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnVecDescr_t)vecX,
                                beta,
                                (const cusparseDnVecDescr_t)vecY,
                                hipDataTypeToCudaDataType(computeType),
                                hipSpMVAlgToCudaSpMVAlg(alg),
                                bufferSize));
}
#endif

#if(CUDART_VERSION >= 10010)
hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnVecDescr_t vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpMV((cusparseHandle_t)handle,
                                                     hipOperationToCudaOperation(opA),
                                                     alpha,
                                                     (const cusparseSpMatDescr_t)matA,
                                                     (const cusparseDnVecDescr_t)vecX,
                                                     beta,
                                                     (const cusparseDnVecDescr_t)vecY,
                                                     hipDataTypeToCudaDataType(computeType),
                                                     hipSpMVAlgToCudaSpMVAlg(alg),
                                                     externalBuffer));
}
#endif

#if(CUDART_VERSION >= 10010)
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
                                           size_t*                     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM_bufferSize((cusparseHandle_t)handle,
                                hipOperationToCudaOperation(opA),
                                hipOperationToCudaOperation(opB),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnMatDescr_t)matB,
                                beta,
                                (const cusparseDnMatDescr_t)matC,
                                hipDataTypeToCudaDataType(computeType),
                                hipSpMMAlgToCudaSpMMAlg(alg),
                                bufferSize));
}
#endif

#if(CUDART_VERSION >= 11021)
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
                                           void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpMM_preprocess((cusparseHandle_t)handle,
                                hipOperationToCudaOperation(opA),
                                hipOperationToCudaOperation(opB),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnMatDescr_t)matB,
                                beta,
                                (const cusparseDnMatDescr_t)matC,
                                hipDataTypeToCudaDataType(computeType),
                                hipSpMMAlgToCudaSpMMAlg(alg),
                                externalBuffer));
}
#endif

#if(CUDART_VERSION >= 10010)
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
                                void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpMM((cusparseHandle_t)handle,
                                                     hipOperationToCudaOperation(opA),
                                                     hipOperationToCudaOperation(opB),
                                                     alpha,
                                                     (const cusparseSpMatDescr_t)matA,
                                                     (const cusparseDnMatDescr_t)matB,
                                                     beta,
                                                     (const cusparseDnMatDescr_t)matC,
                                                     hipDataTypeToCudaDataType(computeType),
                                                     hipSpMMAlgToCudaSpMMAlg(alg),
                                                     externalBuffer));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpGEMM_createDescr((cusparseSpGEMMDescr_t*)descr));
}
#endif

#if(CUDART_VERSION >= 11000)
hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpGEMM_destroyDescr((cusparseSpGEMMDescr_t)descr));
}
#endif

#if(CUDART_VERSION >= 11000)
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
                                                 void*                  externalBuffer1)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpGEMM_workEstimation((cusparseHandle_t)handle,
                                      hipOperationToCudaOperation(opA),
                                      hipOperationToCudaOperation(opB),
                                      alpha,
                                      (cusparseSpMatDescr_t)matA,
                                      (cusparseSpMatDescr_t)matB,
                                      beta,
                                      (cusparseSpMatDescr_t)matC,
                                      computeType,
                                      hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                      (cusparseSpGEMMDescr_t)spgemmDescr,
                                      bufferSize1,
                                      externalBuffer1));
}
#endif

#if(CUDART_VERSION >= 11000)
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
                                          void*                  externalBuffer2)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpGEMM_compute((cusparseHandle_t)handle,
                                                               hipOperationToCudaOperation(opA),
                                                               hipOperationToCudaOperation(opB),
                                                               alpha,
                                                               (cusparseSpMatDescr_t)matA,
                                                               (cusparseSpMatDescr_t)matB,
                                                               beta,
                                                               (cusparseSpMatDescr_t)matC,
                                                               computeType,
                                                               hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                                               (cusparseSpGEMMDescr_t)spgemmDescr,
                                                               bufferSize2,
                                                               externalBuffer2));
}
#endif

#if(CUDART_VERSION >= 11000)
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
                                       hipsparseSpGEMMDescr_t spgemmDescr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpGEMM_copy((cusparseHandle_t)handle,
                                                            hipOperationToCudaOperation(opA),
                                                            hipOperationToCudaOperation(opB),
                                                            alpha,
                                                            (cusparseSpMatDescr_t)matA,
                                                            (cusparseSpMatDescr_t)matB,
                                                            beta,
                                                            (cusparseSpMatDescr_t)matC,
                                                            computeType,
                                                            hipSpGEMMAlgToCudaSpGEMMAlg(alg),
                                                            (cusparseSpGEMMDescr_t)spgemmDescr));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t           handle,
                                 hipsparseOperation_t        opA,
                                 hipsparseOperation_t        opB,
                                 const void*                 alpha,
                                 const hipsparseDnMatDescr_t matA,
                                 const hipsparseDnMatDescr_t matB,
                                 const void*                 beta,
                                 hipsparseSpMatDescr_t       matC,
                                 hipDataType                 computeType,
                                 hipsparseSDDMMAlg_t         alg,
                                 void*                       tempBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSDDMM((cusparseHandle_t)handle,
                                                      hipOperationToCudaOperation(opA),
                                                      hipOperationToCudaOperation(opB),
                                                      alpha,
                                                      (const cusparseDnMatDescr_t)matA,
                                                      (const cusparseDnMatDescr_t)matB,
                                                      beta,
                                                      (cusparseSpMatDescr_t)matC,
                                                      hipDataTypeToCudaDataType(computeType),
                                                      hipSDDMMAlgToCudaSDDMMAlg(alg),
                                                      tempBuffer));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t matA,
                                            const hipsparseDnMatDescr_t matB,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       matC,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            size_t*                     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSDDMM_bufferSize((cusparseHandle_t)handle,
                                 hipOperationToCudaOperation(opA),
                                 hipOperationToCudaOperation(opB),
                                 alpha,
                                 (const cusparseDnMatDescr_t)matA,
                                 (const cusparseDnMatDescr_t)matB,
                                 beta,
                                 (cusparseSpMatDescr_t)matC,
                                 hipDataTypeToCudaDataType(computeType),
                                 hipSDDMMAlgToCudaSDDMMAlg(alg),
                                 bufferSize));
}
#endif

#if(CUDART_VERSION >= 11020)
hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t matA,
                                            const hipsparseDnMatDescr_t matB,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       matC,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            void*                       tempBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSDDMM_preprocess((cusparseHandle_t)handle,
                                 hipOperationToCudaOperation(opA),
                                 hipOperationToCudaOperation(opB),
                                 alpha,
                                 (const cusparseDnMatDescr_t)matA,
                                 (const cusparseDnMatDescr_t)matB,
                                 beta,
                                 (cusparseSpMatDescr_t)matC,
                                 hipDataTypeToCudaDataType(computeType),
                                 hipSDDMMAlgToCudaSDDMMAlg(alg),
                                 tempBuffer));
}
#endif

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_createDescr(hipsparseSpSVDescr_t* descr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpSV_createDescr((cusparseSpSVDescr_t*)descr));
}
#endif

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_destroyDescr(hipsparseSpSVDescr_t descr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpSV_destroyDescr((cusparseSpSVDescr_t)descr));
}
#endif

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t x,
                                           const hipsparseDnVecDescr_t y,
                                           hipDataType                 computeType,
                                           hipsparseSpSVAlg_t          alg,
                                           hipsparseSpSVDescr_t        spsvDescr,
                                           size_t*                     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_bufferSize((cusparseHandle_t)handle,
                                hipOperationToCudaOperation(opA),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnVecDescr_t)x,
                                (const cusparseDnVecDescr_t)y,
                                hipDataTypeToCudaDataType(computeType),
                                hipSpSVAlgToCudaSpSVAlg(alg),
                                (cusparseSpSVDescr_t)spsvDescr,
                                bufferSize));
}
#endif

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_analysis(hipsparseHandle_t           handle,
                                         hipsparseOperation_t        opA,
                                         const void*                 alpha,
                                         const hipsparseSpMatDescr_t matA,
                                         const hipsparseDnVecDescr_t x,
                                         const hipsparseDnVecDescr_t y,
                                         hipDataType                 computeType,
                                         hipsparseSpSVAlg_t          alg,
                                         hipsparseSpSVDescr_t        spsvDescr,
                                         void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpSV_analysis((cusparseHandle_t)handle,
                              hipOperationToCudaOperation(opA),
                              alpha,
                              (const cusparseSpMatDescr_t)matA,
                              (const cusparseDnVecDescr_t)x,
                              (const cusparseDnVecDescr_t)y,
                              hipDataTypeToCudaDataType(computeType),
                              hipSpSVAlgToCudaSpSVAlg(alg),
                              (cusparseSpSVDescr_t)spsvDescr,
                              externalBuffer));
}
#endif

#if(CUDART_VERSION >= 11030)
hipsparseStatus_t hipsparseSpSV_solve(hipsparseHandle_t           handle,
                                      hipsparseOperation_t        opA,
                                      const void*                 alpha,
                                      const hipsparseSpMatDescr_t matA,
                                      const hipsparseDnVecDescr_t x,
                                      const hipsparseDnVecDescr_t y,
                                      hipDataType                 computeType,
                                      hipsparseSpSVAlg_t          alg,
                                      hipsparseSpSVDescr_t        spsvDescr,
                                      void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpSV_solve((cusparseHandle_t)handle,
                                                           hipOperationToCudaOperation(opA),
                                                           alpha,
                                                           (const cusparseSpMatDescr_t)matA,
                                                           (const cusparseDnVecDescr_t)x,
                                                           (const cusparseDnVecDescr_t)y,
                                                           hipDataTypeToCudaDataType(computeType),
                                                           hipSpSVAlgToCudaSpSVAlg(alg),
                                                           (cusparseSpSVDescr_t)spsvDescr));
}
#endif

#if(CUDART_VERSION >= 11031)
hipsparseStatus_t hipsparseSpSM_createDescr(hipsparseSpSMDescr_t* descr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpSM_createDescr((cusparseSpSMDescr_t*)descr));
}
#endif

#if(CUDART_VERSION >= 11031)
hipsparseStatus_t hipsparseSpSM_destroyDescr(hipsparseSpSMDescr_t descr)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpSM_destroyDescr((cusparseSpSMDescr_t)descr));
}
#endif

#if(CUDART_VERSION >= 11031)
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
                                           size_t*                     bufferSize)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpSM_bufferSize((cusparseHandle_t)handle,
                                hipOperationToCudaOperation(opA),
                                hipOperationToCudaOperation(opB),
                                alpha,
                                (const cusparseSpMatDescr_t)matA,
                                (const cusparseDnMatDescr_t)matB,
                                (const cusparseDnMatDescr_t)matC,
                                hipDataTypeToCudaDataType(computeType),
                                hipSpSMAlgToCudaSpSMAlg(alg),
                                (cusparseSpSMDescr_t)spsmDescr,
                                bufferSize));
}
#endif

#if(CUDART_VERSION >= 11031)
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
                                         void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSpSM_analysis((cusparseHandle_t)handle,
                              hipOperationToCudaOperation(opA),
                              hipOperationToCudaOperation(opB),
                              alpha,
                              (const cusparseSpMatDescr_t)matA,
                              (const cusparseDnMatDescr_t)matB,
                              (const cusparseDnMatDescr_t)matC,
                              hipDataTypeToCudaDataType(computeType),
                              hipSpSMAlgToCudaSpSMAlg(alg),
                              (cusparseSpSMDescr_t)spsmDescr,
                              externalBuffer));
}
#endif

#if(CUDART_VERSION >= 11031)
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
                                      void*                       externalBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSpSM_solve((cusparseHandle_t)handle,
                                                           hipOperationToCudaOperation(opA),
                                                           hipOperationToCudaOperation(opB),
                                                           alpha,
                                                           (const cusparseSpMatDescr_t)matA,
                                                           (const cusparseDnMatDescr_t)matB,
                                                           (const cusparseDnMatDescr_t)matC,
                                                           hipDataTypeToCudaDataType(computeType),
                                                           hipSpSMAlgToCudaSpSMAlg(alg),
                                                           (cusparseSpSMDescr_t)spsmDescr));
}
#endif

hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgtsv2StridedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgtsv2StridedBatch_bufferSizeExt(
        (cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCgtsv2StridedBatch_bufferSizeExt((cusparseHandle_t)handle,
                                                 m,
                                                 (const cuComplex*)dl,
                                                 (const cuComplex*)d,
                                                 (const cuComplex*)du,
                                                 (const cuComplex*)x,
                                                 batchCount,
                                                 batchStride,
                                                 pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* x,
                                                            int                     batchCount,
                                                            int                     batchStride,
                                                            size_t* pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZgtsv2StridedBatch_bufferSizeExt((cusparseHandle_t)handle,
                                                 m,
                                                 (const cuDoubleComplex*)dl,
                                                 (const cuDoubleComplex*)d,
                                                 (const cuDoubleComplex*)du,
                                                 (const cuDoubleComplex*)x,
                                                 batchCount,
                                                 batchStride,
                                                 pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgtsv2StridedBatch(
        (cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgtsv2StridedBatch(
        (cusparseHandle_t)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgtsv2StridedBatch((cusparseHandle_t)handle,
                                                                   m,
                                                                   (const cuComplex*)dl,
                                                                   (const cuComplex*)d,
                                                                   (const cuComplex*)du,
                                                                   (cuComplex*)x,
                                                                   batchCount,
                                                                   batchStride,
                                                                   pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t       handle,
                                              int                     m,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       x,
                                              int                     batchCount,
                                              int                     batchStride,
                                              void*                   pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgtsv2StridedBatch((cusparseHandle_t)handle,
                                                                   m,
                                                                   (const cuDoubleComplex*)dl,
                                                                   (const cuDoubleComplex*)d,
                                                                   (const cuDoubleComplex*)du,
                                                                   (cuDoubleComplex*)x,
                                                                   batchCount,
                                                                   batchStride,
                                                                   pBuffer));
}

hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const float*      dl,
                                                const float*      d,
                                                const float*      du,
                                                const float*      B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgtsv2_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const double*     dl,
                                                const double*     d,
                                                const double*     du,
                                                const double*     B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgtsv2_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const hipComplex* dl,
                                                const hipComplex* d,
                                                const hipComplex* du,
                                                const hipComplex* B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgtsv2_bufferSizeExt((cusparseHandle_t)handle,
                                                                     m,
                                                                     n,
                                                                     (const cuComplex*)dl,
                                                                     (const cuComplex*)d,
                                                                     (const cuComplex*)du,
                                                                     (const cuComplex*)B,
                                                                     ldb,
                                                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(hipsparseHandle_t       handle,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* dl,
                                                const hipDoubleComplex* d,
                                                const hipDoubleComplex* du,
                                                const hipDoubleComplex* B,
                                                int                     ldb,
                                                size_t*                 pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgtsv2_bufferSizeExt((cusparseHandle_t)handle,
                                                                     m,
                                                                     n,
                                                                     (const cuDoubleComplex*)dl,
                                                                     (const cuDoubleComplex*)d,
                                                                     (const cuDoubleComplex*)du,
                                                                     (const cuDoubleComplex*)B,
                                                                     ldb,
                                                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const float*      dl,
                                  const float*      d,
                                  const float*      du,
                                  float*            B,
                                  int               ldb,
                                  void*             pBuffer)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSgtsv2((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const double*     dl,
                                  const double*     d,
                                  const double*     du,
                                  double*           B,
                                  int               ldb,
                                  void*             pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDgtsv2((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const hipComplex* dl,
                                  const hipComplex* d,
                                  const hipComplex* du,
                                  hipComplex*       B,
                                  int               ldb,
                                  void*             pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgtsv2((cusparseHandle_t)handle,
                                                       m,
                                                       n,
                                                       (const cuComplex*)dl,
                                                       (const cuComplex*)d,
                                                       (const cuComplex*)du,
                                                       (cuComplex*)B,
                                                       ldb,
                                                       pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* dl,
                                  const hipDoubleComplex* d,
                                  const hipDoubleComplex* du,
                                  hipDoubleComplex*       B,
                                  int                     ldb,
                                  void*                   pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgtsv2((cusparseHandle_t)handle,
                                                       m,
                                                       n,
                                                       (const cuDoubleComplex*)dl,
                                                       (const cuDoubleComplex*)d,
                                                       (const cuDoubleComplex*)du,
                                                       (cuDoubleComplex*)B,
                                                       ldb,
                                                       pBuffer));
}

hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const float*      dl,
                                                        const float*      d,
                                                        const float*      du,
                                                        const float*      B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSgtsv2_nopivot_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const double*     dl,
                                                        const double*     d,
                                                        const double*     du,
                                                        const double*     B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDgtsv2_nopivot_bufferSizeExt(
        (cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const hipComplex* dl,
                                                        const hipComplex* d,
                                                        const hipComplex* du,
                                                        const hipComplex* B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle,
                                             m,
                                             n,
                                             (const cuComplex*)dl,
                                             (const cuComplex*)d,
                                             (const cuComplex*)du,
                                             (const cuComplex*)B,
                                             ldb,
                                             pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t       handle,
                                                        int                     m,
                                                        int                     n,
                                                        const hipDoubleComplex* dl,
                                                        const hipDoubleComplex* d,
                                                        const hipDoubleComplex* du,
                                                        const hipDoubleComplex* B,
                                                        int                     ldb,
                                                        size_t*                 pBufferSizeInBytes)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseZgtsv2_nopivot_bufferSizeExt((cusparseHandle_t)handle,
                                             m,
                                             n,
                                             (const cuDoubleComplex*)dl,
                                             (const cuDoubleComplex*)d,
                                             (const cuDoubleComplex*)du,
                                             (const cuDoubleComplex*)B,
                                             ldb,
                                             pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const float*      dl,
                                          const float*      d,
                                          const float*      du,
                                          float*            B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const double*     dl,
                                          const double*     d,
                                          const double*     du,
                                          double*           B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDgtsv2_nopivot((cusparseHandle_t)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const hipComplex* dl,
                                          const hipComplex* d,
                                          const hipComplex* du,
                                          hipComplex*       B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(cusparseCgtsv2_nopivot((cusparseHandle_t)handle,
                                                               m,
                                                               n,
                                                               (const cuComplex*)dl,
                                                               (const cuComplex*)d,
                                                               (const cuComplex*)du,
                                                               (cuComplex*)B,
                                                               ldb,
                                                               pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2_nopivot(hipsparseHandle_t       handle,
                                          int                     m,
                                          int                     n,
                                          const hipDoubleComplex* dl,
                                          const hipDoubleComplex* d,
                                          const hipDoubleComplex* du,
                                          hipDoubleComplex*       B,
                                          int                     ldb,
                                          void*                   pBuffer)

{
    return hipCUSPARSEStatusToHIPStatus(cusparseZgtsv2_nopivot((cusparseHandle_t)handle,
                                                               m,
                                                               n,
                                                               (const cuDoubleComplex*)dl,
                                                               (const cuDoubleComplex*)d,
                                                               (const cuDoubleComplex*)du,
                                                               (cuDoubleComplex*)B,
                                                               ldb,
                                                               pBuffer));
}

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
                                     hipsparseColorInfo_t      info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseScsrcolor((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (cusparseColorInfo_t)info));
}

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
                                     hipsparseColorInfo_t      info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDcsrcolor((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (cusparseColorInfo_t)info));
}

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
                                     hipsparseColorInfo_t      info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCcsrcolor((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          (const cuComplex*)csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (cusparseColorInfo_t)info));
}

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
                                     hipsparseColorInfo_t      info)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseZcsrcolor((cusparseHandle_t)handle,
                                                          m,
                                                          nnz,
                                                          (const cusparseMatDescr_t)descrA,
                                                          (const cuDoubleComplex*)csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (cusparseColorInfo_t)info));
}

#ifdef __cplusplus
}
#endif
