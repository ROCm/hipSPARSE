/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsparse.h"

#include <hip/hip_runtime_api.h>
#include <cuda_runtime_api.h>
#include <cusparse_v2.h>

#ifdef __cplusplus
extern "C" {
#endif

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
    return hipCUSPARSEStatusToHIPStatus(
        cusparseGetVersion((cusparseHandle_t)handle, version));
}

hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSetStream((cusparseHandle_t)handle, streamId));
}

hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseGetStream((cusparseHandle_t)handle, streamId));
}

hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle,
                                          hipsparsePointerMode_t mode)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSetPointerMode(
        (cusparseHandle_t)handle, hipPointerModeToCudaPointerMode(mode)));
}

hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle,
                                          hipsparsePointerMode_t* mode)
{
    cusparsePointerMode_t cusparseMode;
    cusparseStatus_t status =
        cusparseGetPointerMode((cusparseHandle_t)handle, &cusparseMode);
    *mode = CudaPointerModeToHIPPointerMode(cusparseMode);
    return hipCUSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCreateMatDescr((cusparseMatDescr_t*)descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDestroyMatDescr((cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest,
                                        const hipsparseMatDescr_t src)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCopyMatDescr((cusparseMatDescr_t)dest, (const cusparseMatDescr_t)src));
}

hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA,
                                      hipsparseMatrixType_t type)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSetMatType(
        (cusparseMatDescr_t)descrA, hipMatrixTypeToCudaMatrixType(type)));
}

hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA)
{
    return CudaMatrixTypeToHIPMatrixType(
        cusparseGetMatType((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA,
                                          hipsparseFillMode_t fillMode)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSetMatFillMode((cusparseMatDescr_t)descrA, hipFillToCudaFill(fillMode)));
}

hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA)
{
    return CudaFillToHIPFill(cusparseGetMatFillMode((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA,
                                          hipsparseDiagType_t diagType)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSetMatDiagType(
        (cusparseMatDescr_t)descrA, hipDiagonalToCudaDiagonal(diagType)));
}

hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA)
{
    return CudaDiagonalToHIPDiagonal(
        cusparseGetMatDiagType((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA,
                                           hipsparseIndexBase_t base)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseSetMatIndexBase(
        (cusparseMatDescr_t)descrA, hipIndexBaseToCudaIndexBase(base)));
}

hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA)
{
    return CudaIndexBaseToHIPIndexBase(
        cusparseGetMatIndexBase((const cusparseMatDescr_t)descrA));
}

hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseCreateHybMat((cusparseHybMat_t*)hybA));
}

hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA)
{
    return hipCUSPARSEStatusToHIPStatus(cusparseDestroyHybMat((cusparseHybMat_t)hybA));
}

hipsparseStatus_t hipsparseSaxpyi(hipsparseHandle_t handle,
                                  int nnz,
                                  const float* alpha,
                                  const float* xVal,
                                  const int* xInd,
                                  float* y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseSaxpyi((cusparseHandle_t)handle,
                       nnz,
                       alpha,
                       xVal,
                       xInd,
                       y,
                       hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseDaxpyi(hipsparseHandle_t handle,
                                  int nnz,
                                  const double* alpha,
                                  const double* xVal,
                                  const int* xInd,
                                  double* y,
                                  hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDaxpyi((cusparseHandle_t)handle,
                       nnz,
                       alpha,
                       xVal,
                       xInd,
                       y,
                       hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float* csrSortedValA,
                                  const int* csrSortedRowPtrA,
                                  const int* csrSortedColIndA,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseScsrmv((cusparseHandle_t)handle,
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

hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double* csrSortedValA,
                                  const int* csrSortedRowPtrA,
                                  const int* csrSortedColIndA,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDcsrmv((cusparseHandle_t)handle,
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

hipsparseStatus_t hipsparseScoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float* cooValA,
                                  const int* cooRowIndA,
                                  const int* cooColIndA,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double* cooValA,
                                  const int* cooRowIndA,
                                  const int* cooColIndA,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseSellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float* ellValA,
                                  const int* ellColIndA,
                                  int ellWidth,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDellmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double* ellValA,
                                  const int* ellColIndA,
                                  int ellWidth,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseShybmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t hybA,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseShybmv((cusparseHandle_t)handle,
                       hipOperationToCudaOperation(transA),
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cusparseHybMat_t)hybA,
                       x,
                       beta,
                       y));
}

hipsparseStatus_t hipsparseDhybmv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t hybA,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseDhybmv((cusparseHandle_t)handle,
                       hipOperationToCudaOperation(transA),
                       alpha,
                       (const cusparseMatDescr_t)descrA,
                       (const cusparseHybMat_t)hybA,
                       x,
                       beta,
                       y));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle,
                                    const int* csrRowPtr,
                                    int nnz,
                                    int m,
                                    int* cooRowInd,
                                    hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcsr2coo((cusparseHandle_t)handle,
                         csrRowPtr,
                         nnz,
                         m,
                         cooRowInd,
                         hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseXcsr2ellWidth(hipsparseHandle_t handle,
                                         int m,
                                         const hipsparseMatDescr_t descrA,
                                         const int* csrRowPtrA,
                                         const hipsparseMatDescr_t descrC,
                                         int* ellWidthC)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseScsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t descrA,
                                    const float* csrValA,
                                    const int* csrRowPtrA,
                                    const int* csrColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    int ellWidthC,
                                    float* ellValC,
                                    int* ellColIndC)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseDcsr2ell(hipsparseHandle_t handle,
                                    int m,
                                    const hipsparseMatDescr_t descrA,
                                    const double* csrValA,
                                    const int* csrRowPtrA,
                                    const int* csrColIndA,
                                    const hipsparseMatDescr_t descrC,
                                    int ellWidthC,
                                    double* ellValC,
                                    int* ellColIndC)
{
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descrA,
                                    const float* csrSortedValA,
                                    const int* csrSortedRowPtrA,
                                    const int* csrSortedColIndA,
                                    hipsparseHybMat_t hybA,
                                    int userEllWidth,
                                    hipsparseHybPartition_t partitionType)
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

hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    const hipsparseMatDescr_t descrA,
                                    const double* csrSortedValA,
                                    const int* csrSortedRowPtrA,
                                    const int* csrSortedColIndA,
                                    hipsparseHybMat_t hybA,
                                    int userEllWidth,
                                    hipsparseHybPartition_t partitionType)
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

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle,
                                    const int* cooRowInd,
                                    int nnz,
                                    int m,
                                    int* csrRowPtr,
                                    hipsparseIndexBase_t idxBase)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseXcoo2csr((cusparseHandle_t)handle,
                         cooRowInd,
                         nnz,
                         m,
                         csrRowPtr,
                         hipIndexBaseToCudaIndexBase(idxBase)));
}

hipsparseStatus_t hipsparseCreateIdentityPermutation(hipsparseHandle_t handle,
                                                     int n,
                                                     int* p)
{
    return hipCUSPARSEStatusToHIPStatus(
        cusparseCreateIdentityPermutation((cusparseHandle_t)handle, n, p));
}

#ifdef __cplusplus
}
#endif
