/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "hipsparse.h"

#include <rocsparse.h>
#include <hip/hip_runtime_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                 \
    {                                                               \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;   \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                      \
        {                                                           \
            return hipErrorToHIPSPARSEStatus(TMP_STATUS_FOR_CHECK); \
        }                                                           \
    }

#define RETURN_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        rocsparse_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)            \
        {                                                               \
            return rocSPARSEStatusToHIPStatus(TMP_STATUS_FOR_CHECK);    \
        }                                                               \
    }

hipsparseStatus_t hipErrorToHIPSPARSEStatus(hipError_t status)
{
    switch(status)
    {
    case hipSuccess: return HIPSPARSE_STATUS_SUCCESS;
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources: return HIPSPARSE_STATUS_ALLOC_FAILED;
    case hipErrorInvalidDevicePointer: return HIPSPARSE_STATUS_INVALID_VALUE;
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle: return HIPSPARSE_STATUS_NOT_INITIALIZED;
    case hipErrorInvalidValue: return HIPSPARSE_STATUS_INVALID_VALUE;
    case hipErrorNoDevice:
    case hipErrorUnknown: return HIPSPARSE_STATUS_INTERNAL_ERROR;
    default: return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }
}

hipsparseStatus_t rocSPARSEStatusToHIPStatus(rocsparse_status_ status)
{
    switch(status)
    {
    case rocsparse_status_success: return HIPSPARSE_STATUS_SUCCESS;
    case rocsparse_status_invalid_handle: return HIPSPARSE_STATUS_NOT_INITIALIZED;
    case rocsparse_status_not_implemented: return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case rocsparse_status_invalid_pointer: return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparse_status_invalid_size: return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparse_status_memory_error: return HIPSPARSE_STATUS_ALLOC_FAILED;
    case rocsparse_status_internal_error: return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case rocsparse_status_invalid_value: return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparse_status_arch_mismatch: return HIPSPARSE_STATUS_ARCH_MISMATCH;
    default: throw "Non existent rocsparse_status";
    }
}

rocsparse_pointer_mode_ hipPtrModeToHCCPtrMode(hipsparsePointerMode_t mode)
{
    switch(mode)
    {
    case HIPSPARSE_POINTER_MODE_HOST: return rocsparse_pointer_mode_host;
    case HIPSPARSE_POINTER_MODE_DEVICE: return rocsparse_pointer_mode_device;
    default: throw "Non existent hipsparsePointerMode_t";
    }
}

hipsparsePointerMode_t HCCPtrModeToHIPPtrMode(rocsparse_pointer_mode_ mode)
{
    switch(mode)
    {
    case rocsparse_pointer_mode_host: return HIPSPARSE_POINTER_MODE_HOST;
    case rocsparse_pointer_mode_device: return HIPSPARSE_POINTER_MODE_DEVICE;
    default: throw "Non existent rocsparse_pointer_mode";
    }
}

rocsparse_action_ hipActionToHCCAction(hipsparseAction_t action)
{
    switch(action)
    {
    case HIPSPARSE_ACTION_SYMBOLIC: return rocsparse_action_symbolic;
    case HIPSPARSE_ACTION_NUMERIC: return rocsparse_action_numeric;
    default: throw "Non existent hipsparseAction_t";
    }
}

rocsparse_matrix_type_ hipMatTypeToHCCMatType(hipsparseMatrixType_t type)
{
    switch(type)
    {
    case HIPSPARSE_MATRIX_TYPE_GENERAL: return rocsparse_matrix_type_general;
    case HIPSPARSE_MATRIX_TYPE_SYMMETRIC: return rocsparse_matrix_type_symmetric;
    case HIPSPARSE_MATRIX_TYPE_HERMITIAN: return rocsparse_matrix_type_hermitian;
    case HIPSPARSE_MATRIX_TYPE_TRIANGULAR: return rocsparse_matrix_type_triangular;
    default: throw "Non existent hipsparseMatrixType_t";
    }
}

hipsparseMatrixType_t HCCMatTypeToHIPMatType(rocsparse_matrix_type_ type)
{
    switch(type)
    {
    case rocsparse_matrix_type_general: return HIPSPARSE_MATRIX_TYPE_GENERAL;
    case rocsparse_matrix_type_symmetric: return HIPSPARSE_MATRIX_TYPE_SYMMETRIC;
    case rocsparse_matrix_type_hermitian: return HIPSPARSE_MATRIX_TYPE_HERMITIAN;
    case rocsparse_matrix_type_triangular: return HIPSPARSE_MATRIX_TYPE_TRIANGULAR;
    default: throw "Non existent rocsparse_matrix_type";
    }
}

// TODO fillmode
// TODO diagtype

rocsparse_index_base_ hipBaseToHCCBase(hipsparseIndexBase_t base)
{
    switch(base)
    {
    case HIPSPARSE_INDEX_BASE_ZERO: return rocsparse_index_base_zero;
    case HIPSPARSE_INDEX_BASE_ONE: return rocsparse_index_base_one;
    default: throw "Non existent hipsparseIndexBase_t";
    }
}

hipsparseIndexBase_t HCCBaseToHIPBase(rocsparse_index_base_ base)
{
    switch(base)
    {
    case rocsparse_index_base_zero: return HIPSPARSE_INDEX_BASE_ZERO;
    case rocsparse_index_base_one: return HIPSPARSE_INDEX_BASE_ONE;
    default: throw "Non existent rocsparse_index_base_";
    }
}

rocsparse_operation_ hipOperationToHCCOperation(hipsparseOperation_t op)
{
    switch(op)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE: return rocsparse_operation_none;
    case HIPSPARSE_OPERATION_TRANSPOSE: return rocsparse_operation_transpose;
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE: return rocsparse_operation_conjugate_transpose;
    default: throw "Non existent hipsparseOperation_t";
    }
}

hipsparseOperation_t HCCOperationToHIPOperation(rocsparse_operation_ op)
{
    switch(op)
    {
    case rocsparse_operation_none: return HIPSPARSE_OPERATION_NON_TRANSPOSE;
    case rocsparse_operation_transpose: return HIPSPARSE_OPERATION_TRANSPOSE;
    case rocsparse_operation_conjugate_transpose: return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default: throw "Non existent rocsparse_operation_";
    }
}

rocsparse_hyb_partition_ hipHybPartToHCCHybPart(hipsparseHybPartition_t partition)
{
    switch(partition)
    {
    case HIPSPARSE_HYB_PARTITION_AUTO: return rocsparse_hyb_partition_auto;
    case HIPSPARSE_HYB_PARTITION_USER: return rocsparse_hyb_partition_user;
    case HIPSPARSE_HYB_PARTITION_MAX: return rocsparse_hyb_partition_max;
    default: throw "Non existent hipsparseHybPartition_t";
    }
}

hipsparseHybPartition_t HCCHybPartToHIPHybPart(rocsparse_hyb_partition_ partition)
{
    switch(partition)
    {
    case rocsparse_hyb_partition_auto: return HIPSPARSE_HYB_PARTITION_AUTO;
    case rocsparse_hyb_partition_user: return HIPSPARSE_HYB_PARTITION_USER;
    case rocsparse_hyb_partition_max: return HIPSPARSE_HYB_PARTITION_MAX;
    default: throw "Non existent rocsparse_hyb_partition_";
    }
}

// TODO side

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle)
{
    int deviceId;
    hipError_t err;
    hipsparseStatus_t retval = HIPSPARSE_STATUS_SUCCESS;

    if(handle == nullptr)
    {
        handle = (hipsparseHandle_t*)new rocsparse_handle();
    }

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = rocSPARSEStatusToHIPStatus(rocsparse_create_handle((rocsparse_handle*)handle));
    }
    return retval;
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_handle((rocsparse_handle)handle));
}

hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_get_version((rocsparse_handle)handle, version));
}

hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_set_stream((rocsparse_handle)handle, streamId));
}

hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_get_stream((rocsparse_handle)handle, streamId));
}

hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_set_pointer_mode((rocsparse_handle)handle, hipPtrModeToHCCPtrMode(mode)));
}

hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode)
{
    rocsparse_pointer_mode_ rocsparse_mode;
    rocsparse_status status = rocsparse_get_pointer_mode((rocsparse_handle)handle, &rocsparse_mode);
    *mode                   = HCCPtrModeToHIPPtrMode(rocsparse_mode);
    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_descr((rocsparse_mat_descr*)descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_descr((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src)
{
    // TODO    return rocSPARSEStatusToHIPStatus(
    //        rocsparse_copy_mat_descr((rocsparse_mat_descr) dest,
    //                                 (const rocsparse_mat_descr) src));
    return HIPSPARSE_STATUS_INTERNAL_ERROR;
}

hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_set_mat_type((rocsparse_mat_descr)descrA, hipMatTypeToHCCMatType(type)));
}

hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA)
{
    return HCCMatTypeToHIPMatType(rocsparse_get_mat_type((rocsparse_mat_descr)descrA));
}

// TODO fillmode
// TODO diagtype

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_set_mat_index_base((rocsparse_mat_descr)descrA, hipBaseToHCCBase(base)));
}

hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA)
{
    return HCCBaseToHIPBase(rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_hyb_mat((rocsparse_hyb_mat*)hybA));
}

hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_hyb_mat((rocsparse_hyb_mat)hybA));
}

hipsparseStatus_t hipsparseSaxpyi(hipsparseHandle_t handle,
                                  int nnz,
                                  const float* alpha,
                                  const float* xVal,
                                  const int* xInd,
                                  float* y,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_saxpyi(
        (rocsparse_handle)handle, nnz, alpha, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDaxpyi(hipsparseHandle_t handle,
                                  int nnz,
                                  const double* alpha,
                                  const double* xVal,
                                  const int* xInd,
                                  double* y,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_daxpyi(
        (rocsparse_handle)handle, nnz, alpha, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t handle,
                                 int nnz,
                                 const float* xVal,
                                 const int* xInd,
                                 const float* y,
                                 float* result,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sdoti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, result, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t handle,
                                 int nnz,
                                 const double* xVal,
                                 const int* xInd,
                                 const double* y,
                                 double* result,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_ddoti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, result, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSgthr(hipsparseHandle_t handle,
                                 int nnz,
                                 const float* y,
                                 float* xVal,
                                 const int* xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sgthr((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDgthr(hipsparseHandle_t handle,
                                 int nnz,
                                 const double* y,
                                 double* xVal,
                                 const int* xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dgthr((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSgthrz(hipsparseHandle_t handle,
                                  int nnz,
                                  float* y,
                                  float* xVal,
                                  const int* xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sgthrz((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDgthrz(hipsparseHandle_t handle,
                                  int nnz,
                                  double* y,
                                  double* xVal,
                                  const int* xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dgthrz((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSroti(hipsparseHandle_t handle,
                                 int nnz,
                                 float* xVal,
                                 const int* xInd,
                                 float* y,
                                 const float* c,
                                 const float* s,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sroti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, c, s, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDroti(hipsparseHandle_t handle,
                                 int nnz,
                                 double* xVal,
                                 const int* xInd,
                                 double* y,
                                 const double* c,
                                 const double* s,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_droti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, c, s, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSsctr(hipsparseHandle_t handle,
                                 int nnz,
                                 const float* xVal,
                                 const int* xInd,
                                 float* y,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ssctr((rocsparse_handle)handle, nnz, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDsctr(hipsparseHandle_t handle,
                                 int nnz,
                                 const double* xVal,
                                 const int* xInd,
                                 double* y,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dsctr((rocsparse_handle)handle, nnz, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
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
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
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
                                  const float* cooSortedValA,
                                  const int* cooSortedRowIndA,
                                  const int* cooSortedColIndA,
                                  const float* x,
                                  const float* beta,
                                  float* y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scoomv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       cooSortedValA,
                                                       cooSortedRowIndA,
                                                       cooSortedColIndA,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseDcoomv(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int nnz,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double* cooSortedValA,
                                  const int* cooSortedRowIndA,
                                  const int* cooSortedColIndA,
                                  const double* x,
                                  const double* beta,
                                  double* y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcoomv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       cooSortedValA,
                                                       cooSortedRowIndA,
                                                       cooSortedColIndA,
                                                       x,
                                                       beta,
                                                       y));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_sellmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       ellValA,
                                                       ellColIndA,
                                                       ellWidth,
                                                       x,
                                                       beta,
                                                       y));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_dellmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       ellValA,
                                                       ellColIndA,
                                                       ellWidth,
                                                       x,
                                                       beta,
                                                       y));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_shybmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (rocsparse_hyb_mat)hybA,
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
    return rocSPARSEStatusToHIPStatus(rocsparse_dhybmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (rocsparse_hyb_mat)hybA,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseScsrmm(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int k,
                                  int nnz,
                                  const float* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float* csrSortedValA,
                                  const int* csrSortedRowPtrA,
                                  const int* csrSortedColIndA,
                                  const float* B,
                                  int ldb,
                                  const float* beta,
                                  float* C,
                                  int ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       rocsparse_operation_none,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t handle,
                                  hipsparseOperation_t transA,
                                  int m,
                                  int n,
                                  int k,
                                  int nnz,
                                  const double* alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double* csrSortedValA,
                                  const int* csrSortedRowPtrA,
                                  const int* csrSortedColIndA,
                                  const double* B,
                                  int ldb,
                                  const double* beta,
                                  double* C,
                                  int ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       rocsparse_operation_none,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseScsrmm2(hipsparseHandle_t handle,
                                   hipsparseOperation_t transA,
                                   hipsparseOperation_t transB,
                                   int m,
                                   int n,
                                   int k,
                                   int nnz,
                                   const float* alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const float* csrSortedValA,
                                   const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA,
                                   const float* B,
                                   int ldb,
                                   const float* beta,
                                   float* C,
                                   int ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseDcsrmm2(hipsparseHandle_t handle,
                                   hipsparseOperation_t transA,
                                   hipsparseOperation_t transB,
                                   int m,
                                   int n,
                                   int k,
                                   int nnz,
                                   const double* alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const double* csrSortedValA,
                                   const int* csrSortedRowPtrA,
                                   const int* csrSortedColIndA,
                                   const double* B,
                                   int ldb,
                                   const double* beta,
                                   double* C,
                                   int ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t handle,
                                    const int* csrRowPtr,
                                    int nnz,
                                    int m,
                                    int* cooRowInd,
                                    hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csr2coo(
        (rocsparse_handle)handle, csrRowPtr, nnz, m, cooRowInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const float* csrSortedVal,
                                    const int* csrSortedRowPtr,
                                    const int* csrSortedColInd,
                                    float* cscSortedVal,
                                    int* cscSortedRowInd,
                                    int* cscSortedColPtr,
                                    hipsparseAction_t copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            csrSortedRowPtr,
                                                            csrSortedColInd,
                                                            hipActionToHCCAction(copyValues),
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsr2csc((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 nnz,
                                                 csrSortedVal,
                                                 csrSortedRowPtr,
                                                 csrSortedColInd,
                                                 cscSortedVal,
                                                 cscSortedRowInd,
                                                 cscSortedColPtr,
                                                 hipActionToHCCAction(copyValues),
                                                 hipBaseToHCCBase(idxBase),
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const double* csrSortedVal,
                                    const int* csrSortedRowPtr,
                                    const int* csrSortedColInd,
                                    double* cscSortedVal,
                                    int* cscSortedRowInd,
                                    int* cscSortedColPtr,
                                    hipsparseAction_t copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            csrSortedRowPtr,
                                                            csrSortedColInd,
                                                            hipActionToHCCAction(copyValues),
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsr2csc((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 nnz,
                                                 csrSortedVal,
                                                 csrSortedRowPtr,
                                                 csrSortedColInd,
                                                 cscSortedVal,
                                                 cscSortedRowInd,
                                                 cscSortedColPtr,
                                                 hipActionToHCCAction(copyValues),
                                                 hipBaseToHCCBase(idxBase),
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcsr2ellWidth(hipsparseHandle_t handle,
                                         int m,
                                         const hipsparseMatDescr_t descrA,
                                         const int* csrRowPtrA,
                                         const hipsparseMatDescr_t descrC,
                                         int* ellWidthC)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csr2ell_width((rocsparse_handle)handle,
                                                              m,
                                                              (rocsparse_mat_descr)descrA,
                                                              csrRowPtrA,
                                                              (rocsparse_mat_descr)descrC,
                                                              ellWidthC));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_scsr2ell((rocsparse_handle)handle,
                                                         m,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (rocsparse_mat_descr)descrC,
                                                         ellWidthC,
                                                         ellValC,
                                                         ellColIndC));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsr2ell((rocsparse_handle)handle,
                                                         m,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (rocsparse_mat_descr)descrC,
                                                         ellWidthC,
                                                         ellValC,
                                                         ellColIndC));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_scsr2hyb((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_hyb_mat)hybA,
                                                         userEllWidth,
                                                         hipHybPartToHCCHybPart(partitionType)));
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
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsr2hyb((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_hyb_mat)hybA,
                                                         userEllWidth,
                                                         hipHybPartToHCCHybPart(partitionType)));
}

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t handle,
                                    const int* cooRowInd,
                                    int nnz,
                                    int m,
                                    int* csrRowPtr,
                                    hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coo2csr(
        (rocsparse_handle)handle, cooRowInd, nnz, m, csrRowPtr, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseCreateIdentityPermutation(hipsparseHandle_t handle, int n, int* p)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_identity_permutation((rocsparse_handle)handle, n, p));
}

hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int m,
                                                  int n,
                                                  int nnz,
                                                  const int* csrRowPtr,
                                                  const int* csrColInd,
                                                  size_t* pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csrsort_buffer_size(
        (rocsparse_handle)handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcsrsort(hipsparseHandle_t handle,
                                    int m,
                                    int n,
                                    int nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int* csrRowPtr,
                                    int* csrColInd,
                                    int* P,
                                    void* pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csrsort((rocsparse_handle)handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        (rocsparse_mat_descr)descrA,
                                                        csrRowPtr,
                                                        csrColInd,
                                                        P,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int m,
                                                  int n,
                                                  int nnz,
                                                  const int* cooRows,
                                                  const int* cooCols,
                                                  size_t* pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coosort_buffer_size(
        (rocsparse_handle)handle, m, n, nnz, cooRows, cooCols, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcoosortByRow(hipsparseHandle_t handle,
                                         int m,
                                         int n,
                                         int nnz,
                                         int* cooRows,
                                         int* cooCols,
                                         int* P,
                                         void* pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coosort_by_row(
        (rocsparse_handle)handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}

hipsparseStatus_t hipsparseXcoosortByColumn(hipsparseHandle_t handle,
                                            int m,
                                            int n,
                                            int nnz,
                                            int* cooRows,
                                            int* cooCols,
                                            int* P,
                                            void* pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coosort_by_column(
        (rocsparse_handle)handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}

#ifdef __cplusplus
}
#endif
