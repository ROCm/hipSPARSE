!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Copyright (c) 2020 Advanced Micro Devices, Inc.
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in
! all copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
! THE SOFTWARE.
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module hipsparse
    use iso_c_binding

! ===========================================================================
!   types SPARSE
! ===========================================================================

!   hipsparseStatus_t
    enum, bind(c)
        enumerator :: HIPSPARSE_STATUS_SUCCESS = 0
        enumerator :: HIPSPARSE_STATUS_NOT_INITIALIZED = 1
        enumerator :: HIPSPARSE_STATUS_ALLOC_FAILED = 2
        enumerator :: HIPSPARSE_STATUS_INVALID_VALUE = 3
        enumerator :: HIPSPARSE_STATUS_ARCH_MISMATCH = 4
        enumerator :: HIPSPARSE_STATUS_MAPPING_ERROR = 5
        enumerator :: HIPSPARSE_STATUS_EXECUTION_FAILED = 6
        enumerator :: HIPSPARSE_STATUS_INTERNAL_ERROR = 7
        enumerator :: HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8
        enumerator :: HIPSPARSE_STATUS_ZERO_PIVOT = 9
    end enum

!   hipsparsePointerMode_t
    enum, bind(c)
        enumerator :: HIPSPARSE_POINTER_MODE_HOST = 0
        enumerator :: HIPSPARSE_POINTER_MODE_DEVICE = 1
    end enum

!   hipsparseAction_t
    enum, bind(c)
        enumerator :: HIPSPARSE_ACTION_SYMBOLIC = 0
        enumerator :: HIPSPARSE_ACTION_NUMERIC = 1
    end enum

!   hipsparseMatrixType_t
    enum, bind(c)
        enumerator :: HIPSPARSE_MATRIX_TYPE_GENERAL = 0
        enumerator :: HIPSPARSE_MATRIX_TYPE_SYMMETRIC = 1
        enumerator :: HIPSPARSE_MATRIX_TYPE_HERMITIAN = 2
        enumerator :: HIPSPARSE_MATRIX_TYPE_TRIANGULAR = 3
    end enum

!   hipsparseFillMode_t
    enum, bind(c)
        enumerator :: HIPSPARSE_FILL_MODE_LOWER = 0
        enumerator :: HIPSPARSE_FILL_MODE_UPPER = 1
    end enum

!   hipsparseDiagType_t
    enum, bind(c)
        enumerator :: HIPSPARSE_DIAG_TYPE_NON_UNIT = 0
        enumerator :: HIPSPARSE_DIAG_TYPE_UNIT = 1
    end enum

!   hipsparseIndexBase_t
    enum, bind(c)
        enumerator :: HIPSPARSE_INDEX_BASE_ZERO = 0
        enumerator :: HIPSPARSE_INDEX_BASE_ONE = 1
    end enum

!   hipsparseOperation_t
    enum, bind(c)
        enumerator :: HIPSPARSE_OPERATION_NON_TRANSPOSE = 0
        enumerator :: HIPSPARSE_OPERATION_TRANSPOSE = 1
        enumerator :: HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
    end enum

!   hipsparseHybPartition_t
    enum, bind(c)
        enumerator :: HIPSPARSE_HYB_PARTITION_AUTO = 0
        enumerator :: HIPSPARSE_HYB_PARTITION_USER = 1
        enumerator :: HIPSPARSE_HYB_PARTITION_MAX = 2
    end enum

!   hipsparseSolvePolicy_t
    enum, bind(c)
        enumerator :: HIPSPARSE_SOLVE_POLICY_NO_LEVEL = 0
        enumerator :: HIPSPARSE_SOLVE_POLICY_USE_LEVEL = 1
    end enum

!   hipsparseSideMode_t
    enum, bind(c)
        enumerator :: HIPSPARSE_SIDE_LEFT = 0
        enumerator :: HIPSPARSE_SIDE_RIGHT = 1
    end enum

!   hipsparseDirection_t
    enum, bind(c)
        enumerator :: HIPSPARSE_DIRECTION_ROW = 0
        enumerator :: HIPSPARSE_DIRECTION_COLUMN = 1
    end enum

! ===========================================================================
!   auxiliary SPARSE
! ===========================================================================

    interface

!       hipsparseHandle_t
        function hipsparseCreate(handle) &
                result(c_int) &
                bind(c, name = 'hipsparseCreate')
            use iso_c_binding
            implicit none
            type(c_ptr) :: handle
        end function hipsparseCreate

        function hipsparseDestroy(handle) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroy')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
        end function hipsparseDestroy

!       hipsparseVersion
        function hipsparseGetVersion(handle, version) &
                result(c_int) &
                bind(c, name = 'hipsparseGetVersion')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int) :: version
        end function hipsparseGetVersion

!       hipsparseStream
        function hipsparseSetStream(handle, stream) &
                result(c_int) &
                bind(c, name = 'hipsparseSetStream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: stream
        end function hipsparseSetStream

        function hipsparseGetStream(handle, stream) &
                result(c_int) &
                bind(c, name = 'hipsparseGetStream')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr) :: stream
        end function hipsparseGetStream

!       hipsparsePointerMode_t
        function hipsparseSetPointerMode(handle, mode) &
                result(c_int) &
                bind(c, name = 'hipsparseSetPointerMode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: mode
        end function hipsparseSetPointerMode

        function hipsparseGetPointerMode(handle, mode) &
                result(c_int) &
                bind(c, name = 'hipsparseGetPointerMode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int) :: mode
        end function hipsparseGetPointerMode

!       hipsparseMatDescr_t
        function hipsparseCreateMatDescr(descr) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateMatDescr')
            use iso_c_binding
            implicit none
            type(c_ptr) :: descr
        end function hipsparseCreateMatDescr

        function hipsparseDestroyMatDescr(descr) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroyMatDescr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
        end function hipsparseDestroyMatDescr

        function hipsparseCopyMatDescr(dest, src) &
                result(c_int) &
                bind(c, name = 'hipsparseCopyMatDescr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: dest
            type(c_ptr), intent(in), value :: src
        end function hipsparseCopyMatDescr

!       hipsparseMatrixType_t
        function hipsparseSetMatType(descr, mtype) &
                result(c_int) &
                bind(c, name = 'hipsparseSetMatType')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: mtype
        end function hipsparseSetMatType

        function hipsparseGetMatType(descr) &
                result(c_int) &
                bind(c, name = 'hipsparseGetMatType')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatType

!       hipsparseFillMode_t
        function hipsparseSetMatFillMode(descr, fill) &
                result(c_int) &
                bind(c, name = 'hipsparseSetMatFillMode')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: fill
        end function hipsparseSetMatFillMode

        function hipsparseGetMatFillMode(descr) &
                result(c_int) &
                bind(c, name = 'hipsparseGetMatFillMode')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatFillMode

!       hipsparseDiagType_t
        function hipsparseSetMatDiagType(descr, diag) &
                result(c_int) &
                bind(c, name = 'hipsparseSetMatDiagType')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: diag
        end function hipsparseSetMatDiagType

        function hipsparseGetMatDiagType(descr) &
                result(c_int) &
                bind(c, name = 'hipsparseGetMatDiagType')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatDiagType

!       hipsparseIndexBase_t
        function hipsparseSetMatIndexBase(descr, base) &
                result(c_int) &
                bind(c, name = 'hipsparseSetMatIndexBase')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: descr
            integer(c_int), value :: base
        end function hipsparseSetMatIndexBase

        function hipsparseGetMatIndexBase(descr) &
                result(c_int) &
                bind(c, name = 'hipsparseGetMatIndexBase')
            use iso_c_binding
            implicit none
            type(c_ptr), intent(in), value :: descr
        end function hipsparseGetMatIndexBase

!       hipsparseHybMat_t
        function hipsparseCreateHybMat(hyb) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateHybMat')
            use iso_c_binding
            implicit none
            type(c_ptr) :: hyb
        end function hipsparseCreateHybMat

        function hipsparseDestroyHybMat(hyb) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroyHybMat')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: hyb
        end function hipsparseDestroyHybMat

!       csrsv2Info_t
        function hipsparseCreateCsrsv2Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateCsrsv2Info')
            use iso_c_binding
            implicit none
            type(c_ptr) :: info
        end function hipsparseCreateCsrsv2Info

        function hipsparseDestroyCsrsv2Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroyCsrsv2Info')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrsv2Info

!       csrsm2Info_t
        function hipsparseCreateCsrsm2Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateCsrsm2Info')
            use iso_c_binding
            implicit none
            type(c_ptr) :: info
        end function hipsparseCreateCsrsm2Info

        function hipsparseDestroyCsrsm2Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroyCsrsm2Info')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrsm2Info

!       csrilu02Info_t
        function hipsparseCreateCsrilu02Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateCsrilu02Info')
            use iso_c_binding
            implicit none
            type(c_ptr) :: info
        end function hipsparseCreateCsrilu02Info

        function hipsparseDestroyCsrilu02Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroyCsrilu02Info')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrilu02Info

!       csric02Info_t
        function hipsparseCreateCsric02Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateCsric02Info')
            use iso_c_binding
            implicit none
            type(c_ptr) :: info
        end function hipsparseCreateCsric02Info

        function hipsparseDestroyCsric02Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroyCsric02Info')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: info
        end function hipsparseDestroyCsric02Info

!       csrgemm2Info_t
        function hipsparseCreateCsrgemm2Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateCsrgemm2Info')
            use iso_c_binding
            implicit none
            type(c_ptr) :: info
        end function hipsparseCreateCsrgemm2Info

        function hipsparseDestroyCsrgemm2Info(info) &
                result(c_int) &
                bind(c, name = 'hipsparseDestroyCsrgemm2Info')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: info
        end function hipsparseDestroyCsrgemm2Info

! ===========================================================================
!   level 1 SPARSE
! ===========================================================================

!       hipsparseXaxpyi
        function hipsparseSaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseSaxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseSaxpyi

        function hipsparseDaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseDaxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseDaxpyi

        function hipsparseCaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseCaxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseCaxpyi

        function hipsparseZaxpyi(handle, nnz, alpha, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseZaxpyi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseZaxpyi

!       hipsparseXdoti
        function hipsparseSdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseSdoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseSdoti

        function hipsparseDdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseDdoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseDdoti

        function hipsparseCdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseCdoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseCdoti

        function hipsparseZdoti(handle, nnz, xVal, xInd, y, result, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseZdoti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseZdoti

!       hipsparseXdotci
        function hipsparseCdotci(handle, nnz, xVal, xInd, y, result, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseCdotci')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseCdotci

        function hipsparseZdotci(handle, nnz, xVal, xInd, y, result, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseZdotci')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: result
            integer(c_int), value :: idxBase
        end function hipsparseZdotci

!       hipsparseXgthr
        function hipsparseSgthr(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseSgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseSgthr

        function hipsparseDgthr(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseDgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseDgthr

        function hipsparseCgthr(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseCgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseCgthr

        function hipsparseZgthr(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseZgthr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseZgthr

!       hipsparseXgthrz
        function hipsparseSgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseSgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseSgthrz

        function hipsparseDgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseDgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseDgthrz

        function hipsparseCgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseCgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseCgthrz

        function hipsparseZgthrz(handle, nnz, y, xVal, xInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseZgthrz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: y
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            integer(c_int), value :: idxBase
        end function hipsparseZgthrz

!       hipsparseXroti
        function hipsparseSroti(handle, nnz, xVal, xInd, y, c, s, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseSroti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            type(c_ptr), intent(in), value :: c
            type(c_ptr), intent(in), value :: s
            integer(c_int), value :: idxBase
        end function hipsparseSroti

        function hipsparseDroti(handle, nnz, xVal, xInd, y, c, s, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseDroti')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            type(c_ptr), intent(in), value :: c
            type(c_ptr), intent(in), value :: s
            integer(c_int), value :: idxBase
        end function hipsparseDroti

!       hipsparseXsctr
        function hipsparseSsctr(handle, nnz, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseSsctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseSsctr

        function hipsparseDsctr(handle, nnz, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseDsctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseDsctr

        function hipsparseCsctr(handle, nnz, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseCsctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseCsctr

        function hipsparseZsctr(handle, nnz, xVal, xInd, y, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseZsctr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: xVal
            type(c_ptr), intent(in), value :: xInd
            type(c_ptr), value :: y
            integer(c_int), value :: idxBase
        end function hipsparseZsctr

! ===========================================================================
!   level 2 SPARSE
! ===========================================================================

!       hipsparseXcsrmv
        function hipsparseScsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseScsrmv

        function hipsparseDcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseDcsrmv

        function hipsparseCcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseCcsrmv

        function hipsparseZcsrmv(handle, trans, m, n, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseZcsrmv

!       hipsparseXcsrsv2_zeroPivot
        function hipsparseXcsrsv2_zeroPivot(handle, info, position) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrsv2_zeroPivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrsv2_zeroPivot

!       hipsparseXcsrsv2_bufferSize
        function hipsparseScsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrsv2_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrsv2_bufferSize

        function hipsparseDcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrsv2_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrsv2_bufferSize

        function hipsparseCcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrsv2_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrsv2_bufferSize

        function hipsparseZcsrsv2_bufferSize(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrsv2_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrsv2_bufferSize

!       hipsparseXcsrsv2_bufferSizeExt
        function hipsparseScsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrsv2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrsv2_bufferSizeExt

        function hipsparseDcsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrsv2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrsv2_bufferSizeExt

        function hipsparseCcsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrsv2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrsv2_bufferSizeExt

        function hipsparseZcsrsv2_bufferSizeExt(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrsv2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrsv2_bufferSizeExt

!       hipsparseXcsrsv2_analysis
        function hipsparseScsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrsv2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsv2_analysis

        function hipsparseDcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrsv2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsv2_analysis

        function hipsparseCcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrsv2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsv2_analysis

        function hipsparseZcsrsv2_analysis(handle, trans, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrsv2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsv2_analysis

!       hipsparseXcsrsv2_solve
        function hipsparseScsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrsv2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsv2_solve

        function hipsparseDcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrsv2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsv2_solve

        function hipsparseCcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrsv2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsv2_solve

        function hipsparseZcsrsv2_solve(handle, trans, m, nnz, alpha, descr, csrVal, &
                csrRowPtr, csrColInd, info, x, y, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrsv2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), intent(in), value :: x
            type(c_ptr), value :: y
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsv2_solve

!       hipsparseXhybmv
        function hipsparseShybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseShybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseShybmv

        function hipsparseDhybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseDhybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseDhybmv

        function hipsparseChybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseChybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseChybmv

        function hipsparseZhybmv(handle, trans, alpha, descr, hyb, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseZhybmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: trans
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: hyb
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseZhybmv

!       hipsparseXbsrmv
        function hipsparseSbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseSbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseSbsrmv

        function hipsparseDbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseDbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseDbsrmv

        function hipsparseCbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseCbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseCbsrmv

        function hipsparseZbsrmv(handle, dir, trans, mb, nb, nnzb, alpha, descr, &
                bsrVal, bsrRowPtr, bsrColInd, blockDim, x, beta, y) &
                result(c_int) &
                bind(c, name = 'hipsparseZbsrmv')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: trans
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            integer(c_int), value :: nnzb
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: bsrVal
            type(c_ptr), intent(in), value :: bsrRowPtr
            type(c_ptr), intent(in), value :: bsrColInd
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: x
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: y
        end function hipsparseZbsrmv

! ===========================================================================
!   level 3 SPARSE
! ===========================================================================

!       hipsparseXcsrmm
        function hipsparseScsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseScsrmm

        function hipsparseDcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseDcsrmm

        function hipsparseCcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseCcsrmm

        function hipsparseZcsrmm(handle, transA, m, n, k, nnz, alpha, descrA, csrValA, &
                csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrmm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseZcsrmm

!       hipsparseXcsrmm2
        function hipsparseScsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrmm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseScsrmm2

        function hipsparseDcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrmm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseDcsrmm2

        function hipsparseCcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrmm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseCcsrmm2

        function hipsparseZcsrmm2(handle, transA, transB, m, n, k, nnz, alpha, descrA, &
                csrValA, csrRowPtrA, csrColIndA, B, ldb, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrmm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseZcsrmm2

!       hipsparseXcsrsm2_zeroPivot
        function hipsparseXcsrsm2_zeroPivot(handle, info, position) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrsm2_zeroPivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrsm2_zeroPivot

!       hipsparseXcsrsm2_bufferSizeExt
        function hipsparseScsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrsm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrsm2_bufferSizeExt

        function hipsparseDcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrsm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrsm2_bufferSizeExt

        function hipsparseCcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrsm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrsm2_bufferSizeExt

        function hipsparseZcsrsm2_bufferSizeExt(handle, algo, transA, transB, m, nrhs, &
                nnz, alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, &
                policy, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrsm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrsm2_bufferSizeExt

!       hipsparseXcsrsm2_analysis
        function hipsparseScsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrsm2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsm2_analysis

        function hipsparseDcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrsm2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsm2_analysis

        function hipsparseCcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrsm2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsm2_analysis

        function hipsparseZcsrsm2_analysis(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrsm2_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsm2_analysis

!       hipsparseXcsrsm2_solve
        function hipsparseScsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrsm2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrsm2_solve

        function hipsparseDcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrsm2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrsm2_solve

        function hipsparseCcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrsm2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrsm2_solve

        function hipsparseZcsrsm2_solve(handle, algo, transA, transB, m, nrhs, nnz, &
                alpha, descrA, csrValA, csrRowPtrA, csrColIndA, B, ldb, info, policy, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrsm2_solve')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: algo
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: nrhs
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: B
            integer(c_int), value :: ldb
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrsm2_solve

!       hipsparseXgemmi
        function hipsparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseSgemmi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: cscValB
            type(c_ptr), intent(in), value :: cscColPtrB
            type(c_ptr), intent(in), value :: cscRowIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseSgemmi

        function hipsparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseDgemmi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: cscValB
            type(c_ptr), intent(in), value :: cscColPtrB
            type(c_ptr), intent(in), value :: cscRowIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseDgemmi

        function hipsparseCgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseCgemmi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: cscValB
            type(c_ptr), intent(in), value :: cscColPtrB
            type(c_ptr), intent(in), value :: cscRowIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseCgemmi

        function hipsparseZgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB, &
                cscColPtrB, cscRowIndB, beta, C, ldc) &
                result(c_int) &
                bind(c, name = 'hipsparseZgemmi')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: cscValB
            type(c_ptr), intent(in), value :: cscColPtrB
            type(c_ptr), intent(in), value :: cscRowIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), value :: C
            integer(c_int), value :: ldc
        end function hipsparseZgemmi

! ===========================================================================
!   extra SPARSE
! ===========================================================================

!       hipsparseXcsrgeamNnz
        function hipsparseXcsrgeamNnz(handle, m, n, descrA, nnzA, csrRowPtrA, &
                csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, &
                csrRowPtrC, nnzTotalDevHostPtr) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrgeamNnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseXcsrgeamNnz

!       hipsparseXcsrgeam
        function hipsparseScsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseScsrgeam

        function hipsparseDcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseDcsrgeam

        function hipsparseCcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseCcsrgeam

        function hipsparseZcsrgeam(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrgeam')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseZcsrgeam

!       hipsparseXcsrgeam2_bufferSizeExt
        function hipsparseScsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrgeam2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrgeam2_bufferSizeExt

        function hipsparseDcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrgeam2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrgeam2_bufferSizeExt

        function hipsparseCcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrgeam2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrgeam2_bufferSizeExt

        function hipsparseZcsrgeam2_bufferSizeExt(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrgeam2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), intent(in), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), intent(in), value :: csrColIndC
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrgeam2_bufferSizeExt

!       hipsparseXcsrgeam2Nnz
        function hipsparseXcsrgeam2Nnz(handle, m, n, descrA, nnzA, csrRowPtrA, &
                csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, &
                csrRowPtrC, nnzTotalDevHostPtr, workspace) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrgeam2Nnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
            type(c_ptr), value :: workspace
        end function hipsparseXcsrgeam2Nnz

!       hipsparseXcsrgeam2
        function hipsparseScsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrgeam2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseScsrgeam2

        function hipsparseDcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrgeam2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseDcsrgeam2

        function hipsparseCcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrgeam2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseCcsrgeam2

        function hipsparseZcsrgeam2(handle, m, n, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, beta, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrgeam2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: buffer
        end function hipsparseZcsrgeam2

!       hipsparseXcsrgemmNnz
        function hipsparseXcsrgemmNnz(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrC, &
                csrRowPtrC, nnzTotalDevHostPtr) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrgemmNnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseXcsrgemmNnz

!       hipsparseScsrgemm
        function hipsparseScsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseScsrgemm

        function hipsparseDcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseDcsrgemm

        function hipsparseCcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseCcsrgemm

        function hipsparseZcsrgemm(handle, transA, transB, m, n, k, descrA, nnzA, &
                csrValA, csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, &
                csrColIndB, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrgemm')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: transA
            integer(c_int), value :: transB
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseZcsrgemm

!       hipsparseXcsrgemm2_bufferSizeExt
        function hipsparseScsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrgemm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrgemm2_bufferSizeExt

        function hipsparseDcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrgemm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrgemm2_bufferSizeExt

        function hipsparseCcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrgemm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrgemm2_bufferSizeExt

        function hipsparseZcsrgemm2_bufferSizeExt(handle, m, n, k, alpha, descrA, &
                nnzA, csrRowPtrA, csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrRowPtrD, csrColIndD, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrgemm2_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrgemm2_bufferSizeExt

!       hipsparseXcsrgemm2Nnz
        function hipsparseXcsrgemm2Nnz(handle, m, n, k, descrA, nnzA, csrRowPtrA, &
                csrColIndA, descrB, nnzB, csrRowPtrB, csrColIndB, descrD, nnzD, &
                csrRowPtrD, csrColIndD, descrC, csrRowPtrC, nnzTotalDevHostPtr, info, &
                buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrgemm2Nnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: nnzTotalDevHostPtr
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseXcsrgemm2Nnz

!       hipsparseXcsrgemm2
        function hipsparseScsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrgemm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseScsrgemm2

        function hipsparseDcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrgemm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseDcsrgemm2

        function hipsparseCcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrgemm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseCcsrgemm2

        function hipsparseZcsrgemm2(handle, m, n, k, alpha, descrA, nnzA, csrValA, &
                csrRowPtrA, csrColIndA, descrB, nnzB, csrValB, csrRowPtrB, csrColIndB, &
                beta, descrD, nnzD, csrValD, csrRowPtrD, csrColIndD, descrC, csrValC, &
                csrRowPtrC, csrColIndC, info, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrgemm2')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: k
            type(c_ptr), intent(in), value :: alpha
            type(c_ptr), intent(in), value :: descrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: descrB
            integer(c_int), value :: nnzB
            type(c_ptr), intent(in), value :: csrValB
            type(c_ptr), intent(in), value :: csrRowPtrB
            type(c_ptr), intent(in), value :: csrColIndB
            type(c_ptr), intent(in), value :: beta
            type(c_ptr), intent(in), value :: descrD
            integer(c_int), value :: nnzD
            type(c_ptr), intent(in), value :: csrValD
            type(c_ptr), intent(in), value :: csrRowPtrD
            type(c_ptr), intent(in), value :: csrColIndD
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), intent(in), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), intent(in), value :: info
            type(c_ptr), value :: buffer
        end function hipsparseZcsrgemm2

! ===========================================================================
!   preconditioner SPARSE
! ===========================================================================

!       hipsparseXcsrilu02_zeroPivot
        function hipsparseXcsrilu02_zeroPivot(handle, info, position) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrilu02_zeroPivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsrilu02_zeroPivot

!       hipsparseXcsrilu02_bufferSize
        function hipsparseScsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrilu02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrilu02_bufferSize

        function hipsparseDcsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrilu02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrilu02_bufferSize

        function hipsparseCcsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrilu02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrilu02_bufferSize

        function hipsparseZcsrilu02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrilu02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrilu02_bufferSize

!       hipsparseXcsrilu02_bufferSizeExt
        function hipsparseScsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrilu02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsrilu02_bufferSizeExt

        function hipsparseDcsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrilu02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsrilu02_bufferSizeExt

        function hipsparseCcsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrilu02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsrilu02_bufferSizeExt

        function hipsparseZcsrilu02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrilu02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsrilu02_bufferSizeExt

!       hipsparseXcsrilu02_analysis
        function hipsparseScsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrilu02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrilu02_analysis

        function hipsparseDcsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrilu02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrilu02_analysis

        function hipsparseCcsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrilu02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrilu02_analysis

        function hipsparseZcsrilu02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrilu02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrilu02_analysis

!       hipsparseXcsrilu02
        function hipsparseScsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsrilu02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsrilu02

        function hipsparseDcsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsrilu02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsrilu02

        function hipsparseCcsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsrilu02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsrilu02

        function hipsparseZcsrilu02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsrilu02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsrilu02

!       hipsparseXcsric02_zeroPivot
        function hipsparseXcsric02_zeroPivot(handle, info, position) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsric02_zeroPivot')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), value :: info
            type(c_ptr), value :: position
        end function hipsparseXcsric02_zeroPivot

!       hipsparseXcsric02_bufferSize
        function hipsparseScsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsric02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsric02_bufferSize

        function hipsparseDcsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsric02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsric02_bufferSize

        function hipsparseCcsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsric02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsric02_bufferSize

        function hipsparseZcsric02_bufferSize(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsric02_bufferSize')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsric02_bufferSize

!       hipsparseXcsric02_bufferSizeExt
        function hipsparseScsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseScsric02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseScsric02_bufferSizeExt

        function hipsparseDcsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsric02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseDcsric02_bufferSizeExt

        function hipsparseCcsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsric02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseCcsric02_bufferSizeExt

        function hipsparseZcsric02_bufferSizeExt(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsric02_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            type(c_ptr), value :: bufferSize
        end function hipsparseZcsric02_bufferSizeExt

!       hipsparseXcsric02_analysis
        function hipsparseScsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsric02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsric02_analysis

        function hipsparseDcsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsric02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsric02_analysis

        function hipsparseCcsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsric02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsric02_analysis

        function hipsparseZcsric02_analysis(handle, m, nnz, descr, csrVal, &
                csrRowPtr, csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsric02_analysis')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsric02_analysis

!       hipsparseXcsric02
        function hipsparseScsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseScsric02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseScsric02

        function hipsparseDcsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsric02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseDcsric02

        function hipsparseCcsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsric02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseCcsric02

        function hipsparseZcsric02(handle, m, nnz, descr, csrVal, csrRowPtr, &
                csrColInd, info, policy, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsric02')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: info
            integer(c_int), value :: policy
            type(c_ptr), value :: buffer
        end function hipsparseZcsric02

! ===========================================================================
!   conversion SPARSE
! ===========================================================================

!       hipsparseXnnz
        function hipsparseSnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                result(c_int) &
                bind(c, name = 'hipsparseSnnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseSnnz

        function hipsparseDnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                result(c_int) &
                bind(c, name = 'hipsparseDnnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseDnnz

        function hipsparseCnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                result(c_int) &
                bind(c, name = 'hipsparseCnnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseCnnz

        function hipsparseZnnz(handle, dir, m, n, descrA, A, lda, &
                nnzPerRowColumn, nnzTotalDevHostPtr) &
                result(c_int) &
                bind(c, name = 'hipsparseZnnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dir
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), value :: nnzPerRowColumn
            type(c_ptr), value :: nnzTotalDevHostPtr
        end function hipsparseZnnz

!       hipsparseXdense2csr
        function hipsparseSdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseSdense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseSdense2csr

        function hipsparseDdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseDdense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseDdense2csr

        function hipsparseCdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseCdense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseCdense2csr

        function hipsparseZdense2csr(handle, m, n, descrA, A, lda, nnzPerRows, &
                csrValA, csrRowPtrA, csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseZdense2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerRows
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseZdense2csr

!       hipsparseXdense2csc
        function hipsparseSdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseSdense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseSdense2csc

        function hipsparseDdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseDdense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseDdense2csc

        function hipsparseCdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseCdense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseCdense2csc

        function hipsparseZdense2csc(handle, m, n, descrA, A, lda, nnzPerColumns, &
                cscValA, cscRowPtrA, cscColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseZdense2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: A
            integer(c_int), value :: lda
            type(c_ptr), intent(in), value :: nnzPerColumns
            type(c_ptr), value :: cscValA
            type(c_ptr), value :: cscRowPtrA
            type(c_ptr), value :: cscColIndA
        end function hipsparseZdense2csc

!       hipsparseXcsr2dense
        function hipsparseScsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseScsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseScsr2dense

        function hipsparseDcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseDcsr2dense

        function hipsparseCcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseCcsr2dense

        function hipsparseZcsr2dense(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsr2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseZcsr2dense

!       hipsparseXcsc2dense
        function hipsparseScsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseScsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseScsc2dense

        function hipsparseDcsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseDcsc2dense

        function hipsparseCcsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseCcsc2dense

        function hipsparseZcsc2dense(handle, m, n, descrA, cscValA, cscRowPtrA, &
                cscColIndA, A, lda) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsc2dense')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: cscValA
            type(c_ptr), intent(in), value :: cscRowPtrA
            type(c_ptr), intent(in), value :: cscColIndA
            type(c_ptr), value :: A
            integer(c_int), value :: lda
        end function hipsparseZcsc2dense

!       hipsparseXnnz_compress
        function hipsparseSnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseSnnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            real(c_float), value :: tol
        end function hipsparseSnnz_compress

        function hipsparseDnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseDnnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            real(c_double), value :: tol
        end function hipsparseDnnz_compress

        function hipsparseCnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseCnnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            complex(c_float_complex), value :: tol
        end function hipsparseCnnz_compress

        function hipsparseZnnz_compress(handle, m, descrA, csrValA, csrRowPtrA, &
                nnzPerRow, nnzC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseZnnz_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), value :: nnzPerRow
            type(c_ptr), value :: nnzC
            complex(c_double_complex), value :: tol
        end function hipsparseZnnz_compress

!       hipsparseXcsr2coo
        function hipsparseXcsr2coo(handle, csrRowPtr, nnz, m, cooRowInd, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsr2coo')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: csrRowPtr
            integer(c_int), value :: nnz
            integer(c_int), value :: m
            type(c_ptr), value :: cooRowInd
            integer(c_int), value :: idxBase
        end function hipsparseXcsr2coo

!       hipsparseXcsr2csc
        function hipsparseScsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseScsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseScsr2csc

        function hipsparseDcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseDcsr2csc

        function hipsparseCcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseCcsr2csc

        function hipsparseZcsr2csc(handle, m, n, nnz, csrVal, csrRowPtr, csrColInd, &
                cscVal, cscRowInd, cscColPtr, copyValues, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsr2csc')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrVal
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: cscVal
            type(c_ptr), value :: cscRowInd
            type(c_ptr), value :: cscColPtr
            integer(c_int), value :: copyValues
            integer(c_int), value :: idxBase
        end function hipsparseZcsr2csc

!       hipsparseXcsr2hyb
        function hipsparseScsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                result(c_int) &
                bind(c, name = 'hipsparseScsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseScsr2hyb

        function hipsparseDcsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseDcsr2hyb

        function hipsparseCcsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseCcsr2hyb

        function hipsparseZcsr2hyb(handle, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, hybA, userEllWidth, partitionType) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsr2hyb')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), value :: hybA
            integer(c_int), value :: userEllWidth
            integer(c_int), value :: partitionType
        end function hipsparseZcsr2hyb

!       hipsparseXcsr2bsrNnz
        function hipsparseXcsr2bsrNnz(handle, dirA, m, n, descrA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrRowPtrC, bsrNnzb) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsr2bsrNnz')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrNnzb
        end function hipsparseXcsr2bsrNnz

!       hipsparseXcsr2bsr
        function hipsparseScsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseScsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseScsr2bsr

        function hipsparseDcsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseDcsr2bsr

        function hipsparseCcsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseCcsr2bsr

        function hipsparseZcsr2bsr(handle, dirA, m, n, descrA, csrValA, csrRowPtrA, &
                csrColIndA, blockDim, descrC, bsrValC, bsrRowPtrC, bsrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsr2bsr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrRowPtrA
            type(c_ptr), intent(in), value :: csrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: bsrValC
            type(c_ptr), value :: bsrRowPtrC
            type(c_ptr), value :: bsrColIndC
        end function hipsparseZcsr2bsr

!       hipsparseXbsr2csr
        function hipsparseSbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseSbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseSbsr2csr

        function hipsparseDbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseDbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseDbsr2csr

        function hipsparseCbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseCbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseCbsr2csr

        function hipsparseZbsr2csr(handle, dirA, mb, nb, descrA, bsrValA, bsrRowPtrA, &
                bsrColIndA, blockDim, descrC, csrValC, csrRowPtrC, csrColIndC) &
                result(c_int) &
                bind(c, name = 'hipsparseZbsr2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: dirA
            integer(c_int), value :: mb
            integer(c_int), value :: nb
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: bsrValA
            type(c_ptr), intent(in), value :: bsrRowPtrA
            type(c_ptr), intent(in), value :: bsrColIndA
            integer(c_int), value :: blockDim
            type(c_ptr), intent(in), value :: descrC
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrRowPtrC
            type(c_ptr), value :: csrColIndC
        end function hipsparseZbsr2csr

!       hipsparseXcsr2csr_compress
        function hipsparseScsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseScsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            real(c_float), value :: tol
        end function hipsparseScsr2csr_compress

        function hipsparseDcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseDcsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            real(c_double), value :: tol
        end function hipsparseDcsr2csr_compress

        function hipsparseCcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseCcsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            complex(c_float_complex), value :: tol
        end function hipsparseCcsr2csr_compress

        function hipsparseZcsr2csr_compress(handle, m, n, descrA, csrValA, csrColIndA, &
                csrRowPtrA, nnzA, nnzPerRow, csrValC, csrColIndC, csrRowPtrC, tol) &
                result(c_int) &
                bind(c, name = 'hipsparseZcsr2csr_compress')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: csrValA
            type(c_ptr), intent(in), value :: csrColIndA
            type(c_ptr), intent(in), value :: csrRowPtrA
            integer(c_int), value :: nnzA
            type(c_ptr), intent(in), value :: nnzPerRow
            type(c_ptr), value :: csrValC
            type(c_ptr), value :: csrColIndC
            type(c_ptr), value :: csrRowPtrC
            complex(c_double_complex), value :: tol
        end function hipsparseZcsr2csr_compress

!       hipsparseXhyb2csr
        function hipsparseShyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseShyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseShyb2csr

        function hipsparseDhyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseDhyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseDhyb2csr

        function hipsparseChyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseChyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseChyb2csr

        function hipsparseZhyb2csr(handle, descrA, hybA, csrValA, csrRowPtrA, &
                csrColIndA) &
                result(c_int) &
                bind(c, name = 'hipsparseZhyb2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: descrA
            type(c_ptr), intent(in), value :: hybA
            type(c_ptr), value :: csrValA
            type(c_ptr), value :: csrRowPtrA
            type(c_ptr), value :: csrColIndA
        end function hipsparseZhyb2csr

!       hipsparseXcoo2csr
        function hipsparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr, idxBase) &
                result(c_int) &
                bind(c, name = 'hipsparseXcoo2csr')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            type(c_ptr), intent(in), value :: cooRowInd
            integer(c_int), value :: nnz
            integer(c_int), value :: m
            type(c_ptr), value :: csrRowPtr
            integer(c_int), value :: idxBase
        end function hipsparseXcoo2csr

!       hipsparseCreateIdentityPermutation
        function hipsparseCreateIdentityPermutation(handle, n, p) &
                result(c_int) &
                bind(c, name = 'hipsparseCreateIdentityPermutation')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: n
            type(c_ptr), value :: p
        end function hipsparseCreateIdentityPermutation

!       hipsparseXcsrsort_bufferSizeExt
        function hipsparseXcsrsort_bufferSizeExt(handle, m, n, nnz, csrRowPtr, &
                csrColInd, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrsort_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), intent(in), value :: csrColInd
            type(c_ptr), value :: bufferSize
        end function hipsparseXcsrsort_bufferSizeExt

!       hipsparseXcsrsort
        function hipsparseXcsrsort(handle, m, n, nnz, descr, csrRowPtr, csrColInd, &
                perm, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseXcsrsort')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: csrRowPtr
            type(c_ptr), value :: csrColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcsrsort

!       hipsparseXcscsort_bufferSizeExt
        function hipsparseXcscsort_bufferSizeExt(handle, m, n, nnz, cscRowPtr, &
                cscColInd, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseXcscsort_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: cscRowPtr
            type(c_ptr), intent(in), value :: cscColInd
            type(c_ptr), value :: bufferSize
        end function hipsparseXcscsort_bufferSizeExt

!       hipsparseXcscsort
        function hipsparseXcscsort(handle, m, n, nnz, descr, cscRowPtr, cscColInd, &
                perm, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseXcscsort')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: descr
            type(c_ptr), intent(in), value :: cscRowPtr
            type(c_ptr), value :: cscColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcscsort

!       hipsparseXcoosort_bufferSizeExt
        function hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRowInd, &
                cooColInd, bufferSize) &
                result(c_int) &
                bind(c, name = 'hipsparseXcoosort_bufferSizeExt')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), intent(in), value :: cooRowInd
            type(c_ptr), intent(in), value :: cooColInd
            type(c_ptr), value :: bufferSize
        end function hipsparseXcoosort_bufferSizeExt

!       hipsparseXcoosortByRow
        function hipsparseXcoosortByRow(handle, m, n, nnz, cooRowInd, cooColInd, &
                perm, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseXcoosortByRow')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: cooRowInd
            type(c_ptr), value :: cooColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcoosortByRow

!       hipsparseXcoosortByColumn
        function hipsparseXcoosortByColumn(handle, m, n, nnz, cooRowInd, cooColInd, &
                perm, buffer) &
                result(c_int) &
                bind(c, name = 'hipsparseXcoosortByColumn')
            use iso_c_binding
            implicit none
            type(c_ptr), value :: handle
            integer(c_int), value :: m
            integer(c_int), value :: n
            integer(c_int), value :: nnz
            type(c_ptr), value :: cooRowInd
            type(c_ptr), value :: cooColInd
            type(c_ptr), value :: perm
            type(c_ptr), value :: buffer
        end function hipsparseXcoosortByColumn

    end interface

end module hipsparse
