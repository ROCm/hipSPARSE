# ########################################################################
# Copyright (C) 2018 Advanced Micro Devices, Inc. All rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ########################################################################

# hipSPARSE source
if(NOT USE_CUDA)
  # hipSPARSE source
  set(hipsparse_source 
      src/amd_detail/hipsparse_auxiliary.cpp
      src/amd_detail/hipsparse_generic_auxiliary.cpp
      # Level1
      src/amd_detail/level1/hipsparse_axpyi.cpp
      src/amd_detail/level1/hipsparse_doti.cpp
      src/amd_detail/level1/hipsparse_dotci.cpp
      src/amd_detail/level1/hipsparse_gthr.cpp
      src/amd_detail/level1/hipsparse_gthrz.cpp
      src/amd_detail/level1/hipsparse_roti.cpp
      src/amd_detail/level1/hipsparse_sctr.cpp
      # Level2
      src/amd_detail/level2/hipsparse_bsrmv.cpp
      src/amd_detail/level2/hipsparse_bsrxmv.cpp
      src/amd_detail/level2/hipsparse_bsrsv.cpp
      src/amd_detail/level2/hipsparse_csrmv.cpp
      src/amd_detail/level2/hipsparse_csrsv.cpp
      src/amd_detail/level2/hipsparse_gemvi.cpp
      src/amd_detail/level2/hipsparse_hybmv.cpp
      # Level3
      src/amd_detail/level3/hipsparse_bsrmm.cpp
      src/amd_detail/level3/hipsparse_bsrsm.cpp
      src/amd_detail/level3/hipsparse_csrmm.cpp
      src/amd_detail/level3/hipsparse_csrsm.cpp
      src/amd_detail/level3/hipsparse_gemmi.cpp
      # Precond
      src/amd_detail/precond/hipsparse_bsric0.cpp
      src/amd_detail/precond/hipsparse_bsrilu0.cpp
      src/amd_detail/precond/hipsparse_csric0.cpp
      src/amd_detail/precond/hipsparse_csrilu0.cpp
      src/amd_detail/precond/hipsparse_gpsv_interleaved_batch.cpp
      src/amd_detail/precond/hipsparse_gtsv_interleaved_batch.cpp
      src/amd_detail/precond/hipsparse_gtsv_nopivot.cpp
      src/amd_detail/precond/hipsparse_gtsv_strided_batch.cpp
      src/amd_detail/precond/hipsparse_gtsv.cpp
      # Extra
      src/amd_detail/extra/hipsparse_csrgeam.cpp
      src/amd_detail/extra/hipsparse_csrgemm.cpp
      # Generic
      src/amd_detail/generic/hipsparse_axpby.cpp
      src/amd_detail/generic/hipsparse_gather.cpp
      src/amd_detail/generic/hipsparse_scatter.cpp
      src/amd_detail/generic/hipsparse_rot.cpp
      src/amd_detail/generic/hipsparse_spvv.cpp
      src/amd_detail/generic/hipsparse_spmv.cpp
      src/amd_detail/generic/hipsparse_spsv.cpp
      src/amd_detail/generic/hipsparse_spmm.cpp
      src/amd_detail/generic/hipsparse_spsm.cpp
      src/amd_detail/generic/hipsparse_sddmm.cpp
      src/amd_detail/generic/hipsparse_spgemm.cpp
      src/amd_detail/generic/hipsparse_spgemm_reuse.cpp
      src/amd_detail/generic/hipsparse_sparse2dense.cpp
      src/amd_detail/generic/hipsparse_dense2sparse.cpp
      # Conversion
      src/amd_detail/conversion/hipsparse_bsr2csr.cpp
      src/amd_detail/conversion/hipsparse_coo2csr.cpp
      src/amd_detail/conversion/hipsparse_coosort.cpp
      src/amd_detail/conversion/hipsparse_create_identity_permutation.cpp
      src/amd_detail/conversion/hipsparse_csc2dense.cpp
      src/amd_detail/conversion/hipsparse_cscsort.cpp
      src/amd_detail/conversion/hipsparse_csr2bsr.cpp
      src/amd_detail/conversion/hipsparse_csr2coo.cpp
      src/amd_detail/conversion/hipsparse_csr2csc.cpp
      src/amd_detail/conversion/hipsparse_csr2csr_compress.cpp
      src/amd_detail/conversion/hipsparse_csr2csru.cpp
      src/amd_detail/conversion/hipsparse_csr2dense.cpp
      src/amd_detail/conversion/hipsparse_csr2gebsr.cpp
      src/amd_detail/conversion/hipsparse_csr2hyb.cpp
      src/amd_detail/conversion/hipsparse_csrsort.cpp
      src/amd_detail/conversion/hipsparse_csru2csr.cpp
      src/amd_detail/conversion/hipsparse_dense2csc.cpp
      src/amd_detail/conversion/hipsparse_dense2csr.cpp
      src/amd_detail/conversion/hipsparse_gebsr2csr.cpp
      src/amd_detail/conversion/hipsparse_gebsr2gebsc.cpp
      src/amd_detail/conversion/hipsparse_gebsr2gebsr.cpp
      src/amd_detail/conversion/hipsparse_hyb2csr.cpp
      src/amd_detail/conversion/hipsparse_nnz_compress.cpp
      src/amd_detail/conversion/hipsparse_nnz.cpp
      src/amd_detail/conversion/hipsparse_prune_csr2csr_by_percentage.cpp
      src/amd_detail/conversion/hipsparse_prune_csr2csr.cpp
      src/amd_detail/conversion/hipsparse_prune_dense2csr_by_percentage.cpp
      src/amd_detail/conversion/hipsparse_prune_dense2csr.cpp
      # Reorder
      src/amd_detail/reorder/hipsparse_csrcolor.cpp
      )
else()
  # hipSPARSE CUDA source
  set(hipsparse_source src/nvidia_detail/hipsparse.cpp)
endif()

# hipSPARSE Fortran source
set(hipsparse_fortran_source src/hipsparse.f90 src/hipsparse_enums.f90)
