# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

if(NOT BUILD_CUDA)
  # hipSPARSE source
  set(hipsparse_source src/hcc_detail/hipsparse.cpp)
else()
  # hipSPARSE CUDA source
  set(hipsparse_source src/nvcc_detail/hipsparse.cpp)
endif()
