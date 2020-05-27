# ########################################################################
# Copyright (c) 2018-2020 Advanced Micro Devices, Inc.
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

set(HIPSPARSE_CLIENTS_COMMON ../common/utility.cpp)

# Function to add hipsparse examples
function(add_hipsparse_example EXAMPLE_SOURCE)
  get_filename_component(EXAMPLE_TARGET ${EXAMPLE_SOURCE} NAME_WE)
  add_executable(${EXAMPLE_TARGET} ${EXAMPLE_SOURCE} ${HIPSPARSE_CLIENTS_COMMON})

  # hipSPARSE headers
  target_include_directories(${EXAMPLE_TARGET}
    PRIVATE
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
      $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
  )

  # hipSPARSE library
  if(NOT TARGET hipsparse)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${HIPSPARSE_LIBRARIES})
  else()
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE roc::hipsparse)
  endif()

  if(NOT BUILD_CUDA)
    target_compile_definitions(${EXAMPLE_TARGET} PRIVATE __HIP_PLATFORM_HCC__)
    if( CMAKE_CXX_COMPILER MATCHES ".*/hcc$" )
        target_link_libraries( ${EXAMPLE_TARGET} PRIVATE ${HIP_LIBRARIES} )
    endif( )
    if ( LIBAMDHIP64_LIBRARY )
      target_include_directories(${EXAMPLE_TARGET} PRIVATE ${ROCM_PATH}/hip/include)
      target_link_libraries( ${EXAMPLE_TARGET} PRIVATE hip::amdhip64 )
    else ( )
      target_include_directories(${EXAMPLE_TARGET} PRIVATE ${ROCM_PATH}/hsa/include)
      get_target_property( HIP_HCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE )
      target_link_libraries( ${EXAMPLE_TARGET} PRIVATE ${HIP_HCC_LOCATION} )
    endif( )
  else()
    target_compile_definitions(${EXAMPLE_TARGET} PRIVATE __HIP_PLATFORM_NVCC__)
    target_link_libraries(${EXAMPLE_TARGET} PRIVATE ${CUDA_LIBRARIES})
  endif()

  set_target_properties(${EXAMPLE_TARGET} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
endfunction()

# Examples
add_hipsparse_example(example_handle.cpp)
add_hipsparse_example(example_csrmv.cpp)
add_hipsparse_example(example_hybmv.cpp)
add_hipsparse_example(example_fortran_auxiliary.f90)
add_hipsparse_example(example_fortran_csrsv2.f90)
add_hipsparse_example(example_fortran_csric02.f90)
add_hipsparse_example(example_fortran_csrilu02.f90)
add_hipsparse_example(example_fortran_csrsm2.f90)
add_hipsparse_example(example_fortran_dotci.f90)
add_hipsparse_example(example_fortran_roti.f90)
add_hipsparse_example(example_fortran_spmv.f90)
