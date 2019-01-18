# ########################################################################
# Copyright (c) 2018 Advanced Micro Devices, Inc.
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

# Download some test matrices
set(TEST_MATRICES
  Bova/rma10
  JGD_BIBD/bibd_22_8
  Williams/mac_econ_fwd500
  Williams/mc2depi
  Hamm/scircuit
  Sandia/ASIC_320k
  GHS_psdef/bmwcra_1
  HB/nos1
  HB/nos2
  HB/nos3
  HB/nos4
  HB/nos5
  HB/nos6
  HB/nos7
)

if(NOT TARGET hipsparse)
  set(CONVERT ${CMAKE_SOURCE_DIR}/../deps/convert)
else()
  set(CONVERT ${CMAKE_SOURCE_DIR}/deps/convert)
endif()

foreach(m ${TEST_MATRICES})
  string(REPLACE "/" ";" sep_m ${m})
  list(GET sep_m 0 dir)
  list(GET sep_m 1 mat)

  # Download test matrices if not already downloaded
  if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/matrices/${mat}.bin")
    message("  Downloading and extracting test matrix ${m}.tar.gz")
    file(DOWNLOAD http://www.cise.ufl.edu/research/sparse/MM/${m}.tar.gz
         ${CMAKE_CURRENT_BINARY_DIR}/matrices/${mat}.tar.gz)
    execute_process(COMMAND tar xf ${mat}.tar.gz
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
    execute_process(COMMAND mv ${mat}/${mat}.mtx .
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
    execute_process(COMMAND ${CONVERT} ${mat}.mtx ${mat}.bin
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
    execute_process(COMMAND rm ${mat}.tar.gz ${mat} ${mat}.mtx -rf
                    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
  endif()
endforeach()


find_package(GTest REQUIRED)

set(HIPSPARSE_TEST_SOURCES
  hipsparse_gtest_main.cpp
  test_axpyi.cpp
  test_doti.cpp
  test_gthr.cpp
  test_gthrz.cpp
  test_roti.cpp
  test_sctr.cpp
  test_csrmv.cpp
  test_csrsv2.cpp
  test_csrmm.cpp
  test_csrilu02.cpp
  test_csr2coo.cpp
  test_csr2csc.cpp
  test_coo2csr.cpp
  test_identity.cpp
  test_csrsort.cpp
  test_coosort.cpp
  test_csrilusv.cpp
)

if(NOT BUILD_CUDA)
    list(APPEND HIPSPARSE_TEST_SOURCES
        test_hybmv.cpp
        test_csr2hyb.cpp
    )
endif()

set(HIPSPARSE_CLIENTS_COMMON
  ../common/arg_check.cpp
  ../common/unit.cpp
  ../common/utility.cpp
  ../common/hipsparse_template_specialization.cpp
)

add_executable(hipsparse-test ${HIPSPARSE_TEST_SOURCES} ${HIPSPARSE_CLIENTS_COMMON})

target_include_directories(hipsparse-test
                             PRIVATE
                               $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
                               $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                               $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                               $<BUILD_INTERFACE:${hip_INCLUDE_DIRS}>)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
target_link_libraries(hipsparse-test PRIVATE Threads::Threads)
target_link_libraries(hipsparse-test PRIVATE ${GTEST_BOTH_LIBRARIES})

if(NOT TARGET hipsparse)
  target_link_libraries(hipsparse-test PRIVATE ${HIPSPARSE_LIBRARIES})
else()
  target_link_libraries(hipsparse-test PRIVATE roc::hipsparse)
endif()

set_target_properties(hipsparse-test PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED ON)

target_compile_definitions(hipsparse-test PRIVATE GOOGLE_TEST)

target_include_directories(hipsparse-test
  SYSTEM
    PRIVATE
      $<BUILD_INTERFACE:${GTEST_INCLUDE_DIRS}>
      $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
)

if(NOT BUILD_CUDA)
  target_compile_definitions(hipsparse-test PRIVATE __HIP_PLATFORM_HCC__)
  get_target_property(HIP_HCC_LOCATION hip::hip_hcc IMPORTED_LOCATION_RELEASE)
  if(HIP_COMPILER STREQUAL "hcc")
    get_target_property(HCC_AM_LOCATION hcc::hc_am IMPORTED_LOCATION_RELEASE)
  endif()
  target_link_libraries(hipsparse-test PRIVATE ${HIP_HCC_LOCATION} ${HCC_AM_LOCATION})
else()
  target_compile_definitions(hipsparse-test PRIVATE __HIP_PLATFORM_NVCC__)
  target_link_libraries(hipsparse-test PRIVATE ${CUDA_LIBRARIES})
endif()

add_test(hipsparse-test hipsparse-test)
