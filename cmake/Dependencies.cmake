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

# Dependencies

# Git
find_package(Git REQUIRED)

# DownloadProject package
include(cmake/DownloadProject/DownloadProject.cmake)

# Either rocSPARSE or cuSPARSE is required
if(NOT BUILD_CUDA)
  find_package(rocSPARSE 0.1.3 REQUIRED) # ROCm 1.9
  find_package(hip 1.5.18353 REQUIRED CONFIG PATHS /opt/rocm) # ROCm 1.9
else()
  find_package(CUDA REQUIRED)
endif()

# Test dependencies
if(BUILD_TEST)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(GTest QUIET)
  endif()
  if(NOT GTEST_FOUND)
    message(STATUS "GTest not found. Downloading and building GTest.")
    set(GTEST_ROOT ${CMAKE_CURRENT_BINARY_DIR}/gtest CACHE PATH "")
    download_project(PROJ        googletest
             GIT_REPOSITORY      https://github.com/google/googletest.git
             GIT_TAG             master
             INSTALL_DIR         ${GTEST_ROOT}
             CMAKE_ARGS          -DBUILD_GTEST=ON -DINSTALL_GTEST=ON -Dgtest_force_shared_crt=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             LOG_DOWNLOAD        TRUE
             LOG_CONFIGURE       TRUE
             LOG_BUILD           TRUE
             LOG_INSTALL         TRUE
             BUILD_PROJECT       TRUE
             UPDATE_DISCONNECTED TRUE
    )
  endif()
  find_package(GTest REQUIRED)
  # Download some test matrices
  set(TEST_MATRICES
    nos1
    nos2
    nos3
    nos4
    nos5
    nos6
    nos7
  )
  foreach(m ${TEST_MATRICES})
    if(NOT EXISTS "${CMAKE_CURRENT_BINARY_DIR}/matrices/${m}.mtx")
      file(DOWNLOAD ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/lanpro/${m}.mtx.gz
           ${CMAKE_CURRENT_BINARY_DIR}/matrices/${m}.mtx.gz)
      execute_process(COMMAND gzip -d -f ${m}.mtx.gz
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/matrices)
    endif()
  endforeach()
endif()

# Benchmark dependencies
if(BUILD_BENCHMARK)
  if(NOT DEPENDENCIES_FORCE_DOWNLOAD)
    find_package(benchmark QUIET)
  endif()
  if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found. Downloading and building Google Benchmark.")
    set(GOOGLEBENCHMARK_ROOT ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark CACHE PATH "")
    download_project(PROJ        googlebenchmark
             GIT_REPOSITORY      https://github.com/google/benchmark.git
             GIT_TAG             master
             INSTALL_DIR         ${GOOGLEBENCHMARK_ROOT}
             CMAKE_ARGS          -DCMAKE_BUILD_TYPE=RELEASE -DBENCHMARK_ENABLE_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             LOG_DOWNLOAD        TRUE
             LOG_CONFIGURE       TRUE
             LOG_BUILD           TRUE
             LOG_INSTALL         TRUE
             BUILD_PROJECT       TRUE
             UPDATE_DISCONNECTED TRUE
    )
  endif()
  find_package(benchmark REQUIRED CONFIG PATHS ${GOOGLEBENCHMARK_ROOT})
endif()

# ROCm package
find_package(ROCM QUIET CONFIG PATHS /opt/rocm)
if(NOT ROCM_FOUND)
  set(rocm_cmake_tag "master" CACHE STRING "rocm-cmake tag to download")
  file(DOWNLOAD https://github.com/RadeonOpenCompute/rocm-cmake/archive/${rocm_cmake_tag}.zip
       ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
  )
  execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag}.zip
                  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )
  find_package(ROCM REQUIRED CONFIG PATHS ${CMAKE_CURRENT_BINARY_DIR}/rocm-cmake-${rocm_cmake_tag})
endif()

include(ROCMSetupVersion)
include(ROCMCreatePackage)
include(ROCMInstallTargets)
include(ROCMPackageConfigHelpers)
include(ROCMInstallSymlinks)
