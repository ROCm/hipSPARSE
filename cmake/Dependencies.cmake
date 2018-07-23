# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

# Dependencies

# Git
find_package(Git REQUIRED)

# DownloadProject package
include(cmake/DownloadProject.cmake)

# Either rocSPARSE or cuSPARSE is required
if(NOT BUILD_CUDA)
  find_package(rocSPARSE REQUIRED)
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

# rocPRIM package
#set(ROCPRIM_ROOT ${CMAKE_CURRENT_BINARY_DIR}/rocPRIM CACHE PATH "")
#message(STATUS "Downloading rocPRIM.")
#download_project(PROJ    rocPRIM
#     GIT_REPOSITORY      https://github.com/ROCmSoftwarePlatform/rocPRIM.git
#     GIT_TAG             master
#     INSTALL_DIR         ${ROCPRIM_ROOT}
#     CMAKE_ARGS          -DCMAKE_BUILD_TYPE=RELEASE -DBUILD_TEST=OFF -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
#     LOG_DOWNLOAD        TRUE
#     LOG_CONFIGURE       TRUE
#     LOG_INSTALL         TRUE
#     BUILD_PROJECT       TRUE
#     UPDATE_DISCONNECT   TRUE
#)

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
