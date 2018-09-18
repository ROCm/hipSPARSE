# ########################################################################
# Copyright 2018 Advanced Micro Devices, Inc.
# ########################################################################

cmake_minimum_required(VERSION 3.5 FATAL_ERROR)

# Consider removing this in the future
# This should appear before the project command, because it does not use FORCE
if(WIN32)
  set(CMAKE_INSTALL_PREFIX "${PROJECT_BINARY_DIR}/package" CACHE PATH "Install path prefix, prepended onto install directories")
else()
  set(CMAKE_INSTALL_PREFIX "/opt/rocm" CACHE PATH "Install path prefix, prepended onto install directories")
endif()

# hipSPARSE project
project(hipsparse VERSION 0.1.3.0 LANGUAGES CXX)
set(hipsparse_SOVERSION 0)

# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE CACHE BOOLEAN "Add paths to linker search and installed rpath")

# Build options
option(BUILD_CLIENTS_TESTS "Build tests (requires googletest)" OFF)
option(BUILD_CLIENTS_SAMPLES "Build examples" ON)
option(BUILD_CUDA "Build hipSPARSE for CUDA" OFF)
option(BUILD_SHARED_LIBS "Build hipSPARSE as a shared library" ON)
option(BUILD_VERBOSE "Output additional build information" OFF)

# Dependencies
include(cmake/Dependencies.cmake)

# hipSPARSE library
add_subdirectory(library)

if(BUILD_CLIENTS_SAMPLES OR BUILD_CLIENTS_TESTS)
  enable_testing()
  add_subdirectory(clients)
endif()
