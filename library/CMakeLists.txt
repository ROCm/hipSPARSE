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

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Get full git commit SHA-1
set(GIT_COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD)
execute_process(COMMAND ${GIT_COMMAND}
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                OUTPUT_VARIABLE hipsparse_GIT_REVISION
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)

# Configure a header file to pass the hipSPARSE version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparse-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipsparse-version.h"
               @ONLY
)

# Public hipSPARSE headers
set(hipsparse_headers_public
  include/hipsparse.h
  ${PROJECT_BINARY_DIR}/include/hipsparse-version.h
  ${PROJECT_BINARY_DIR}/include/hipsparse-export.h
)

# Include sources
include(src/CMakeLists.txt)

# Create hipSPARSE library
add_library(hipsparse ${hipsparse_source} ${hipsparse_headers_public})
add_library(roc::hipsparse ALIAS hipsparse)

# Target compile definitions
if(NOT BUILD_CUDA)
  target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_HCC__)
else()
  target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_NVCC__)
endif()

# Target compile features
target_compile_features(hipsparse PRIVATE cxx_nullptr)

# Target include directories
target_include_directories(hipsparse
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>
                                   $<INSTALL_INTERFACE:include>
)

if(BUILD_CUDA)
  target_include_directories(hipsparse PUBLIC $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)
endif()

# Target link libraries
if(NOT BUILD_CUDA)
  target_link_libraries(hipsparse PRIVATE ${ROCSPARSE_LIBRARIES})
else()
  target_link_libraries(hipsparse PRIVATE ${CUDA_cusparse_LIBRARY})
endif()

# Target properties
set_target_properties(hipsparse PROPERTIES VERSION ${hipsparse_VERSION} SOVERSION ${hipsparse_SOVERSION} CXX_EXTENSIONS NO)
set_target_properties(hipsparse PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipsparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_propertieS(hipsparse PROPERTIES DEBUG_POSTFIX "-d")

# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
  set_target_properties(hipsparse PROPERTIES PREFIX "lib")
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(hipsparse EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipsparse-export.h)

# Install targets
rocm_install_targets(TARGETS hipsparse
                     INCLUDE
                       ${CMAKE_SOURCE_DIR}/library/include
                       ${CMAKE_BINARY_DIR}/include
                     PREFIX hipsparse
)

# Export targets
rocm_export_targets(TARGETS roc::hipsparse
                    PREFIX hipsparse
                    DEPENDS PACKAGE hip
                    NAMESPACE roc::
)

# Symbolic links
rocm_install_symlink_subdir(hipsparse)

# Package specific CPACK vars
if(NOT BUILD_CUDA)
  set(CPACK_DEBIAN_PACKAGE_DEPENDS "hip_hcc (>= 1.5.19055), rocsparse (>= 1.1.4)")
  set(CPACK_RPM_PACKAGE_REQUIRES "hip_hcc >= 1.5.19055, rocsparse >= 1.1.4")
endif()
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE.md")

if(NOT CPACK_PACKAGING_INSTALL_PREFIX)
  set(CPACK_PACKAGING_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
endif()

set(CPACK_RPM_EXCLUDE_FROM_AUTO_FILELIST_ADDITION "\${CPACK_PACKAGING_INSTALL_PREFIX}" "\${CPACK_PACKAGING_INSTALL_PREFIX}/include")

if(NOT BUILD_CUDA)
  set(package_name hipsparse)
else()
  set(package_name hipsparse-alt)
endif()

set(HIPSPARSE_CONFIG_DIR "\${CPACK_PACKAGING_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" CACHE PATH "Path placed into ldconfig file")

rocm_create_package(
  NAME ${package_name}
  DESCRIPTION "Radeon Open Compute SPARSE library"
  MAINTAINER "Nico Trost"
  LDCONFIG
  LDCONFIG_DIR ${HIPSPARSE_CONFIG_DIR}
)
