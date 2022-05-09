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

# This option only works for make/nmake and the ninja generators, but no reason it shouldn't be on all the time
# This tells cmake to create a compile_commands.json file that can be used with clang tooling or vim
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Print verbose compiler flags
if(BUILD_VERBOSE)
  include(../cmake/Verbose.cmake)
endif()

# Configure a header file to pass the hipSPARSE version
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparse-version.h.in"
               "${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-version.h")

# Copy Header files to build directory
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/hipsparse.h"
               "${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse.h" COPYONLY)

# Public hipSPARSE headers
set(hipsparse_headers_public
  include/hipsparse.h
  ${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-version.h)

source_group("Header Files\\Public" FILES ${hipsparse_headers_public})

include(GNUInstallDirs)

set(BIN_INSTALL_DIR ${CMAKE_INSTALL_BINDIR})
set(LIB_INSTALL_DIR ${CMAKE_INSTALL_LIBDIR})
set(INCLUDE_INSTALL_DIR ${CMAKE_INSTALL_INCLUDEDIR})

# Include sources
include(src/CMakeLists.txt)

if (NOT WIN32)
# Set Fortran module output directory
set(CMAKE_Fortran_MODULE_DIRECTORY ${PROJECT_BINARY_DIR}/include/hipsparse)

# Create hipSPARSE Fortran module
add_library(hipsparse_fortran OBJECT ${hipsparse_fortran_source})

# Target compile options
target_compile_options(hipsparse_fortran PRIVATE -std=f2003 -ffree-form -cpp)
endif()

# Create hipSPARSE library
add_library(hipsparse ${hipsparse_source} ${hipsparse_headers_public})
add_library(roc::hipsparse ALIAS hipsparse)

# Target compile options
target_compile_options(hipsparse PRIVATE -Wno-unused-command-line-argument -Wall)

# Target include directories
target_include_directories(hipsparse
                           PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/include>
                           PUBLIC  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/library/include>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include/hipsparse>
                                   $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                                   $<INSTALL_INTERFACE:include>)

if(NOT USE_CUDA)
  target_link_libraries(hipsparse PRIVATE roc::rocsparse hip::host)
else()
  target_compile_definitions(hipsparse PRIVATE __HIP_PLATFORM_NVIDIA__)
  target_include_directories(hipsparse
                               PUBLIC
                                 $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>
                                 $<BUILD_INTERFACE:${HIP_INCLUDE_DIRS}>)
  target_link_libraries(hipsparse PRIVATE ${CUDA_cusparse_LIBRARY})
endif()

# Target properties
rocm_set_soversion(hipsparse ${hipsparse_SOVERSION})
set_target_properties(hipsparse PROPERTIES CXX_EXTENSIONS NO)
set_target_properties(hipsparse PROPERTIES CXX_VISIBILITY_PRESET "hidden" VISIBILITY_INLINES_HIDDEN ON)
set_target_properties(hipsparse PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/staging")
set_target_propertieS(hipsparse PROPERTIES DEBUG_POSTFIX "-d")

if (WIN32 AND BUILD_SHARED_LIBS)
  add_custom_command( TARGET hipsparse POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/$<TARGET_FILE_NAME:hipsparse> ${PROJECT_BINARY_DIR}/clients/staging/$<TARGET_FILE_NAME:hipsparse> )
  if( ${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_custom_command( TARGET hipsparse POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${PROJECT_BINARY_DIR}/staging/hipsparse.pdb ${PROJECT_BINARY_DIR}/clients/staging/hipsparse.pdb )
  endif()
endif()

# Following boost conventions of prefixing 'lib' on static built libraries
if(NOT BUILD_SHARED_LIBS)
  set_target_properties(hipsparse PROPERTIES PREFIX "lib")
endif()

# Generate export header
include(GenerateExportHeader)
generate_export_header(hipsparse EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-export.h)

if (BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_wrap_header_file(
    hipsparse-version.h hipsparse-export.h
    GUARDS SYMLINK WRAPPER
    WRAPPER_LOCATIONS include hipsparse/include
    ORIGINAL_FILES ${PROJECT_BINARY_DIR}/include/hipsparse/hipsparse-version.h
  )
endif( )

# Install targets
rocm_install_targets(TARGETS hipsparse
                     INCLUDE
                       ${CMAKE_BINARY_DIR}/include
                     )

if(BUILD_FILE_REORG_BACKWARD_COMPATIBILITY)
  rocm_install(
    DIRECTORY
       "${PROJECT_BINARY_DIR}/hipsparse"
        DESTINATION "." )
  message( STATUS "Backward Compatible Sym Link Created for include directories" )
endif()


# Export targets
if(NOT USE_CUDA)
  rocm_export_targets(TARGETS roc::hipsparse
                      DEPENDS PACKAGE hip
                      NAMESPACE roc::)
else()
  rocm_export_targets(TARGETS roc::hipsparse
                      NAMESPACE roc::)
endif()
