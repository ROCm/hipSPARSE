.. _user_manual:

***********
User Manual
***********

.. toctree::
   :maxdepth: 3
   :caption: Contents:

Introduction
============

hipSPARSE is a library that contains basic linear algebra subroutines for sparse matrices and vectors written in HIP for GPU devices.
It is designed to be used from C and C++ code.
The functionality of hipSPARSE is organized in the following categories:

* :ref:`hipsparse_auxiliary_functions_` describe available helper functions that are required for subsequent library calls.
* :ref:`hipsparse_level1_functions_` describe operations between a vector in sparse format and a vector in dense format.
* :ref:`hipsparse_level2_functions_` describe operations between a matrix in sparse format and a vector in dense format.
* :ref:`hipsparse_level3_functions_` describe operations between a matrix in sparse format and multiple vectors in dense format.
* :ref:`hipsparse_extra_functions_` describe operations that manipulate sparse matrices.
* :ref:`hipsparse_precond_functions_` describe manipulations on a matrix in sparse format to obtain a preconditioner.
* :ref:`hipsparse_conversion_functions_` describe operations on a matrix in sparse format to obtain a different matrix format.
* :ref:`hipsparse_reordering_functions_` describe operations on a matrix in sparse format to obtain a reordering.
* :ref:`hipsparse_generic_functions_` describe operations that manipulate sparse matrices.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/hipSPARSE

hipSPARSE is a SPARSE marshalling library, with multiple supported backends.
It sits between the application and a `worker` SPARSE library, marshalling inputs into the backend library and marshalling results back to the application.
hipSPARSE exports an interface that does not require the client to change, regardless of the chosen backend.
Currently, hipSPARSE supports rocSPARSE and cuSPARSE as backends.
hipSPARSE focuses on convenience and portability.
If performance outweighs these factors, then using rocSPARSE itself is highly recommended.
rocSPARSE can also be found on `GitHub <https://github.com/ROCmSoftwarePlatform/rocSPARSE/>`_.

.. _hipsparse_building:

Building and Installing
=======================

Prerequisites
-------------
hipSPARSE requires a ROCm enabled platform, more information `here <https://rocm.github.io/>`_.

Installing pre-built packages
-----------------------------
hipSPARSE can be installed from `AMD ROCm repository <https://rocm.github.io/ROCmInstall.html#installing-from-amd-rocm-repositories>`_.
For detailed instructions on how to set up ROCm on different platforms, see the `AMD ROCm Platform Installation Guide for Linux <https://rocm.github.io/ROCmInstall.html>`_.

hipSPARSE can be installed on e.g. Ubuntu using

::

    $ sudo apt-get update
    $ sudo apt-get install hipsparse

Once installed, hipSPARSE can be used just like any other library with a C API.
The header file will need to be included in the user code in order to make calls into hipSPARSE, and the hipSPARSE shared library will become link-time and run-time dependent for the user application.

Building hipSPARSE from source
------------------------------
Building from source is not necessary, as hipSPARSE can be used after installing the pre-built packages as described above.
If desired, the following instructions can be used to build hipSPARSE from source.
Furthermore, the following compile-time dependencies must be met

- `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_
- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_
- `googletest <https://github.com/google/googletest>`_ (optional, for clients)

Download hipSPARSE
``````````````````
The hipSPARSE source code is available at the `hipSPARSE GitHub page <https://github.com/ROCmSoftwarePlatform/hipSPARSE>`_.
Download the master branch using:

::

  $ git clone -b master https://github.com/ROCmSoftwarePlatform/hipSPARSE.git
  $ cd hipSPARSE

Below are steps to build different packages of the library, including dependencies and clients.
It is recommended to install hipSPARSE using the `install.sh` script.

Using `install.sh` to build hipSPARSE with dependencies
```````````````````````````````````````````````````````
The following table lists common uses of `install.sh` to build dependencies + library.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

================= ====
Command           Description
================= ====
`./install.sh -h` Print help information.
`./install.sh -d` Build dependencies and library in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh`    Build library in your local directory. It is assumed dependencies are available.
`./install.sh -i` Build library, then build and install hipSPARSE package in `/opt/rocm/hipsparse`. You will be prompted for sudo access. This will install for all users.
================= ====

Using `install.sh` to build hipSPARSE with dependencies and clients
```````````````````````````````````````````````````````````````````
The client contains example code and unit tests. Common uses of `install.sh` to build them are listed in the table below.

.. tabularcolumns::
      |\X{1}{6}|\X{5}{6}|

=================== ====
Command             Description
=================== ====
`./install.sh -h`   Print help information.
`./install.sh -dc`  Build dependencies, library and client in your local directory. The `-d` flag only needs to be used once. For subsequent invocations of `install.sh` it is not necessary to rebuild the dependencies.
`./install.sh -c`   Build library and client in your local directory. It is assumed dependencies are available.
`./install.sh -idc` Build library, dependencies and client, then build and install hipSPARSE package in `/opt/rocm/hipsparse`. You will be prompted for sudo access. This will install for all users.
`./install.sh -ic`  Build library and client, then build and install hipSPARSE package in `opt/rocm/hipsparse`. You will be prompted for sudo access. This will install for all users.
=================== ====

Using individual commands to build hipSPARSE
````````````````````````````````````````````
CMake 3.5 or later is required in order to build hipSPARSE.

hipSPARSE can be built using the following commands:

::

  # Create and change to build directory
  $ mkdir -p build/release ; cd build/release

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ cmake ../..

  # Compile hipSPARSE library
  $ make -j$(nproc)

  # Install hipSPARSE to /opt/rocm
  $ make install

GoogleTest is required in order to build hipSPARSE clients.

hipSPARSE with dependencies and clients can be built using the following commands:

::

  # Install googletest
  $ mkdir -p build/release/deps ; cd build/release/deps
  $ cmake ../../../deps
  $ make -j$(nproc) install

  # Change to build directory
  $ cd ..

  # Default install path is /opt/rocm, use -DCMAKE_INSTALL_PREFIX=<path> to adjust it
  $ cmake ../.. -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_SAMPLES=ON

  # Compile hipSPARSE library
  $ make -j$(nproc)

  # Install hipSPARSE to /opt/rocm
  $ make install

Simple Test
```````````
You can test the installation by running one of the hipSPARSE examples, after successfully compiling the library with clients.

::

   # Navigate to clients binary directory
   $ cd hipSPARSE/build/release/clients/staging

   # Execute hipSPARSE example
   $ ./example_csrmv 1000

Supported Targets
-----------------
Currently, hipSPARSE is supported under the following operating systems

- `Ubuntu 18.04 <https://ubuntu.com/>`_
- `Ubuntu 20.04 <https://ubuntu.com/>`_
- `CentOS 7 <https://www.centos.org/>`_
- `CentOS 8 <https://www.centos.org/>`_
- `SLES 15 <https://www.suse.com/solutions/enterprise-linux/>`_

To compile and run hipSPARSE, `AMD ROCm Platform <https://github.com/RadeonOpenCompute/ROCm>`_ is required.

Device and Stream Management
============================
:cpp:func:`hipSetDevice` and :cpp:func:`hipGetDevice` are HIP device management APIs.
They are NOT part of the hipSPARSE API.

Asynchronous Execution
----------------------
All hipSPARSE library functions, unless otherwise stated, are non blocking and executed asynchronously with respect to the host. They may return before the actual computation has finished. To force synchronization, :cpp:func:`hipDeviceSynchronize` or :cpp:func:`hipStreamSynchronize` can be used. This will ensure that all previously executed hipSPARSE functions on the device / this particular stream have completed.

HIP Device Management
---------------------
Before a HIP kernel invocation, users need to call :cpp:func:`hipSetDevice` to set a device, e.g. device 1. If users do not explicitly call it, the system by default sets it as device 0. Unless users explicitly call :cpp:func:`hipSetDevice` to set to another device, their HIP kernels are always launched on device 0.

The above is a HIP (and CUDA) device management approach and has nothing to do with hipSPARSE. hipSPARSE honors the approach above and assumes users have already set the device before a hipSPARSE routine call.

Once users set the device, they create a handle with :ref:`hipsparse_create_handle_`.

Subsequent hipSPARSE routines take this handle as an input parameter. hipSPARSE ONLY queries (by :cpp:func:`hipGetDevice`) the user's device; hipSPARSE does NOT set the device for users. If hipSPARSE does not see a valid device, it returns an error message. It is the users' responsibility to provide a valid device to hipSPARSE and ensure the device safety.

Users CANNOT switch devices between :ref:`hipsparse_create_handle_` and :ref:`hipsparse_destroy_handle_`. If users want to change device, they must destroy the current handle and create another hipSPARSE handle.

HIP Stream Management
---------------------
HIP kernels are always launched in a queue (also known as stream).

If users do not explicitly specify a stream, the system provides a default stream, maintained by the system. Users cannot create or destroy the default stream. However, users can freely create new streams (with :cpp:func:`hipStreamCreate`) and bind it to the hipSPARSE handle using :ref:`hipsparse_set_stream_`. HIP kernels are invoked in hipSPARSE routines. The hipSPARSE handle is always associated with a stream, and hipSPARSE passes its stream to the kernels inside the routine. One hipSPARSE routine only takes one stream in a single invocation. If users create a stream, they are responsible for destroying it.

Multiple Streams and Multiple Devices
-------------------------------------
If the system under test has multiple HIP devices, users can run multiple hipSPARSE handles concurrently, but can NOT run a single hipSPARSE handle on different discrete devices. Each handle is associated with a particular singular device, and a new handle should be created for each additional device.

Storage Formats
===============

COO storage format
------------------
The Coordinate (COO) storage format represents a :math:`m \times n` matrix by

=========== ==================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
coo_val     array of ``nnz`` elements containing the data (floating point).
coo_row_ind array of ``nnz`` elements containing the row indices (integer).
coo_col_ind array of ``nnz`` elements containing the column indices (integer).
=========== ==================================================================

The COO matrix is expected to be sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using zero based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{coo_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{coo_row_ind}[8] & = \{0, 0, 0, 1, 1, 2, 2, 2\} \\
    \text{coo_col_ind}[8] & = \{0, 1, 3, 1, 2, 0, 3, 4\}
  \end{array}

COO (AoS) storage format
------------------------
The Coordinate (COO) Array of Structure (AoS) storage format represents a :math:`m \times n` matrix by

======= ==========================================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
coo_val     array of ``nnz`` elements containing the data (floating point).
coo_ind     array of ``2 * nnz`` elements containing alternating row and column indices (integer).
======= ==========================================================================================

The COO (AoS) matrix is expected to be sorted by row indices and column indices per row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding COO (AoS) structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using zero based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{coo_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{coo_ind}[16] & = \{0, 0, 0, 1, 0, 3, 1, 1, 1, 2, 2, 0, 2, 3, 2, 4\} \\
  \end{array}

CSR storage format
------------------
The Compressed Sparse Row (CSR) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements (integer).
csr_val     array of ``nnz`` elements containing the data (floating point).
csr_row_ptr array of ``m+1`` elements that point to the start of every row (integer).
csr_col_ind array of ``nnz`` elements containing the column indices (integer).
=========== =========================================================================

The CSR matrix is expected to be sorted by column indices within each row. Furthermore, each pair of indices should appear only once.
Consider the following :math:`3 \times 5` matrix and the corresponding CSR structures, with :math:`m = 3, n = 5` and :math:`\text{nnz} = 8` using one based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{csr_val}[8] & = \{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0\} \\
    \text{csr_row_ptr}[4] & = \{1, 4, 6, 9\} \\
    \text{csr_col_ind}[8] & = \{1, 2, 4, 2, 3, 1, 4, 5\}
  \end{array}

BSR storage format
------------------
The Block Compressed Sparse Row (BSR) storage format represents a :math:`(mb \cdot \text{bsr_dim}) \times (nb \cdot \text{bsr_dim})` matrix by

=========== ====================================================================================================================================
mb          number of block rows (integer)
nb          number of block columns (integer)
nnzb        number of non-zero blocks (integer)
bsr_val     array of ``nnzb * bsr_dim * bsr_dim`` elements containing the data (floating point). Blocks can be stored column-major or row-major.
bsr_row_ptr array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind array of ``nnzb`` elements containing the block column indices (integer).
bsr_dim     dimension of each block (integer).
=========== ====================================================================================================================================

The BSR matrix is expected to be sorted by column indices within each row. If :math:`m` or :math:`n` are not evenly divisible by the block dimension, then zeros are padded to the matrix, such that :math:`mb = (m + \text{bsr_dim} - 1) / \text{bsr_dim}` and :math:`nb = (n + \text{bsr_dim} - 1) / \text{bsr_dim}`.
Consider the following :math:`4 \times 3` matrix and the corresponding BSR structures, with :math:`\text{bsr_dim} = 2, mb = 2, nb = 2` and :math:`\text{nnzb} = 4` using zero based indexing and column-major storage:

.. math::

  A = \begin{pmatrix}
        1.0 & 0.0 & 2.0 \\
        3.0 & 0.0 & 4.0 \\
        5.0 & 6.0 & 0.0 \\
        7.0 & 0.0 & 8.0 \\
      \end{pmatrix}

with the blocks :math:`A_{ij}`

.. math::

  A_{00} = \begin{pmatrix}
             1.0 & 0.0 \\
             3.0 & 0.0 \\
           \end{pmatrix},
  A_{01} = \begin{pmatrix}
             2.0 & 0.0 \\
             4.0 & 0.0 \\
           \end{pmatrix},
  A_{10} = \begin{pmatrix}
             5.0 & 6.0 \\
             7.0 & 0.0 \\
           \end{pmatrix},
  A_{11} = \begin{pmatrix}
             0.0 & 0.0 \\
             8.0 & 0.0 \\
           \end{pmatrix}

such that

.. math::

  A = \begin{pmatrix}
        A_{00} & A_{01} \\
        A_{10} & A_{11} \\
      \end{pmatrix}

with arrays representation

.. math::

  \begin{array}{ll}
    \text{bsr_val}[16] & = \{1.0, 3.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0, 5.0, 7.0, 6.0, 0.0, 0.0, 8.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
  \end{array}

GEBSR storage format
--------------------
The General Block Compressed Sparse Row (GEBSR) storage format represents a :math:`(mb \cdot \text{bsr_row_dim}) \times (nb \cdot \text{bsr_col_dim})` matrix by

=========== ====================================================================================================================================
mb          number of block rows (integer)
nb          number of block columns (integer)
nnzb        number of non-zero blocks (integer)
bsr_val     array of ``nnzb * bsr_row_dim * bsr_col_dim`` elements containing the data (floating point). Blocks can be stored column-major or row-major.
bsr_row_ptr array of ``mb+1`` elements that point to the start of every block row (integer).
bsr_col_ind array of ``nnzb`` elements containing the block column indices (integer).
bsr_row_dim row dimension of each block (integer).
bsr_col_dim column dimension of each block (integer).
=========== ====================================================================================================================================

The GEBSR matrix is expected to be sorted by column indices within each row. If :math:`m` is not evenly divisible by the row block dimension or :math:`n` is not evenly
divisible by the column block dimension, then zeros are padded to the matrix, such that :math:`mb = (m + \text{bsr_row_dim} - 1) / \text{bsr_row_dim}` and
:math:`nb = (n + \text{bsr_col_dim} - 1) / \text{bsr_col_dim}`. Consider the following :math:`4 \times 5` matrix and the corresponding GEBSR structures,
with :math:`\text{bsr_row_dim} = 2`, :math:`\text{bsr_col_dim} = 3`, mb = 2, nb = 2` and :math:`\text{nnzb} = 4` using zero based indexing and column-major storage:

.. math::

  A = \begin{pmatrix}
        1.0 & 0.0 & 0.0 & 2.0 & 0.0 \\
        3.0 & 0.0 & 4.0 & 0.0 & 0.0 \\
        5.0 & 6.0 & 0.0 & 7.0 & 0.0 \\
        0.0 & 0.0 & 8.0 & 0.0 & 9.0 \\
      \end{pmatrix}

with the blocks :math:`A_{ij}`

.. math::

  A_{00} = \begin{pmatrix}
             1.0 & 0.0 & 0.0 \\
             3.0 & 0.0 & 4.0 \\
           \end{pmatrix},
  A_{01} = \begin{pmatrix}
             2.0 & 0.0 & 0.0 \\
             0.0 & 0.0 & 0.0 \\
           \end{pmatrix},
  A_{10} = \begin{pmatrix}
             5.0 & 6.0 & 0.0 \\
             0.0 & 0.0 & 8.0 \\
           \end{pmatrix},
  A_{11} = \begin{pmatrix}
             7.0 & 0.0 & 0.0 \\
             0.0 & 9.0 & 0.0 \\
           \end{pmatrix}

such that

.. math::

  A = \begin{pmatrix}
        A_{00} & A_{01} \\
        A_{10} & A_{11} \\
      \end{pmatrix}

with arrays representation

.. math::

  \begin{array}{ll}
    \text{bsr_val}[24] & = \{1.0, 3.0, 0.0, 0.0, 0.0, 4.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 6.0, 0.0, 0.0, 8.0, 7.0, 0.0, 0.0, 9.0, 0.0, 0.0\} \\
    \text{bsr_row_ptr}[3] & = \{0, 2, 4\} \\
    \text{bsr_col_ind}[4] & = \{0, 1, 0, 1\}
  \end{array}

ELL storage format
------------------
The Ellpack-Itpack (ELL) storage format represents a :math:`m \times n` matrix by

=========== ================================================================================
m           number of rows (integer).
n           number of columns (integer).
ell_width   maximum number of non-zero elements per row (integer)
ell_val     array of ``m times ell_width`` elements containing the data (floating point).
ell_col_ind array of ``m times ell_width`` elements containing the column indices (integer).
=========== ================================================================================

The ELL matrix is assumed to be stored in column-major format. Rows with less than ``ell_width`` non-zero elements are padded with zeros (``ell_val``) and :math:`-1` (``ell_col_ind``).
Consider the following :math:`3 \times 5` matrix and the corresponding ELL structures, with :math:`m = 3, n = 5` and :math:`\text{ell_width} = 3` using zero based indexing:

.. math::

  A = \begin{pmatrix}
        1.0 & 2.0 & 0.0 & 3.0 & 0.0 \\
        0.0 & 4.0 & 5.0 & 0.0 & 0.0 \\
        6.0 & 0.0 & 0.0 & 7.0 & 8.0 \\
      \end{pmatrix}

where

.. math::

  \begin{array}{ll}
    \text{ell_val}[9] & = \{1.0, 4.0, 6.0, 2.0, 5.0, 7.0, 3.0, 0.0, 8.0\} \\
    \text{ell_col_ind}[9] & = \{0, 1, 0, 1, 2, 3, 3, -1, 4\}
  \end{array}

.. _HYB storage format:

HYB storage format
------------------
The Hybrid (HYB) storage format represents a :math:`m \times n` matrix by

=========== =========================================================================================
m           number of rows (integer).
n           number of columns (integer).
nnz         number of non-zero elements of the COO part (integer)
ell_width   maximum number of non-zero elements per row of the ELL part (integer)
ell_val     array of ``m times ell_width`` elements containing the ELL part data (floating point).
ell_col_ind array of ``m times ell_width`` elements containing the ELL part column indices (integer).
coo_val     array of ``nnz`` elements containing the COO part data (floating point).
coo_row_ind array of ``nnz`` elements containing the COO part row indices (integer).
coo_col_ind array of ``nnz`` elements containing the COO part column indices (integer).
=========== =========================================================================================

The HYB format is a combination of the ELL and COO sparse matrix formats. Typically, the regular part of the matrix is stored in ELL storage format, and the irregular part of the matrix is stored in COO storage format. Three different partitioning schemes can be applied when converting a CSR matrix to a matrix in HYB storage format. For further details on the partitioning schemes, see :ref:`hipsparse_hyb_partition_`.

Types
=====

hipsparseHandle_t
-----------------

.. doxygentypedef:: hipsparseHandle_t

hipsparseMatDescr_t
-------------------

.. doxygentypedef:: hipsparseMatDescr_t

hipsparseHybMat_t
-----------------

.. doxygentypedef:: hipsparseHybMat_t

For more details on the HYB format, see :ref:`HYB storage format`.

.. _hipsparse_color_:

hipsparseColorInfo_t
--------------------

.. doxygentypedef:: hipsparseColorInfo_t

bsrsv2Info_t
------------

.. doxygentypedef:: bsrsv2Info_t

bsrsm2Info_t
------------

.. doxygentypedef:: bsrsm2Info_t

bsrilu02Info_t
--------------

.. doxygentypedef:: bsrilu02Info_t

bsric02Info_t
-------------

.. doxygentypedef:: bsric02Info_t

csrsv2Info_t
------------

.. doxygentypedef:: csrsv2Info_t

csrsm2Info_t
------------

.. doxygentypedef:: csrsm2Info_t

csrilu02Info_t
--------------

.. doxygentypedef:: csrilu02Info_t

csric02Info_t
-------------

.. doxygentypedef:: csric02Info_t

csrgemm2Info_t
--------------

.. doxygentypedef:: csrgemm2Info_t

pruneInfo_t
-----------

.. doxygentypedef:: pruneInfo_t

csru2csrInfo_t
--------------

.. doxygentypedef:: csru2csrInfo_t

hipsparseSpVecDescr_t
---------------------

.. doxygentypedef:: hipsparseSpVecDescr_t

hipsparseSpMatDescr_t
---------------------

.. doxygentypedef:: hipsparseSpMatDescr_t

hipsparseDnVecDescr_t
---------------------

.. doxygentypedef:: hipsparseDnVecDescr_t

hipsparseDnMatDescr_t
---------------------

.. doxygentypedef:: hipsparseDnMatDescr_t

hipsparseSpGEMMDescr_t
----------------------

.. doxygentypedef:: hipsparseSpGEMMDescr_t

hipsparseSpSVDescr_t
--------------------

.. doxygentypedef:: hipsparseSpSVDescr_t

hipsparseSpSMDescr_t
--------------------

.. doxygentypedef:: hipsparseSpSMDescr_t

hipsparseStatus_t
-----------------

.. doxygenenum:: hipsparseStatus_t

hipsparsePointerMode_t
----------------------

.. doxygenenum:: hipsparsePointerMode_t

.. _hipsparse_action_:

hipsparseAction_t
-----------------

.. doxygenenum:: hipsparseAction_t

hipsparseMatrixType_t
---------------------

.. doxygenenum:: hipsparseMatrixType_t

.. _hipsparse_fill_mode_:

hipsparseFillMode_t
-------------------

.. doxygenenum:: hipsparseFillMode_t

.. _hipsparse_diag_type_:

hipsparseDiagType_t
-------------------

.. doxygenenum:: hipsparseDiagType_t

.. _hipsparse_index_base_:

hipsparseIndexBase_t
--------------------

.. doxygenenum:: hipsparseIndexBase_t

.. _hipsparse_operation_:

hipsparseOperation_t
--------------------

.. doxygenenum:: hipsparseOperation_t

.. _hipsparse_hyb_partition_:

hipsparseHybPartition_t
-----------------------

.. doxygenenum:: hipsparseHybPartition_t

hipsparseSolvePolicy_t
----------------------

.. doxygenenum:: hipsparseSolvePolicy_t

hipsparseSideMode_t
-------------------

.. doxygenenum:: hipsparseSideMode_t

hipsparseDirection_t
--------------------

.. doxygenenum:: hipsparseDirection_t

hipsparseFormat_t
-----------------

.. doxygenenum:: hipsparseFormat_t

hipsparseOrder_t
----------------

.. doxygenenum:: hipsparseOrder_t

hipsparseIndextype_t
--------------------

.. doxygenenum:: hipsparseIndexType_t

hipsparseCsr2CscAlg_t
---------------------

.. doxygenenum:: hipsparseCsr2CscAlg_t

hipsparseSpMVAlg_t
------------------

.. doxygenenum:: hipsparseSpMVAlg_t

hipsparseSpMMAlg_t
------------------

.. doxygenenum:: hipsparseSpMMAlg_t

hipsparseSparseToDenseAlg_t
---------------------------

.. doxygenenum:: hipsparseSparseToDenseAlg_t

hipsparseDenseToSparseAlg_t
---------------------------

.. doxygenenum:: hipsparseDenseToSparseAlg_t

hipsparseSDDMMAlg_t
-------------------

.. doxygenenum:: hipsparseSDDMMAlg_t

hipsparseSpSVAlg_t
------------------

.. doxygenenum:: hipsparseSpSVAlg_t

hipsparseSpSMAlg_t
------------------

.. doxygenenum:: hipsparseSpSMAlg_t

hipsparseSpMatAttribute_t
-------------------------

.. doxygenenum:: hipsparseSpMatAttribute_t

hipsparseSpGEMMAlg_t
--------------------

.. doxygenenum:: hipsparseSpGEMMAlg_t

.. _api:

Exported Sparse Functions
=========================

Auxiliary Functions
-------------------

+------------------------------------------+
|Function name                             |
+------------------------------------------+
|:cpp:func:`hipsparseCreate`               |
+------------------------------------------+
|:cpp:func:`hipsparseDestroy`              |
+------------------------------------------+
|:cpp:func:`hipsparseGetVersion`           |
+------------------------------------------+
|:cpp:func:`hipsparseGetGitRevision`       |
+------------------------------------------+
|:cpp:func:`hipsparseSetStream`            |
+------------------------------------------+
|:cpp:func:`hipsparseGetStream`            |
+------------------------------------------+
|:cpp:func:`hipsparseSetPointerMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseGetPointerMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseCreateMatDescr`       |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyMatDescr`      |
+------------------------------------------+
|:cpp:func:`hipsparseCopyMatDescr`         |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatType`           |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatType`           |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatFillMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatFillMode`       |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatDiagType`       |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatDiagType`       |
+------------------------------------------+
|:cpp:func:`hipsparseSetMatIndexBase`      |
+------------------------------------------+
|:cpp:func:`hipsparseGetMatIndexBase`      |
+------------------------------------------+
|:cpp:func:`hipsparseCreateHybMat`         |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyHybMat`        |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsrsv2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsrsv2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsrsm2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsrsm2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsrilu02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsrilu02Info`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBsric02Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyBsric02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrsv2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrsv2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrsm2Info`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrsm2Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrilu02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrilu02Info`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsric02Info`    |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsric02Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsru2csrInfo`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsru2csrInfo`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreateColorInfo`      |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyColorInfo`     |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsrgemm2Info`   |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyCsrgemm2Info`  |
+------------------------------------------+
|:cpp:func:`hipsparseCreatePruneInfo`      |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyPruneInfo`     |
+------------------------------------------+
|:cpp:func:`hipsparseCreateSpVec`          |
+------------------------------------------+
|:cpp:func:`hipsparseDestroySpVec`         |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecGet`             |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecGetIndexBase`    |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpVecSetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCoo`            |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCooAoS`         |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsr`            |
+------------------------------------------+
|:cpp:func:`hipsparseCreateCsc`            |
+------------------------------------------+
|:cpp:func:`hipsparseCreateBlockedEll`     |
+------------------------------------------+
|:cpp:func:`hipsparseDestroySpMat`         |
+------------------------------------------+
|:cpp:func:`hipsparseCooGet`               |
+------------------------------------------+
|:cpp:func:`hipsparseCooAoSGet`            |
+------------------------------------------+
|:cpp:func:`hipsparseCsrGet`               |
+------------------------------------------+
|:cpp:func:`hipsparseBlockedEllGet`        |
+------------------------------------------+
|:cpp:func:`hipsparseCsrSetPointers`       |
+------------------------------------------+
|:cpp:func:`hipsparseCscSetPointers`       |
+------------------------------------------+
|:cpp:func:`hipsparseCooSetPointers`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetSize`         |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetFormat`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetIndexBase`    |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatSetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatGetAttribute`    |
+------------------------------------------+
|:cpp:func:`hipsparseSpMatSetAttribute`    |
+------------------------------------------+
|:cpp:func:`hipsparseCreateDnVec`          |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyDnVec`         |
+------------------------------------------+
|:cpp:func:`hipsparseDnVecGet`             |
+------------------------------------------+
|:cpp:func:`hipsparseDnVecGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseDnVecSetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseCreateDnMat`          |
+------------------------------------------+
|:cpp:func:`hipsparseDestroyDnMat`         |
+------------------------------------------+
|:cpp:func:`hipsparseDnMatGet`             |
+------------------------------------------+
|:cpp:func:`hipsparseDnMatGetValues`       |
+------------------------------------------+
|:cpp:func:`hipsparseDnMatSetValues`       |
+------------------------------------------+

Sparse Level 1 Functions
------------------------

================================================ ====== ====== ============== ==============
Function name                                    single double single complex double complex
================================================ ====== ====== ============== ==============
:cpp:func:`hipsparseXaxpyi() <hipsparseSaxpyi>`  x      x      x              x
:cpp:func:`hipsparseXdoti() <hipsparseSdoti>`    x      x      x              x
:cpp:func:`hipsparseXdotci() <hipsparseCdotci>`                x              x
:cpp:func:`hipsparseXgthr() <hipsparseSgthr>`    x      x      x              x
:cpp:func:`hipsparseXgthrz() <hipsparseSgthrz>`  x      x      x              x
:cpp:func:`hipsparseXroti() <hipsparseSroti>`    x      x
:cpp:func:`hipsparseXsctr() <hipsparseSsctr>`    x      x      x              x
================================================ ====== ====== ============== ==============

Sparse Level 2 Functions
------------------------

============================================================================== ====== ====== ============== ==============
Function name                                                                  single double single complex double complex
============================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXcsrmv() <hipsparseScsrmv>`                                x      x      x              x
:cpp:func:`hipsparseXcsrsv2_zeroPivot`
:cpp:func:`hipsparseXcsrsv2_bufferSize() <hipsparseScsrsv2_bufferSize>`        x      x      x              x
:cpp:func:`hipsparseXcsrsv2_bufferSizeExt() <hipsparseScsrsv2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXcsrsv2_analysis() <hipsparseScsrsv2_analysis>`            x      x      x              x
:cpp:func:`hipsparseXcsrsv2_solve() <hipsparseScsrsv2_solve>`                  x      x      x              x
:cpp:func:`hipsparseXhybmv() <hipsparseShybmv>`                                x      x      x              x
:cpp:func:`hipsparseXbsrmv() <hipsparseSbsrmv>`                                x      x      x              x
:cpp:func:`hipsparseXbsrxmv() <hipsparseSbsrxmv>`                              x      x      x              x
:cpp:func:`hipsparseXbsrsv2_zeroPivot`
:cpp:func:`hipsparseXbsrsv2_bufferSize() <hipsparseSbsrsv2_bufferSize>`        x      x      x              x
:cpp:func:`hipsparseXbsrsv2_bufferSizeExt() <hipsparseSbsrsv2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXbsrsv2_analysis() <hipsparseSbsrsv_analysis>`             x      x      x              x
:cpp:func:`hipsparseXbsrsv2_solve() <hipsparseSbsrsv2_solve>`                  x      x      x              x
:cpp:func:`hipsparseXgemvi_bufferSize() <hipsparseSgemvi_bufferSize>`          x      x      x              x
:cpp:func:`hipsparseXgemvi() <hipsparseSgemvi>`                                x      x      x              x
============================================================================== ====== ====== ============== ==============

Sparse Level 3 Functions
------------------------

============================================================================= ====== ====== ============== ==============
Function name                                                                 single double single complex double complex
============================================================================= ====== ====== ============== ==============
:cpp:func:`hipsparseXbsrmm() <hipsparseSbsrmm>`                               x      x      x              x
:cpp:func:`hipsparseXcsrmm() <hipsparseScsrmm>`                               x      x      x              x
:cpp:func:`hipsparseXcsrmm2() <hipsparseScsrmm2>`                             x      x      x              x
:cpp:func:`hipsparseXbsrsm2_zeroPivot`
:cpp:func:`hipsparseXbsrsm2_bufferSize() <hipsparseSbsrsm2_bufferSize>`       x      x      x              x
:cpp:func:`hipsparseXbsrsm2_analysis() <hipsparseSbsrsm2_analysis>`           x      x      x              x
:cpp:func:`hipsparseXbsrsm2_solve() <hipsparseSbsrsm2_solve>`                 x      x      x              x
:cpp:func:`hipsparseXcsrsm2_zeroPivot`
:cpp:func:`hipsparseXcsrsm2_bufferSizeExt() <hipsparseScsrsm2_bufferSizeExt>` x      x      x              x
:cpp:func:`hipsparseXcsrsm2_analysis() <hipsparseScsrsm2_analysis>`           x      x      x              x
:cpp:func:`hipsparseXcsrsm2_solve() <hipsparseScsrsm2_solve>`                 x      x      x              x
:cpp:func:`hipsparseXgemmi() <hipsparseSgemmi>`                               x      x      x              x
============================================================================= ====== ====== ============== ==============

Sparse Extra Functions
----------------------

================================================================================== ====== ====== ============== ==============
Function name                                                                      single double single complex double complex
================================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXcsrgeamNnz()`
:cpp:func:`hipsparseXcsrgeam() <hipsparseScsrgeam>`                                x      x      x              x
:cpp:func:`hipsparseXcsrgeam2_bufferSizeExt() <hipsparseScsrgeam2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXcsrgeam2Nnz()`
:cpp:func:`hipsparseXcsrgeam2() <hipsparseScsrgeam2>`                              x      x      x              x
:cpp:func:`hipsparseXcsrgemmNnz`
:cpp:func:`hipsparseXcsrgemm() <hipsparseScsrgemm>`                                x      x      x              x
:cpp:func:`hipsparseXcsrgemm2_bufferSizeExt() <hipsparseScsrgemm2_bufferSizeExt>`  x      x      x              x
:cpp:func:`hipsparseXcsrgemm2Nnz`
:cpp:func:`hipsparseXcsrgemm2() <hipsparseScsrgemm2>`                              x      x      x              x
================================================================================== ====== ====== ============== ==============

Preconditioner Functions
------------------------

===================================================================================================================== ====== ====== ============== ==============
Function name                                                                                                         single double single complex double complex
===================================================================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXbsrilu02_zeroPivot`
:cpp:func:`hipsparseXbsrilu02_numericBoost() <hipsparseSbsrilu02_numericBoost>`                                       x      x      x              x
:cpp:func:`hipsparseXbsrilu02_bufferSize() <hipsparseSbsrilu02_bufferSize>`                                           x      x      x              x
:cpp:func:`hipsparseXbsrilu02_analysis() <hipsparseSbsrilu02_analysis>`                                               x      x      x              x
:cpp:func:`hipsparseXbsrilu02() <hipsparseSbsrilu02>`                                                                 x      x      x              x
:cpp:func:`hipsparseXcsrilu02_zeroPivot`
:cpp:func:`hipsparseXcsrilu02_numericBoost() <hipsparseScsrilu02_numericBoost>`                                       x      x      x              x
:cpp:func:`hipsparseXcsrilu02_bufferSize() <hipsparseScsrilu02_bufferSize>`                                           x      x      x              x
:cpp:func:`hipsparseXcsrilu02_bufferSizeExt() <hipsparseScsrilu02_bufferSizeExt>`                                     x      x      x              x
:cpp:func:`hipsparseXcsrilu02_analysis() <hipsparseScsrilu02_analysis>`                                               x      x      x              x
:cpp:func:`hipsparseXcsrilu02() <hipsparseScsrilu02>`                                                                 x      x      x              x
:cpp:func:`hipsparseXbsric02_zeroPivot`
:cpp:func:`hipsparseXbsric02_bufferSize() <hipsparseSbsric02_bufferSize>`                                             x      x      x              x
:cpp:func:`hipsparseXbsric02_analysis() <hipsparseSbsric02_analysis>`                                                 x      x      x              x
:cpp:func:`hipsparseXbsric02() <hipsparseSbsric02>`                                                                   x      x      x              x
:cpp:func:`hipsparseXcsric02_zeroPivot`
:cpp:func:`hipsparseXcsric02_bufferSize() <hipsparseScsric02_bufferSize>`                                             x      x      x              x
:cpp:func:`hipsparseXcsric02_bufferSizeExt() <hipsparseScsric02_bufferSizeExt>`                                       x      x      x              x
:cpp:func:`hipsparseXcsric02_analysis() <hipsparseScsric02_analysis>`                                                 x      x      x              x
:cpp:func:`hipsparseXcsric02() <hipsparseScsric02>`                                                                   x      x      x              x
:cpp:func:`hipsparseXgtsv2_bufferSizeExt() <hipsparseSgtsv2_bufferSizeExt>`                                           x      x      x              x
:cpp:func:`hipsparseXgtsv2() <hipsparseSgtsv2>`                                                                       x      x      x              x
:cpp:func:`hipsparseXgtsv2_nopivot_bufferSizeExt() <hipsparseSgtsv2_nopivot_bufferSizeExt>`                           x      x      x              x
:cpp:func:`hipsparseXgtsv2_nopivot() <hipsparseSgtsv2_nopivot>`                                                       x      x      x              x
:cpp:func:`hipsparseXgtsv2StridedBatch_bufferSizeExt() <hipsparseSgtsv2StridedBatch_bufferSizeExt>`                   x      x      x              x
:cpp:func:`hipsparseXgtsv2StridedBatch() <hipsparseSgtsv2StridedBatch>`                                               x      x      x              x
:cpp:func:`hipsparseXgtsvInterleavedBatch_bufferSizeExt() <hipsparseSgtsvInterleavedBatch_bufferSizeExt>`             x      x      x              x
:cpp:func:`hipsparseXgtsvInterleavedBatch() <hipsparseSgtsvInterleavedBatch>`                                         x      x      x              x
:cpp:func:`hipsparseXgpsvInterleavedBatch_bufferSizeExt() <hipsparseSgpsvInterleavedBatch_bufferSizeExt>`             x      x      x              x
:cpp:func:`hipsparseXgpsvInterleavedBatch() <hipsparseSgpsvInterleavedBatch>`                                         x      x      x              x
===================================================================================================================== ====== ====== ============== ==============

Conversion Functions
--------------------

====================================================================================================================== ====== ====== ============== ==============
Function name                                                                                                          single double single complex double complex
====================================================================================================================== ====== ====== ============== ==============
:cpp:func:`hipsparseXnnz() <hipsparseSnnz>`                                                                            x      x      x              x
:cpp:func:`hipsparseXdense2csr() <hipsparseSdense2csr>`                                                                x      x      x              x
:cpp:func:`hipsparseXpruneDense2csr_bufferSize() <hipsparseSpruneDense2csr_bufferSize>`                                x      x
:cpp:func:`hipsparseXpruneDense2csrNnz() <hipsparseSpruneDense2csrNnz>`                                                x      x
:cpp:func:`hipsparseXpruneDense2csr() <hipsparseSpruneDense2csr>`                                                      x      x
:cpp:func:`hipsparseXpruneDense2csrByPercentage_bufferSize() <hipsparseSpruneDense2csrByPercentage_bufferSize>`        x      x
:cpp:func:`hipsparseXpruneDense2csrByPercentage_bufferSizeExt() <hipsparseSpruneDense2csrByPercentage_bufferSizeExt>`  x      x
:cpp:func:`hipsparseXpruneDense2csrNnzByPercentage() <hipsparseSpruneDense2csrNnzByPercentage>`                        x      x
:cpp:func:`hipsparseXpruneDense2csrByPercentage() <hipsparseSpruneDense2csrByPercentage>`                              x      x
:cpp:func:`hipsparseXdense2csc() <hipsparseSdense2csc>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsr2dense() <hipsparseScsr2dense>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsc2dense() <hipsparseScsc2dense>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsr2bsrNnz`
:cpp:func:`hipsparseXcsr2bsr() <hipsparseScsr2bsr>`                                                                    x      x      x              x
:cpp:func:`hipsparseXnnz_compress() <hipsparseSnnz_compress>`                                                          x      x      x              x
:cpp:func:`hipsparseXcsr2coo`
:cpp:func:`hipsparseXcsr2csc() <hipsparseScsr2csc>`                                                                    x      x      x              x
:cpp:func:`hipsparseXcsr2hyb() <hipsparseScsr2hyb>`                                                                    x      x      x              x
:cpp:func:`hipsparseXgebsr2gebsc_bufferSize <hipsparseSgebsr2gebsc_bufferSize>`                                        x      x      x              x
:cpp:func:`hipsparseXgebsr2gebsc() <hipsparseSgebsr2gebsc>`                                                            x      x      x              x
:cpp:func:`hipsparseXcsr2gebsr_bufferSize() <hipsparseScsr2gebsr_bufferSize>`                                          x      x      x              x
:cpp:func:`hipsparseXcsr2gebsrNnz`
:cpp:func:`hipsparseXcsr2gebsr() <hipsparseScsr2gebsr>`                                                                x      x      x              x
:cpp:func:`hipsparseXbsr2csr() <hipsparseSbsr2csr>`                                                                    x      x      x              x
:cpp:func:`hipsparseXgebsr2csr() <hipsparseSgebsr2csr>`                                                                x      x      x              x
:cpp:func:`hipsparseXcsr2csr_compress() <hipsparseScsr2csr_compress>`                                                  x      x      x              x
:cpp:func:`hipsparseXpruneCsr2csr_bufferSize() <hipsparseSpruneCsr2csr_bufferSize>`                                    x      x
:cpp:func:`hipsparseXpruneCsr2csr_bufferSizeExt() <hipsparseSpruneCsr2csr_bufferSizeExt>`                              x      x
:cpp:func:`hipsparseXpruneCsr2csrNnz() <hipsparseSpruneCsr2csrNnz>`                                                    x      x
:cpp:func:`hipsparseXpruneCsr2csr() <hipsparseSpruneCsr2csr>`                                                          x      x
:cpp:func:`hipsparseXpruneCsr2csrByPercentage_bufferSize() <hipsparseSpruneCsr2csrByPercentage_bufferSize>`            x      x
:cpp:func:`hipsparseXpruneCsr2csrByPercentage_bufferSizeExt() <hipsparseSpruneCsr2csrByPercentage_bufferSizeExt>`      x      x
:cpp:func:`hipsparseXpruneCsr2csrNnzByPercentage() <hipsparseSpruneCsr2csrNnzByPercentage>`                            x      x
:cpp:func:`hipsparseXpruneCsr2csrByPercentage() <hipsparseSpruneCsr2csrByPercentage>`                                  x      x
:cpp:func:`hipsparseXhyb2csr() <hipsparseShyb2csr>`                                                                    x      x      x              x
:cpp:func:`hipsparseXcoo2csr`
:cpp:func:`hipsparseCreateIdentityPermutation`
:cpp:func:`hipsparseXcsrsort_bufferSizeExt`
:cpp:func:`hipsparseXcsrsort`
:cpp:func:`hipsparseXcscsort_bufferSizeExt`
:cpp:func:`hipsparseXcscsort`
:cpp:func:`hipsparseXcoosort_bufferSizeExt`
:cpp:func:`hipsparseXcoosortByRow`
:cpp:func:`hipsparseXcoosortByColumn`
:cpp:func:`hipsparseXgebsr2gebsr_bufferSize() <hipsparseSgebsr2gebsr_bufferSize>`                                      x      x      x              x
:cpp:func:`hipsparseXgebsr2gebsrNnz()`
:cpp:func:`hipsparseXgebsr2gebsr() <hipsparseSgebsr2gebsr>`                                                            x      x      x              x
:cpp:func:`hipsparseXcsru2csr_bufferSizeExt() <hipsparseScsru2csr_bufferSizeExt>`                                      x      x      x              x
:cpp:func:`hipsparseXcsru2csr() <hipsparseScsru2csr>`                                                                  x      x      x              x
:cpp:func:`hipsparseXcsr2csru() <hipsparseScsr2csru>`                                                                  x      x      x              x
====================================================================================================================== ====== ====== ============== ==============

Reordering Functions
--------------------

======================================================= ====== ====== ============== ==============
Function name                                           single double single complex double complex
======================================================= ====== ====== ============== ==============
:cpp:func:`hipsparseXcsrcolor() <hipsparseScsrcolor>`   x      x      x              x
======================================================= ====== ====== ============== ==============

Sparse Generic Functions
------------------------

=============================================== ====== ====== ============== ==============
Function name                                   single double single complex double complex
=============================================== ====== ====== ============== ==============
:cpp:func:`hipsparseAxpby()`                    x      x      x              x
:cpp:func:`hipsparseGather()`                   x      x      x              x
:cpp:func:`hipsparseScatter()`                  x      x      x              x
:cpp:func:`hipsparseRot()`                      x      x      x              x
:cpp:func:`hipsparseSparseToDense_bufferSize()` x      x      x              x
:cpp:func:`hipsparseSparseToDense()`            x      x      x              x
:cpp:func:`hipsparseDenseToSparse_bufferSize()` x      x      x              x
:cpp:func:`hipsparseDenseToSparse_analysis()`   x      x      x              x
:cpp:func:`hipsparseDenseToSparse_convert()`    x      x      x              x
:cpp:func:`hipsparseSpVV_bufferSize()`          x      x      x              x
:cpp:func:`hipsparseSpVV()`                     x      x      x              x
:cpp:func:`hipsparseSpMV_bufferSize()`          x      x      x              x
:cpp:func:`hipsparseSpMV()`                     x      x      x              x
:cpp:func:`hipsparseSpMM_bufferSize()`          x      x      x              x
:cpp:func:`hipsparseSpMM_preprocess()`          x      x      x              x
:cpp:func:`hipsparseSpMM()`                     x      x      x              x
:cpp:func:`hipsparseSpGEMM_createDescr()`       x      x      x              x
:cpp:func:`hipsparseSpGEMM_destroyDescr()`      x      x      x              x
:cpp:func:`hipsparseSpGEMM_workEstimation()`    x      x      x              x
:cpp:func:`hipsparseSpGEMM_compute()`           x      x      x              x
:cpp:func:`hipsparseSpGEMM_copy()`              x      x      x              x
:cpp:func:`hipsparseSDDMM_bufferSize()`         x      x      x              x
:cpp:func:`hipsparseSDDMM_preprocess()`         x      x      x              x
:cpp:func:`hipsparseSDDMM()`                    x      x      x              x
:cpp:func:`hipsparseSpSV_createDescr()`         x      x      x              x
:cpp:func:`hipsparseSpSV_destroyDescr()`        x      x      x              x
:cpp:func:`hipsparseSpSV_bufferSize()`          x      x      x              x
:cpp:func:`hipsparseSpSV_analysis()`            x      x      x              x
:cpp:func:`hipsparseSpSV_solve()`               x      x      x              x
:cpp:func:`hipsparseSpSM_createDescr()`         x      x      x              x
:cpp:func:`hipsparseSpSM_destroyDescr()`        x      x      x              x
:cpp:func:`hipsparseSpSM_bufferSize()`          x      x      x              x
:cpp:func:`hipsparseSpSM_analysis()`            x      x      x              x
:cpp:func:`hipsparseSpSM_solve()`               x      x      x              x
=============================================== ====== ====== ============== ==============

Storage schemes and indexing base
---------------------------------
hipSPARSE supports 0 and 1 based indexing.
The index base is selected by the :cpp:enum:`hipsparseIndexBase_t` type which is either passed as standalone parameter or as part of the :cpp:type:`hipsparseMatDescr_t` type.

Furthermore, dense vectors are represented with a 1D array, stored linearly in memory.
Sparse vectors are represented by a 1D data array stored linearly in memory that hold all non-zero elements and a 1D indexing array stored linearly in memory that hold the positions of the corresponding non-zero elements.

Pointer mode
------------
The auxiliary functions :cpp:func:`hipsparseSetPointerMode` and :cpp:func:`hipsparseGetPointerMode` are used to set and get the value of the state variable :cpp:enum:`hipsparsePointerMode_t`.
If :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_HOST`, then scalar parameters must be allocated on the host.
If :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_DEVICE`, then scalar parameters must be allocated on the device.

There are two types of scalar parameter:

  1. Scaling parameters, such as `alpha` and `beta` used in e.g. :cpp:func:`hipsparseScsrmv`, :cpp:func:`hipsparseSbsrmv`, ...
  2. Scalar results from functions such as :cpp:func:`hipsparseSdoti`, :cpp:func:`hipsparseCdotci`, ...

For scalar parameters such as alpha and beta, memory can be allocated on the host heap or stack, when :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_HOST`.
The kernel launch is asynchronous, and if the scalar parameter is on the heap, it can be freed after the return from the kernel launch.
When :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_DEVICE`, the scalar parameter must not be changed till the kernel completes.

For scalar results, when :cpp:enum:`hipsparsePointerMode_t` is equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_HOST`, the function blocks the CPU till the GPU has copied the result back to the host.
Using :cpp:enum:`hipsparsePointerMode_t` equal to :cpp:enumerator:`HIPSPARSE_POINTER_MODE_DEVICE`, the function will return after the asynchronous launch.
Similarly to vector and matrix results, the scalar result is only available when the kernel has completed execution.

Asynchronous API
----------------
Except a functions having memory allocation inside preventing asynchronicity, all hipSPARSE functions are configured to operate in non-blocking fashion with respect to CPU, meaning these library functions return immediately.

.. _hipsparse_auxiliary_functions_:

Sparse Auxiliary Functions
==========================

This module holds all sparse auxiliary functions.

The functions that are contained in the auxiliary module describe all available helper functions that are required for subsequent library calls.

.. _hipsparse_create_handle_:

hipsparseCreate()
-----------------

.. doxygenfunction:: hipsparseCreate

.. _hipsparse_destroy_handle_:

hipsparseDestroy()
------------------

.. doxygenfunction:: hipsparseDestroy

.. _hipsparse_set_stream_:

hipsparseGetVersion()
---------------------

.. doxygenfunction:: hipsparseGetVersion

hipsparseGetGitRevision()
-------------------------

.. doxygenfunction:: hipsparseGetGitRevision

hipsparseSetStream()
--------------------

.. doxygenfunction:: hipsparseSetStream

hipsparseGetStream()
--------------------

.. doxygenfunction:: hipsparseGetStream

hipsparseSetPointerMode()
-------------------------

.. doxygenfunction:: hipsparseSetPointerMode

hipsparseGetPointerMode()
-------------------------

.. doxygenfunction:: hipsparseGetPointerMode

hipsparseCreateMatDescr()
-------------------------

.. doxygenfunction:: hipsparseCreateMatDescr

hipsparseDestroyMatDescr()
--------------------------

.. doxygenfunction:: hipsparseDestroyMatDescr

hipsparseCopyMatDescr()
-----------------------

.. doxygenfunction:: hipsparseCopyMatDescr

hipsparseSetMatType()
---------------------

.. doxygenfunction:: hipsparseSetMatType

hipsparseGetMatType()
---------------------

.. doxygenfunction:: hipsparseGetMatType

hipsparseSetMatFillMode()
-------------------------

.. doxygenfunction:: hipsparseSetMatFillMode

hipsparseGetMatFillMode()
-------------------------

.. doxygenfunction:: hipsparseGetMatFillMode

hipsparseSetMatDiagType()
-------------------------

.. doxygenfunction:: hipsparseSetMatDiagType

hipsparseGetMatDiagType()
-------------------------

.. doxygenfunction:: hipsparseGetMatDiagType

hipsparseSetMatIndexBase()
--------------------------

.. doxygenfunction:: hipsparseSetMatIndexBase

hipsparseGetMatIndexBase()
--------------------------

.. doxygenfunction:: hipsparseGetMatIndexBase

hipsparseCreateHybMat()
-----------------------

.. doxygenfunction:: hipsparseCreateHybMat

hipsparseDestroyHybMat()
------------------------

.. doxygenfunction:: hipsparseDestroyHybMat

hipsparseCreateBsrsv2Info()
---------------------------

.. doxygenfunction:: hipsparseCreateBsrsv2Info

hipsparseDestroyBsrsv2Info()
----------------------------

.. doxygenfunction:: hipsparseDestroyBsrsv2Info

hipsparseCreateBsrsm2Info()
---------------------------

.. doxygenfunction:: hipsparseCreateBsrsm2Info

hipsparseDestroyBsrsm2Info()
----------------------------

.. doxygenfunction:: hipsparseDestroyBsrsm2Info

hipsparseCreateBsrilu02Info()
-----------------------------

.. doxygenfunction:: hipsparseCreateBsrilu02Info

hipsparseDestroyBsrilu02Info()
------------------------------

.. doxygenfunction:: hipsparseDestroyBsrilu02Info

hipsparseCreateBsric02Info()
----------------------------

.. doxygenfunction:: hipsparseCreateBsric02Info

hipsparseDestroyBsric02Info()
-----------------------------

.. doxygenfunction:: hipsparseDestroyBsric02Info

hipsparseCreateCsrsv2Info()
---------------------------

.. doxygenfunction:: hipsparseCreateCsrsv2Info

hipsparseDestroyCsrsv2Info()
----------------------------

.. doxygenfunction:: hipsparseDestroyCsrsv2Info

hipsparseCreateCsrsm2Info()
---------------------------

.. doxygenfunction:: hipsparseCreateCsrsm2Info

hipsparseDestroyCsrsm2Info()
----------------------------

.. doxygenfunction:: hipsparseDestroyCsrsm2Info

hipsparseCreateCsrilu02Info()
-----------------------------

.. doxygenfunction:: hipsparseCreateCsrilu02Info

hipsparseDestroyCsrilu02Info()
------------------------------

.. doxygenfunction:: hipsparseDestroyCsrilu02Info

hipsparseCreateCsric02Info()
----------------------------

.. doxygenfunction:: hipsparseCreateCsric02Info

hipsparseDestroyCsric02Info()
-----------------------------

.. doxygenfunction:: hipsparseDestroyCsric02Info

hipsparseCreateCsru2csrInfo()
-----------------------------

.. doxygenfunction:: hipsparseCreateCsru2csrInfo

hipsparseDestroyCsru2csrInfo()
------------------------------

.. doxygenfunction:: hipsparseDestroyCsru2csrInfo

hipsparseCreateColorInfo()
--------------------------

.. doxygenfunction:: hipsparseCreateColorInfo

hipsparseDestroyColorInfo()
---------------------------

.. doxygenfunction:: hipsparseDestroyColorInfo

hipsparseCreateCsrgemm2Info()
-----------------------------

.. doxygenfunction:: hipsparseCreateCsrgemm2Info

hipsparseDestroyCsrgemm2Info()
------------------------------

.. doxygenfunction:: hipsparseDestroyCsrgemm2Info

hipsparseCreatePruneInfo()
--------------------------

.. doxygenfunction:: hipsparseCreatePruneInfo

hipsparseDestroyPruneInfo()
---------------------------

.. doxygenfunction:: hipsparseDestroyPruneInfo

hipsparseCreateSpVec()
----------------------

.. doxygenfunction:: hipsparseCreateSpVec

hipsparseDestroySpVec()
-----------------------

.. doxygenfunction:: hipsparseDestroySpVec

hipsparseSpVecGet()
-------------------

.. doxygenfunction:: hipsparseSpVecGet

hipsparseSpVecGetIndexBase()
----------------------------

.. doxygenfunction:: hipsparseSpVecGetIndexBase

hipsparseSpVecGetValues()
-------------------------

.. doxygenfunction:: hipsparseSpVecGetValues

hipsparseSpVecSetValues()
-------------------------

.. doxygenfunction:: hipsparseSpVecSetValues

hipsparseCreateCoo()
--------------------

.. doxygenfunction:: hipsparseCreateCoo

hipsparseCreateCooAoS()
-----------------------

.. doxygenfunction:: hipsparseCreateCooAoS

hipsparseCreateCsr()
--------------------

.. doxygenfunction:: hipsparseCreateCsr

hipsparseCreateCsc()
--------------------

.. doxygenfunction:: hipsparseCreateCsc

hipsparseCreateBlockedEll()
---------------------------

.. doxygenfunction:: hipsparseCreateBlockedEll

hipsparseDestroySpMat()
-----------------------

.. doxygenfunction:: hipsparseDestroySpMat

hipsparseCooGet()
-----------------

.. doxygenfunction:: hipsparseCooGet

hipsparseCooAoSGet()
--------------------

.. doxygenfunction:: hipsparseCooAoSGet

hipsparseCsrGet()
-----------------

.. doxygenfunction:: hipsparseCsrGet

hipsparseBlockedEllGet()
------------------------

.. doxygenfunction:: hipsparseBlockedEllGet

hipsparseCsrSetPointers()
-------------------------

.. doxygenfunction:: hipsparseCsrSetPointers

hipsparseCscSetPointers()
-------------------------

.. doxygenfunction:: hipsparseCscSetPointers

hipsparseCooSetPointers()
-------------------------

.. doxygenfunction:: hipsparseCooSetPointers

hipsparseSpMatGetSize()
-----------------------

.. doxygenfunction:: hipsparseSpMatGetSize

hipsparseSpMatGetFormat()
-------------------------

.. doxygenfunction:: hipsparseSpMatGetFormat

hipsparseSpMatGetIndexBase()
----------------------------

.. doxygenfunction:: hipsparseSpMatGetIndexBase

hipsparseSpMatGetValues()
-------------------------

.. doxygenfunction:: hipsparseSpMatGetValues

hipsparseSpMatSetValues()
-------------------------

.. doxygenfunction:: hipsparseSpMatSetValues

hipsparseSpMatGetAttribute()
----------------------------

.. doxygenfunction:: hipsparseSpMatGetAttribute

hipsparseSpMatSetAttribute()
----------------------------

.. doxygenfunction:: hipsparseSpMatSetAttribute

hipsparseCreateDnVec()
----------------------

.. doxygenfunction:: hipsparseCreateDnVec

hipsparseDestroyDnVec()
-----------------------

.. doxygenfunction:: hipsparseDestroyDnVec

hipsparseDnVecGet()
-------------------

.. doxygenfunction:: hipsparseDnVecGet

hipsparseDnVecGetValues()
-------------------------

.. doxygenfunction:: hipsparseDnVecGetValues

hipsparseDnVecSetValues()
-------------------------

.. doxygenfunction:: hipsparseDnVecSetValues

hipsparseCreateDnMat()
----------------------

.. doxygenfunction:: hipsparseCreateDnMat

hipsparseDestroyDnMat()
-----------------------

.. doxygenfunction:: hipsparseDestroyDnMat

hipsparseDnMatGet()
-------------------

.. doxygenfunction:: hipsparseDnMatGet

hipsparseDnMatGetValues()
-------------------------

.. doxygenfunction:: hipsparseDnMatGetValues

hipsparseDnMatSetValues()
-------------------------

.. doxygenfunction:: hipsparseDnMatSetValues

.. _hipsparse_level1_functions_:

Sparse Level 1 Functions
========================

The sparse level 1 routines describe operations between a vector in sparse format and a vector in dense format. This section describes all hipSPARSE level 1 sparse linear algebra functions.

hipsparseXaxpyi()
-----------------

.. doxygenfunction:: hipsparseSaxpyi
  :outline:
.. doxygenfunction:: hipsparseDaxpyi
  :outline:
.. doxygenfunction:: hipsparseCaxpyi
  :outline:
.. doxygenfunction:: hipsparseZaxpyi

hipsparseXdoti()
----------------

.. doxygenfunction:: hipsparseSdoti
  :outline:
.. doxygenfunction:: hipsparseDdoti
  :outline:
.. doxygenfunction:: hipsparseCdoti
  :outline:
.. doxygenfunction:: hipsparseZdoti

hipsparseXdotci()
-----------------

.. doxygenfunction:: hipsparseCdotci
  :outline:
.. doxygenfunction:: hipsparseZdotci

hipsparseXgthr()
----------------

.. doxygenfunction:: hipsparseSgthr
  :outline:
.. doxygenfunction:: hipsparseDgthr
  :outline:
.. doxygenfunction:: hipsparseCgthr
  :outline:
.. doxygenfunction:: hipsparseZgthr

hipsparseXgthrz()
-----------------

.. doxygenfunction:: hipsparseSgthrz
  :outline:
.. doxygenfunction:: hipsparseDgthrz
  :outline:
.. doxygenfunction:: hipsparseCgthrz
  :outline:
.. doxygenfunction:: hipsparseZgthrz

hipsparseXroti()
----------------

.. doxygenfunction:: hipsparseSroti
  :outline:
.. doxygenfunction:: hipsparseDroti

hipsparseXsctr()
----------------

.. doxygenfunction:: hipsparseSsctr
  :outline:
.. doxygenfunction:: hipsparseDsctr
  :outline:
.. doxygenfunction:: hipsparseCsctr
  :outline:
.. doxygenfunction:: hipsparseZsctr

.. _hipsparse_level2_functions_:

Sparse Level 2 Functions
========================

This module holds all sparse level 2 routines.

The sparse level 2 routines describe operations between a matrix in sparse format and a vector in dense format.

hipsparseXcsrmv()
-----------------

.. doxygenfunction:: hipsparseScsrmv
  :outline:
.. doxygenfunction:: hipsparseDcsrmv
  :outline:
.. doxygenfunction:: hipsparseCcsrmv
  :outline:
.. doxygenfunction:: hipsparseZcsrmv

hipsparseXcsrsv2_zeroPivot()
----------------------------

.. doxygenfunction:: hipsparseXcsrsv2_zeroPivot

hipsparseXcsrsv2_bufferSize()
-----------------------------

.. doxygenfunction:: hipsparseScsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_bufferSize

hipsparseXcsrsv2_bufferSizeExt()
--------------------------------

.. doxygenfunction:: hipsparseScsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_bufferSizeExt

hipsparseXcsrsv2_analysis()
---------------------------

.. doxygenfunction:: hipsparseScsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_analysis

hipsparseXcsrsv2_solve()
------------------------

.. doxygenfunction:: hipsparseScsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseDcsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseCcsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseZcsrsv2_solve

hipsparseXhybmv()
-----------------

.. doxygenfunction:: hipsparseShybmv
  :outline:
.. doxygenfunction:: hipsparseDhybmv
  :outline:
.. doxygenfunction:: hipsparseChybmv
  :outline:
.. doxygenfunction:: hipsparseZhybmv

hipsparseXbsrmv()
-----------------

.. doxygenfunction:: hipsparseSbsrmv
  :outline:
.. doxygenfunction:: hipsparseDbsrmv
  :outline:
.. doxygenfunction:: hipsparseCbsrmv
  :outline:
.. doxygenfunction:: hipsparseZbsrmv

hipsparseXbsrxmv()
------------------

.. doxygenfunction:: hipsparseSbsrxmv
  :outline:
.. doxygenfunction:: hipsparseDbsrxmv
  :outline:
.. doxygenfunction:: hipsparseCbsrxmv
  :outline:
.. doxygenfunction:: hipsparseZbsrxmv

hipsparseXbsrsv2_zeroPivot()
----------------------------

.. doxygenfunction:: hipsparseXbsrsv2_zeroPivot

hipsparseXbsrsv2_bufferSize()
-----------------------------

.. doxygenfunction:: hipsparseSbsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_bufferSize

hipsparseXbsrsv2_bufferSizeExt()
--------------------------------

.. doxygenfunction:: hipsparseSbsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_bufferSizeExt

hipsparseXbsrsv2_analysis()
---------------------------

.. doxygenfunction:: hipsparseSbsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_analysis

hipsparseXbsrsv2_solve()
------------------------

.. doxygenfunction:: hipsparseSbsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseDbsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseCbsrsv2_solve
  :outline:
.. doxygenfunction:: hipsparseZbsrsv2_solve

hipsparseXgemvi_bufferSize()
----------------------------

.. doxygenfunction:: hipsparseSgemvi_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDgemvi_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCgemvi_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZgemvi_bufferSize

hipsparseXgemvi()
-----------------

.. doxygenfunction:: hipsparseSgemvi
  :outline:
.. doxygenfunction:: hipsparseDgemvi
  :outline:
.. doxygenfunction:: hipsparseCgemvi
  :outline:
.. doxygenfunction:: hipsparseZgemvi

.. _hipsparse_level3_functions_:

Sparse Level 3 Functions
========================

This module holds all sparse level 3 routines.

The sparse level 3 routines describe operations between a matrix in sparse format and multiple vectors in dense format that can also be seen as a dense matrix.

hipsparseXbsrmm()
-----------------

.. doxygenfunction:: hipsparseSbsrmm
  :outline:
.. doxygenfunction:: hipsparseDbsrmm
  :outline:
.. doxygenfunction:: hipsparseCbsrmm
  :outline:
.. doxygenfunction:: hipsparseZbsrmm

hipsparseXcsrmm()
-----------------

.. doxygenfunction:: hipsparseScsrmm
  :outline:
.. doxygenfunction:: hipsparseDcsrmm
  :outline:
.. doxygenfunction:: hipsparseCcsrmm
  :outline:
.. doxygenfunction:: hipsparseZcsrmm

hipsparseXcsrmm2()
------------------

.. doxygenfunction:: hipsparseScsrmm2
  :outline:
.. doxygenfunction:: hipsparseDcsrmm2
  :outline:
.. doxygenfunction:: hipsparseCcsrmm2
  :outline:
.. doxygenfunction:: hipsparseZcsrmm2

hipsparseXbsrsm2_zeroPivot()
----------------------------

.. doxygenfunction:: hipsparseXbsrsm2_zeroPivot

hipsparseXbsrsm2_bufferSize()
-----------------------------

.. doxygenfunction:: hipsparseSbsrsm2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsrsm2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsrsm2_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsrsm2_bufferSize

hipsparseXbsrsm2_analysis()
---------------------------

.. doxygenfunction:: hipsparseSbsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsrsm2_analysis

hipsparseXbsrsm2_solve()
------------------------

.. doxygenfunction:: hipsparseSbsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseDbsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseCbsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseZbsrsm2_solve

hipsparseXcsrsm2_zeroPivot()
----------------------------

.. doxygenfunction:: hipsparseXcsrsm2_zeroPivot

hipsparseXcsrsm2_bufferSizeExt()
--------------------------------

.. doxygenfunction:: hipsparseScsrsm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrsm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrsm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrsm2_bufferSizeExt

hipsparseXcsrsm2_analysis()
---------------------------

.. doxygenfunction:: hipsparseScsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsrsm2_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsrsm2_analysis

hipsparseXcsrsm2_solve()
------------------------

.. doxygenfunction:: hipsparseScsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseDcsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseCcsrsm2_solve
  :outline:
.. doxygenfunction:: hipsparseZcsrsm2_solve

hipsparseXgemmi()
-----------------

.. doxygenfunction:: hipsparseSgemmi
  :outline:
.. doxygenfunction:: hipsparseDgemmi
  :outline:
.. doxygenfunction:: hipsparseCgemmi
  :outline:
.. doxygenfunction:: hipsparseZgemmi

.. _hipsparse_extra_functions_:

Sparse Extra Functions
======================

This module holds all sparse extra routines.

The sparse extra routines describe operations that manipulate sparse matrices.

hipsparseXcsrgeamNnz()
----------------------

.. doxygenfunction:: hipsparseXcsrgeamNnz

hipsparseXcsrgeam()
-------------------

.. doxygenfunction:: hipsparseScsrgeam
  :outline:
.. doxygenfunction:: hipsparseDcsrgeam
  :outline:
.. doxygenfunction:: hipsparseCcsrgeam
  :outline:
.. doxygenfunction:: hipsparseZcsrgeam

hipsparseXcsrgeam2_bufferSizeExt()
----------------------------------

.. doxygenfunction:: hipsparseScsrgeam2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrgeam2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrgeam2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrgeam2_bufferSizeExt

hipsparseXcsrgeam2Nnz()
-----------------------

.. doxygenfunction:: hipsparseXcsrgeam2Nnz

hipsparseXcsrgeam2()
--------------------

.. doxygenfunction:: hipsparseScsrgeam2
  :outline:
.. doxygenfunction:: hipsparseDcsrgeam2
  :outline:
.. doxygenfunction:: hipsparseCcsrgeam2
  :outline:
.. doxygenfunction:: hipsparseZcsrgeam2

hipsparseXcsrgemmNnz()
----------------------

.. doxygenfunction:: hipsparseXcsrgemmNnz

hipsparseXcsrgemm()
-------------------

.. doxygenfunction:: hipsparseScsrgemm
  :outline:
.. doxygenfunction:: hipsparseDcsrgemm
  :outline:
.. doxygenfunction:: hipsparseCcsrgemm
  :outline:
.. doxygenfunction:: hipsparseZcsrgemm

hipsparseXcsrgemm2_bufferSizeExt()
----------------------------------

.. doxygenfunction:: hipsparseScsrgemm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrgemm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrgemm2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrgemm2_bufferSizeExt

hipsparseXcsrgemm2Nnz()
-----------------------

.. doxygenfunction:: hipsparseXcsrgemm2Nnz

hipsparseXcsrgemm2()
--------------------

.. doxygenfunction:: hipsparseScsrgemm2
  :outline:
.. doxygenfunction:: hipsparseDcsrgemm2
  :outline:
.. doxygenfunction:: hipsparseCcsrgemm2
  :outline:
.. doxygenfunction:: hipsparseZcsrgemm2

.. _hipsparse_precond_functions_:

Preconditioner Functions
========================

This module holds all sparse preconditioners.

The sparse preconditioners describe manipulations on a matrix in sparse format to obtain a sparse preconditioner matrix.

hipsparseXbsrilu02_zeroPivot()
------------------------------

.. doxygenfunction:: hipsparseXbsrilu02_zeroPivot

hipsparseXbsrilu02_numericBoost()
---------------------------------

.. doxygenfunction:: hipsparseSbsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02_numericBoost

hipsparseXbsrilu02_bufferSize()
-------------------------------

.. doxygenfunction:: hipsparseSbsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02_bufferSize

hipsparseXbsrilu02_analysis()
-----------------------------

.. doxygenfunction:: hipsparseSbsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02_analysis

hipsparseXbsrilu02()
--------------------

.. doxygenfunction:: hipsparseSbsrilu02
  :outline:
.. doxygenfunction:: hipsparseDbsrilu02
  :outline:
.. doxygenfunction:: hipsparseCbsrilu02
  :outline:
.. doxygenfunction:: hipsparseZbsrilu02

hipsparseXcsrilu02_zeroPivot()
------------------------------

.. doxygenfunction:: hipsparseXcsrilu02_zeroPivot

hipsparseXcsrilu02_numericBoost()
---------------------------------

.. doxygenfunction:: hipsparseScsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_numericBoost
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_numericBoost

hipsparseXcsrilu02_bufferSize()
-------------------------------

.. doxygenfunction:: hipsparseScsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_bufferSize

hipsparseXcsrilu02_bufferSizeExt()
----------------------------------

.. doxygenfunction:: hipsparseScsrilu02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_bufferSizeExt

hipsparseXcsrilu02_analysis()
-----------------------------

.. doxygenfunction:: hipsparseScsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02_analysis

hipsparseXcsrilu02()
--------------------

.. doxygenfunction:: hipsparseScsrilu02
  :outline:
.. doxygenfunction:: hipsparseDcsrilu02
  :outline:
.. doxygenfunction:: hipsparseCcsrilu02
  :outline:
.. doxygenfunction:: hipsparseZcsrilu02

hipsparseXbsric02_zeroPivot()
-----------------------------

.. doxygenfunction:: hipsparseXbsric02_zeroPivot

hipsparseXbsric02_bufferSize()
------------------------------

.. doxygenfunction:: hipsparseSbsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDbsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCbsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZbsric02_bufferSize

hipsparseXbsric02_analysis()
----------------------------

.. doxygenfunction:: hipsparseSbsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseDbsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseCbsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseZbsric02_analysis

hipsparseXbsric02()
-------------------

.. doxygenfunction:: hipsparseSbsric02
  :outline:
.. doxygenfunction:: hipsparseDbsric02
  :outline:
.. doxygenfunction:: hipsparseCbsric02
  :outline:
.. doxygenfunction:: hipsparseZbsric02

hipsparseXcsric02_zeroPivot()
-----------------------------

.. doxygenfunction:: hipsparseXcsric02_zeroPivot

hipsparseXcsric02_bufferSize()
------------------------------

.. doxygenfunction:: hipsparseScsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsric02_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsric02_bufferSize

hipsparseXcsric02_bufferSizeExt()
---------------------------------

.. doxygenfunction:: hipsparseScsric02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsric02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsric02_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsric02_bufferSizeExt

hipsparseXcsric02_analysis()
----------------------------

.. doxygenfunction:: hipsparseScsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseDcsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseCcsric02_analysis
  :outline:
.. doxygenfunction:: hipsparseZcsric02_analysis

hipsparseXcsric02()
-------------------

.. doxygenfunction:: hipsparseScsric02
  :outline:
.. doxygenfunction:: hipsparseDcsric02
  :outline:
.. doxygenfunction:: hipsparseCcsric02
  :outline:
.. doxygenfunction:: hipsparseZcsric02

hipsparseXgtsv2_bufferSizeExt()
-------------------------------

.. doxygenfunction:: hipsparseSgtsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsv2_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsv2_bufferSizeExt

hipsparseXgtsv2()
-----------------

.. doxygenfunction:: hipsparseSgtsv2
  :outline:
.. doxygenfunction:: hipsparseDgtsv2
  :outline:
.. doxygenfunction:: hipsparseCgtsv2
  :outline:
.. doxygenfunction:: hipsparseZgtsv2

hipsparseXgtsv2_nopivot_bufferSizeExt()
---------------------------------------

.. doxygenfunction:: hipsparseSgtsv2_nopivot_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsv2_nopivot_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsv2_nopivot_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsv2_nopivot_bufferSizeExt

hipsparseXgtsv2_nopivot()
-------------------------

.. doxygenfunction:: hipsparseSgtsv2_nopivot
  :outline:
.. doxygenfunction:: hipsparseDgtsv2_nopivot
  :outline:
.. doxygenfunction:: hipsparseCgtsv2_nopivot
  :outline:
.. doxygenfunction:: hipsparseZgtsv2_nopivot

hipsparseXgtsv2StridedBatch_bufferSizeExt()
-------------------------------------------

.. doxygenfunction:: hipsparseSgtsv2StridedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsv2StridedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsv2StridedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsv2StridedBatch_bufferSizeExt

hipsparseXgtsv2StridedBatch()
-----------------------------

.. doxygenfunction:: hipsparseSgtsv2StridedBatch
  :outline:
.. doxygenfunction:: hipsparseDgtsv2StridedBatch
  :outline:
.. doxygenfunction:: hipsparseCgtsv2StridedBatch
  :outline:
.. doxygenfunction:: hipsparseZgtsv2StridedBatch

hipsparseXgtsvInterleavedBatch_bufferSizeExt()
----------------------------------------------

.. doxygenfunction:: hipsparseSgtsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgtsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgtsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgtsvInterleavedBatch_bufferSizeExt

hipsparseXgtsvInterleavedBatch()
--------------------------------

.. doxygenfunction:: hipsparseSgtsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseDgtsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseCgtsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseZgtsvInterleavedBatch

hipsparseXgpsvInterleavedBatch_bufferSizeExt()
----------------------------------------------

.. doxygenfunction:: hipsparseSgpsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDgpsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCgpsvInterleavedBatch_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZgpsvInterleavedBatch_bufferSizeExt

hipsparseXgpsvInterleavedBatch()
--------------------------------

.. doxygenfunction:: hipsparseSgpsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseDgpsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseCgpsvInterleavedBatch
  :outline:
.. doxygenfunction:: hipsparseZgpsvInterleavedBatch

.. _hipsparse_conversion_functions_:

Sparse Conversion Functions
===========================

This module holds all sparse conversion routines.

The sparse conversion routines describe operations on a matrix in sparse format to obtain a matrix in a different sparse format.

hipsparseXnnz()
---------------

.. doxygenfunction:: hipsparseSnnz
  :outline:
.. doxygenfunction:: hipsparseDnnz
  :outline:
.. doxygenfunction:: hipsparseCnnz
  :outline:
.. doxygenfunction:: hipsparseZnnz

hipsparseXdense2csr()
---------------------

.. doxygenfunction:: hipsparseSdense2csr
  :outline:
.. doxygenfunction:: hipsparseDdense2csr
  :outline:
.. doxygenfunction:: hipsparseCdense2csr
  :outline:
.. doxygenfunction:: hipsparseZdense2csr

hipsparseXpruneDense2csr_bufferSize()
-------------------------------------

.. doxygenfunction:: hipsparseSpruneDense2csr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csr_bufferSize

hipsparseXpruneDense2csrNnz()
-----------------------------

.. doxygenfunction:: hipsparseSpruneDense2csrNnz
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrNnz

hipsparseXpruneDense2csr()
--------------------------

.. doxygenfunction:: hipsparseSpruneDense2csr
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csr

hipsparseXpruneDense2csrByPercentage_bufferSize()
-------------------------------------------------

.. doxygenfunction:: hipsparseSpruneDense2csrByPercentage_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrByPercentage_bufferSize

hipsparseXpruneDense2csrByPercentage_bufferSizeExt()
----------------------------------------------------

.. doxygenfunction:: hipsparseSpruneDense2csrByPercentage_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrByPercentage_bufferSizeExt

hipsparseXpruneDense2csrNnzByPercentage()
-----------------------------------------

.. doxygenfunction:: hipsparseSpruneDense2csrNnzByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrNnzByPercentage

hipsparseXpruneDense2csrByPercentage()
--------------------------------------

.. doxygenfunction:: hipsparseSpruneDense2csrByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneDense2csrByPercentage

hipsparseXdense2csc()
---------------------

.. doxygenfunction:: hipsparseSdense2csc
  :outline:
.. doxygenfunction:: hipsparseDdense2csc
  :outline:
.. doxygenfunction:: hipsparseCdense2csc
  :outline:
.. doxygenfunction:: hipsparseZdense2csc

hipsparseXcsr2dense()
---------------------

.. doxygenfunction:: hipsparseScsr2dense
  :outline:
.. doxygenfunction:: hipsparseDcsr2dense
  :outline:
.. doxygenfunction:: hipsparseCcsr2dense
  :outline:
.. doxygenfunction:: hipsparseZcsr2dense

hipsparseXcsc2dense()
---------------------

.. doxygenfunction:: hipsparseScsc2dense
  :outline:
.. doxygenfunction:: hipsparseDcsc2dense
  :outline:
.. doxygenfunction:: hipsparseCcsc2dense
  :outline:
.. doxygenfunction:: hipsparseZcsc2dense

hipsparseXcsr2bsrNnz()
----------------------

.. doxygenfunction:: hipsparseXcsr2bsrNnz

hipsparseXcsr2bsr()
-------------------

.. doxygenfunction:: hipsparseScsr2bsr
  :outline:
.. doxygenfunction:: hipsparseDcsr2bsr
  :outline:
.. doxygenfunction:: hipsparseCcsr2bsr
  :outline:
.. doxygenfunction:: hipsparseZcsr2bsr

hipsparseXnnz_compress()
------------------------

.. doxygenfunction:: hipsparseSnnz_compress
  :outline:
.. doxygenfunction:: hipsparseDnnz_compress
  :outline:
.. doxygenfunction:: hipsparseCnnz_compress
  :outline:
.. doxygenfunction:: hipsparseZnnz_compress

hipsparseXcsr2coo()
-------------------

.. doxygenfunction:: hipsparseXcsr2coo

hipsparseXcsr2csc()
-------------------

.. doxygenfunction:: hipsparseScsr2csc
  :outline:
.. doxygenfunction:: hipsparseDcsr2csc
  :outline:
.. doxygenfunction:: hipsparseCcsr2csc
  :outline:
.. doxygenfunction:: hipsparseZcsr2csc

hipsparseXcsr2cscEx2_bufferSize()
---------------------------------

.. doxygenfunction:: hipsparseCsr2cscEx2_bufferSize

hipsparseXcsr2cscEx2()
----------------------

.. doxygenfunction:: hipsparseCsr2cscEx2

hipsparseXcsr2hyb()
-------------------

.. doxygenfunction:: hipsparseScsr2hyb
  :outline:
.. doxygenfunction:: hipsparseDcsr2hyb
  :outline:
.. doxygenfunction:: hipsparseCcsr2hyb
  :outline:
.. doxygenfunction:: hipsparseZcsr2hyb

hipsparseXgebsr2gebsc_bufferSize()
----------------------------------

.. doxygenfunction:: hipsparseSgebsr2gebsc_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsc_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsc_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsc_bufferSize

hipsparseXgebsr2gebsc()
-----------------------

.. doxygenfunction:: hipsparseSgebsr2gebsc
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsc
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsc
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsc

hipsparseXcsr2gebsr_bufferSize()
--------------------------------

.. doxygenfunction:: hipsparseScsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDcsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCcsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZcsr2gebsr_bufferSize

hipsparseXcsr2gebsrNnz()
------------------------

.. doxygenfunction:: hipsparseXcsr2gebsrNnz

hipsparseXcsr2gebsr()
---------------------

.. doxygenfunction:: hipsparseScsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseDcsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseCcsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseZcsr2gebsr

hipsparseXbsr2csr()
-------------------

.. doxygenfunction:: hipsparseSbsr2csr
  :outline:
.. doxygenfunction:: hipsparseDbsr2csr
  :outline:
.. doxygenfunction:: hipsparseCbsr2csr
  :outline:
.. doxygenfunction:: hipsparseZbsr2csr

hipsparseXgebsr2csr()
---------------------

.. doxygenfunction:: hipsparseSgebsr2csr
  :outline:
.. doxygenfunction:: hipsparseDgebsr2csr
  :outline:
.. doxygenfunction:: hipsparseCgebsr2csr
  :outline:
.. doxygenfunction:: hipsparseZgebsr2csr

hipsparseXcsr2csr_compress()
----------------------------

.. doxygenfunction:: hipsparseScsr2csr_compress
  :outline:
.. doxygenfunction:: hipsparseDcsr2csr_compress
  :outline:
.. doxygenfunction:: hipsparseCcsr2csr_compress
  :outline:
.. doxygenfunction:: hipsparseZcsr2csr_compress

hipsparseXpruneCsr2csr_bufferSize()
-----------------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csr_bufferSize

hipsparseXpruneCsr2csr_bufferSizeExt()
--------------------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csr_bufferSizeExt

hipsparseXpruneCsr2csrNnz()
---------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csrNnz
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrNnz

hipsparseXpruneCsr2csr()
------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csr
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csr

hipsparseXpruneCsr2csrByPercentage_bufferSize()
-----------------------------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csrByPercentage_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrByPercentage_bufferSize

hipsparseXpruneCsr2csrByPercentage_bufferSizeExt()
--------------------------------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csrByPercentage_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrByPercentage_bufferSizeExt

hipsparseXpruneCsr2csrNnzByPercentage()
---------------------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csrNnzByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrNnzByPercentage

hipsparseXpruneCsr2csrByPercentage()
------------------------------------

.. doxygenfunction:: hipsparseSpruneCsr2csrByPercentage
  :outline:
.. doxygenfunction:: hipsparseDpruneCsr2csrByPercentage

hipsparseXhyb2csr()
-------------------

.. doxygenfunction:: hipsparseShyb2csr
  :outline:
.. doxygenfunction:: hipsparseDhyb2csr
  :outline:
.. doxygenfunction:: hipsparseChyb2csr
  :outline:
.. doxygenfunction:: hipsparseZhyb2csr

hipsparseXcoo2csr()
-------------------

.. doxygenfunction:: hipsparseXcoo2csr

hipsparseCreateIdentityPermutation()
------------------------------------

.. doxygenfunction:: hipsparseCreateIdentityPermutation

hipsparseXcsrsort_bufferSizeExt()
---------------------------------

.. doxygenfunction:: hipsparseXcsrsort_bufferSizeExt

hipsparseXcsrsort()
-------------------

.. doxygenfunction:: hipsparseXcsrsort

hipsparseXcscsort_bufferSizeExt()
---------------------------------

.. doxygenfunction:: hipsparseXcscsort_bufferSizeExt

hipsparseXcscsort()
-------------------

.. doxygenfunction:: hipsparseXcscsort

hipsparseXcoosort_bufferSizeExt()
---------------------------------

.. doxygenfunction:: hipsparseXcoosort_bufferSizeExt

hipsparseXcoosortByRow()
------------------------

.. doxygenfunction:: hipsparseXcoosortByRow

hipsparseXcoosortByColumn()
---------------------------

.. doxygenfunction:: hipsparseXcoosortByColumn

hipsparseXgebsr2gebsr_bufferSize()
----------------------------------

.. doxygenfunction:: hipsparseSgebsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsr_bufferSize
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsr_bufferSize

hipsparseXgebsr2gebsrNnz()
--------------------------

.. doxygenfunction:: hipsparseXgebsr2gebsrNnz

hipsparseXgebsr2gebsr()
-----------------------

.. doxygenfunction:: hipsparseSgebsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseDgebsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseCgebsr2gebsr
  :outline:
.. doxygenfunction:: hipsparseZgebsr2gebsr

hipsparseXcsru2csr_bufferSizeExt()
----------------------------------

.. doxygenfunction:: hipsparseScsru2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseDcsru2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseCcsru2csr_bufferSizeExt
  :outline:
.. doxygenfunction:: hipsparseZcsru2csr_bufferSizeExt

hipsparseXcsru2csr()
--------------------

.. doxygenfunction:: hipsparseScsru2csr
  :outline:
.. doxygenfunction:: hipsparseDcsru2csr
  :outline:
.. doxygenfunction:: hipsparseCcsru2csr
  :outline:
.. doxygenfunction:: hipsparseZcsru2csr

hipsparseXcsr2csru()
--------------------

.. doxygenfunction:: hipsparseScsr2csru
  :outline:
.. doxygenfunction:: hipsparseDcsr2csru
  :outline:
.. doxygenfunction:: hipsparseCcsr2csru
  :outline:
.. doxygenfunction:: hipsparseZcsr2csru

.. _hipsparse_reordering_functions_:

Sparse Reordering Functions
===========================

This module holds all sparse reordering routines.

hipsparseXcsrcolor()
--------------------

.. doxygenfunction:: hipsparseScsrcolor
  :outline:
.. doxygenfunction:: hipsparseDcsrcolor
  :outline:
.. doxygenfunction:: hipsparseCcsrcolor
  :outline:
.. doxygenfunction:: hipsparseZcsrcolor

.. _hipsparse_generic_functions_:

Sparse Generic Functions
========================

This module holds all sparse generic routines.

The sparse generic routines describe operations that manipulate sparse matrices.

hipsparseAxpby()
----------------

.. doxygenfunction:: hipsparseAxpby

hipsparseGather()
-----------------

.. doxygenfunction:: hipsparseGather

hipsparseScatter()
------------------

.. doxygenfunction:: hipsparseScatter

hipsparseRot()
--------------

.. doxygenfunction:: hipsparseRot

hipsparseSparseToDense_bufferSize()
-----------------------------------

.. doxygenfunction:: hipsparseSparseToDense_bufferSize

hipsparseSparseToDense()
------------------------

.. doxygenfunction:: hipsparseSparseToDense

hipsparseDenseToSparse_bufferSize()
-----------------------------------

.. doxygenfunction:: hipsparseDenseToSparse_bufferSize

hipsparseDenseToSparse_analysis()
---------------------------------

.. doxygenfunction:: hipsparseDenseToSparse_analysis

hipsparseDenseToSparse_convert()
--------------------------------

.. doxygenfunction:: hipsparseDenseToSparse_convert

hipsparseSpVV_bufferSize()
--------------------------

.. doxygenfunction:: hipsparseSpVV_bufferSize

hipsparseSpVV()
---------------

.. doxygenfunction:: hipsparseSpVV

hipsparseSpMV_bufferSize()
--------------------------

.. doxygenfunction:: hipsparseSpMV_bufferSize

hipsparseSpMV()
---------------

.. doxygenfunction:: hipsparseSpMV

hipsparseSpMM_bufferSize()
--------------------------

.. doxygenfunction:: hipsparseSpMM_bufferSize

hipsparseSpMM_preprocess()
--------------------------

.. doxygenfunction:: hipsparseSpMM_preprocess

hipsparseSpMM()
---------------

.. doxygenfunction:: hipsparseSpMM

hipsparseSpGEMM_createDescr()
-----------------------------

.. doxygenfunction:: hipsparseSpGEMM_createDescr

hipsparseSpGEMM_destroyDescr()
------------------------------

.. doxygenfunction:: hipsparseSpGEMM_destroyDescr

hipsparseSpGEMM_workEstimation()
--------------------------------

.. doxygenfunction:: hipsparseSpGEMM_workEstimation

hipsparseSpGEMM_compute()
-------------------------

.. doxygenfunction:: hipsparseSpGEMM_compute

hipsparseSpGEMM_copy()
----------------------

.. doxygenfunction:: hipsparseSpGEMM_copy

hipsparseSDDMM_bufferSize()
---------------------------

.. doxygenfunction:: hipsparseSDDMM_bufferSize

hipsparseSDDMM_preprocess()
---------------------------

.. doxygenfunction:: hipsparseSDDMM_preprocess

hipsparseSDDMM()
----------------

.. doxygenfunction:: hipsparseSDDMM

hipsparseSpSV_createDescr()
---------------------------

.. doxygenfunction:: hipsparseSpSV_createDescr

hipsparseSpSV_destroyDescr()
----------------------------

.. doxygenfunction:: hipsparseSpSV_destroyDescr

hipsparseSpSV_bufferSize()
--------------------------

.. doxygenfunction:: hipsparseSpSV_bufferSize

hipsparseSpSV_analysis()
------------------------

.. doxygenfunction:: hipsparseSpSV_analysis

hipsparseSpSV_solve()
---------------------

.. doxygenfunction:: hipsparseSpSV_solve

hipsparseSpSM_createDescr()
---------------------------

.. doxygenfunction:: hipsparseSpSM_createDescr

hipsparseSpSM_destroyDescr()
----------------------------

.. doxygenfunction:: hipsparseSpSM_destroyDescr

hipsparseSpSM_bufferSize()
--------------------------

.. doxygenfunction:: hipsparseSpSM_bufferSize

hipsparseSpSM_analysis()
------------------------

.. doxygenfunction:: hipsparseSpSM_analysis

hipsparseSpSM_solve()
---------------------

.. doxygenfunction:: hipsparseSpSM_solve
