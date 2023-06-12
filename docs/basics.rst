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
