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