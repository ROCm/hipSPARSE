.. meta::
  :description: hipSPARSE documentation and API reference library
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation

.. _hipsparse:

********************************************************************
hipSPARSE documentation
********************************************************************

hipSPARSE exposes a common interface that provides basic linear algebra subroutines for sparse computation implemented on top of the AMD ROCm runtime and toolchains. hipSPARSE is a SPARSE marshalling library supporting both rocSPARSE and cuSPARSE as backends.
It sits between the application and a `worker` SPARSE library, marshalling inputs into the backend library and marshalling results back to the application. hipSPARSE is created using the HIP programming language and optimized for AMD's latest discrete GPUs.

The code is open and hosted here: https://github.com/ROCmSoftwarePlatform/hipSPARSE

The hipSPARSE documentation is structured as follows:

.. grid:: 2

  .. grid-item-card:: Installation
  
    * :ref:`hipsparse_building`
  
  .. grid-item-card:: How-To
  
    * :ref:`hipsparse-docs`
  
  .. grid-item-card:: API reference
  
    * :ref:`api`
    * :ref:`hipsparse-types`
    * :ref:`hipsparse_auxiliary_functions` required for subsequent library calls
    * :ref:`hipsparse_level1_functions` between a vector in sparse format and a vector in dense format
    * :ref:`hipsparse_level2_functions` between a matrix in sparse format and a vector in dense format
    * :ref:`hipsparse_level3_functions` between a matrix in sparse format and multiple vectors (matrix) in dense format
    * :ref:`hipsparse_extra_functions` for proposing linear algebra operations
    * :ref:`hipsparse_precond_functions` on a matrix in sparse format to obtain a preconditioner
    * :ref:`hipsparse_conversion_functions` to convert a matrix in sparse format to a different storage format
    * :ref:`hipsparse_reordering_functions` for reordering sparse matrices
    * :ref:`hipsparse_generic_functions` for manipulating sparse matrices 

.. note::
  hipSPARSE exports an interface that does not require the client to change, regardless of the chosen backend. hipSPARSE focuses on convenience and portability. If performance outweighs these factors, then using rocSPARSE itself is highly recommended. rocSPARSE can also be found on `GitHub <https://github.com/ROCmSoftwarePlatform/rocSPARSE/>`_.

To contribute to the documentation refer to `Contributing to ROCm  <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the `Licensing <https://rocm.docs.amd.com/en/latest/about/license.html>`_ page.


