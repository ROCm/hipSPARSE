# Changelog for hipSPARSE

Documentation for hipSPARSE is available at
[https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/).

## hipSPARSE 3.1.1 for ROCm 6.2.0

### Additions

* Added missing `hipsparseCscGet()` routine

### Changes

* All internal hipSPARSE functions now exist inside a namespace
* Match deprecations found in cusparse 12.x.x when using cusparse backend.

### Fixes

* Fixed SpGEMM and SpGEMM_reuse routines which were not matching cuSPARSE behaviour

### Optimizations

* Improved user manual 
* Improved contribution guidelines

### Known issues

* In `hipsparseSpSM_solve()`, we currently pass the external buffer as a parameter. This does not match the cuSPARSE API and this extra external buffer parameter will be removed in a future release. For now this extra parameter can be ignored and nullptr passed as it is unused internally by `hipsparseSpSM_solve()`.

## hipSPARSE 3.0.1 for ROCm 6.1.0

### Fixes
* Fixes to the build chain

## hipSPARSE 3.0.0 for ROCm 6.0.0

### Additions

* Added `hipsparseGetErrorName` and `hipsparseGetErrorString`

### Changes

* Changed the `hipsparseSpSV_solve()` API function to match the cuSPARSE API
* Changed generic API functions to use const descriptors
* Improved documentation

## hipSPARSE 2.3.8 for ROCm 5.7.0

### Fixes

* Compilation failures when using the cuSPARSE 12.1.0 and 12.0.0 backends
* Compilation failures when using the cuSPARSE 10.1 (non-update version) backend

## hipSPARSE 2.3.7 for ROCm 5.6.1

### Fixes

* Reverted an undocumented API change in hipSPARSE 2.3.6 that affected the `hipsparseSpSV_solve`
  function

## hipSPARSE 2.3.6 for ROCm 5.6.0

### Additions

* Added SpGEMM algorithms

### Changes

* blockDim == 0 now returns `HIPSPARSE_STATUS_INVALID_SIZE` for `hipsparseXbsr2csr` and
  `hipsparseXcsr2bsr`

## hipSPARSE 2.3.5 for ROCm 5.5.0

### Fixes

* Fixed an issue where the `rocm` folder was not removed after upgrading meta packages
* Fixed a compilation issue with the cuSPARSE backend
* Added more detailed messages for unit test failures due related to missing input data
* Improved documentation
* Fixed a bug with deprecation messages when using gcc9

## hipSPARSE 2.3.3 for ROCm 5.4.0

### Additions

* Added `hipsparseCsr2cscEx2_bufferSize` and `hipsparseCsr2cscEx2` routines

### Changes

* `HIPSPARSE_ORDER_COLUMN` has been renamed to `HIPSPARSE_ORDER_COL` in order to match
    cuSPARSE

## hipSPARSE 2.3.1 for ROCm 5.3.0

### Additions

* Added SpMM and SpMM batched for CSC format

## hipSPARSE 2.2.0 for ROCm 5.2.0

### Additions

* New packages for test and benchmark executables on all supported operating systems using CPack

## hipSPARSE 2.1.0 for ROCm 5.1.0

### Additions

* Added `gtsv_interleaved_batch` and `gpsv_interleaved_batch` routines
* Added `SpGEMM_reuse`

### Changes

* Changed `BUILD_CUDA` with `USE_CUDA` in the install script and CMake files
* Updated GoogleTest to 11.1

### Fixes

* Fixed a bug in SpMM Alg versioning

## hipSPARSE 2.0.0 for ROCm 5.0.0

### Additions

* Added (conjugate) transpose support for `csrmv`, `hybmv`, and `spmv` routines

## hipSPARSE 1.11.2 for ROCm 4.5.0

### Additions

* Triangular solve for multiple right-hand sides using BSR format
* SpMV for BSRX format
* Enhanced SpMM in CSR format to work with transposed A
* Matrix coloring for CSR matrices
* Batched tridiagonal solve (`gtsv_strided_batch`)
* SpMM for BLOCKED ELL format
* Generic routines for SpSV and SpSM
* Beta support for Windows 10
* Additional atomic-based algorithms for SpMM in COO format
* Additional algorithm for SpMM in CSR format

### Changes

* Packaging has been split into a runtime package (`hipsparse`) and a development package
  (`hipsparse-devel`):
  The development package depends on the runtime package. When installing the runtime package,
  the package manager will suggest the installation of the development package to aid users
  transitioning from the previous version's combined package. This suggestion by package manager is
  for all supported operating systems (except CentOS 7) to aid in the transition. The `suggestion`
  feature in the runtime package is introduced as a deprecated feature and will be removed in a future
  ROCm release.

* GTest dependency has been updated to v1.10.0

### Fixes

* Fixed a bug with `gemvi` on Navi21

### Optimizations

* Optimization for pivot-based GTSV

## hipSPARSE 1.10.7 for ROCm 4.3.0

### Additions

* Tridiagonal solve with and without pivoting (batched)
* Dense matrix sparse vector multiplication (`gemvi`)
* Sampled dense-dense matrix multiplication (`sddmm`)

## hipSPARSE 1.10.6 for ROCm 4.2.0

### Additions

* Generic API support, including SpMM

## hipSPARSE 1.10.4 for ROCm 4.1.0

### Additions

* Generic API support, including Axpby, Gather, Scatter, Rot, SpVV, SpMV, SparseToDense,
  DenseToSparse, and SpGEMM

## hipSPARSE 1.9.6 for ROCm 4.0.0

### Additions

* Changelog file
* `csr2gebsr`
* `gebsr2csr`
* `gebsr2gebsc`
* `gebsr2gebsr`

### Changes

* Updates to Debian package name.

## hipSPARSE 1.9.4 for ROCm 3.9

### Additions

* `prune_csr2csr, prune_dense2csr_percentage` and `prune_csr2csr_percentage`
* `bsrilu0`

## hipSPARSE 1.8.1 for ROCm 3.8

### Additions

* `bsric0`

## hipSPARSE 1.7.1 for ROCm 3.7

### Additions

* Fortran bindings
* Triangular solve for BSR format (bsrsv)
* CentOS 6 support

## hipSPARSE 1.7.1 for ROCm 3.6

### Additions

* Fortran bindings
* Triangular solve for BSR format (bsrsv)
* CentOS 6 support

## hipSPARSE 1.6.5 for ROCm 3.5

### Additions

* Switched to HIP-Clang as default compiler
* `csr2dense`, `csc2dense`, `csr2csr_compress`, `nnz_compress`, `bsr2csr`, `csr2bsr`, `bsrmv`, and
  `csrgeam`
* static build
* New examples

### Optimizations

* `dense2csr`, `dense2csc`

### Fixes

* Installation process
