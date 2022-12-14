# Change Log for hipSPARSE

## hipSPARSE 2.3.5 for ROCm 5.5.0
### Improved
- Fixed an issue, where the rocm folder was not removed on upgrade of meta packages
- Fixed a compilation issue with cusparse backend
- Added more detailed messages on unit test failures due to missing input data
- Improved documentation
- Fixed a bug with deprecation messages when using gcc9 (Thanks @Maetveis)

## hipSPARSE 2.3.3 for ROCm 5.4.0
### Added
- Added hipsparseCsr2cscEx2_bufferSize and hipsparseCsr2cscEx2 routines
### Changed
- HIPSPARSE_ORDER_COLUMN has been renamed to HIPSPARSE_ORDER_COL to match cusparse

## hipSPARSE 2.3.1 for ROCm 5.3.0
### Added
- Add SpMM and SpMM batched for CSC format

## hipSPARSE 2.2.0 for ROCm 5.2.0
### Added
- Packages for test and benchmark executables on all supported OSes using CPack.

## hipSPARSE 2.1.0 for ROCm 5.1.0
### Added
- Added gtsv_interleaved_batch and gpsv_interleaved_batch routines
- Add SpGEMM_reuse
### Changed
- Changed BUILD_CUDA with USE_CUDA in install script and cmake files
- Update googletest to 11.1
### Improved
- Fixed a bug in SpMM Alg versioning
### Known Issues
- none

## hipSPARSE 2.0.0 for ROCm 5.0.0
### Added
- Added (conjugate) transpose support for csrmv, hybmv and spmv routines

## hipSPARSE 1.11.2 for ROCm 4.5.0
### Added
- Triangular solve for multiple right-hand sides using BSR format
- SpMV for BSRX format
- SpMM in CSR format enhanced to work with transposed A
- Matrix coloring for CSR matrices
- Added batched tridiagonal solve (gtsv\_strided\_batch)
- SpMM for BLOCKED ELL format
- Generic routines for SpSV and SpSM
- Enabling beta support for Windows 10
- Additional atomic based algorithms for SpMM in COO format
- Additional algorithm for SpMM in CSR format
### Changed
- Packaging split into a runtime package called hipsparse and a development package called hipsparse-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
- GTest dependency updated to v1.10.0
### Improved
- Fixed a bug with gemvi on Navi21
- Optimization for pivot based gtsv
### Known Issues
- none

## hipSPARSE 1.10.7 for ROCm 4.3.0
### Added
- (batched) tridiagonal solve with and without pivoting
- dense matrix sparse vector multiplication (gemvi)
- sampled dense-dense matrix multiplication (sddmm)

## hipSPARSE 1.10.6 for ROCm 4.2.0
### Added
- Generic API support, including SpMM

## hipSPARSE 1.10.4 for ROCm 4.1.0
### Added
- Generic API support, including Axpby, Gather, Scatter, Rot, SpVV, SpMV, SparseToDense, DenseToSparse and SpGEMM

## hipSPARSE 1.9.6 for ROCm 4.0.0
### Added
- changelog
- csr2gebsr
- gebsr2csr
- gebsr2gebsc
- gebsr2gebsr
### Improved
- Updates to debian package name.

## hipSPARSE 1.9.4 for ROCm 3.9
### Added
- prune\_csr2csr, prune\_dense2csr\_percentage and prune\_csr2csr\_percentage
- bsrilu0
### Known Issues
- none

## hipSPARSE 1.8.1 for ROCm 3.8
### Added
- bsric0 added.
### Known Issues
- none

## hipSPARSE 1.7.1 for ROCm 3.7
### Added
- Fortran bindings
- Triangular solve for BSR format (bsrsv)
- CentOS 6 support.
### Known Issues
- none

## hipSPARSE 1.7.1 for ROCm 3.6
### Added
- Fortran bindings
- Triangular solve for BSR format (bsrsv)
- CentOS 6 support.
### Known Issues
- none

## hipSPARSE 1.6.5 for ROCm 3.5
### Added
- Switched to hip-clang as default compiler
- csr2dense, csc2dense, csr2csr\_compress, nnz\_compress, bsr2csr, csr2bsr, bsrmv, csrgeam
- static build
- more examples
### Optimized
- dense2csr, dense2csc
### Improved
- Installation process
### Known Issues
- none
