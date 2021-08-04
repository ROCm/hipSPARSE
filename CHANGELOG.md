# Change Log for hipSPARSE

## [hipSPARSE 1.10.9 for ROCm 4.4.0]
### Added
- Triangular solve for multiple right-hand sides using BSR format
- SpMV for BSRX format
- SpMM in CSR format enhanced to work with transposed A
- Matrix coloring for CSR matrices
- Added batched tridiagonal solve (gtsv\_strided\_batch)
### Improved
- Fixed a bug with gemvi on Navi21
- Optimization for pivot based gtsv
### Known Issues
- none

## [hipSPARSE 1.10.7 for ROCm 4.3.0]
### Added
- (batched) tridiagonal solve with and without pivoting
- dense matrix sparse vector multiplication (gemvi)
- sampled dense-dense matrix multiplication (sddmm)

## [hipSPARSE 1.10.6 for ROCm 4.2.0]
### Added
- Generic API support, including SpMM

## [hipSPARSE 1.10.4 for ROCm 4.1.0]
### Added
- Generic API support, including Axpby, Gather, Scatter, Rot, SpVV, SpMV, SparseToDense, DenseToSparse and SpGEMM

## [hipSPARSE 1.9.6 for ROCm 4.0.0]
### Added
- changelog
- csr2gebsr
- gebsr2csr
- gebsr2gebsc
- gebsr2gebsr
### Improved
- Updates to debian package name.

## [hipSPARSE 1.9.4 for ROCm 3.9]
### Added
- prune_csr2csr, prune_dense2csr_percentage and prune_csr2csr_percentage
- bsrilu0
### Known Issues
- none

## [hipSPARSE 1.8.1 for ROCm 3.8]
### Added
- bsric0 added.
### Known Issues
- none

## [hipSPARSE 1.7.1 for ROCm 3.7]
### Added
- Fortran bindings
- Triangular solve for BSR format (bsrsv)
- CentOS 6 support.
### Known Issues
- none

## [hipSPARSE 1.7.1 for ROCm 3.6]
### Added
- Fortran bindings
- Triangular solve for BSR format (bsrsv)
- CentOS 6 support.
### Known Issues
- none

## [hipSPARSE 1.6.5 for ROCm 3.5]
### Added
- Switched to hip-clang as default compiler
- csr2dense, csc2dense, csr2csr_compress, nnz_compress, bsr2csr, csr2bsr, bsrmv, csrgeam
- static build
- more examples
### Optimized
- dense2csr, dense2csc
### Improved
- Installation process
### Known Issues
- none
