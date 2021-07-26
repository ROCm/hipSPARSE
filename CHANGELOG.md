# Change Log for hipSPARSE

## (Unreleased) hipSPARSE 1.10.9

### Changed
- Packaging split into a runtime package called hipsparse and a development package called hipsparse-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.

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
