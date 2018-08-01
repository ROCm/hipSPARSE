# hipSPARSE
hipSPARSE is a SPARSE marshalling library, with multiple supported backends. It sits between the application and a 'worker' SPARSE library, marshalling inputs into the backend library and marshalling results back to the application. hipSPARSE exports an interface that does not require the client to change, regardless of the chosen backend. Currently, hipSPARSE supports [rocSPARSE](https://github.com/ROCmSoftwarePlatform/rocSPARSE) and [cuSPARSE](https://developer.nvidia.com/cusparse) as backends.

## Installing pre-built packages
Download pre-built packages either from [ROCm's package servers](https://rocm.github.io/install.html#installing-from-amd-rocm-repositories) or by clicking the github releases tab and manually downloading, which could be newer. Release notes are available for each release on the releases tab.
* `sudo apt update && sudo apt install hipsparse`

## Quickstart hipSPARSE build

#### Bash helper build script (Ubuntu only)
The root of this repository has a helper bash script `install.sh` to build and install hipSPARSE on Ubuntu with a single command.  It does not take a lot of options and hard-codes configuration that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install. A few commands in the script need sudo access, so it may prompt you for a password.
*  `./install -h`  -- shows help
*  `./install -id` -- build library, build dependencies and install (-d flag only needs to be passed once on a system)

## Manual build (all supported platforms)
If you use a distro other than Ubuntu, or would like more control over the build process, the [hipSPARSE build wiki](https://github.com/ROCmSoftwarePlatform/hipSPARSE/wiki/Build) has helpful information on how to configure cmake and manually build.

### Functions supported
A list of [exported functions](https://github.com/ROCmSoftwarePlatform/hipSPARSE/wiki/Exported-functions) from hipSPARSE can be found on the wiki.

## hipSPARSE interface examples
The hipSPARSE interface is compatible with rocSPARSE and cuSPARSE-v2 APIs. Porting a CUDA application which originally calls the cuSPARSE API to an application calling hipSPARSE API should be relatively straightforward. For example, the hipSPARSE SCSRMV interface is

### CSRMV API

```c
hipsparseStatus_t
hipsparseScsrmv(hipsparseHandle_t handle,
                hipsparseOperation_t transA,
                int m, int n, int nnz, const float *alpha,
                const hipsparseMatDescr_t descrA,
                const float *csrValA,
                const int *csrRowPtrA, const int *csrColIndA,
                const float *x, const float *beta,
                float *y);
```

hipSPARSE assumes matrix A and vectors x, y are allocated in GPU memory space filled with data. Users are responsible for copying data from/to the host and device memory.
