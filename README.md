# hipSPARSE

hipSPARSE is a SPARSE marshalling library with multiple supported backends. It sits between your
application and a 'worker' SPARSE library, where it marshals inputs to the backend library and marshals
results to your application. hipSPARSE exports an interface that doesn't require the client to change,
regardless of the chosen backend. Currently, hipSPARSE supports
[rocSPARSE](https://github.com/ROCmSoftwarePlatform/rocSPARSE) and
[cuSPARSE](https://developer.nvidia.com/cusparse) backends.

## Documentation

Documentation for hipSPARSE is available at
[https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/](https://rocm.docs.amd.com/projects/hipSPARSE/en/latest/).

To build our documentation locally, run the following code:

```bash
cd docs

pip3 install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

Alternatively, build with CMake:

```bash
cmake -DBUILD_DOCS=ON ...
```

## Installing pre-built packages

Download pre-built packages from
[ROCm's package servers](https://rocm.docs.amd.com/en/latest/deploy/linux/index.html) using the
following code:

```bash
`sudo apt update && sudo apt install hipsparse`
```

## Build hipSPARSE

To build hipSPARSE, you can use our bash helper script (for Ubuntu only) or you can perform a manual
build (for all supported platforms).

* Bash helper script (`install.sh`):
  This script, which is located in the root of this repository, builds and installs hipSPARSE on Ubuntu
  with a single command. Note that this option doesn't allow much customization and hard-codes
  configurations that can be specified through invoking CMake directly. Some commands in the script
  require sudo access, so it may prompt you for a password.

    ```bash
    `./install -h`  # shows help
    `./install -id` # builds library, dependencies, then installs (the `-d` flag only needs to be passed once on a system)
    ```

* Manual build:
    If you use a distribution other than Ubuntu, or would like more control over the build process,
    the [hipSPARSE build wiki](https://github.com/ROCmSoftwarePlatform/hipSPARSE/wiki/Build)
    provides information on how to configure CMake and build hipSPARSE manually.

### Supported functions

You can find a list of
[exported functions](https://github.com/ROCmSoftwarePlatform/hipSPARSE/wiki/Exported-functions)
on our wiki.

## Interface examples

The hipSPARSE interface is compatible with rocSPARSE and cuSPARSE-v2 APIs. Porting a CUDA
application that calls the cuSPARSE API to an application that calls the hipSPARSE API is relatively
straightforward. For example, the hipSPARSE SCSRMV interface is:

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

hipSPARSE assumes matrix A and vectors x, y are allocated in GPU memory space filled with data. Users
are responsible for copying data to and from the host and device memory.
