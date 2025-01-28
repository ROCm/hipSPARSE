.. meta::
  :description: hipSPARSE documentation and API reference library
  :keywords: hipSPARSE, rocSPARSE, ROCm, API, documentation

.. _hipsparse_building:

********************************************************************
Installation
********************************************************************

Prerequisites
=============
hipSPARSE requires a ROCm enabled platform.

Installing pre-built packages
=============================

hipSPARSE can be installed from AMD ROCm repository.
For detailed instructions on how to set up ROCm on different platforms, see the `AMD ROCm Platform Installation Guide for Linux <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_.

hipSPARSE can be installed on Ubuntu using

::

    $ sudo apt-get update
    $ sudo apt-get install hipsparse

Once installed, hipSPARSE can be used just like any other library with a C API.
The header file will need to be included in the user code in order to make calls into hipSPARSE, and the hipSPARSE shared library will become link-time and run-time dependent for the user application.

Building hipSPARSE from source
==============================

Building from source is not necessary, as hipSPARSE can be used after installing the pre-built packages as described above.
If desired, the following instructions can be used to build hipSPARSE from source.
Furthermore, the following compile-time dependencies must be met

- `rocSPARSE <https://github.com/ROCmSoftwarePlatform/rocSPARSE>`_
- `git <https://git-scm.com/>`_
- `CMake <https://cmake.org/>`_ 3.5 or later
- `AMD ROCm <https://github.com/ROCm/ROCm>`_
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

- Ubuntu 20.04
- Ubuntu 22.04
- RHEL 8
- RHEL 9
- SLES 15

To compile and run hipSPARSE, `AMD ROCm Platform <https://github.com/ROCm/ROCm>`_ is required.
