/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include <stdio.h>
#include <hipsparse.h>

int main(int argc, char* argv[])
{
    hipsparseHandle_t handle;
    hipsparseCreate(&handle);

    int version;
    hipsparseGetVersion(handle, &version);

    printf("hipSPARSE version %d.%d.%d\n", version / 100000, version / 100 % 1000, version % 100);

    hipsparseDestroy(handle);

    return 0;
}
