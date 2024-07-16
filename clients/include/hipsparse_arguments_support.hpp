/*! \file */
/* ************************************************************************
* Copyright (C) 2024 Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */
#pragma once

#include <vector>

#include <hipsparse.h>

struct csr2csc_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_CSR2CSC_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_CSR2CSC_ALG_DEFAULT;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
        return HIPSPARSE_CSR2CSC_ALG1;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running csr2csc. Possible choices are "
               "default: 0, Alg1: 1, Alg2: 2 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running csr2csc. Possible choices are "
               "default: 0, Alg1: 1 (default:0)";
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running csr2csc. Possible choices are "
               "default: 1, Alg1: 1, Alg2: 2 (default:1)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }

    static std::vector<int> get_supported_algorithms()
    {
#if(!defined(CUDART_VERSION))
        return std::vector<int>({HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1, HIPSPARSE_CSR2CSC_ALG2});
#else
#if(CUDART_VERSION >= 12000)
        return std::vector<int>({HIPSPARSE_CSR2CSC_ALG_DEFAULT, HIPSPARSE_CSR2CSC_ALG1});
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 12000)
        return std::vector<int>({HIPSPARSE_CSR2CSC_ALG1, HIPSPARSE_CSR2CSC_ALG2});
#endif
#endif
        return std::vector<int>();
    }
};

struct dense2sparse_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return "Indicates what algorithm to use when running dense2sparse. Possible choices are "
               "default: 0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct sparse2dense_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11020)
        return "Indicates what algorithm to use when running sparse2dense. Possible choices are "
               "default: 0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct sddmm_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
        return HIPSPARSE_SDDMM_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11022)
        return "Indicates what algorithm to use when running sddmm. Possible choices are default: "
               "0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct spgemm_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_SPGEMM_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_SPGEMM_DEFAULT;
#elif(CUDART_VERSION >= 11031 && CUDART_VERSION < 12000)
        return HIPSPARSE_SPGEMM_DEFAULT;
#elif(CUDART_VERSION >= 11000)
        return HIPSPARSE_SPGEMM_DEFAULT;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0, Deterministic: 1, Non-Deterministic: 2, Alg1: 3, Alg2: 4, Alg3: 5 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0, Deterministic: 1, Non-Deterministic: 2, Alg1: 3, Alg2: 4, Alg3: 5 (default:0)";
#elif(CUDART_VERSION >= 11031 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0, Deterministic: 1, Non-Deterministic: 2 (default:0)";
#elif(CUDART_VERSION >= 11000)
        return "Indicates what algorithm to use when running spgemm. Possible choices are default: "
               "0 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }
};

struct spmm_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
        return HIPSPARSE_SPMM_ALG_DEFAULT;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
        return HIPSPARSE_MM_ALG_DEFAULT;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5, CSR Alg2: 6, "
               "CSR Alg3: 12, Blocked ELL Alg1: 13 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5 CSR Alg2: 6, "
               "CSR Alg3: 12, Blocked ELL Alg1: 13 (default:0)";
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5, CSR Alg2: 6, "
               "CSR Alg3: 12, Blocked ELL Alg1: 13 (default:0)";
#elif(CUDART_VERSION >= 11003 && CUDART_VERSION < 11021)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4, COO Alg4: 5, CSR Alg2: 6, "
               "Blocked ELL Alg1: 13 (default:0)";
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11003)
        return "Indicates what algorithm to use when running spmm. Possible choices are default: "
               "0, COO Alg1: 1, COO Alg2: 2, COO Alg3: 3, CSR Alg1: 4  (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }
};

struct spmv_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION))
        return HIPSPARSE_SPMV_ALG_DEFAULT;
#else
#if(CUDART_VERSION >= 12000)
        return HIPSPARSE_SPMV_ALG_DEFAULT;
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return HIPSPARSE_SPMV_ALG_DEFAULT;
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
        return HIPSPARSE_MV_ALG_DEFAULT;
#else
        return -1;
#endif
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION))
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg1: 1, CSR Alg1: 2, CSR Alg2: 3, COO Alg2: 4 (default:0)";
#else
#if(CUDART_VERSION >= 12000)
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg1: 1, CSR Alg1: 2, CSR Alg2: 3, COO Alg2: 4 (default:0)";
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg1: 1, CSR Alg1: 2, CSR Alg2: 3, COO Alg2: 4 (default:0)";
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
        return "Indicates what algorithm to use when running spmv. Possible choices are default: "
               "0, COO Alg: 1, CSR Alg1: 2, CSR Alg2: 3 (default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
#endif
    }
};

struct spsm_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
        return HIPSPARSE_SPSM_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11031)
        return "Indicates what algorithm to use when running spsm. Possible choices are default: 0 "
               "(default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};

struct spsv_alg_support
{
    static int get_default_algorithm()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
        return HIPSPARSE_SPSV_ALG_DEFAULT;
#else
        return -1;
#endif
    }

    static std::string get_description()
    {
#if(!defined(CUDART_VERSION) || CUDART_VERSION >= 11030)
        return "Indicates what algorithm to use when running spsv. Possible choices are default: 0 "
               "(default:0)";
#else
        return "No algorithm supported in selected cusparse version";
#endif
    }
};
