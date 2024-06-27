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

/*! \file
 *  \brief hipsparse_arguments.hpp provides a class to parse command arguments in both,
 *  clients and gtest. If class structure is changed, rocsparse_common.yaml must also be
 *  changed.
 */

#pragma once

#include <cstring>
#include <iomanip>
#include <cmath>
#include <iostream>

#include "hipsparse_datatype2string.hpp"

template <typename T>
static T convert_alpha_beta(double r, double i)
{
    return static_cast<T>(r);
}

struct Arguments
{
    int M;
    int N;
    int K;
    int nnz;
    int block_dim;
    int row_block_dimA;
    int col_block_dimA;
    int row_block_dimB;
    int col_block_dimB;

    int lda;
    int ldb;
    int ldc;

    int batch_count;

    hipsparseIndexType_t index_type_I;
    hipsparseIndexType_t index_type_J;
    hipDataType compute_type;

    double alpha;
    double alphai;
    double beta;
    double betai;
    double threshold;
    double percentage;

    hipsparseOperation_t transA;
    hipsparseOperation_t transB;
    hipsparseIndexBase_t baseA;
    hipsparseIndexBase_t baseB;
    hipsparseIndexBase_t baseC;
    hipsparseIndexBase_t baseD;

    hipsparseAction_t       action;
    hipsparseHybPartition_t part;
    hipsparseDiagType_t     diag_type;
    hipsparseFillMode_t     fill_mode;

    hipsparseDirection_t dirA;
    hipsparseOrder_t     orderA;
    hipsparseOrder_t     orderB;
    hipsparseOrder_t     orderC;
    hipsparseFormat_t    formatA;
    hipsparseFormat_t    formatB;
   
    int    numericboost;
    double boosttol;
    double boostval;
    double boostvali;

    int ell_width;
    int permute;

    int unit_check;
    int timing;
    int iters;

    std::string filename;
    std::string function_name;

    Arguments()
    {
        this->M = -1;
        this->N = -1;
        this->K = -1;
        this->nnz = -1;
        this->block_dim = 2;
        this->row_block_dimA = 2;
        this->col_block_dimA = 2;
        this->row_block_dimB = 2;
        this->col_block_dimB = 2;

        this->lda = -1;
        this->ldb = -1;
        this->ldc = -1;

        this->batch_count = -1;

        this->index_type_I = HIPSPARSE_INDEX_32I;
        this->index_type_J = HIPSPARSE_INDEX_32I;
        this->compute_type = HIP_R_32F;

        this->alpha = 0.0;
        this->alphai = 0.0;
        this->beta = 0.0;
        this->betai = 0.0;
        this->threshold = 0.0;
        this->percentage = 0.0;

        this->transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->baseA = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseB = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseC = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseD = HIPSPARSE_INDEX_BASE_ZERO;

        this->action = HIPSPARSE_ACTION_NUMERIC;
        this->part = HIPSPARSE_HYB_PARTITION_AUTO;
        this->diag_type = HIPSPARSE_DIAG_TYPE_NON_UNIT;
        this->fill_mode = HIPSPARSE_FILL_MODE_LOWER;

        this->dirA = HIPSPARSE_DIRECTION_ROW;
        this->orderA = HIPSPARSE_ORDER_COL;
        this->orderB = HIPSPARSE_ORDER_COL;
        this->orderC = HIPSPARSE_ORDER_COL;
        this->formatA = HIPSPARSE_FORMAT_COO;
        this->formatB = HIPSPARSE_FORMAT_COO;
    
        this->numericboost = 0;
        this->boosttol = 0.0;
        this->boostval = 1.0;
        this->boostvali = 0.0;

        this->ell_width = 0;
        this->permute = 0;

        this->unit_check = 1;
        this->timing = 0;
        this->iters = 10;

        this->filename = "";
        this->function_name = "";
    }

    // Validate input format.
    static void validate(std::istream& ifs)
    {
        auto error = [](auto name) {
            std::cerr << "Arguments field " << name << " does not match format.\n\n"
                      << "Fatal error: Binary test data does match input format.\n"
                         "Ensure that rocsparse_arguments.hpp and rocsparse_common.yaml\n"
                         "define exactly the same Arguments, that rocsparse_gentest.py\n"
                         "generates the data correctly, and that endianness is the same.\n";
            abort();
        };

        char      header[10]{}, trailer[10]{};
        Arguments arg{};
        ifs.read(header, sizeof(header));
        ifs >> arg;
        ifs.read(trailer, sizeof(trailer));

        if(strcmp(header, "rocSPARSE"))
            error("header");
        else if(strcmp(trailer, "ROCsparse"))
            error("trailer");

        auto check_func = [&, sig = (uint8_t)0](const auto& elem, auto name) mutable {
            static_assert(sizeof(elem) <= 255,
                          "One of the fields of Arguments is too large (> 255 bytes)");
            for(uint8_t i = 0; i < sizeof(elem); ++i)
                if(reinterpret_cast<const uint8_t*>(&elem)[i] ^ sig ^ i)
                    error(name);
            sig += 89;
        };

#define ROCSPARSE_FORMAT_CHECK(x) check_func(arg.x, #x)

        // Order is important
        ROCSPARSE_FORMAT_CHECK(M);
        ROCSPARSE_FORMAT_CHECK(N);
        ROCSPARSE_FORMAT_CHECK(K);
        ROCSPARSE_FORMAT_CHECK(nnz);
        ROCSPARSE_FORMAT_CHECK(block_dim);
        ROCSPARSE_FORMAT_CHECK(row_block_dimA);
        ROCSPARSE_FORMAT_CHECK(col_block_dimA);
        ROCSPARSE_FORMAT_CHECK(row_block_dimB);
        ROCSPARSE_FORMAT_CHECK(col_block_dimB);

        ROCSPARSE_FORMAT_CHECK(lda);
        ROCSPARSE_FORMAT_CHECK(ldb);
        ROCSPARSE_FORMAT_CHECK(ldc);

        ROCSPARSE_FORMAT_CHECK(batch_count);

        ROCSPARSE_FORMAT_CHECK(index_type_I);
        ROCSPARSE_FORMAT_CHECK(index_type_J);
        ROCSPARSE_FORMAT_CHECK(compute_type);
        ROCSPARSE_FORMAT_CHECK(alpha);
        ROCSPARSE_FORMAT_CHECK(alphai);
        ROCSPARSE_FORMAT_CHECK(beta);
        ROCSPARSE_FORMAT_CHECK(betai);
        ROCSPARSE_FORMAT_CHECK(threshold);
        ROCSPARSE_FORMAT_CHECK(percentage);
        ROCSPARSE_FORMAT_CHECK(transA);
        ROCSPARSE_FORMAT_CHECK(transB);
        ROCSPARSE_FORMAT_CHECK(baseA);
        ROCSPARSE_FORMAT_CHECK(baseB);
        ROCSPARSE_FORMAT_CHECK(baseC);
        ROCSPARSE_FORMAT_CHECK(baseD);
        ROCSPARSE_FORMAT_CHECK(action);
        ROCSPARSE_FORMAT_CHECK(part);
        ROCSPARSE_FORMAT_CHECK(diag_type);
        ROCSPARSE_FORMAT_CHECK(fill_mode);
        ROCSPARSE_FORMAT_CHECK(dirA);
        ROCSPARSE_FORMAT_CHECK(orderA);
        ROCSPARSE_FORMAT_CHECK(orderB);
        ROCSPARSE_FORMAT_CHECK(orderC);
        ROCSPARSE_FORMAT_CHECK(formatA);
        ROCSPARSE_FORMAT_CHECK(formatB);
        ROCSPARSE_FORMAT_CHECK(numericboost);
        ROCSPARSE_FORMAT_CHECK(boosttol);
        ROCSPARSE_FORMAT_CHECK(boostval);
        ROCSPARSE_FORMAT_CHECK(boostvali);
        ROCSPARSE_FORMAT_CHECK(ell_width);
        ROCSPARSE_FORMAT_CHECK(permute);
        ROCSPARSE_FORMAT_CHECK(unit_check);
        ROCSPARSE_FORMAT_CHECK(timing);
        ROCSPARSE_FORMAT_CHECK(iters);
        ROCSPARSE_FORMAT_CHECK(filename);
        ROCSPARSE_FORMAT_CHECK(function_name);
    }

    template <typename T>
    T get_alpha() const
    {
        return convert_alpha_beta<T>(alpha, alphai);
    }

    template <typename T>
    T get_beta() const
    {
        return convert_alpha_beta<T>(beta, betai);
    }

    template <typename T>
    T get_threshold() const
    {
        return threshold;
    }

    template <typename T>
    T get_percentage() const
    {
        return percentage;
    }

private:
    // Function to read Structures data from stream
    friend std::istream& operator>>(std::istream& str, Arguments& arg)
    {
        str.read(reinterpret_cast<char*>(&arg), sizeof(arg));
        return str;
    }

    // print_value is for formatting different data types

    // Default output
    template <typename T>
    static void print_value(std::ostream& str, const T& x)
    {
        str << x;
    }

    // Floating-point output
    static void print_value(std::ostream& str, double x)
    {
        if(std::isnan(x))
            str << ".nan";
        else if(std::isinf(x))
            str << (x < 0 ? "-.inf" : ".inf");
        else
        {
            char s[32];
            snprintf(s, sizeof(s) - 2, "%.17g", x);

            // If no decimal point or exponent, append .0
            char* end = s + strcspn(s, ".eE");
            if(!*end)
                strcat(end, ".0");
            str << s;
        }
    }

    // Character output
    static void print_value(std::ostream& str, char c)
    {
        char s[]{c, 0};
        str << std::quoted(s, '\'');
    }

    // bool output
    static void print_value(std::ostream& str, bool b)
    {
        str << (b ? "true" : "false");
    }

    // string output
    static void print_value(std::ostream& str, const char* s)
    {
        str << std::quoted(s);
    }

    // Function to print Arguments out to stream in YAML format
    // Google Tests uses this automatically to dump parameters
    friend std::ostream& operator<<(std::ostream& str, const Arguments& arg)
    {
        // delim starts as '{' opening brace and becomes ',' afterwards
        auto print = [&, delim = '{'](const char* name, auto x) mutable {
            str << delim << " " << name << ": ";
            print_value(str, x);
            delim = ',';
        };

        print("function_name", arg.function_name);
        print("index_type_I", hipsparse_indextype2string(arg.index_type_I));
        print("index_type_J", hipsparse_indextype2string(arg.index_type_J));
        print("compute_type", hipsparse_datatype2string(arg.compute_type));
        print("transA", hipsparse_operation2string(arg.transA));
        print("transB", hipsparse_operation2string(arg.transB));
        print("baseA", hipsparse_indexbase2string(arg.baseA));
        print("baseB", hipsparse_indexbase2string(arg.baseB));
        print("baseC", hipsparse_indexbase2string(arg.baseC));
        print("baseD", hipsparse_indexbase2string(arg.baseD));
        print("M", arg.M);
        print("N", arg.N);
        print("K", arg.K);
        print("nnz", arg.nnz);
        print("block_dim", arg.block_dim);
        print("row_block_dimA", arg.row_block_dimA);
        print("col_block_dimA", arg.col_block_dimA);
        print("row_block_dimB", arg.row_block_dimB);
        print("col_block_dimB", arg.col_block_dimB);

        print("lda", arg.lda);
        print("ldb", arg.ldb);
        print("ldc", arg.ldc);

        print("batch_count", arg.batch_count);

        print("alpha", arg.alpha);
        print("alphai", arg.alphai);
        print("beta", arg.beta);
        print("betai", arg.betai);
        print("threshold", arg.threshold);
        print("percentage", arg.percentage);

        print("action", hipsparse_action2string(arg.action));
        print("part", hipsparse_partition2string(arg.part));
        print("diag_type", hipsparse_diagtype2string(arg.diag_type));
        print("fill_mode", hipsparse_fillmode2string(arg.fill_mode));

        print("dirA", hipsparse_direction2string(arg.dirA));
        print("orderA", hipsparse_order2string(arg.orderA));
        print("orderB", hipsparse_order2string(arg.orderB));
        print("orderC", hipsparse_order2string(arg.orderC));
        print("formatA", hipsparse_format2string(arg.formatA));
        print("formatB", hipsparse_format2string(arg.formatB));

        print("numeric_boost", arg.numericboost);
        print("boost_tol", arg.boosttol);
        print("boost_val", arg.boostval);
        print("boost_vali", arg.boostvali);

        print("ell_width", arg.ell_width);
        print("permute", arg.permute);

        print("file", arg.filename);
        print("unit_check", arg.unit_check);
        print("timing", arg.timing);
        print("iters", arg.iters);
        return str << " }\n";
    }
};