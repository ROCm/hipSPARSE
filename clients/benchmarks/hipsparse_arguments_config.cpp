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

#include "hipsparse_arguments_config.hpp"
//#include "rocsparse_clients_matrices_dir.hpp"
//#include "rocsparse_enum.hpp"
//#include "rocsparse_importer_format_t.hpp"

hipsparse_arguments_config::hipsparse_arguments_config()
{
    //
    // Arguments must be a C-compatible struct so cppcheck complains about non-initialized member variables.
    // Then we need to initialize.
    {
        this->M               = 0;
        this->N               = 0;
        this->K               = 0;
        this->nnz             = 0;
        this->block_dim       = 0;
        this->row_block_dimA  = 0;
        this->col_block_dimA  = 0;
        this->row_block_dimB  = 0;
        this->col_block_dimB  = 0;

        this->lda  = 0;
        this->ldb  = 0;
        this->ldc  = 0;
        
        this->filename = "";
        this->function_name = "";

        this->index_type_I    = HIPSPARSE_INDEX_32I;
        this->index_type_J    = HIPSPARSE_INDEX_32I;
        this->compute_type    = HIP_R_32F;

        this->alpha           = 0.0;
        this->alphai          = 0.0;
        this->beta            = 0.0;
        this->betai           = 0.0;
        this->threshold       = 0.0;
        this->percentage      = 0.0;

        this->transA          = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->transB          = HIPSPARSE_OPERATION_NON_TRANSPOSE;
        this->baseA           = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseB           = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseC           = HIPSPARSE_INDEX_BASE_ZERO;
        this->baseD           = HIPSPARSE_INDEX_BASE_ZERO;

        this->action          = HIPSPARSE_ACTION_NUMERIC;
        this->part            = HIPSPARSE_HYB_PARTITION_AUTO;
        this->diag_type       = HIPSPARSE_DIAG_TYPE_NON_UNIT;
        this->fill_mode       = HIPSPARSE_FILL_MODE_LOWER;

        this->dirA            = HIPSPARSE_DIRECTION_ROW;
        this->order           = HIPSPARSE_ORDER_ROW;
        this->orderB          = HIPSPARSE_ORDER_ROW;
        this->orderC          = HIPSPARSE_ORDER_ROW;
        this->formatA         = HIPSPARSE_FORMAT_CSR;
        this->formatB         = HIPSPARSE_FORMAT_CSR;

        this->numericboost     = 0;
        this->boosttol         = 0.0;
        this->boostval         = 0.0;
        this->boostvali        = 0.0;

        this->ell_width = 0;
        this->permute = 0;

        this->precision = 's';
        this->indextype = 's';
    }
}

void hipsparse_arguments_config::set_description(options_description& desc)
{
    desc.add_options()("help,h", "produces this help message")
        // clang-format off
    ("sizem,m",
     value<int>(&this->M)->default_value(128),
     "Specific matrix size testing: sizem is only applicable to SPARSE-2 "
     "& SPARSE-3: the number of rows.")

    ("sizen,n",
     value<int>(&this->N)->default_value(128),
     "Specific matrix/vector size testing: SPARSE-1: the length of the "
     "dense vector. SPARSE-2 & SPARSE-3: the number of columns")

    ("sizek,k",
     value<int>(&this->K)->default_value(128),
     "Specific matrix/vector size testing: SPARSE-3: the number of columns")

    ("sizennz,z",
     value<int>(&this->nnz)->default_value(32),
     "Specific vector size testing, LEVEL-1: the number of non-zero elements "
     "of the sparse vector.")

    ("blockdim",
     value<int>(&this->block_dim)->default_value(2),
     "BSR block dimension (default: 2)")

    ("row-blockdimA",
     value<int>(&this->row_block_dimA)->default_value(2),
     "General BSR row block dimension (default: 2)")

    ("col-blockdimA",
     value<int>(&this->col_block_dimA)->default_value(2),
     "General BSR col block dimension (default: 2)")

    ("row-blockdimB",
     value<int>(&this->row_block_dimB)->default_value(2),
     "General BSR row block dimension (default: 2)")

    ("col-blockdimB",
     value<int>(&this->col_block_dimB)->default_value(2),
     "General BSR col block dimension (default: 2)")

    ("lda",
     value<int>(&this->lda)->default_value(2),
     "Leading dimension (default: 2)")

    ("ldb",
     value<int>(&this->ldb)->default_value(2),
     "Leading dimension (default: 2)")

    ("ldc",
     value<int>(&this->ldc)->default_value(2),
     "Leading dimension (default: 2)")

    ("file",
     value<std::string>(&this->filename)->default_value(""),
     "read from file with file extension detection.")

    ("alpha",
     value<double>(&this->alpha)->default_value(1.0), "specifies the scalar alpha")

    ("beta",
     value<double>(&this->beta)->default_value(0.0), "specifies the scalar beta")

    ("threshold",
     value<double>(&this->threshold)->default_value(1.0), "specifies the scalar threshold")

    ("percentage",
     value<double>(&this->percentage)->default_value(0.0), "specifies the scalar percentage")

    ("transposeA",
     value<char>(&this->b_transA)->default_value('N'),
     "N = no transpose, T = transpose, C = conjugate transpose")

    ("transposeB",
     value<char>(&this->b_transB)->default_value('N'),
     "N = no transpose, T = transpose, C = conjugate transpose, (default = N)")

    ("indexbaseA",
     value<int>(&this->b_baseA)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseB",
     value<int>(&this->b_baseB)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseC",
     value<int>(&this->b_baseC)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("indexbaseD",
     value<int>(&this->b_baseD)->default_value(0),
     "0 = zero-based indexing, 1 = one-based indexing, (default: 0)")

    ("action",
     value<int>(&this->b_action)->default_value(0),
     "0 = rocsparse_action_numeric, 1 = rocsparse_action_symbolic, (default: 0)")

    ("hybpart",
     value<int>(&this->b_part)->default_value(0),
     "0 = rocsparse_hyb_partition_auto, 1 = rocsparse_hyb_partition_user,\n"
     "2 = rocsparse_hyb_partition_max, (default: 0)")

    ("diag",
     value<char>(&this->b_diag)->default_value('N'),
     "N = non-unit diagonal, U = unit diagonal, (default = N)")

    ("uplo",
     value<char>(&this->b_uplo)->default_value('L'),
     "L = lower fill, U = upper fill, (default = L)")

    ("function,f",
     value<std::string>(&this->function_name)->default_value("axpyi"),
     "SPARSE function to test. Options:\n"
     "  Level1: axpyi, doti, dotci, gthr, gthrz, roti, sctr\n"
     "  Level2: bsrsv2, bsrxmv, coomv, csrmv, csrsv, gemvi, hybmv\n"
     "  Level3: bsrmm, bsrsm2, coomm, cscmm, csrmm, coosm, csrsm, gemmi\n"
     "  Extra: csrgeam, csrgemm\n"
     "  Preconditioner: bsric02, bsrilu02, csric02, csrilu02, gtsv2, gtsv2_nopivot, gtsv2_strided_batch, gtsv_interleaved_batch, gpsv_interleaved_batch\n"
     "  Conversion: bsr2csr, csr2coo, csr2csc, csr2hyb, csr2bsr, csr2gebsr, csr2csr_compress, coo2csr, hyb2csr, csr2dense, csc2dense, coo2dense\n"
     "              dense2csr, dense2csc, dense2coo, gebsr2csr, gebsr2gebsc, gebsr2gebsr\n")

    ("verify,v",
     value<int>(&this->unit_check)->default_value(0),
     "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

    ("indextype",
     value<char>(&this->indextype)->default_value('s'),
     "Specify index types to be int32_t (s), int64_t (d) or mixed (m). Options: s,d,m")

    ("precision,r",
     value<char>(&this->precision)->default_value('s'), "Options: s,d,c,z")

    ("iters,i",
     value<int>(&this->iters)->default_value(10),
     "Iterations to run inside timing loop")

    ("device,d",
     value<int>(&this->device_id)->default_value(0),
     "Set default device to be used for subsequent program runs")

    ("dirA",
     value<int>(&this->b_dir)->default_value(HIPSPARSE_DIRECTION_ROW),
     "Indicates whether BSR blocks should be laid out in row-major storage or by column-major storage: row-major storage = 0, column-major storage = 1 (default: 0)")

    ("order",
     value<int>(&this->b_order)->default_value(HIPSPARSE_ORDER_COL),
     "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

    ("orderB",
     value<int>(&this->b_orderB)->default_value(HIPSPARSE_ORDER_COL),
     "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

    ("orderC",
     value<int>(&this->b_orderC)->default_value(HIPSPARSE_ORDER_COL),
     "Indicates whether a dense matrix is laid out in column-major storage: 1, or row-major storage 0 (default: 1)")

    ("format",
     value<int>(&this->b_formatA)->default_value(HIPSPARSE_FORMAT_COO),
     "Indicates whether a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3, bell format: 4 (default:0)")

    ("formatA",
     value<int>(&this->b_formatA)->default_value(HIPSPARSE_FORMAT_COO),
     "Indicates whether a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3, bell format: 4 (default:0)")

    ("formatB",
     value<int>(&this->b_formatB)->default_value(HIPSPARSE_FORMAT_COO),
     "Indicates whether a sparse matrix is laid out in coo format: 0, coo_aos format: 1, csr format: 2, csc format: 3, bell format: 4 (default:0)")

    ("ell_width",
     value<int>(&this->ell_width)->default_value(0),
     "ELL width (default 0)");

    ("ell_width",
     value<int>(&this->ell_width)->default_value(0),
     "Using permutation vector in coosort, csrsort, cscsort. Do not use vector: 0, Use vector: 1 (default 0)");
}

int hipsparse_arguments_config::parse(int&argc,char**&argv, options_description&desc)
{
    variables_map vm;
    store(parse_command_line(argc, argv, desc,  sizeof(hipsparse_arguments_config)), vm);
    notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return -2;
    }

    if(this->b_dir != HIPSPARSE_DIRECTION_ROW && this->b_dir != HIPSPARSE_DIRECTION_COLUMN)
    {
        std::cerr << "Invalid value for --dirA" << std::endl;
        return -1;
    }

    if(this->b_order != HIPSPARSE_ORDER_ROW && this->b_order != HIPSPARSE_ORDER_COL)
    {
        std::cerr << "Invalid value for --order" << std::endl;
        return -1;
    }

    if(this->b_orderB != HIPSPARSE_ORDER_ROW && this->b_orderB != HIPSPARSE_ORDER_COL)
    {
        std::cerr << "Invalid value for --orderB" << std::endl;
        return -1;
    }

    if(this->b_orderC != HIPSPARSE_ORDER_ROW && this->b_orderC != HIPSPARSE_ORDER_COL)
    {
        std::cerr << "Invalid value for --orderC" << std::endl;
        return -1;
    }

    bool is_formatA_invalid = true;
    switch(this->b_formatA)
    {
    case HIPSPARSE_FORMAT_CSR:
    case HIPSPARSE_FORMAT_CSC:
    case HIPSPARSE_FORMAT_COO:
    case HIPSPARSE_FORMAT_COO_AOS:
    case HIPSPARSE_FORMAT_BLOCKED_ELL:
    {
        is_formatA_invalid = false;
        break;
    }
    }

    if(is_formatA_invalid)
    {
        std::cerr << "Invalid value for --formatA" << std::endl;
        return -1;
    }

    bool is_formatB_invalid = true;
    switch(this->b_formatB)
    {
    case HIPSPARSE_FORMAT_CSR:
    case HIPSPARSE_FORMAT_CSC:
    case HIPSPARSE_FORMAT_COO:
    case HIPSPARSE_FORMAT_COO_AOS:
    case HIPSPARSE_FORMAT_BLOCKED_ELL:
    {
        is_formatB_invalid = false;
        break;
    }
    }

    if(is_formatB_invalid)
    {
        std::cerr << "Invalid value for --formatB" << std::endl;
        return -1;
    }

    if(this->b_transA == 'N')
    {
        this->transA = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    }
    else if(this->b_transA == 'T')
    {
        this->transA = HIPSPARSE_OPERATION_TRANSPOSE;
    }
    else if(this->b_transA == 'C')
    {
        this->transA = HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }

    if(this->b_transB == 'N')
    {
        this->transB = HIPSPARSE_OPERATION_NON_TRANSPOSE;
    }
    else if(this->b_transB == 'T')
    {
        this->transB = HIPSPARSE_OPERATION_TRANSPOSE;
    }
    else if(this->b_transB == 'C')
    {
        this->transB = HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    }

    this->baseA = (this->b_baseA == 0) ? HIPSPARSE_INDEX_BASE_ZERO : HIPSPARSE_INDEX_BASE_ONE;
    this->baseB = (this->b_baseB == 0) ? HIPSPARSE_INDEX_BASE_ZERO : HIPSPARSE_INDEX_BASE_ONE;
    this->baseC = (this->b_baseC == 0) ? HIPSPARSE_INDEX_BASE_ZERO : HIPSPARSE_INDEX_BASE_ONE;
    this->baseD = (this->b_baseD == 0) ? HIPSPARSE_INDEX_BASE_ZERO : HIPSPARSE_INDEX_BASE_ONE;


    this->action      = (this->b_action == 0) ? HIPSPARSE_ACTION_NUMERIC : HIPSPARSE_ACTION_SYMBOLIC;
    this->part        = (this->b_part == 0)   ? HIPSPARSE_HYB_PARTITION_AUTO
        : (this->b_part == 1) ? HIPSPARSE_HYB_PARTITION_USER
        : HIPSPARSE_HYB_PARTITION_MAX;
    this->diag_type = (this->b_diag == 'N') ? HIPSPARSE_DIAG_TYPE_NON_UNIT : HIPSPARSE_DIAG_TYPE_UNIT;
    this->fill_mode = (this->b_uplo == 'L') ? HIPSPARSE_FILL_MODE_LOWER : HIPSPARSE_FILL_MODE_UPPER;


    this->dirA = (this->b_dir == 0) ? HIPSPARSE_DIRECTION_ROW : HIPSPARSE_DIRECTION_COLUMN;
    this->order  = (this->b_order == 0) ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;
    this->orderB  = (this->b_orderB == 0) ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;
    this->orderC  = (this->b_orderC == 0) ? HIPSPARSE_ORDER_ROW : HIPSPARSE_ORDER_COL;
    this->formatA = (hipsparseFormat_t)this->b_formatA;
    this->formatB = (hipsparseFormat_t)this->b_formatB;

    if(this->M < 0 || this->N < 0)
    {
        std::cerr << "Invalid dimension" << std::endl;
        return -1;
    }

    if(this->block_dim < 1)
    {
        std::cerr << "Invalid value for --blockdim" << std::endl;
        return -1;
    }

    if(this->row_block_dimA < 1)
    {
        std::cerr << "Invalid value for --row-blockdimA" << std::endl;
        return -1;
    }

    if(this->col_block_dimA < 1)
    {
        std::cerr << "Invalid value for --col-blockdimA" << std::endl;
        return -1;
    }

    if(this->row_block_dimB < 1)
    {
        std::cerr << "Invalid value for --row-blockdimB" << std::endl;
        return -1;
    }

    if(this->col_block_dimB < 1)
    {
        std::cerr << "Invalid value for --col-blockdimB" << std::endl;
        return -1;
    }

    switch(this->indextype)
    {
    case 's':
    {
        this->index_type_I   = HIPSPARSE_INDEX_32I;
        this->index_type_J   = HIPSPARSE_INDEX_32I;
        break;
    }
    case 'd':
    {
	    this->index_type_I   = HIPSPARSE_INDEX_64I;
	    this->index_type_J   = HIPSPARSE_INDEX_64I;
	    break;
    }

    case 'm':
    {
	    this->index_type_I   = HIPSPARSE_INDEX_64I;
	    this->index_type_J   = HIPSPARSE_INDEX_32I;
	    break;
    }
    default:
    {
	    std::cerr << "Invalid value for --indextype" << std::endl;
	    return -1;
    }
    }

    switch(this->precision)
    {
    case 's':
    {
	    this->compute_type = HIP_R_32F;
        break;
    }
    case 'd':
    {
	    this->compute_type = HIP_R_64F;
	    break;
    }

    case 'c':
    {
	    this->compute_type = HIP_C_32F;
	    break;
    }
    case 'z':
    {
	    this->compute_type = HIP_C_64F;
	    break;
    }
    default:
    {
	    std::cerr << "Invalid value for --precision" << std::endl;
	    return -1;
    }
    }

    return 0;
}

