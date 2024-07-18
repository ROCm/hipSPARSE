/*! \file */
/* ************************************************************************
 * Copyright (C) 2019-2024 Advanced Micro Devices, Inc. All rights Reserved.
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

#include <hipsparse.h>
#include <string>

constexpr auto hipsparse_indextype2string(hipsparseIndexType_t type)
{
    switch(type)
    {
    case HIPSPARSE_INDEX_16U:
        return "u16";
    case HIPSPARSE_INDEX_32I:
        return "i32";
    case HIPSPARSE_INDEX_64I:
        return "i64";
    }
    return "invalid";
}

constexpr auto hipsparse_datatype2string(hipDataType type)
{
    switch(type)
    {
    case HIP_R_32F:
        return "f32_r";
    case HIP_R_64F:
        return "f64_r";
    case HIP_C_32F:
        return "f32_c";
    case HIP_C_64F:
        return "f64_c";
    default:
        return "invalid";
    }
}

constexpr auto hipsparse_indexbase2string(hipsparseIndexBase_t base)
{
    switch(base)
    {
    case HIPSPARSE_INDEX_BASE_ZERO:
        return "0b";
    case HIPSPARSE_INDEX_BASE_ONE:
        return "1b";
    }
    return "invalid";
}

constexpr auto hipsparse_operation2string(hipsparseOperation_t trans)
{
    switch(trans)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return "NT";
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return "T";
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return "CT";
    }
    return "invalid";
}

constexpr auto hipsparse_direction2string(hipsparseDirection_t direction)
{
    switch(direction)
    {
    case HIPSPARSE_DIRECTION_ROW:
        return "row";
    case HIPSPARSE_DIRECTION_COLUMN:
        return "column";
    }
    return "invalid";
}

#if(!defined(CUDART_VERSION))
constexpr auto hipsparse_order2string(hipsparseOrder_t order)
{
    switch(order)
    {
    case HIPSPARSE_ORDER_ROW:
        return "row";
    case HIPSPARSE_ORDER_COL:
        return "col";
    }
    return "invalid";
}
#else
#if(CUDART_VERSION >= 11000)
constexpr auto hipsparse_order2string(hipsparseOrder_t order)
{
    switch(order)
    {
    case HIPSPARSE_ORDER_ROW:
        return "row";
    case HIPSPARSE_ORDER_COL:
        return "col";
    }
    return "invalid";
}
#elif(CUDART_VERSION >= 10010)
constexpr auto hipsparse_order2string(hipsparseOrder_t order)
{
    switch(order)
    {
    case HIPSPARSE_ORDER_COL:
        return "col";
    }
    return "invalid";
}
#endif
#endif

#if(!defined(CUDART_VERSION))
constexpr auto hipsparse_format2string(hipsparseFormat_t format)
{
    switch(format)
    {
    case HIPSPARSE_FORMAT_COO:
        return "coo";
    case HIPSPARSE_FORMAT_COO_AOS:
        return "coo_aos";
    case HIPSPARSE_FORMAT_CSR:
        return "csr";
    case HIPSPARSE_FORMAT_CSC:
        return "csc";
    case HIPSPARSE_FORMAT_BLOCKED_ELL:
        return "bell";
    }
    return "invalid";
}
#else
#if(CUDART_VERSION >= 12000)
constexpr auto hipsparse_format2string(hipsparseFormat_t format)
{
    switch(format)
    {
    case HIPSPARSE_FORMAT_COO:
        return "coo";
    case HIPSPARSE_FORMAT_CSR:
        return "csr";
    case HIPSPARSE_FORMAT_CSC:
        return "csc";
    case HIPSPARSE_FORMAT_BLOCKED_ELL:
        return "bell";
    }
    return "invalid";
}
#elif(CUDART_VERSION >= 11021 && CUDART_VERSION < 12000)
constexpr auto hipsparse_format2string(hipsparseFormat_t format)
{
    switch(format)
    {
    case HIPSPARSE_FORMAT_COO:
        return "coo";
    case HIPSPARSE_FORMAT_COO_AOS:
        return "coo_aos";
    case HIPSPARSE_FORMAT_CSR:
        return "csr";
    case HIPSPARSE_FORMAT_CSC:
        return "csc";
    case HIPSPARSE_FORMAT_BLOCKED_ELL:
        return "bell";
    }
    return "invalid";
}
#elif(CUDART_VERSION >= 10010 && CUDART_VERSION < 11021)
constexpr auto hipsparse_format2string(hipsparseFormat_t format)
{
    switch(format)
    {
    case HIPSPARSE_FORMAT_COO:
        return "coo";
    case HIPSPARSE_FORMAT_COO_AOS:
        return "coo_aos";
    case HIPSPARSE_FORMAT_CSR:
        return "csr";
    }
    return "invalid";
}
#endif
#endif

constexpr auto hipsparse_action2string(hipsparseAction_t action)
{
    switch(action)
    {
    case HIPSPARSE_ACTION_SYMBOLIC:
        return "sym";
    case HIPSPARSE_ACTION_NUMERIC:
        return "num";
    }
    return "invalid";
}

constexpr auto hipsparse_partition2string(hipsparseHybPartition_t part)
{
    switch(part)
    {
    case HIPSPARSE_HYB_PARTITION_AUTO:
        return "auto";
    case HIPSPARSE_HYB_PARTITION_USER:
        return "user";
    case HIPSPARSE_HYB_PARTITION_MAX:
        return "max";
    }
    return "invalid";
}

constexpr auto hipsparse_diagtype2string(hipsparseDiagType_t diag_type)
{
    switch(diag_type)
    {
    case HIPSPARSE_DIAG_TYPE_NON_UNIT:
        return "ND";
    case HIPSPARSE_DIAG_TYPE_UNIT:
        return "UD";
    }
    return "invalid";
}

constexpr auto hipsparse_fillmode2string(hipsparseFillMode_t fill_mode)
{
    switch(fill_mode)
    {
    case HIPSPARSE_FILL_MODE_LOWER:
        return "L";
    case HIPSPARSE_FILL_MODE_UPPER:
        return "U";
    }
    return "invalid";
}
