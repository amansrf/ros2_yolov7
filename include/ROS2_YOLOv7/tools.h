
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef __TOOLS_H__
#define __TOOLS_H__

#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include "NvInfer.h"

void checkCudaErrors(cudaError_t err);

// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger {
public:
    Logger(Severity severity = Severity::kWARNING);
    void log(Severity severity, const char* msg) noexcept override;
    Severity reportableSeverity;
};

template <typename T>
struct TrtDeleter {
    void operator()(T* p) noexcept {
        if (p != nullptr) delete p;
    }
};

template <typename T>
struct CuMemDeleter {
    void operator()(T* p) noexcept { checkCudaErrors(cudaFree(p)); }
};

// template <typename T>
// std::unique_ptr<T, CuMemDeleter<T>> mallocCudaMem(size_t nbElems);

#ifndef __IMAGE_INFERENCE_CPP__
template <typename T>
std::unique_ptr<T, CuMemDeleter<T>> mallocCudaMem(size_t nbElems) {
    T* ptr = nullptr;
    checkCudaErrors(cudaMalloc((void**)&ptr, sizeof(T) * nbElems));
    return std::unique_ptr<T, CuMemDeleter<T>>{ptr};
};
#endif

struct EventDeleter {
    void operator()(CUevent_st* event) noexcept { checkCudaErrors(cudaEventDestroy(event)); }
};
struct StreamDeleter {
    void operator()(CUstream_st* stream) noexcept { checkCudaErrors(cudaStreamDestroy(stream)); }
};

std::unique_ptr<CUevent_st, EventDeleter> makeCudaEvent(int flags);

std::unique_ptr<CUstream_st, StreamDeleter> makeCudaStream(int flags, int priority);



#endif