/******************************************************************************
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved. # SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Implementation based on FlashInfer library.
#
******************************************************************************/

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <driver_types.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

void embedding_lookup_cuda(at::Tensor indices,
                           at::Tensor embedding_table,
                           at::Tensor output,
                           int num_sms,
                           int max_threads_per_sm);

PYBIND11_MODULE(pcie_lookup_poc, m) {
  m.def("embedding_lookup_cuda", &embedding_lookup_cuda,
        "embedding lookup on GPU");
}