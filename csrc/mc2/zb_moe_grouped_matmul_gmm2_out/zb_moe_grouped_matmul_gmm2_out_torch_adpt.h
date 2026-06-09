/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef ZB_MOE_GROUPED_MATMUL_GMM2_OUT_TORCH_ADPT_H
#define ZB_MOE_GROUPED_MATMUL_GMM2_OUT_TORCH_ADPT_H

#include "aclnn_torch_adapter/op_api_common.h"

namespace vllm_ascend {

inline at::TensorList EmptyTensorList()
{
    static std::vector<at::Tensor> kEmpty;
    return at::TensorList(kEmpty);
}

inline at::TensorList SingleTensorList(const at::Tensor &tensor)
{
    static thread_local std::vector<at::Tensor> holder;
    holder.assign(1, tensor);
    return at::TensorList(holder);
}

inline at::TensorList ToTensorList(const at::Tensor &tensor)
{
    return SingleTensorList(tensor);
}

inline at::TensorList ToTensorList(const at::TensorList &tensor_list)
{
    return tensor_list;
}

// ZB-only grouped matmul: write gmm2 output into caller-provided SHMEM ``out``.
// Uses CANN ``aclnnGroupedMatmulWeightNz`` (NZ weight, W8A8/per-token quant path).
at::Tensor &zb_moe_grouped_matmul_gmm2_out(
    const at::Tensor &x,
    const at::TensorList &weight,
    const c10::optional<at::TensorList> &scale,
    const c10::optional<at::TensorList> &per_token_scale,
    const c10::optional<at::TensorList> &bias,
    const at::Tensor &group_list,
    at::Tensor &out,
    int64_t split_item,
    int64_t group_type,
    int64_t group_list_type,
    int64_t act_type)
{
    TORCH_CHECK(out.defined(), "zb_moe_grouped_matmul_gmm2_out: out must be defined");
    TORCH_CHECK(out.is_npu(), "zb_moe_grouped_matmul_gmm2_out: out must be on NPU");
    TORCH_CHECK(x.is_npu(), "zb_moe_grouped_matmul_gmm2_out: x must be on NPU");
    TORCH_CHECK(weight.size() > 0, "zb_moe_grouped_matmul_gmm2_out: weight must not be empty");

    const at::TensorList x_list = ToTensorList(x);
    const at::TensorList y_list = ToTensorList(out);
    const at::TensorList scale_list = scale.has_value() ? scale.value() : EmptyTensorList();
    const at::TensorList per_token_scale_list =
        per_token_scale.has_value() ? per_token_scale.value() : EmptyTensorList();
    const at::TensorList bias_list = bias.has_value() ? bias.value() : EmptyTensorList();
    const bool use_quant = scale.has_value() && scale.value().size() > 0;

    if (use_quant) {
        EXEC_NPU_CMD(aclnnGroupedMatmulWeightNz,
                     x_list,
                     weight,
                     bias_list,
                     scale_list,
                     EmptyTensorList(),
                     EmptyTensorList(),
                     EmptyTensorList(),
                     per_token_scale_list,
                     group_list,
                     split_item,
                     group_type,
                     group_list_type,
                     act_type,
                     y_list);
    } else {
        EXEC_NPU_CMD(aclnnGroupedMatmulV4,
                     x_list,
                     weight,
                     bias_list,
                     EmptyTensorList(),
                     EmptyTensorList(),
                     EmptyTensorList(),
                     EmptyTensorList(),
                     per_token_scale_list,
                     group_list,
                     EmptyTensorList(),
                     EmptyTensorList(),
                     EmptyTensorList(),
                     split_item,
                     group_type,
                     group_list_type,
                     act_type,
                     y_list,
                     EmptyTensorList(),
                     EmptyTensorList());
    }

    return out;
}

}  // namespace vllm_ascend

#endif
