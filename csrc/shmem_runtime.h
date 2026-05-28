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

#pragma once

#include <cstdint>
#include <string>

namespace vllm_ascend {

int64_t zb_shmem_init(int64_t rank, int64_t world_size, int64_t local_mem_size, const std::string &server_ip_port);
int64_t zb_shmem_alloc(int64_t element_count, int64_t element_size);
void zb_shmem_free(int64_t ptr);
void zb_shmem_finalize();
int64_t zb_shmem_get_ext_info();
bool zb_shmem_is_initialized();

} // namespace vllm_ascend
