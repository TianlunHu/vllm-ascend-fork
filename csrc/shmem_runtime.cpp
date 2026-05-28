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

#include "shmem_runtime.h"

#include <algorithm>
#include <mutex>
#include <string>

#include <c10/util/Exception.h>

#ifdef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
#include "shmem.h"
#endif

namespace vllm_ascend {
namespace {

std::mutex g_shmem_mutex;
bool g_initialized = false;
void *g_ext_info = nullptr;

#ifndef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
void throw_shmem_unavailable()
{
    TORCH_CHECK(false,
                "zero-buffer SHMEM runtime is not available in this build. "
                "Please install Ascend SHMEM and rebuild vllm_ascend_C.");
}
#else
int32_t fill_init_attr(int32_t rank, int32_t world_size, uint64_t local_mem_size,
                       const std::string &server_ip_port, aclshmemx_init_attr_t *attributes)
{
    aclshmemx_uniqueid_t default_flag_uid = {};
    size_t ip_len = 0;
    if (!server_ip_port.empty()) {
        ip_len = std::min(server_ip_port.size(), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN) - 1);
        std::copy_n(server_ip_port.data(), ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }

    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = rank;
    attributes->n_pes = world_size;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT,
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    return ACLSHMEM_SUCCESS;
}
#endif

} // namespace

int64_t zb_shmem_init(int64_t rank, int64_t world_size, int64_t local_mem_size, const std::string &server_ip_port)
{
#ifndef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    throw_shmem_unavailable();
#else
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    TORCH_CHECK(rank >= 0, "rank must be non-negative, got ", rank);
    TORCH_CHECK(world_size > 0, "world_size must be positive, got ", world_size);
    TORCH_CHECK(rank < world_size, "rank must be smaller than world_size, got rank=", rank,
                ", world_size=", world_size);
    TORCH_CHECK(local_mem_size > 0, "local_mem_size must be positive, got ", local_mem_size);

    if (!g_initialized) {
        aclshmemx_set_conf_store_tls(false, nullptr, 0);
        aclshmemx_init_attr_t attributes = {};
        int32_t status = fill_init_attr(static_cast<int32_t>(rank), static_cast<int32_t>(world_size),
                                        static_cast<uint64_t>(local_mem_size), server_ip_port, &attributes);
        TORCH_CHECK(status == ACLSHMEM_SUCCESS, "failed to fill SHMEM init attributes, status=", status);

        status = aclshmemx_init_attr(ACLSHMEMX_INIT_WITH_DEFAULT, &attributes);
        TORCH_CHECK(status == ACLSHMEM_SUCCESS, "aclshmemx_init_attr failed, status=", status);
        TORCH_CHECK(aclshmemx_init_status() == ACLSHMEM_STATUS_IS_INITIALIZED,
                    "aclshmem runtime is not initialized after aclshmemx_init_attr");
        g_initialized = true;
    }

    return static_cast<int64_t>(aclshmem_my_pe());
#endif
}

int64_t zb_shmem_alloc(int64_t element_count, int64_t element_size)
{
#ifndef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    throw_shmem_unavailable();
#else
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    TORCH_CHECK(g_initialized, "SHMEM runtime must be initialized before allocation");
    TORCH_CHECK(element_count > 0, "element_count must be positive, got ", element_count);
    TORCH_CHECK(element_size > 0, "element_size must be positive, got ", element_size);
    TORCH_CHECK(g_ext_info == nullptr, "zero-buffer SHMEM runtime currently supports one active allocation");

    g_ext_info = aclshmemx_calloc(static_cast<size_t>(element_count), static_cast<size_t>(element_size));
    TORCH_CHECK(g_ext_info != nullptr, "aclshmemx_calloc failed");
    return reinterpret_cast<int64_t>(g_ext_info);
#endif
}

void zb_shmem_free(int64_t ptr)
{
#ifdef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    void *raw_ptr = reinterpret_cast<void *>(ptr);
    if (raw_ptr != nullptr) {
        aclshmem_free(raw_ptr);
    }
    if (raw_ptr == g_ext_info) {
        g_ext_info = nullptr;
    }
#else
    (void)ptr;
    throw_shmem_unavailable();
#endif
}

void zb_shmem_finalize()
{
#ifdef VLLM_ASCEND_ENABLE_SHMEM_RUNTIME
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    if (g_ext_info != nullptr) {
        aclshmem_free(g_ext_info);
        g_ext_info = nullptr;
    }
    if (g_initialized) {
        int32_t status = aclshmem_finalize();
        TORCH_CHECK(status == ACLSHMEM_SUCCESS, "aclshmem_finalize failed, status=", status);
        g_initialized = false;
    }
#else
    throw_shmem_unavailable();
#endif
}

int64_t zb_shmem_get_ext_info()
{
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    return reinterpret_cast<int64_t>(g_ext_info);
}

bool zb_shmem_is_initialized()
{
    std::lock_guard<std::mutex> guard(g_shmem_mutex);
    return g_initialized;
}

} // namespace vllm_ascend
