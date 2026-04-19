// Copyright (c) TorchMoE.
// SPDX-License-Identifier: Apache-2.0

// TorchMoE Team

#include "expert_dispatcher.h"
#include "aio/archer_tensor_index.h"
#include "common/pytorch.h"
#include "common/time.h"
#include "prefetch/task_scheduler.h"
#include "prefetch/task_thread.h"
#include "utils/archer_logger.h"
#include "utils/cuda_utils.h"

#include <c10/core/ScalarType.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include <future>

ExpertDispatcher::ExpertDispatcher(int num_experts, int num_layers, int dtype, int expert_type)
    : pending_(0), num_enqueued_(0), start_(false), expert_type_(expert_type)
{
    main_thread_stop_flag_.store(false);

    int num_gpu = GetDeviceCount();
    for (int i = 0; i < num_gpu; ++i) {
        std::thread t(&ExpertDispatcher::GPUFetchFunc, this, i);
        SetThreadAffinity(t);
        threads_.emplace_back(std::move(t));

        cudaSetDevice(i);
        cudaStream_t fetch_stream;
        cudaStreamCreateWithFlags(&fetch_stream, cudaStreamNonBlocking);
        fetch_streams_.emplace_back(fetch_stream);

        cudaStream_t out_stream;
        cudaStreamCreateWithFlags(&out_stream, cudaStreamNonBlocking);
        out_streams_.emplace_back(out_stream);
    }

    for (int i = 0; i < num_gpu; ++i) {
        std::thread t(&ExpertDispatcher::GPUThreadFunc, this, i);
        SetThreadAffinity(t);
        threads_.emplace_back(std::move(t));

        gpu_overload_.emplace_back(false);
    }

    for (int i = 0; i < num_gpu; ++i) {
        std::thread t(&ExpertDispatcher::GPUExecFunc, this, i);
        SetThreadAffinity(t);
        threads_.emplace_back(std::move(t));

        cudaSetDevice(i);
        cudaStream_t exec_stream;
        cudaStreamCreateWithFlags(&exec_stream, cudaStreamNonBlocking);
        exec_streams_.emplace_back(exec_stream);
    }

    at::InferenceMode infer_guard(0);

    for (int i = 0; i < num_experts; ++i) {
        experts_.emplace_back();
        for (int j = 0; j < num_layers; ++j) {
            experts_[i].emplace_back();
            experts_[i][j] = std::make_shared<ExpertNode>();
            experts_[i][j]->expert_type = expert_type;
            int expert_type = expert_type_;
            switch (expert_type) {
                case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
                    experts_[i][j]->module = new SwitchTransformersDenseActDense(dtype);
                    break;
                case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
                    experts_[i][j]->module = new SwitchTransformersDenseGatedActDense(dtype);
                    break;
                case NLLB_MOE_DENSE_ACT_DENSE:
                    experts_[i][j]->module = new NllbMoeDenseActDense(dtype);
                    break;
                case FSGPT_MOE_DENSE_ACT_DENSE:
                    experts_[i][j]->module = new FSGPTMoEDenseActDense(dtype);
                    break;
                case MIXTRAL_MOE_DENSE_ACT_DENSE:
                    experts_[i][j]->module = new MixtralMoEDenseActDense(dtype);
                    break;
                default:
                    ARCHER_LOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                                     expert_type);
            }
            experts_[i][j]->module->eval();
            experts_[i][j]->layer_idx = j;
            experts_[i][j]->expert_idx = i;
        }
    }
}

void ExpertDispatcher::DispatchBatch(const torch::Tensor& hidden_states,
                                     const torch::Tensor& router_mask,
                                     int layer_idx,
                                     const std::vector<int>& expert_indices,
                                     const std::vector<torch::Tensor>& token_indices,
                                     int gpu_id,
                                     bool remote)
{
    SetInputs(hidden_states, router_mask);
    SetAssignments(expert_indices, token_indices);
    SetExpectedQueue(expert_indices.size());
    for (auto expert_idx : expert_indices) {
        EnqueueExpert(layer_idx, expert_idx, gpu_id, remote);
    }
}

void ExpertDispatcher::EnqueueExpert(int layer_idx, int expert_idx, int gpu_id, bool remote)
{
    ExpertDispatcher::CallArgs args;
    args.layer_idx = layer_idx;
    args.expert_idx = expert_idx;
    args.gpu_id = gpu_id;
    args.remote = remote;
    Enqueue(args);
}

void ExpertDispatcher::Enqueue(const CallArgs& args)
{
    int layer_idx = args.layer_idx;
    int expert_idx = args.expert_idx;
    auto expert_node = experts_[expert_idx][layer_idx];

    expert_node->node->active_service_users.fetch_add(1);
    expert_node->node->last_access_time = MCIROSECONDS_SINCE_EPOCH;

    {
        std::lock_guard<std::mutex> lock(input_mutex_);
        input_queue_.push_back(std::move(args));
        num_enqueued_.fetch_add(1);

        auto& a = input_queue_.back();
        if (expert_node->node->device.is_cuda()) { a.gpu_id = expert_node->node->device.index(); }
        ARCHER_LOG_DEBUG(
            "ExpertDispatcher::Enqueue: num_enqueued_ ", num_enqueued_.load(),
            "input_queue_ ", input_queue_.size(), \
            "gpu_id ", a.gpu_id,
            "layer_idx ", a.layer_idx,
            "expert_idx ", a.expert_idx,
            "remote ", a.remote);
    }
    input_cv_.notify_all();
}

void ExpertDispatcher::RegisterExpert(int layer_idx,
                                      int expert_idx,
                                      const std::vector<std::uint32_t>& tensor_ids)
{
    NodePtr cached_node = nullptr;
    for (auto tensor_id : tensor_ids) {
        auto node = kTopologyHandle->GetNodeFromTensorID(tensor_id);
        if (cached_node == nullptr) {
            cached_node = node;
            experts_[expert_idx][layer_idx]->node = node;
        } else if (cached_node != node) {
            ARCHER_LOG_FATAL("RegisterExpert: tensor_id has multiple nodes", tensor_id);
        }
    }
}

void ExpertDispatcher::GPUThreadFunc(int gpu_id)
{
    while (!main_thread_stop_flag_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void ExpertDispatcher::GPUFetchFunc(int gpu_id)
{
    while (!main_thread_stop_flag_.load()) {
        std::unique_lock<std::mutex> lock(input_mutex_);
        input_cv_.wait(lock, [this]() {
            return main_thread_stop_flag_.load() || !input_queue_.empty();
        });
        if (main_thread_stop_flag_.load()) { break; }

        CallArgs args;

        // find all the args with gpu_id
        std::vector<CallArgs> args_list;
        auto it = input_queue_.begin();
        while (it != input_queue_.end()) {
            if (it->gpu_id == gpu_id) {
                args_list.emplace_back(std::move(*it));
                it = input_queue_.erase(it);
            } else {
                ++it;
            }
        }

        lock.unlock();

        if (args_list.empty()) { continue; }
        auto device = CUDA_DEVICE(gpu_id);

        struct PendingExecItem {
            CallArgs args;
            ExpertNodePtr expert_node = nullptr;
            bool cache_hit = false;
        };

        std::vector<PendingExecItem> pending_items;
        pending_items.reserve(args_list.size());

        for (auto& a : args_list) {
            int layer_idx = a.layer_idx;
            int expert_idx = a.expert_idx;
            auto expert_node = experts_[expert_idx][layer_idx];

            bool cache_hit = expert_node->node->device.is_cuda();
            if (!expert_node->node->device.is_cuda()) {
                expert_node->node->state = 1;
                auto task = std::make_shared<Task>();
                task->priority = 0;
                task->node = expert_node->node;
                task->on_demand = true;
                task->src_device = expert_node->node->device;
                task->dst_device = device;
                task->remove_layer = true;
                kTaskPool->EnqueueTask(task);
            }

            PendingExecItem item;
            item.args = std::move(a);
            item.expert_node = expert_node;
            item.cache_hit = cache_hit;
            pending_items.emplace_back(std::move(item));
        }

        for (auto& item : pending_items) {
            if (!item.cache_hit) {
                std::unique_lock<std::mutex> node_lock(item.expert_node->node->mutex);
                item.expert_node->node->cv.wait(node_lock, [&item]() {
                    return item.expert_node->node->state == 0 &&
                           item.expert_node->node->device.is_cuda();
                });
            }

            item.expert_node->SetTensorsFromBlob(device);

            int expert_idx = item.args.expert_idx;
            torch::Tensor token_indices;
            auto assignment_it = assignment_tokens_.find(expert_idx);
            if (assignment_it != assignment_tokens_.end()) {
                token_indices = assignment_it->second.to(item.expert_node->node->device);
            } else {
                token_indices = router_mask_.index({"...", expert_idx}).to(torch::kBool);
            }

            torch::Tensor input;
            switch (expert_type_) {
                case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
                case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
                case NLLB_MOE_DENSE_ACT_DENSE:
                case FSGPT_MOE_DENSE_ACT_DENSE:
                case MIXTRAL_MOE_DENSE_ACT_DENSE:
                    input = hidden_states_.index({token_indices}).to(item.expert_node->node->device);
                    break;
                default:
                    ARCHER_LOG_FATAL("ExpertDispatcher::ExpertDispatcher: unknown expert type ",
                                     expert_type_);
            }

            ARCHER_LOG_DEBUG(
                "ExpertDispatcher::GPUFetchFunc gpu_id ",  gpu_id,
                "layer_idx ", item.args.layer_idx,
                "expert_idx ", expert_idx,
                "input ", input.device().str(),
                "node ", item.expert_node->node->device.str());

            ExecArgs exec_args;
            exec_args.hidden_states = std::move(input);
            exec_args.expert_node = item.expert_node;
            exec_args.out_gpu_id =
                ((item.args.remote) ? CPU_DEVICE : hidden_states_.device()).index();
            exec_args.out_dtype = c10::typeMetaToScalarType(hidden_states_.dtype());
            exec_args.evict = false;
            exec_args.hit = item.cache_hit;
            std::lock_guard<std::mutex> exec_lock(exec_mutex_);
            exec_queue_.emplace_back(std::move(exec_args));
        }
        exec_cv_.notify_all();
    }
}

void ExpertDispatcher::GPUExecFunc(int gpu_id)
{
    while (!main_thread_stop_flag_.load()) {
        std::unique_lock<std::mutex> lock(exec_mutex_);
        exec_cv_.wait(lock, [this]() {
            return main_thread_stop_flag_.load() || !exec_queue_.empty();
        });
        if (main_thread_stop_flag_.load()) { break; }

        ExecArgs args;

        for (auto it = exec_queue_.begin(); it != exec_queue_.end(); ++it) {
            if (it->expert_node->node->device.index() == gpu_id) {
                args = std::move(*it);
                exec_queue_.erase(it);
                break;
            }
        }

        lock.unlock();

        if (args.expert_node == nullptr) { continue; }

        torch::Tensor output;

        at::InferenceMode infer_guard(true);

        c10::cuda::CUDAStream stream =
            c10::cuda::getStreamFromExternal(fetch_streams_[gpu_id], gpu_id);

        {
            c10::cuda::CUDAStreamGuard guard(stream);

            auto* expert_module = args.expert_node->module;
            int expert_type = expert_type_;

            try {
                switch (expert_type) {
                    case SWITCH_TRANSFORMERS_DENSE_ACT_DENSE:
                        output = reinterpret_cast<SwitchTransformersDenseActDense*>(expert_module)
                                     ->forward(args.hidden_states);
                        break;
                    case SWITCH_TRANSFORMERS_DENSE_GATED_ACT_DENSE:
                        output =
                            reinterpret_cast<SwitchTransformersDenseGatedActDense*>(expert_module)
                                ->forward(args.hidden_states);
                        break;
                    case NLLB_MOE_DENSE_ACT_DENSE:
                        output = reinterpret_cast<NllbMoeDenseActDense*>(expert_module)
                                     ->forward(args.hidden_states);
                        break;
                    case FSGPT_MOE_DENSE_ACT_DENSE:
                        output = reinterpret_cast<FSGPTMoEDenseActDense*>(expert_module)
                                     ->forward(args.hidden_states);
                        break;
                    case MIXTRAL_MOE_DENSE_ACT_DENSE:
                        output = reinterpret_cast<MixtralMoEDenseActDense*>(expert_module)
                                     ->forward(args.hidden_states);
                        break;
                    default:
                        ARCHER_LOG_FATAL(
                            "ExpertDispatcher::ExpertDispatcher: unknown expert type",
                            expert_type);
                }

            } catch (const std::exception& e) {
                std::stringstream ss;
                ss << "DenseActDense tensor_ids: [";
                for (auto& id : args.expert_node->node->tensor_ids) { ss << id << " "; }
                ss << "]";
                ARCHER_LOG_FATAL("ExpertDispatcher::GPUExecFunc", ss.str(), "expert_type", expert_type, e.what());
            }
        }

        OutputFunc(std::move(args), std::move(output), gpu_id);
    }
}

void ExpertDispatcher::OutputFunc(ExecArgs args, torch::Tensor output, int gpu_id)
{
    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromExternal(out_streams_[gpu_id], gpu_id);
    c10::cuda::CUDAStreamGuard guard(stream);

    auto output_device = (args.out_gpu_id < 0) ? CPU_DEVICE : CUDA_DEVICE(args.out_gpu_id);
    torch::Tensor output_tensor = output.to(output_device).to(args.out_dtype);

    if (args.evict) {
        args.expert_node->node->SetDevice(args.expert_node->node->default_host, true, nullptr);
        {
            std::lock_guard<std::mutex> lock(gpu_overload_mutex_);
            gpu_overload_[gpu_id] = false;
        }
    }

    args.expert_node->node->active_service_users.fetch_sub(1);

    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        output_queue_.emplace_back(std::move(output_tensor),
                                   args.expert_node->layer_idx,
                                   args.expert_node->expert_idx,
                                   args.hit);
        ARCHER_LOG_DEBUG(
            "ExpertDispatcher::OutputFunc: output_queue_", output_queue_.size(),
            "output", std::get<0>(output_queue_.back()).device().str(),
            "evict", args.evict,
            "(", 
            args.expert_node->layer_idx,
            args.expert_node->expert_idx,
            gpu_id,
            args.hit, ")");
    }
    if (pending_.fetch_sub(1) == 1) { pending_cv_.notify_all(); }
}

std::vector<ExpertDispatcher::CallResult> ExpertDispatcher::Wait()
{
    std::unique_lock<std::mutex> pending_lock(output_mutex_);
    pending_cv_.wait(pending_lock, [this]() { return pending_.load() == 0; });
    pending_lock.unlock();
    {
        std::lock_guard<std::mutex> input_lock(input_mutex_);
        input_queue_.clear();
    }
    {
        std::lock_guard<std::mutex> exec_lock(exec_mutex_);
        exec_queue_.clear();
    }
    assignment_tokens_.clear();
    num_enqueued_.store(0);
    std::vector<CallResult> output_queue;
    {
        std::lock_guard<std::mutex> lock(output_mutex_);
        output_queue.swap(output_queue_);
    }

    std::stringstream ss;
    ss << "ExpertDispatcher::Wait: output_queue_ " << output_queue.size();
    for (auto& output : output_queue) {
        ss << " (" << std::get<0>(output).sizes() << "," << std::get<1>(output) << ","
           << std::get<2>(output) << "," << std::get<3>(output) << ")";
    }
    ARCHER_LOG_DEBUG(ss.str());
    return output_queue;
}
