// Same-engine solo/partition comparison for GGUF models; not a speed benchmark.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
constexpr float logit_atol = 0.1f;
constexpr float logit_rtol = 0.01f;
constexpr int max_new = 16;

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::cerr << "Usage: engine_correctness MODEL solo|layer|tensor|tensor-local|tensor-single RPC_ENDPOINT OUTPUT_DIR REFERENCE_DIR\n";
        return 2;
    }
    try {
        const std::string model_path = argv[1], mode = argv[2], endpoint = argv[3];
        const fs::path output = argv[4], reference = argv[5];
        if (mode != "solo" && mode != "layer" && mode != "tensor" && mode != "tensor-local" && mode != "tensor-single") throw std::runtime_error("invalid mode");
        fs::create_directories(output);
        ggml_backend_load_all();
        llama_backend_init();
        auto metal = ggml_backend_dev_by_name("MTL0");
        if (!metal) throw std::runtime_error("Metal device unavailable");
        std::vector<ggml_backend_dev_t> devices = {metal};
        if (mode == "tensor-local") {
            devices.push_back(metal);
        } else if (mode != "solo" && mode != "tensor-single") {
            auto rpc = ggml_backend_rpc_add_server(endpoint.c_str());
            if (!rpc || ggml_backend_reg_dev_count(rpc) != 1) throw std::runtime_error("expected one RPC device");
            devices.push_back(ggml_backend_reg_dev_get(rpc, 0));
        }
        devices.push_back(nullptr);
        std::vector<float> split(llama_max_devices(), 0.0f);
        split[0] = 1.0f; split[1] = 1.0f;
        auto mp = llama_model_default_params();
        mp.devices = devices.data();
        mp.n_gpu_layers = 999;
        mp.split_mode = mode == "solo" ? LLAMA_SPLIT_MODE_NONE : mode == "layer" ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_TENSOR;
        mp.tensor_split = split.data();
        mp.load_mtp = false;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(model_path.c_str(), mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 256; cp.n_batch = 128; cp.n_ubatch = 128; cp.n_seq_max = 1;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.type_k = GGML_TYPE_F16; cp.type_v = GGML_TYPE_F16;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context creation failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int nv = llama_vocab_n_tokens(vocab);
        const std::vector<std::string> prompts = {"The capital of France is", "In a shocking finding, scientists discovered", "def add(a, b):"};
        json report = {{"mode", mode}, {"model", model_path}, {"vocabulary_size", nv}, {"max_new_tokens", max_new},
                       {"logit_atol", logit_atol}, {"logit_rtol", logit_rtol}, {"reference", "same pinned GGUF on unpartitioned Metal"},
                       {"scope", "Three prompts; teacher-forced reference contexts for partition comparison. Not a benchmark or universal quality guarantee."},
                       {"cases", json::array()}};
        bool all_ok = true;
        for (size_t pi = 0; pi < prompts.size(); ++pi) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            const auto & prompt = prompts[pi];
            int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, false);
            if (count <= 0 || count + max_new > 256) throw std::runtime_error("invalid prompt token count");
            std::vector<llama_token> ids(count);
            if (llama_tokenize(vocab, prompt.data(), prompt.size(), ids.data(), count, true, false) != count) throw std::runtime_error("tokenization failed");
            const auto stem = "prompt-" + std::to_string(pi);
            std::ofstream saved;
            std::ifstream expected_logits;
            json expected;
            if (mode == "solo") {
                saved.open(output / (stem + ".f32"), std::ios::binary);
                if (!saved) throw std::runtime_error("cannot create reference logits");
            } else {
                std::ifstream meta(reference / (stem + ".json"));
                meta >> expected;
                if (expected.at("prompt_ids") != json(ids) || expected.at("vocabulary_size") != nv || expected.at("model") != model_path) throw std::runtime_error("reference mismatch");
                expected_logits.open(reference / (stem + ".f32"), std::ios::binary);
                if (!expected_logits) throw std::runtime_error("reference logits missing");
            }
            json choices = json::array(), forced = json::array(), step_errors = json::array();
            float max_error = 0, min_margin = std::numeric_limits<float>::infinity();
            bool finite = true, within = true, same_ids = true;
            const int steps = mode == "solo" ? max_new : expected.at("chosen_ids").size();
            for (int step = 0; step < steps; ++step) {
                auto batch = llama_batch_get_one(ids.data(), ids.size());
                fprintf(stderr, "ENGINE_STEP_BEGIN {\"prompt\":%zu,\"step\":%d,\"input_tokens\":%zu}\n", pi, step, ids.size());
                const int64_t step_start_us = ggml_time_us();
                if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("decode failed");
                llama_synchronize(ctx.get());
                fprintf(stderr, "ENGINE_STEP_END {\"prompt\":%zu,\"step\":%d,\"wall_us\":%lld}\n", pi, step, (long long) (ggml_time_us()-step_start_us));
                float * logits = llama_get_logits_ith(ctx.get(), -1);
                if (!logits) throw std::runtime_error("logits missing");
                int best = std::max_element(logits, logits + nv) - logits;
                choices.push_back(best);
                int next = best;
                std::vector<float> ref(nv);
                if (mode == "solo") {
                    saved.write(reinterpret_cast<const char *>(logits), nv * sizeof(float));
                    if (!saved) throw std::runtime_error("failed to save reference logits");
                    std::copy(logits, logits + nv, ref.begin());
                } else {
                    expected_logits.read(reinterpret_cast<char *>(ref.data()), nv * sizeof(float));
                    if (!expected_logits) throw std::runtime_error("truncated reference logits");
                    next = expected.at("chosen_ids").at(step).get<int>();
                    same_ids = same_ids && best == next;
                }
                float largest = -std::numeric_limits<float>::infinity(), second = largest;
                float step_max_error = 0;
                int step_outside_tolerance = 0;
                for (int j = 0; j < nv; ++j) {
                    finite = finite && std::isfinite(logits[j]) && std::isfinite(ref[j]);
                    const float error = std::abs(logits[j] - ref[j]);
                    max_error = std::max(max_error, error);
                    step_max_error = std::max(step_max_error, error);
                    step_outside_tolerance += !std::isfinite(error) || error > logit_atol + logit_rtol * std::abs(ref[j]);
                    within = within && error <= logit_atol + logit_rtol * std::abs(ref[j]);
                    if (ref[j] > largest) { second = largest; largest = ref[j]; }
                    else second = std::max(second, ref[j]);
                }
                step_errors.push_back({{"step", step}, {"max_abs_error", step_max_error}, {"outside_tolerance", step_outside_tolerance}});
                min_margin = std::min(min_margin, largest - second);
                forced.push_back(next);
                if (llama_vocab_is_eog(vocab, next)) break;
                ids = {next};
            }
            std::vector<llama_token> prompt_ids(count);
            llama_tokenize(vocab, prompt.data(), prompt.size(), prompt_ids.data(), count, true, false);
            bool ok = finite && within && same_ids && !choices.empty();
            json result = {{"prompt", prompt}, {"prompt_ids", prompt_ids}, {"model", model_path}, {"vocabulary_size", nv},
                           {"chosen_ids", choices}, {"reference_context_ids", forced}, {"finite", finite},
                           {"tokens_match", same_ids}, {"logits_within_tolerance", within}, {"max_abs_logit_error", max_error},
                           {"min_reference_top2_margin", min_margin}, {"step_errors", step_errors}, {"valid", ok}};
            std::ofstream(output / (stem + ".json")) << result.dump(2) << '\n';
            report["cases"].push_back(result);
            all_ok = all_ok && ok;
        }
        report["valid"] = all_ok;
        std::ofstream(output / "report.json") << report.dump(2) << '\n';
        std::cout << report.dump(2) << '\n';
        return all_ok ? 0 : 1;
    } catch (const std::exception & e) {
        std::cerr << "engine correctness: " << e.what() << '\n';
        return 2;
    }
}
