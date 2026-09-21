// Private shared-model scheduling probe. Host intervals do not prove GPU overlap.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-rpc.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using json = nlohmann::ordered_json;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
static double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

int main(int argc, char ** argv) {
    if (argc != 6) {
        std::cerr << "Usage: gguf_contexts MODEL solo|layer RPC_ENDPOINT REFERENCE_DIR OUTPUT_JSON\n";
        return 2;
    }
    try {
        const std::string path=argv[1], mode=argv[2], endpoint=argv[3];
        if (mode!="solo" && mode!="layer") throw std::runtime_error("invalid mode");
        ggml_backend_load_all(); llama_backend_init();
        const bool no_blas=std::getenv("GGUF_PROBE_NO_BLAS") != nullptr;
        if (no_blas) {
            auto blas=ggml_backend_reg_by_name("BLAS");
            if (blas) ggml_backend_unload(blas);
        }
        json device_caps=json::array();
        for (size_t i=0;i<ggml_backend_dev_count();++i) {
            auto dev=ggml_backend_dev_get(i); ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev,&props);
            device_caps.push_back({{"name",props.name},{"type",int(props.type)},
                {"async",props.caps.async},{"events",props.caps.events}});
        }
        auto metal=ggml_backend_dev_by_name("MTL0");
        if (!metal) throw std::runtime_error("Metal unavailable");
        std::vector<ggml_backend_dev_t> devices={metal};
        if (mode=="layer") {
            auto rpc=ggml_backend_rpc_add_server(endpoint.c_str());
            if (!rpc || ggml_backend_reg_dev_count(rpc)!=1) throw std::runtime_error("expected one RPC device");
            devices.push_back(ggml_backend_reg_dev_get(rpc,0));
        }
        devices.push_back(nullptr);
        std::vector<float> split(llama_max_devices(),0); split[0]=split[1]=1;
        auto mp=llama_model_default_params(); mp.devices=devices.data(); mp.n_gpu_layers=999;
        mp.split_mode=mode=="layer" ? LLAMA_SPLIT_MODE_LAYER : LLAMA_SPLIT_MODE_NONE;
        mp.tensor_split=split.data(); mp.load_mtp=false;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(path.c_str(),mp),llama_model_free);
        if (!model) throw std::runtime_error("load failed");
        const auto * vocab=llama_model_get_vocab(model.get()); const int nv=llama_vocab_n_tokens(vocab);
        std::vector<Context> contexts;
        std::vector<json> refs;
        std::vector<std::vector<float>> ref_logits;
        for (int i=0;i<3;++i) {
            auto cp=llama_context_default_params(); cp.n_ctx=256; cp.n_batch=128; cp.n_ubatch=128;
            cp.n_seq_max=1; cp.n_threads=cp.n_threads_batch=4;
            cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED; cp.type_k=cp.type_v=GGML_TYPE_F16;
            contexts.emplace_back(llama_init_from_model(model.get(),cp),llama_free);
            if (!contexts.back()) throw std::runtime_error("context init failed");
            const fs::path stem=fs::path(argv[4])/("prompt-"+std::to_string(i));
            std::ifstream meta(stem.string()+".json"); json ref; meta>>ref;
            if (ref.at("model")!=path || ref.at("vocabulary_size")!=nv || !ref.at("valid").get<bool>())
                throw std::runtime_error("reference identity mismatch");
            const std::string prompt=ref.at("prompt");
            std::vector<llama_token> ids(ref.at("prompt_ids").size());
            int n=llama_tokenize(vocab,prompt.data(),prompt.size(),ids.data(),ids.size(),true,false);
            if (n!=int(ids.size()) || json(ids)!=ref.at("prompt_ids")) throw std::runtime_error("prompt mismatch");
            const size_t count=ref.at("chosen_ids").size()*nv;
            std::ifstream bin(stem.string()+".f32",std::ios::binary);
            std::vector<float> values(count); bin.read(reinterpret_cast<char *>(values.data()),count*sizeof(float));
            if (!bin || bin.peek()!=EOF) throw std::runtime_error("bad reference logits");
            refs.push_back(ref); ref_logits.push_back(std::move(values));
        }
        json runs=json::array(); bool valid=true;
        // First pair warms all contexts and both scheduling modes; subsequent pairs alternate order.
        const std::vector<std::string> order={"serial","deferred","serial","deferred","deferred","serial"};
        for (size_t ri=0;ri<order.size();++ri) {
            const bool deferred=order[ri]=="deferred";
            std::vector<std::vector<llama_token>> input(3),chosen(3);
            std::vector<std::vector<std::vector<float>>> captured(3);
            std::vector<int> step(3,0); std::vector<bool> pending(3,false),done(3,false);
            json events=json::array();
            for (int i=0;i<3;++i) {
                llama_synchronize(contexts[i].get());
                llama_memory_clear(llama_get_memory(contexts[i].get()),true);
                input[i]=refs[i].at("prompt_ids").get<std::vector<llama_token>>();
                captured[i].resize(refs[i].at("chosen_ids").size(),std::vector<float>(nv));
            }
            const double start=now();
            auto submit=[&](int i) {
                const double before=now();
                auto batch=llama_batch_get_one(input[i].data(),input[i].size());
                if (llama_decode(contexts[i].get(),batch)!=0) throw std::runtime_error("decode failed");
                pending[i]=true;
                events.push_back({{"event","submit"},{"context",i},{"step",step[i]},
                    {"start_s",before-start},{"end_s",now()-start}});
            };
            auto collect=[&](int i) {
                if (!pending[i]) throw std::runtime_error("missing pending graph");
                const double before=now();
                float * logits=llama_get_logits_ith(contexts[i].get(),-1);
                const double ready=now();
                if (!logits) throw std::runtime_error("missing logits");
                std::copy(logits,logits+nv,captured[i][step[i]].begin());
                const llama_token next=std::max_element(logits,logits+nv)-logits;
                chosen[i].push_back(next); input[i]={next}; pending[i]=false;
                events.push_back({{"event","collect"},{"context",i},{"step",step[i]},
                    {"start_s",before-start},{"ready_s",ready-start},{"end_s",now()-start}});
                ++step[i];
                done[i]=llama_vocab_is_eog(vocab,next) || step[i]>=int(refs[i].at("chosen_ids").size());
            };
            if (deferred) {
                for (int i=0;i<3;++i) submit(i);
                while (!std::all_of(done.begin(),done.end(),[](bool x){return x;})) {
                    for (int i=0;i<3;++i) if (!done[i]) {
                        collect(i);
                        if (!done[i]) submit(i);
                    }
                }
            } else {
                while (!std::all_of(done.begin(),done.end(),[](bool x){return x;})) {
                    for (int i=0;i<3;++i) if (!done[i]) { submit(i); collect(i); }
                }
            }
            const double elapsed=now()-start;
            bool run_ok=true; float max_error=0; size_t outside=0,tokens=0;
            for (int i=0;i<3;++i) {
                run_ok=run_ok && json(chosen[i])==refs[i].at("chosen_ids"); tokens+=chosen[i].size();
                for (size_t s=0;s<chosen[i].size();++s) for (int j=0;j<nv;++j) {
                    float value=captured[i][s][j], ref=ref_logits[i][s*nv+j];
                    float error=std::abs(value-ref);
                    if (!std::isfinite(value) || !std::isfinite(ref) || error>0.1f+0.01f*std::abs(ref)) ++outside;
                    max_error=std::max(max_error,error);
                }
            }
            run_ok=run_ok && outside==0; valid=valid && run_ok;
            runs.push_back({{"schedule",order[ri]},{"warmup",ri<2},{"valid",run_ok},{"wall_s",elapsed},
                {"chosen_ids",chosen},{"generated_tokens",tokens},{"max_abs_logit_error",max_error},
                {"outside_tolerance",outside},{"events",events}});
            std::cerr << "RUN " << ri << ' ' << order[ri] << " valid=" << run_ok << " wall=" << elapsed << '\n';
        }
        json report={{"model",path},{"mode",mode},{"blas_disabled",no_blas},{"device_caps_before_rpc",device_caps},{"contexts",3},{"valid",valid},{"runs",runs},
            {"scope","One model shared by three independent contexts; same-GGUF solo full-logit reference. Includes prefill, output copying and host event recording. Diagnostic wall times only, no GPU overlap or physical-pair speedup claim."}};
        std::ofstream(argv[5])<<report.dump(2)<<'\n'; return valid?0:1;
    } catch (const std::exception & e) { std::cerr<<e.what()<<'\n'; return 2; }
}
