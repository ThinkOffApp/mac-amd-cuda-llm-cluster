// Validate an HTTP slot state by continuing it through the engine API.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
using json=nlohmann::ordered_json;
using Context=std::unique_ptr<llama_context,decltype(&llama_free)>;
int main(int argc,char **argv) {
    if(argc!=6 && argc!=7) { std::cerr<<"Usage: gguf_state_check MODEL STATE PROMPT_JSON FULL_SWA_0_OR_1 OUTPUT [PREFILL_CHUNK_ENDS_JSON]\n";return 2; }
    try {
        std::ifstream input(argv[3]); json prompt_json;input>>prompt_json;
        auto prompt=prompt_json.get<std::vector<llama_token>>();
        if(prompt.size()<2 || prompt.size()+16>4096) throw std::runtime_error("invalid prompt length");
        ggml_backend_load_all();llama_backend_init();
        auto metal=ggml_backend_dev_by_name("MTL0");if(!metal)throw std::runtime_error("Metal unavailable");
        ggml_backend_dev_t devices[]={metal,nullptr};
        auto mp=llama_model_default_params();mp.devices=devices;mp.n_gpu_layers=999;mp.load_mtp=false;
        std::unique_ptr<llama_model,decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1],mp),llama_model_free);
        if(!model)throw std::runtime_error("model load failed");
        auto cp=llama_context_default_params();cp.n_ctx=4096;cp.n_batch=512;cp.n_ubatch=512;cp.n_seq_max=1;
        cp.n_threads=cp.n_threads_batch=4;cp.type_k=cp.type_v=GGML_TYPE_F16;
        cp.flash_attn_type=LLAMA_FLASH_ATTN_TYPE_ENABLED;cp.swa_full=std::string(argv[4])=="1";
        Context native(llama_init_from_model(model.get(),cp),llama_free),restored(llama_init_from_model(model.get(),cp),llama_free);
        if(!native||!restored)throw std::runtime_error("context failed");
        size_t packed_count=0;
        if(!llama_state_seq_load_file(restored.get(),argv[2],0,nullptr,0,&packed_count))throw std::runtime_error("state header failed");
        std::vector<llama_token> packed(packed_count);
        size_t bytes=llama_state_seq_load_file(restored.get(),argv[2],0,packed.data(),packed.size(),&packed_count);
        if(!bytes)throw std::runtime_error("state restore failed");
        auto max_pos=llama_memory_seq_pos_max(llama_get_memory(restored.get()),0);
        if(max_pos!=int(prompt.size())-2)throw std::runtime_error("state not at expected prefix boundary");
        std::vector<size_t> ends;
        if(argc==7) {
            std::ifstream plan(argv[6]);json value;plan>>value;ends=value.get<std::vector<size_t>>();
        } else {
            for(size_t n=512;n<prompt.size();n+=512)ends.push_back(n);
            ends.push_back(prompt.size());
        }
        if(ends.empty() || ends.back()!=prompt.size())throw std::runtime_error("bad prefill plan end");
        size_t offset=0;
        for(size_t end:ends) {
            if(end<=offset || end-offset>512 || end>prompt.size())throw std::runtime_error("bad prefill chunk");
            int count=end-offset;
            if(llama_decode(native.get(),llama_batch_get_one(prompt.data()+offset,count)))throw std::runtime_error("native prefill failed");
            offset=end;
        }
        llama_token next=prompt.back();
        if(llama_decode(restored.get(),llama_batch_get_one(&next,1)))throw std::runtime_error("restore tail failed");
        const auto *vocab=llama_model_get_vocab(model.get());const int nv=llama_vocab_n_tokens(vocab);
        json steps=json::array();bool valid=true;
        for(int step=0;step<16;++step) {
            const float *a=llama_get_logits_ith(native.get(),-1),*b=llama_get_logits_ith(restored.get(),-1);
            int ia=std::max_element(a,a+nv)-a,ib=std::max_element(b,b+nv)-b;
            size_t outside=0;float max_error=0;
            for(int j=0;j<nv;++j) {
                float error=std::abs(a[j]-b[j]);max_error=std::max(max_error,error);
                if(!std::isfinite(a[j])||!std::isfinite(b[j])||error>0.1f+0.01f*std::abs(a[j]))++outside;
            }
            valid=valid && ia==ib && outside==0;
            steps.push_back({{"step",step},{"native_token",ia},{"restored_token",ib},{"max_abs_error",max_error},{"outside_tolerance",outside}});
            if(llama_vocab_is_eog(vocab,ia))break;
            if(step<15) {
                llama_token ta=ia,tb=ib;
                if(llama_decode(native.get(),llama_batch_get_one(&ta,1))||llama_decode(restored.get(),llama_batch_get_one(&tb,1)))throw std::runtime_error("continuation failed");
            }
        }
        json report={{"valid",valid},{"model",argv[1]},{"state",argv[2]},{"state_bytes_loaded",bytes},
            {"prefix_pos_max",max_pos},{"prompt_tokens",prompt.size()},{"full_swa",cp.swa_full},
            {"prefill_chunk_ends",ends},{"atol",0.1f},{"rtol",0.01f},{"steps",steps},
            {"scope","Same-host Metal; native full prefill versus HTTP slot state restored in a fresh executable. Not cross-backend or network performance."}};
        std::ofstream(argv[5])<<report.dump(2)<<'\n';std::cout<<report.dump(2)<<'\n';return valid?0:1;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 2;}
}
