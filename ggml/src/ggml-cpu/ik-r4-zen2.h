// Linux/AVX2 optional R4 matrix kernels. The extension has an independent ggml ABI.
#include <cstdlib>
#include <dlfcn.h>
#include <link.h>
#include <string>

namespace ggml::cpu::repack {
struct ik_r4_library {
    using repack_fn = bool (*)(int,int,int,const void *,void *,size_t);
    using multiply_fn = bool (*)(int,int,int,int,const void *,size_t,const void *,size_t,float *,size_t,int,int);
    repack_fn repack = nullptr;
    multiply_fn multiply = nullptr;
    multiply_fn broadcast = nullptr;
    multiply_fn integer_minima = nullptr;
    int row_broadcast_mask = 0;
};
static const ik_r4_library & ik_r4_ops() {
    static const ik_r4_library ops = [] {
        ik_r4_library result;
        const char *enabled=std::getenv("GGML_QWEN_R4");
        if (!enabled || std::strcmp(enabled, "1") != 0) return result;
        Dl_info info{};
        if (!dladdr(reinterpret_cast<const void *>(&ik_r4_ops), &info) || !info.dli_fname) return result;
        std::string path(info.dli_fname);
        const auto slash=path.rfind('/');
        if (slash==std::string::npos) return result;
        path.resize(slash+1);
        path+="iqk-zen2/libqwen-ik-ops.so";
        void *handle=dlmopen(LM_ID_NEWLM,path.c_str(),RTLD_NOW|RTLD_LOCAL);
        if (!handle) { GGML_LOG_WARN("CPU R4 load: %s\n", dlerror()); return result; }
        auto version=reinterpret_cast<int(*)()>(dlsym(handle,"qwen_ik_abi"));
        result.repack=reinterpret_cast<ik_r4_library::repack_fn>(dlsym(handle,"qwen_ik_repack"));
        result.multiply=reinterpret_cast<ik_r4_library::multiply_fn>(dlsym(handle,"qwen_ik_multiply"));
        if(!version || version()!=1 || !result.repack || !result.multiply) {
            dlclose(handle);
            return ik_r4_library{};
        }
        result.broadcast = reinterpret_cast<ik_r4_library::multiply_fn>(dlsym(handle, "qwen_ik_multiply_broadcast"));
        const char * use_mins = std::getenv("GGML_QWEN_Q4_INT_MINS");
        if (use_mins && std::strcmp(use_mins, "1") == 0) {
            result.integer_minima = reinterpret_cast<ik_r4_library::multiply_fn>(dlsym(handle, "qwen_ik_multiply_integer_minima_v1"));
        }
        const char * mask = std::getenv("GGML_QWEN_R4_BROADCAST");
        if (result.broadcast && mask && (std::strcmp(mask,"1")==0 || std::strcmp(mask,"2")==0 || std::strcmp(mask,"3")==0)) {
            result.row_broadcast_mask = *mask-'0';
            GGML_LOG_INFO("CPU R4 broadcast activation, mask=%d\n", result.row_broadcast_mask);
        }
        GGML_LOG_INFO("CPU R4 enabled with upstream Q8_K activation quantization\n");
        return result;
    }();
    return ops;
}
struct ik_r4_activation {
    float d, sum;
    int8_t q[256];
    union { float groups32[8]; int16_t groups16[16]; } sums;
};
static_assert(sizeof(ik_r4_activation)==296 && sizeof(block_q8_K)==292);
class ik_r4_traits final : public tensor_traits_base {
    static int bits(const ggml_tensor *w) {return w->type==GGML_TYPE_Q4_K?4:6;}
    static bool broadcast_rows(const ggml_tensor * w) {
        return (ik_r4_ops().row_broadcast_mask & (bits(w)==4 ? 1 : 2)) != 0;
    }
    static size_t row_bytes(int64_t k, bool broadcast) {
        return size_t(k/256)*(sizeof(ik_r4_activation)+(broadcast ? 1024 : 0));
    }
public:
    int repack(ggml_tensor *t,const void *input,size_t size) override {
        return ik_r4_ops().repack(bits(t),t->ne[0],t->ne[1],input,t->data,size)?0:-1;
    }
    bool work_size(int nth,const ggml_tensor *op,size_t &s) override {
        if(op->op!=GGML_OP_MUL_MAT)return false;
        auto *x=op->src[1];
        s=row_bytes(x->ne[0],broadcast_rows(op->src[0]))*ggml_nrows(x);
        if(x->nb[0]!=sizeof(float))s+=size_t(nth)*x->ne[0]*sizeof(float);
        return true;
    }
    bool compute_forward(ggml_compute_params *params,ggml_tensor *op) override {
        if(op->op!=GGML_OP_MUL_MAT)return false;
        const ggml_tensor *w=op->src[0],*x=op->src[1];
        const int k=w->ne[0],rows=w->ne[1],nr=ggml_nrows(x),bw=bits(w);
        GGML_ASSERT(x->type==GGML_TYPE_F32 && w->ne[2]==1 && w->ne[3]==1 && x->ne[0]==k && op->nb[0]==sizeof(float));
        const bool broadcast=broadcast_rows(w);
        const size_t stride=row_bytes(k,broadcast);
        const bool integer_minima = broadcast && bw == 4 && ik_r4_ops().integer_minima;
        const auto multiply = integer_minima ? ik_r4_ops().integer_minima :
                              broadcast ? ik_r4_ops().broadcast : ik_r4_ops().multiply;
        auto *space=static_cast<unsigned char *>(params->wdata);
        size_t required=0;work_size(params->nth,op,required);
        GGML_ASSERT(params->wsize>=required);
        const auto quantize=ggml_get_type_traits_cpu(GGML_TYPE_Q8_K)->from_float;
        for(int row=params->ith;row<nr;row+=params->nth){
            int64_t i1=row%x->ne[1],plane=row/x->ne[1],i2=plane%x->ne[2],i3=plane/x->ne[2];
            const char *src=static_cast<const char *>(x->data)+i1*x->nb[1]+i2*x->nb[2]+i3*x->nb[3];
            const float *input=reinterpret_cast<const float *>(src);
            if(x->nb[0]!=sizeof(float)){
                auto *gather=reinterpret_cast<float *>(space+stride*nr)+size_t(params->ith)*k;
                for(int i=0;i<k;++i)std::memcpy(gather+i,src+i*x->nb[0],sizeof(float));
                input=gather;
            }
            unsigned char *buf=space+stride*row;
            quantize(input,buf,k);
            // Expand backwards in place. Only metadata layout changes.
            for(int b=k/256-1;b>=0;--b){
                block_q8_K original;std::memcpy(&original,buf+size_t(b)*sizeof(original),sizeof(original));
                if (broadcast) {
                    // One shared expansion of Q8 for all output groups. The prefix keeps the
                    // original upstream Q8 scale, quants and block sums used by the IQK kernel.
                    auto * dest=buf+size_t(k/256)*sizeof(ik_r4_activation)+size_t(b)*1024;
                    for (int group=0;group<8;++group) {
                        const auto y=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(original.qs+group*32));
                        auto * dst=reinterpret_cast<__m256i *>(dest+group*128);
                        _mm256_storeu_si256(dst+0,_mm256_shuffle_epi32(y,0x00));
                        _mm256_storeu_si256(dst+1,_mm256_shuffle_epi32(y,0x55));
                        _mm256_storeu_si256(dst+2,_mm256_shuffle_epi32(y,0xaa));
                        _mm256_storeu_si256(dst+3,_mm256_shuffle_epi32(y,0xff));
                    }
                }
                ik_r4_activation expanded = {};
                expanded.d=original.d;std::memcpy(expanded.q,original.qs,sizeof(expanded.q));
                int total=0;
                for(int g=0;g<8;++g){
                    const int s=original.bsums[2*g]+original.bsums[2*g+1];total+=s;
                    if (integer_minima) {
                        // Each 128-bit SIMD lane contains the sums for four successive groups.
                        expanded.sums.groups16[(g/4)*8 + g%4] = (int16_t) s;
                    } else if (bw==4) {
                        expanded.sums.groups32[g] = original.d*s;
                    }
                }
                if(bw==6)std::memcpy(expanded.sums.groups16,original.bsums,sizeof(original.bsums));
                expanded.sum=original.d*total;
                std::memcpy(buf+size_t(b)*sizeof(expanded),&expanded,sizeof(expanded));
            }
        }
        ggml_barrier(params->threadpool);
        static const size_t tile_bytes = []() -> size_t {
            const char * value = std::getenv("GGML_QWEN_R4_TILE_KIB");
            if (!value) return 0;
            char * end = nullptr;
            const long kib = std::strtol(value, &end, 10);
            return end && *end == 0 && kib >= 32 && kib <= 2048 ? size_t(kib) * 1024 : 0;
        }();
        int begin = 0, end = rows, step = rows;
        int worker = params->ith, workers = params->nth;
        if (nr > 8 && tile_bytes > 0) {
            // Four logical weight rows form one contiguous packed group.
            // The worker keeps its own row range and reuses a cache-sized tile across all positions.
            const int groups_per_worker = (rows / 4 + workers - 1) / workers;
            begin = groups_per_worker * worker * 4;
            end = std::min(rows, begin + groups_per_worker * 4);
            step = 4 * std::max<int>(1, int(tile_bytes / (4 * w->nb[1])));
            worker = 0;
            workers = 1;
        }
        for (int row = begin; row < end; row += step) {
            const int tile_rows = std::min(step, end - row);
            const auto * weights = static_cast<const unsigned char *>(w->data) + size_t(row) * w->nb[1];
            for (int64_t i3 = 0; i3 < x->ne[3]; ++i3) {
                for (int64_t i2 = 0; i2 < x->ne[2]; ++i2) {
                    const int64_t first = x->ne[1] * (i2 + x->ne[2] * i3);
                    for (int64_t j = 0; j < x->ne[1]; j += 8) {
                        const int width = std::min<int64_t>(8, x->ne[1] - j);
                        char * dst = static_cast<char *>(op->data) + i3 * op->nb[3] + i2 * op->nb[2] + j * op->nb[1];
                        const bool ok = multiply(bw, k, tile_rows, width, weights, w->nb[1],
                                space + stride * (first + j), stride, reinterpret_cast<float *>(dst) + row,
                                op->nb[1] / sizeof(float), worker, workers);
                        GGML_ASSERT(ok);
                    }
                }
            }
        }
        return true;
    }
};
}
