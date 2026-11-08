// Mikoshi BitWeave R03 - exact event-delta inference.
// y_new = y_old + W*(x_new - x_old) for INT8 activation and ternary W.
// W packed using two uint32 bit planes (sign + nonzero).
// One warp per output neuron, each lane visits strided events; exact INT32.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <algorithm>
#include <cmath>
#define CU(expr) do {cudaError_t e=(expr); if(e!=cudaSuccess){fprintf(stderr,"CUDA at %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
#define BL(expr) do {cublasStatus_t e=(expr);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"BLAS at %d: %d\n",__LINE__,int(e));exit(3);}}while(0)
__global__ void pack_ternary_weight(const int8_t* __restrict__ weight,
                                    uint32_t* __restrict__ sign,
                                    uint32_t* __restrict__ nz,int N,int K){
    int lane=threadIdx.x&31,word=blockIdx.x*8+(threadIdx.x>>5),row=blockIdx.y;
    if(row>=N||word>=K/32)return;
    const int8_t v=weight[row*K+word*32+lane];
    uint32_t s=__ballot_sync(0xffffffffu,v<0);
    uint32_t z=__ballot_sync(0xffffffffu,v!=0);
    if(lane==0){sign[row*(K/32)+word]=s;nz[row*(K/32)+word]=z;}
}
__global__ void delta_warp(const uint32_t* __restrict__ weightSign,
                           const uint32_t* __restrict__ weightNz,
                           const int32_t* __restrict__ prev,
                           const int32_t* __restrict__ indices,
                           const int16_t* __restrict__ delta,
                           int32_t* __restrict__ next,
                           int N,int K,int count){
    const int lane=threadIdx.x&31;
    const int row=blockIdx.x*8+(threadIdx.x>>5);
    if(row>=N)return;
    int64_t acc=0;
    int words=K/32;
    for(int e=lane;e<count;e+=32){
        const int q=indices[e];
        const int32_t d=delta[e];
        const int word=q>>5;
        const uint32_t mask=1u<<(q&31);
        const uint32_t nz=weightNz[row*words+word]&mask;
        if(nz){
            const uint32_t negative=weightSign[row*words+word]&mask;
            acc+=negative?-d:d;
        }
    }
    #pragma unroll
    for(int shift=16;shift>0;shift>>=1)
        acc+=__shfl_down_sync(0xffffffffu,acc,shift);
    if(lane==0)next[row]=prev[row]+int32_t(acc);
}
// One-time-per-token 8-bit activation to eight binary bitplanes.
// INT8 signed two's complement represented by bits 0..7.
__global__ void pack_activation_i8(const int8_t* __restrict__ input,
                                  uint32_t* __restrict__ planes,int K) {
    const int lane=threadIdx.x&31;
    const int word=blockIdx.x*8+(threadIdx.x>>5);
    if(word>=K/32)return;
    const uint8_t v=static_cast<uint8_t>(input[word*32+lane]);
    #pragma unroll
    for(int bit=0;bit<8;++bit){
        const uint32_t mask=__ballot_sync(0xffffffffu,(v>>bit)&1);
        if(lane==0)planes[bit*(K/32)+word]=mask;
    }
}
template<bool TERNARY_WEIGHT>
__global__ void warp_gemv_mixed_i8(const uint32_t* __restrict__ actplane,
                            const uint32_t* __restrict__ weights_sign,
                            const uint32_t* __restrict__ weights_nonzero,
                            int32_t* __restrict__ out,int N,int K) {
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    const int row=blockIdx.x*8+warp;
    if(row>=N)return;
    const int words=K/32;
    int score=0;
    for(int q=lane;q<words;q+=32) {
        const uint32_t wsign=weights_sign[row*words+q];
        const uint32_t nz=TERNARY_WEIGHT?weights_nonzero[row*words+q]:0xffffffffu;
        #pragma unroll
        for(int bit=0;bit<8;++bit){
            const uint32_t both=nz & actplane[bit*words+q];
            const int value=__popc(both)-2*__popc(both & wsign);
            score += (bit==7 ? -128 : (1<<bit))*value;
        }
    }
    #pragma unroll
    for(int delta=16;delta>0;delta>>=1)
        score+=__shfl_down_sync(0xffffffffu,score,delta);
    if(lane==0)out[row]=score;
}


template<class F>float benchmark(int iterations,F f){
    f();f();CU(cudaDeviceSynchronize());
    cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));
    CU(cudaEventRecord(a));
    for(int i=0;i<iterations;++i)f();
    CU(cudaEventRecord(b));CU(cudaEventSynchronize(b));
    float ms=0;CU(cudaEventElapsedTime(&ms,a,b));
    CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));
    return ms/iterations;
}
int main(int argc,char**argv){
    int N=4096,K=4096,iters=100;
    for(int i=1;i<argc;++i){
        if(i+1>=argc)return 1;
        std::string key=argv[i];int v=atoi(argv[++i]);
        if(key=="--n")N=v;else if(key=="--k")K=v;else if(key=="--iters")iters=v;else return 1;
    }
    if(N<256||N>8192||N%256||K<256||K>8192||K%256||iters<1||iters>1000)return 1;
    std::mt19937 rng(20261028);
    std::uniform_int_distribution<int> trits(-1,1),activ(-64,63);
    size_t weightsCount=size_t(N)*K;
    std::vector<int8_t> W(weightsCount),oldA(K),newA(K);
    for(auto&v:W)v=int8_t(trits(rng));
    for(auto&v:oldA)v=int8_t(activ(rng));
    int8_t *Wd=nullptr,*Ad=nullptr,*ANewd=nullptr;
    uint32_t *Ws=nullptr,*Wn=nullptr,*Aplanes=nullptr;
    int32_t *prev=nullptr,*next=nullptr,*truth=nullptr,*evt_index=nullptr;
    int16_t *evt_delta=nullptr;
    CU(cudaMalloc(reinterpret_cast<void**>(&Wd),weightsCount));
    CU(cudaMalloc(reinterpret_cast<void**>(&Ad),K));
    CU(cudaMalloc(reinterpret_cast<void**>(&ANewd),K));
    CU(cudaMalloc(reinterpret_cast<void**>(&Ws),weightsCount/8));
    CU(cudaMalloc(reinterpret_cast<void**>(&Wn),weightsCount/8));
    CU(cudaMalloc(reinterpret_cast<void**>(&Aplanes),size_t(K)));
    CU(cudaMalloc(reinterpret_cast<void**>(&prev),size_t(N)*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&next),size_t(N)*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&truth),size_t(N)*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&evt_index),size_t(K)*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&evt_delta),size_t(K)*2));
    CU(cudaMemcpy(Wd,W.data(),weightsCount,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(Ad,oldA.data(),K,cudaMemcpyHostToDevice));
    pack_ternary_weight<<<dim3((K/32+7)/8,N),256>>>(Wd,Ws,Wn,N,K);
    CU(cudaGetLastError());CU(cudaDeviceSynchronize());
    cublasHandle_t handle;BL(cublasCreate(&handle));
    int alpha=1,beta=0;
    auto gemm_old=[&]{
        BL(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,1,K,
                       &alpha,Wd,CUDA_R_8I,K,Ad,CUDA_R_8I,K,
                       &beta,prev,CUDA_R_32I,N,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
    };
    auto gemm_new=[&]{
        BL(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,1,K,
                       &alpha,Wd,CUDA_R_8I,K,ANewd,CUDA_R_8I,K,
                       &beta,truth,CUDA_R_32I,N,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
    };
    gemm_old(); CU(cudaDeviceSynchronize());
    const float fullms=benchmark(iters,gemm_old);
    printf("n,k,event_count,delta_ms,full_bitweave_ms,cublas_full_ms,chosen,speedup_vs_bitweave,validation\n");
    for(int count: {1,4,16,64,128,256,384,512,1024,2048}){
        if(count>K)continue;
        newA=oldA;
        std::vector<int32_t> indices(count);
        std::vector<int16_t> deltas(count);
        // Deterministic spread: unique indices, 32 is coprime to K? use odd stride 97.
        for(int e=0;e<count;++e){
            int idx=(int64_t(e)*97+7)%K;
            int newv=int(newA[idx])+((e&1)?-3:5);
            if(newv>127)newv=127;
            if(newv< -128)newv=-128;
            if(newv==int(oldA[idx]))newv=(newv==127)?126:newv+1;
            indices[e]=idx;
            deltas[e]=int16_t(newv-int(oldA[idx]));
            newA[idx]=int8_t(newv);
        }
        CU(cudaMemcpy(evt_index,indices.data(),size_t(count)*4,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(evt_delta,deltas.data(),size_t(count)*2,cudaMemcpyHostToDevice));
        CU(cudaMemcpy(ANewd,newA.data(),size_t(K),cudaMemcpyHostToDevice));
        auto delta_kernel=[&]{
            delta_warp<<<N/8,256>>>(Ws,Wn,prev,evt_index,evt_delta,next,N,K,count);
            CU(cudaGetLastError());
        };
        auto dense_kernel=[&]{
            pack_activation_i8<<<(K/32+7)/8,256>>>(ANewd,Aplanes,K);
            CU(cudaGetLastError());
            warp_gemv_mixed_i8<true><<<N/8,256>>>(Aplanes,Ws,Wn,next,N,K);
            CU(cudaGetLastError());
        };
        delta_kernel();gemm_new();CU(cudaDeviceSynchronize());
        std::vector<int32_t> computed(N),ref(N);
        CU(cudaMemcpy(computed.data(),next,size_t(N)*4,cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(ref.data(),truth,size_t(N)*4,cudaMemcpyDeviceToHost));
        size_t errors=0,cpuerros=0;
        for(int row=0;row<N;++row)errors+=(computed[row]!=ref[row]);
        for(int t=0;t<32;++t){
            int row=(t*131+17)%N;
            int64_t expected=0;
            for(int q=0;q<K;++q)expected+=int(W[size_t(row)*K+q])*int(newA[q]);
            cpuerros+=(expected!=computed[row]);
        }
        if(errors||cpuerros){
            fprintf(stderr,"FAIL events=%d mismatches=%zu cpu=%zu\n",count,errors,cpuerros);return 5;
        }
        float ms=benchmark(iters,delta_kernel);
        // Dense output computed independently, using 8 bitplanes and ternary weight masks.
        dense_kernel();CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(computed.data(),next,size_t(N)*4,cudaMemcpyDeviceToHost));
        size_t denseErrors=0;
        for(int row=0;row<N;++row)denseErrors+=(computed[row]!=ref[row]);
        if(denseErrors){fprintf(stderr,"DENSE FAIL count=%d errors=%zu\n",count,denseErrors);return 5;}
        const float densems=benchmark(iters,dense_kernel);
        const bool useEvent=(ms<densems);
        const char* chosen=useEvent?"event_delta":"full_bitweave";
        // Demonstrate real dispatch, not merely print a choice.
        if(useEvent)delta_kernel(); else dense_kernel();
        CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(computed.data(),next,size_t(N)*4,cudaMemcpyDeviceToHost));
        size_t dispatchedErrors=0;
        for(int row=0;row<N;++row)dispatchedErrors+=(computed[row]!=ref[row]);
        if(dispatchedErrors){fprintf(stderr,"DISPATCH FAIL count=%d errors=%zu\n",count,dispatchedErrors);return 5;}
        printf("%d,%d,%d,%.6f,%.6f,%.6f,%s,%.4f,PASS\n",
               N,K,count,ms,densems,fullms,chosen,densems/ms);
        fflush(stdout);
    }
    BL(cublasDestroy(handle));
    CU(cudaFree(Wd));CU(cudaFree(Ad));CU(cudaFree(ANewd));
    CU(cudaFree(Ws));CU(cudaFree(Wn));CU(cudaFree(Aplanes));CU(cudaFree(prev));CU(cudaFree(next));CU(cudaFree(truth));
    CU(cudaFree(evt_index));CU(cudaFree(evt_delta));
    return 0;
}
