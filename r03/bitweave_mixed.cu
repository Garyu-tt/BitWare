// BitWeave-GEMV R01: low-batch token decoding on Pascal sm_61.
// A[1,K] binary +/-1 (dynamic), W[N,K] binary or ternary (static).
// Compare exact INT32 outputs to cuBLAS GEMMEx signed INT8 reference.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#define CU(x) do { cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA %d %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
#define BL(x) do { cublasStatus_t e=(x);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"BLAS %d status %d\n",__LINE__,int(e));exit(3);}}while(0)

__global__ void pack(const int8_t* __restrict__ input,uint32_t* __restrict__ sign,
                     uint32_t* __restrict__ nz,int N,int K) {
    const int tid=threadIdx.x, lane=tid&31;
    const int word=blockIdx.x*8+(tid>>5),row=blockIdx.y;
    if(row>=N||word>=K/32) return;
    int8_t v=input[row*K+word*32+lane];
    uint32_t s=__ballot_sync(0xffffffffu,v<0);
    uint32_t z=__ballot_sync(0xffffffffu,v!=0);
    if(lane==0) {
        sign[row*K/32+word]=s;
        if(nz)nz[row*K/32+word]=z;
    }
}
template<bool TERNARY>
__global__ void warp_gemv(const uint32_t* __restrict__ activ,
                          const uint32_t* __restrict__ weights,
                          const uint32_t* __restrict__ activeweights,
                          int32_t* __restrict__ output,int N,int K) {
    const int lane=threadIdx.x&31;
    const int warp=threadIdx.x>>5;
    const int row=blockIdx.x*8+warp;
    if(row>=N)return;
    const int words=K/32;
    int score=0;
    for(int q=lane;q<words;q+=32) {
        uint32_t x=activ[q],w=weights[row*words+q];
        if(TERNARY){
            uint32_t nz=activeweights[row*words+q];
            score+=__popc(nz)-2*__popc(nz & (x^w));
        } else {
            score+=__popc(x^w);
        }
    }
    #pragma unroll
    for(int delta=16;delta>0;delta>>=1)
        score+=__shfl_down_sync(0xffffffffu,score,delta);
    if(lane==0)output[row]=TERNARY?score:(K-2*score);
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

template<class F> float bench(int iters,F fn) {
    fn();fn();CU(cudaDeviceSynchronize());
    cudaEvent_t a,b;CU(cudaEventCreate(&a));CU(cudaEventCreate(&b));
    CU(cudaEventRecord(a));
    for(int i=0;i<iters;++i)fn();
    CU(cudaEventRecord(b));CU(cudaEventSynchronize(b));
    float total;CU(cudaEventElapsedTime(&total,a,b));
    CU(cudaEventDestroy(a));CU(cudaEventDestroy(b));
    return total/iters;
}
int main(int argc,char**argv){
    int N=4096,K=4096,iterations=100,mode=0;
    for(int i=1;i<argc;++i){
        if(i+1>=argc)return 1;
        std::string key=argv[i];int v=atoi(argv[++i]);
        if(key=="--n")N=v;
        else if(key=="--k")K=v;
        else if(key=="--iters")iterations=v;
        else if(key=="--ternary")mode=v;
        else return 1;
    }
    if(N<256||N>16384||N%256||K<256||K>16384||K%256||iterations<1||iterations>2000||mode<0||mode>3)return 1;
    const size_t weightsCount=size_t(N)*K;
    std::vector<int8_t> a(K),w(weightsCount);
    std::mt19937 rng(20261028);
    std::uniform_int_distribution<int> bits(0,1),trits(-1,1),activ8(-64,63);
    for(auto &v:a)v=int8_t(mode>=2?activ8(rng):(bits(rng)*2-1));
    for(auto &v:w)v=int8_t((mode==1||mode==2)?trits(rng):(bits(rng)*2-1));
    int8_t *ad=nullptr,*wd=nullptr;
    uint32_t *ap=nullptr,*ap8=nullptr,*wp=nullptr,*wnz=nullptr;
    int32_t *out=nullptr,*reference=nullptr;
    CU(cudaMalloc(reinterpret_cast<void**>(&ad),size_t(K)));
    CU(cudaMalloc(reinterpret_cast<void**>(&wd),weightsCount));
    CU(cudaMalloc(reinterpret_cast<void**>(&ap),size_t(K)/8));
    CU(cudaMalloc(reinterpret_cast<void**>(&ap8),size_t(K)));
    CU(cudaMalloc(reinterpret_cast<void**>(&wp),weightsCount/8));
    CU(cudaMalloc(reinterpret_cast<void**>(&wnz),weightsCount/8));
    CU(cudaMalloc(reinterpret_cast<void**>(&out),size_t(N)*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&reference),size_t(N)*4));
    CU(cudaMemcpy(ad,a.data(),K,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(wd,w.data(),weightsCount,cudaMemcpyHostToDevice));
    auto packa=[&]{
        if(mode>=2)
            pack_activation_i8<<<(K/32+7)/8,256>>>(ad,ap8,K);
        else
            pack<<<dim3((K/32+7)/8,1),256>>>(ad,ap,nullptr,1,K);
        CU(cudaGetLastError());
    };
    auto packw=[&]{pack<<<dim3((K/32+7)/8,N),256>>>(wd,wp,(mode==1||mode==2)?wnz:nullptr,N,K);CU(cudaGetLastError());};
    packa();packw();
    auto kernel=[&]{
        if(mode==2)warp_gemv_mixed_i8<true><<<N/8,256>>>(ap8,wp,wnz,out,N,K);
        else if(mode==3)warp_gemv_mixed_i8<false><<<N/8,256>>>(ap8,wp,wnz,out,N,K);
        else if(mode==1)warp_gemv<true><<<N/8,256>>>(ap,wp,wnz,out,N,K);
        else warp_gemv<false><<<N/8,256>>>(ap,wp,wnz,out,N,K);
        CU(cudaGetLastError());
    };
    cublasHandle_t handle;BL(cublasCreate(&handle));
    const int alpha=1,beta=0;
    auto blas=[&]{
        BL(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,1,K,
                       &alpha,wd,CUDA_R_8I,K,ad,CUDA_R_8I,K,
                       &beta,reference,CUDA_R_32I,N,
                       CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
    };
    kernel();blas();CU(cudaDeviceSynchronize());
    std::vector<int32_t> result(N),ref(N);
    CU(cudaMemcpy(result.data(),out,size_t(N)*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(ref.data(),reference,size_t(N)*4,cudaMemcpyDeviceToHost));
    int errors=0,cpu_errors=0;
    for(int j=0;j<N;++j)errors+=result[j]!=ref[j];
    for(int j=0;j<32;++j) {
        const int row=(j*131+7)%N;
        int expected=0;
        for(int q=0;q<K;++q)expected+=int(a[q])*int(w[size_t(row)*K+q]);
        cpu_errors+=result[row]!=expected;
    }
    if(errors||cpu_errors){
        fprintf(stderr,"FAIL N=%d K=%d mismatches=%d cpu_errors=%d\n",N,K,errors,cpu_errors);
        return 5;
    }
    fprintf(stderr,"PASS mode=%s N=%d K=%d all-cuBLAS and 32 CPU refs\n",
            mode==2?"ternary_weight_i8_activation":mode==3?"binary_weight_i8_activation":mode?"ternary_weight":"binary_weight",N,K);
    float kms=bench(iterations,kernel);
    float bms=bench(iterations,blas);
    float ams=bench(iterations,packa);
    float wms=bench(iterations,packw);
    printf("mode,n,k,method,ms,throughput_glogical_ops,pass\n");
    auto row=[&](const char* name,float ms) {
        printf("%s,%d,%d,%s,%.6f,%.6f,PASS\n",mode==2?"ternaryXint8":mode==3?"binaryXint8":mode?"ternary":"binary",N,K,name,ms,
            (2.0*double(N)*K)/(ms*1e6));
    };
    row("warp_popcount",kms);
    row("cublas_int8",bms);
    row("activation_pack_dynamic",ams);
    row("weight_pack_once",wms);
    fprintf(stderr,"SPEEDUP kernel=%.3fx, with dynamic input pack=%.3fx\n",
            bms/kms,bms/(kms+ams));
    BL(cublasDestroy(handle));
    CU(cudaFree(ad));CU(cudaFree(wd));CU(cudaFree(ap));CU(cudaFree(wp));
    CU(cudaFree(wnz));CU(cudaFree(ap8));CU(cudaFree(out));CU(cudaFree(reference));
    return 0;
}
