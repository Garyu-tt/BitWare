// BitWeave R04 - block-sparse 1-bit GEMM, experimental Pascal sm_61.
// Each GROUP=64 output channels has its own independently sampled active
// K/32 word positions. The remaining signed weights are exactly zero.
// Outputs verified against full dense cuBLAS GEMMEx INT8.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>
#include <string>
#include <numeric>
#define CU(x) do{cudaError_t e=(x);if(e!=cudaSuccess){fprintf(stderr,"CUDA line %d: %s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
#define BL(x) do{cublasStatus_t e=(x);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"BLAS line %d: %d\n",__LINE__,int(e));exit(3);}}while(0)


// GPU input pack: each warp converts eight-bit +/-1 activation values into
// one 32-bit packed sign word using warp ballot; 8 warps/block, persistent loops.
__global__ void pack_binary_activations(const int8_t* __restrict__ A,
                                         uint32_t* __restrict__ packed, int K) {
    const int row=blockIdx.x;
    const int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    const int words=K/32;
    for(int w=warp;w<words;w+=8){
        const int8_t v=A[row*K+w*32+lane];
        const uint32_t sign=__ballot_sync(0xffffffffu,v<0);
        if(lane==0)packed[row*words+w]=sign;
    }
}
template<int SP>
__global__ void binary_bs_gemm(
  const uint32_t* __restrict__ A,
  const uint32_t* __restrict__ Bcompressed,
  const int32_t* __restrict__ activeKwords,
  int16_t* __restrict__ C,int N,int K){
    __shared__ uint32_t as[32][8], bs[64][8];
    const int tx=threadIdx.x,ty=threadIdx.y,tid=ty*16+tx;
    const int rowBase=blockIdx.y*32, colBase=blockIdx.x*64;
    const int words=K/32;
    int p00=0,p01=0,p02=0,p03=0,p10=0,p11=0,p12=0,p13=0;
    #pragma unroll
    for(int base=0;base<SP;base+=8){
        const int ix=activeKwords[blockIdx.x*SP+base+(tid%8)];
        as[tid/8][tid%8]=A[(rowBase+tid/8)*words+ix];
        bs[tid/8][tid%8]=Bcompressed[(colBase+tid/8)*SP+base+(tid%8)];
        const int z=tid+256;
        bs[z/8][z%8]=Bcompressed[(colBase+z/8)*SP+base+(z%8)];
        __syncthreads();
        #pragma unroll
        for(int q=0;q<8;++q){
            const uint32_t av0=as[ty][q],av1=as[ty+16][q];
            const uint32_t b0=bs[tx][q],b1=bs[tx+16][q],b2=bs[tx+32][q],b3=bs[tx+48][q];
            p00+=__popc(av0^b0);p01+=__popc(av0^b1);
            p02+=__popc(av0^b2);p03+=__popc(av0^b3);
            p10+=__popc(av1^b0);p11+=__popc(av1^b1);
            p12+=__popc(av1^b2);p13+=__popc(av1^b3);
        }
        __syncthreads();
    }
    const int row0=rowBase+ty,row1=row0+16,col=colBase+tx;
    const int baseCount=SP*32;
    C[row0*N+col]=baseCount-2*p00;
    C[row0*N+col+16]=baseCount-2*p01;
    C[row0*N+col+32]=baseCount-2*p02;
    C[row0*N+col+48]=baseCount-2*p03;
    C[row1*N+col]=baseCount-2*p10;
    C[row1*N+col+16]=baseCount-2*p11;
    C[row1*N+col+32]=baseCount-2*p12;
    C[row1*N+col+48]=baseCount-2*p13;
}
template<class F> float timed(F f,int iters) {
    f();f();CU(cudaDeviceSynchronize());
    cudaEvent_t start,end; CU(cudaEventCreate(&start));CU(cudaEventCreate(&end));
    CU(cudaEventRecord(start));
    for(int i=0;i<iters;++i)f();
    CU(cudaEventRecord(end));CU(cudaEventSynchronize(end));
    float ms=0;CU(cudaEventElapsedTime(&ms,start,end));
    CU(cudaEventDestroy(start));CU(cudaEventDestroy(end));return ms/iters;
}
template<int SP> float perform(int N,int K,int iters,int8_t *ad,int8_t*bd,
                   uint32_t *pad,int32_t *answer,
                   const std::vector<int8_t>& A,
                   const std::vector<int8_t>& B,
                   const std::vector<int32_t>& cref){
    const int words=K/32;
    const int groups=N/64;
    std::mt19937 rng(210426+SP);
    std::uniform_int_distribution<int> bit(0,1);
    std::vector<int32_t> indices(size_t(groups)*SP);
    std::vector<uint32_t> packedB(size_t(N)*SP);
    std::vector<int8_t> sparseB(size_t(N)*K,0);
    for(int g=0;g<groups;++g){
        std::vector<int> chosen(words);
        std::iota(chosen.begin(),chosen.end(),0);
        std::shuffle(chosen.begin(),chosen.end(),rng);
        chosen.resize(SP);
        std::sort(chosen.begin(),chosen.end());
        for(int w=0;w<SP;++w)indices[g*SP+w]=chosen[w];
        for(int ch=0;ch<64;++ch){
            uint32_t* dest=packedB.data()+(g*64+ch)*SP;
            for(int i=0;i<SP;++i){
                uint32_t sign=0;
                for(int lane=0;lane<32;++lane){
                    const int idx=chosen[i]*32+lane;
                    const int8_t value=bit(rng)?-1:1;
                    sparseB[size_t(g*64+ch)*K+idx]=value;
                    if(value<0)sign|=(1u<<lane);
                }
                dest[i]=sign;
            }
        }
    }
    // Validate Bcompressed and active positions versus full dense reference.
    CU(cudaMemcpy(bd,sparseB.data(),size_t(N)*K,cudaMemcpyHostToDevice));
    // Reference with B transposed in row-major: output A * sparseB^T.
    int *indsGPU=nullptr; int16_t *dC=nullptr;
    uint32_t* bGPU=nullptr;
    CU(cudaMalloc(reinterpret_cast<void**>(&indsGPU),indices.size()*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&bGPU),packedB.size()*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&dC),size_t(N)*N*2));
    CU(cudaMemcpy(indsGPU,indices.data(),indices.size()*4,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(bGPU,packedB.data(),packedB.size()*4,cudaMemcpyHostToDevice));
    cublasHandle_t blas; BL(cublasCreate(&blas));
    const int alpha=1,beta=0;
    auto reference=[&]{
      BL(cublasGemmEx(blas,CUBLAS_OP_T,CUBLAS_OP_N,N,N,K,
         &alpha,bd,CUDA_R_8I,K,ad,CUDA_R_8I,K,&beta,answer,CUDA_R_32I,N,
         CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
    };
    dim3 block(16,16),grid(N/64,N/32);
    auto kernel=[&]{
        binary_bs_gemm<SP><<<grid,block>>>(pad,bGPU,indsGPU,dC,N,K);
        CU(cudaGetLastError());
    };
    reference();kernel();CU(cudaDeviceSynchronize());
    std::vector<int16_t> actual(size_t(N)*N);std::vector<int32_t> expected(size_t(N)*N);
    CU(cudaMemcpy(actual.data(),dC,actual.size()*2,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(expected.data(),answer,expected.size()*4,cudaMemcpyDeviceToHost));
    int wrong=0;
    for(size_t i=0;i<actual.size();++i) if(actual[i]!=expected[i])++wrong;
    int cpuWrong=0;
    for(int s=0;s<48;++s){
        int r=(s*73+5)%N,c=(s*197+11)%N;
        int referenceCPU=0;
        for(int j=0;j<K;++j)
            referenceCPU+=int(A[size_t(r)*K+j])*int(sparseB[size_t(c)*K+j]);
        if(referenceCPU!=actual[size_t(r)*N+c])++cpuWrong;
    }
    fprintf(stderr,"INT16_OUTPUT N=%d K=%d active=%d/%d (%g%%) validation=%s all_diff=%d cpu_samples=%d\n",
        N,K,SP,words,100.f*SP/words,(wrong||cpuWrong?"FAIL":"PASS"),wrong,cpuWrong);
    if(wrong||cpuWrong)exit(5);
    const float ms=timed(kernel,iters);
    const float refms=timed(reference,iters);
    const double denseOps=2.0*double(N)*N*K;
    printf("%d,%d,%d,%.6f,%.4f,%.6f,%.4f,PASS\n",
        N,K,SP,ms,denseOps/(ms*1e9),refms,refms/ms);
    BL(cublasDestroy(blas));
    CU(cudaFree(indsGPU));CU(cudaFree(bGPU));CU(cudaFree(dC));
    return ms;
}
int main(int argc,char** argv){
    int N=2048,K=2048,iters=100;
    for(int i=1;i<argc;++i){
      if(i+1>=argc) return 1;
      std::string flag=argv[i]; int value=atoi(argv[++i]);
      if(flag=="--n")N=value; else if(flag=="--k")K=value;
      else if(flag=="--iters")iters=value; else return 1;
    }
    if(N<512||N>4096||N%64||K<512||K>4096||K%256||iters<1||iters>1000)return 1;
    const int words=K/32;
    std::vector<int8_t> A(size_t(N)*K);
    std::mt19937 rng(20261028);std::uniform_int_distribution<int> bit(0,1);
    std::vector<uint32_t> Apacked(size_t(N)*words);
    for(int r=0;r<N;++r){
        for(int w=0;w<words;++w){
            uint32_t mask=0;
            for(int b=0;b<32;++b){
                int8_t value=bit(rng)?-1:1;
                A[size_t(r)*K+w*32+b]=value;
                if(value<0)mask|=1u<<b;
            }
            Apacked[size_t(r)*words+w]=mask;
        }
    }
    int8_t *ad=nullptr,*bd=nullptr;int32_t* reference=nullptr;uint32_t* pad=nullptr;
    CU(cudaMalloc(reinterpret_cast<void**>(&ad),A.size()));
    CU(cudaMalloc(reinterpret_cast<void**>(&bd),size_t(N)*K));
    CU(cudaMalloc(reinterpret_cast<void**>(&reference),size_t(N)*N*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&pad),Apacked.size()*4));
    CU(cudaMemcpy(ad,A.data(),A.size(),cudaMemcpyHostToDevice));
    auto packGPU=[&]{
        pack_binary_activations<<<N,256>>>(ad,pad,K);
        CU(cudaGetLastError());
    };
    packGPU();CU(cudaDeviceSynchronize());
    std::vector<uint32_t> packedGPU(Apacked.size());
    CU(cudaMemcpy(packedGPU.data(),pad,packedGPU.size()*4,cudaMemcpyDeviceToHost));
    if(packedGPU!=Apacked){fprintf(stderr,"PACK VALIDATION FAILED\n");return 7;}
    const float packms=timed(packGPU,iters);
    fprintf(stderr,"GPU_PACK N=%d K=%d ms=%.6f correctness=PASS\n",N,K,packms);
    std::vector<int8_t> unused; std::vector<int32_t> unusedRef;
    printf("n,k,active_words_per_group,ms,dense_equivalent_tops,cublas_dense_ms,speedup_vs_dense,validation\n");
    if(words>=8)perform<8>(N,K,iters,ad,bd,pad,reference,A,unused,unusedRef);
    if(words>=16)perform<16>(N,K,iters,ad,bd,pad,reference,A,unused,unusedRef);
    if(words>=24)perform<24>(N,K,iters,ad,bd,pad,reference,A,unused,unusedRef);
    if(words>=32)perform<32>(N,K,iters,ad,bd,pad,reference,A,unused,unusedRef);
    CU(cudaFree(ad));CU(cudaFree(bd));CU(cudaFree(pad));CU(cudaFree(reference));
    return 0;
}
