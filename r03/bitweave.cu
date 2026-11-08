// Mikoshi BitWeave R01: Pascal sm_61 binary/ternary GEMM experiment
// Both operands {-1,+1} or {-1,0,+1}, signed Int8 cuBLAS reference.
// Representation: row-major [rows, K/32] uint32 words, sign=1 for -1, nonzero=1 for +/-1.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#define CU(expr) do{cudaError_t e=(expr);if(e!=cudaSuccess){fprintf(stderr,"CUDA:%d:%s\n",__LINE__,cudaGetErrorString(e));exit(2);}}while(0)
#define BL(expr) do{cublasStatus_t e=(expr);if(e!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"CUBLAS:%d:%d\n",__LINE__,int(e));exit(3);}}while(0)

__global__ void pack_gpu(const int8_t* __restrict__ src,
                         uint32_t* __restrict__ signs,
                         uint32_t* __restrict__ nonzero,
                         int rows,int K) {
    const int lane=threadIdx.x&31;
    const int warp=threadIdx.x>>5;
    const int word=blockIdx.x*8+warp;
    const int row=blockIdx.y;
    if(row>=rows || word>=K/32) return;
    const int8_t v=src[row*K+word*32+lane];
    const uint32_t s=__ballot_sync(0xffffffffu, v<0);
    const uint32_t nz=__ballot_sync(0xffffffffu, v!=0);
    if(lane==0) {
        const int at=row*(K/32)+word;
        signs[at]=s;
        if(nonzero) nonzero[at]=nz;
    }
}

__global__ void bin_gemm_32x64(const uint32_t* __restrict__ A,
                               const uint32_t* __restrict__ B,
                               int32_t* __restrict__ C,int M,int N,int K) {
    __shared__ uint32_t as[32][8],bs[64][8];
    const int tx=threadIdx.x,ty=threadIdx.y,tid=ty*16+tx;
    const int aRow=blockIdx.y*32;
    const int bRow=blockIdx.x*64;
    const int words=K/32;
    const int row0=aRow+ty,row1=row0+16,col=bRow+tx;
    int p00=0,p01=0,p02=0,p03=0,p10=0,p11=0,p12=0,p13=0;
    for(int base=0;base<words;base+=8) {
        as[tid/8][tid%8]=A[(aRow+tid/8)*words+base+tid%8];
        bs[tid/8][tid%8]=B[(bRow+tid/8)*words+base+tid%8];
        const int z=tid+256;
        bs[z/8][z%8]=B[(bRow+z/8)*words+base+z%8];
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;++i) {
            uint32_t v0=as[ty][i],v1=as[ty+16][i];
            uint32_t b0=bs[tx][i],b1=bs[tx+16][i],b2=bs[tx+32][i],b3=bs[tx+48][i];
            p00+=__popc(v0^b0); p01+=__popc(v0^b1);
            p02+=__popc(v0^b2); p03+=__popc(v0^b3);
            p10+=__popc(v1^b0); p11+=__popc(v1^b1);
            p12+=__popc(v1^b2); p13+=__popc(v1^b3);
        }
        __syncthreads();
    }
    C[row0*N+col]=K-2*p00; C[row0*N+col+16]=K-2*p01;
    C[row0*N+col+32]=K-2*p02; C[row0*N+col+48]=K-2*p03;
    C[row1*N+col]=K-2*p10; C[row1*N+col+16]=K-2*p11;
    C[row1*N+col+32]=K-2*p12; C[row1*N+col+48]=K-2*p13;
}

__device__ __forceinline__ int ternary_score(uint32_t az,uint32_t as,
                                              uint32_t bz,uint32_t bs) {
    const uint32_t common=az&bz;
    return __popc(common)-2*__popc(common & (as^bs));
}

__global__ void ter_gemm_32x64(const uint32_t* __restrict__ As,
                               const uint32_t* __restrict__ Az,
                               const uint32_t* __restrict__ Bs,
                               const uint32_t* __restrict__ Bz,
                               int32_t* __restrict__ C,int M,int N,int K) {
    __shared__ uint32_t sa[32][8],za[32][8],sb[64][8],zb[64][8];
    const int tx=threadIdx.x,ty=threadIdx.y,tid=ty*16+tx;
    const int aRow=blockIdx.y*32,bRow=blockIdx.x*64,words=K/32;
    const int row0=aRow+ty,row1=row0+16,col=bRow+tx;
    int p00=0,p01=0,p02=0,p03=0,p10=0,p11=0,p12=0,p13=0;
    for(int base=0;base<words;base+=8) {
        const int ai=(aRow+tid/8)*words+base+tid%8;
        const int bi=(bRow+tid/8)*words+base+tid%8;
        const int z=tid+256;
        const int bj=(bRow+z/8)*words+base+z%8;
        sa[tid/8][tid%8]=As[ai]; za[tid/8][tid%8]=Az[ai];
        sb[tid/8][tid%8]=Bs[bi]; zb[tid/8][tid%8]=Bz[bi];
        sb[z/8][z%8]=Bs[bj]; zb[z/8][z%8]=Bz[bj];
        __syncthreads();
        #pragma unroll
        for(int i=0;i<8;++i) {
            uint32_t a0=sa[ty][i], z0=za[ty][i],a1=sa[ty+16][i],z1=za[ty+16][i];
            uint32_t b0=sb[tx][i], v0=zb[tx][i],b1=sb[tx+16][i],v1=zb[tx+16][i];
            uint32_t b2=sb[tx+32][i],v2=zb[tx+32][i],b3=sb[tx+48][i],v3=zb[tx+48][i];
            p00+=ternary_score(z0,a0,v0,b0); p01+=ternary_score(z0,a0,v1,b1);
            p02+=ternary_score(z0,a0,v2,b2); p03+=ternary_score(z0,a0,v3,b3);
            p10+=ternary_score(z1,a1,v0,b0); p11+=ternary_score(z1,a1,v1,b1);
            p12+=ternary_score(z1,a1,v2,b2); p13+=ternary_score(z1,a1,v3,b3);
        }
        __syncthreads();
    }
    C[row0*N+col]=p00; C[row0*N+col+16]=p01;
    C[row0*N+col+32]=p02; C[row0*N+col+48]=p03;
    C[row1*N+col]=p10; C[row1*N+col+16]=p11;
    C[row1*N+col+32]=p12; C[row1*N+col+48]=p13;
}

template<class FN> float duration_ms(FN f,int iters) {
    f();f(); CU(cudaDeviceSynchronize());
    cudaEvent_t s,t;CU(cudaEventCreate(&s));CU(cudaEventCreate(&t));
    CU(cudaEventRecord(s));
    for(int i=0;i<iters;++i)f();
    CU(cudaEventRecord(t));CU(cudaEventSynchronize(t));
    float ms=0;CU(cudaEventElapsedTime(&ms,s,t));
    CU(cudaEventDestroy(s));CU(cudaEventDestroy(t));
    return ms/iters;
}

int main(int argc,char**argv) {
    int N=1024,device=0,iters=20,mode=0;
    for(int i=1;i<argc;i++) {
        if(i+1>=argc) {fprintf(stderr,"missing arg %s\n",argv[i]);return 1;}
        const std::string key=argv[i];
        int v=std::atoi(argv[++i]);
        if(key=="--n")N=v;
        else if(key=="--device")device=v;
        else if(key=="--iters")iters=v;
        else if(key=="--ternary")mode=v;
        else {fprintf(stderr,"bad arg %s\n",key.c_str());return 1;}
    }
    if(N<256||N>4096||N%256!=0||iters<1||iters>200||mode<0||mode>1) {
        fprintf(stderr,"--n 256..4096 divisible by 256; --iters 1..200; --ternary 0|1\n");return 1;
    }
    CU(cudaSetDevice(device));
    cudaDeviceProp prop{};CU(cudaGetDeviceProperties(&prop,device));
    if(prop.major!=6||prop.minor!=1) {
        fprintf(stderr,"WARNING: R01 tuned for Pascal sm_61, found sm_%d%d\n",prop.major,prop.minor);
    }
    const size_t elems=size_t(N)*N, packed=elems/32;
    std::vector<int8_t> a(elems),b(elems);
    std::mt19937 gen(0xC0FFEE);
    std::uniform_int_distribution<int> vdist(mode?-1:0,mode?1:1);
    std::uniform_int_distribution<int> bdist(0,1);
    for(size_t i=0;i<elems;++i){
        a[i]= mode?int8_t(vdist(gen)):int8_t(bdist(gen)*2-1);
        b[i]= mode?int8_t(vdist(gen)):int8_t(bdist(gen)*2-1);
    }
    int8_t *ad=nullptr,*bd=nullptr;
    uint32_t *as=nullptr,*az=nullptr,*bs=nullptr,*bz=nullptr;
    int32_t *c=nullptr,*ref=nullptr;
    CU(cudaMalloc(reinterpret_cast<void**>(&ad),elems));
    CU(cudaMalloc(reinterpret_cast<void**>(&bd),elems));
    CU(cudaMalloc(reinterpret_cast<void**>(&as),packed*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&bs),packed*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&az),packed*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&bz),packed*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&c),elems*4));
    CU(cudaMalloc(reinterpret_cast<void**>(&ref),elems*4));
    CU(cudaMemcpy(ad,a.data(),elems,cudaMemcpyHostToDevice));
    CU(cudaMemcpy(bd,b.data(),elems,cudaMemcpyHostToDevice));
    auto packa=[&]{pack_gpu<<<dim3((N/32+7)/8,N),256>>>(ad,as,mode?az:nullptr,N,N);CU(cudaGetLastError());};
    auto packb=[&]{pack_gpu<<<dim3((N/32+7)/8,N),256>>>(bd,bs,mode?bz:nullptr,N,N);CU(cudaGetLastError());};
    packa();packb();
    const dim3 block(16,16),grid(N/64,N/32);
    auto bitgemm=[&]{
        if(mode)ter_gemm_32x64<<<grid,block>>>(as,az,bs,bz,c,N,N,N);
        else bin_gemm_32x64<<<grid,block>>>(as,bs,c,N,N,N);
        CU(cudaGetLastError());
    };
    cublasHandle_t handle;BL(cublasCreate(&handle));
    const int alpha=1,beta=0;
    auto gemmex=[&]{
        BL(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,N,N,N,
           &alpha,bd,CUDA_R_8I,N,ad,CUDA_R_8I,N,
           &beta,ref,CUDA_R_32I,N,CUBLAS_COMPUTE_32I,CUBLAS_GEMM_DEFAULT));
    };
    bitgemm();gemmex();CU(cudaDeviceSynchronize());
    std::vector<int32_t> h(elems),r(elems);
    CU(cudaMemcpy(h.data(),c,elems*4,cudaMemcpyDeviceToHost));
    CU(cudaMemcpy(r.data(),ref,elems*4,cudaMemcpyDeviceToHost));
    size_t mismatch=0;int32_t max_diff=0;
    for(size_t i=0;i<elems;++i){
        int32_t diff=std::abs(h[i]-r[i]);
        if(diff){++mismatch;max_diff=std::max(max_diff,diff);}
    }
    // Full output verification against independent cuBLAS GEMMEx INT8.
    const bool pass=mismatch==0;
    fprintf(stderr,"VERIFY %s N=%d MISMATCH=%zu MAX_DIFF=%d\n",
            pass?"PASS":"FAIL",N,mismatch,max_diff);
    if(!pass)return 5;
    float bitms=duration_ms(bitgemm,iters);
    float intms=duration_ms(gemmex,iters);
    float packams=duration_ms(packa,iters);
    float packbms=duration_ms(packb,iters);
    const double logical_ops=2.0*double(N)*N*N;
    printf("type,n,method,ms,logical_tops,check\n");
    auto row=[&](const char* meth,float ms) {
        printf("%s,%d,%s,%.6f,%.6f,PASS\n",mode?"ternary":"binary",N,meth,ms,
               logical_ops/ms/1.e9);
    };
    row(mode?"ternary_popc":"binary_xor_popc",bitms);
    row("cublas_int8",intms);
    row("pack_A_dynamic",packams);
    row("pack_B_static",packbms);
    fprintf(stderr,"SPEEDUP bitpacked/cuBLAS: %.3fx, effective with A repack: %.3fx\n",
            intms/bitms,intms/(bitms+packams));
    BL(cublasDestroy(handle));
    CU(cudaFree(ad));CU(cudaFree(bd));
    CU(cudaFree(as));CU(cudaFree(az));CU(cudaFree(bs));CU(cudaFree(bz));
    CU(cudaFree(c));CU(cudaFree(ref));
    return 0;
}
