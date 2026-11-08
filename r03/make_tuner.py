from pathlib import Path
p=Path(__file__).resolve().parent / "bitweave.cu"
dst=p.with_name("bitweave_tune.cu")
src=p.read_text()
kernel=r'''
// Tuned candidate tiles: increase arithmetic-to-memory reuse by 8..32 outputs/thread.
template<int TM,int TN,int BW>
__global__ void bin_tile(const uint32_t* __restrict__ A,
                         const uint32_t* __restrict__ B,
                         int32_t* __restrict__ C,int M,int N,int K) {
    __shared__ uint32_t as[TM][BW],bs[TN][BW];
    const int tx=threadIdx.x,ty=threadIdx.y,tid=ty*16+tx;
    const int aRow=blockIdx.y*TM,bRow=blockIdx.x*TN,words=K/32;
    int acc[TM/16][TN/16];
    #pragma unroll
    for(int r=0;r<TM/16;++r)
        #pragma unroll
        for(int c=0;c<TN/16;++c)
            acc[r][c]=0;
    for(int base=0;base<words;base+=BW) {
        #pragma unroll
        for(int z=tid;z<TM*BW;z+=256)
            as[z/BW][z%BW]=A[(aRow+z/BW)*words+base+z%BW];
        #pragma unroll
        for(int z=tid;z<TN*BW;z+=256)
            bs[z/BW][z%BW]=B[(bRow+z/BW)*words+base+z%BW];
        __syncthreads();
        #pragma unroll
        for(int q=0;q<BW;++q) {
            #pragma unroll
            for(int r=0;r<TM/16;++r) {
                uint32_t av=as[ty+16*r][q];
                #pragma unroll
                for(int c=0;c<TN/16;++c) {
                    uint32_t bv=bs[tx+16*c][q];
                    acc[r][c]+=__popc(av^bv);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for(int r=0;r<TM/16;++r)
        #pragma unroll
        for(int c=0;c<TN/16;++c)
            C[(aRow+ty+16*r)*N+bRow+tx+16*c]=K-2*acc[r][c];
}
'''
anchor='__device__ __forceinline__ int ternary_score('
assert src.count(anchor)==1
src=src.replace(anchor,kernel+'\n'+anchor)
anchor2='    const float PLACEHOLDER_XXX=0;'
assert anchor2 not in src
anchor3='    float bitms=duration_ms(bitgemm,iters);'
add=r'''
    std::vector<std::pair<std::string,float>> tuned;
    if(!mode) {
        auto tune=[&](const char* name, auto run) {
            run();CU(cudaDeviceSynchronize());
            CU(cudaMemcpy(h.data(),c,elems*4,cudaMemcpyDeviceToHost));
            size_t wrong=0;
            for(size_t i=0;i<elems;++i)wrong+=(h[i]!=r[i]);
            fprintf(stderr,"TUNE_VERIFY %s %s mismatches=%zu\n",
                    name,wrong?"FAIL":"PASS",wrong);
            if(wrong) std::exit(5);
            const float ms=duration_ms(run,iters);
            tuned.emplace_back(name,ms);
        };
        tune("bin_64x64_k8",[&]{
            bin_tile<64,64,8><<<dim3(N/64,N/64),block>>>(as,bs,c,N,N,N);
            CU(cudaGetLastError());
        });
        tune("bin_32x128_k8",[&]{
            bin_tile<32,128,8><<<dim3(N/128,N/32),block>>>(as,bs,c,N,N,N);
            CU(cudaGetLastError());
        });
        tune("bin_64x128_k8",[&]{
            bin_tile<64,128,8><<<dim3(N/128,N/64),block>>>(as,bs,c,N,N,N);
            CU(cudaGetLastError());
        });
        if(N%512==0) {
            tune("bin_32x64_k16",[&]{
                bin_tile<32,64,16><<<dim3(N/64,N/32),block>>>(as,bs,c,N,N,N);
                CU(cudaGetLastError());
            });
            tune("bin_64x64_k16",[&]{
                bin_tile<64,64,16><<<dim3(N/64,N/64),block>>>(as,bs,c,N,N,N);
                CU(cudaGetLastError());
            });
            tune("bin_32x128_k16",[&]{
                bin_tile<32,128,16><<<dim3(N/128,N/32),block>>>(as,bs,c,N,N,N);
                CU(cudaGetLastError());
            });
        }
    }
'''
assert src.count(anchor3)==1
src=src.replace(anchor3,add+anchor3)
anchor4='    row("cublas_int8",intms);'
assert src.count(anchor4)==1
src=src.replace(anchor4,anchor4+'\n    for(const auto& [name,ms]:tuned)row(name.c_str(),ms);')
dst.write_text(src)
print(f"Wrote {dst}, {len(src)} bytes")
