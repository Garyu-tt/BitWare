from pathlib import Path
base=Path(__file__).resolve().parent
p=base/"sparse_gemm.cu"
s=p.read_text()
anchor="template<int SP>\n__global__ void binary_bs_gemm("
assert s.count(anchor)==1
new=r"""
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
"""
s=s.replace(anchor,new+anchor,1)
old='    CU(cudaMemcpy(pad,Apacked.data(),Apacked.size()*4,cudaMemcpyHostToDevice));'
new=r"""    auto packGPU=[&]{
        pack_binary_activations<<<N,256>>>(ad,pad,K);
        CU(cudaGetLastError());
    };
    packGPU();CU(cudaDeviceSynchronize());
    std::vector<uint32_t> packedGPU(Apacked.size());
    CU(cudaMemcpy(packedGPU.data(),pad,packedGPU.size()*4,cudaMemcpyDeviceToHost));
    if(packedGPU!=Apacked){fprintf(stderr,"PACK VALIDATION FAILED\n");return 7;}
    const float packms=timed(packGPU,iters);
    fprintf(stderr,"GPU_PACK N=%d K=%d ms=%.6f correctness=PASS\n",N,K,packms);"""
assert s.count(old)==1
s=s.replace(old,new,1)
old='    printf("n,k,active_words_per_group,ms,dense_equivalent_tops,cublas_dense_ms,speedup_vs_dense,validation\\n");'
new='    printf("n,k,active_words_per_group,ms,dense_equivalent_tops,cublas_dense_ms,speedup_vs_dense,validation\\n");'
s=s.replace(old,new)
dst=base/"sparse_gemm_pack.cu"
dst.write_text(s)
print(dst,len(s))
