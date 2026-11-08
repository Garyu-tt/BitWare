from pathlib import Path
p=Path(__file__).resolve().parent
src=(p/"bitweave_events.cu").read_text()
lib=(p/"bitweave_mixed.cu").read_text()
start=lib.index("// One-time-per-token")
end=lib.index("template<class F> float bench")
kernels=lib[start:end]
anchor='template<class F>float benchmark'
assert src.count(anchor)==1
src=src.replace(anchor,kernels+'\n'+anchor)
old='    uint32_t *Ws=nullptr,*Wn=nullptr;'
new='    uint32_t *Ws=nullptr,*Wn=nullptr,*Aplanes=nullptr;'
assert src.count(old)==1;src=src.replace(old,new)
old='    CU(cudaMalloc(reinterpret_cast<void**>(&Wn),weightsCount/8));'
new=old+'\n    CU(cudaMalloc(reinterpret_cast<void**>(&Aplanes),size_t(K)));'
assert src.count(old)==1;src=src.replace(old,new)
old='    printf("n,k,event_count,delta_ms,cublas_full_ms,relative_speedup,validation\\n");'
new='    printf("n,k,event_count,delta_ms,full_bitweave_ms,cublas_full_ms,chosen,speedup_vs_bitweave,validation\\n");'
assert src.count(old)==1;src=src.replace(old,new)
old='        delta_kernel();gemm_new();CU(cudaDeviceSynchronize());'
new='''        auto dense_kernel=[&]{
            pack_activation_i8<<<(K/32+7)/8,256>>>(ANewd,Aplanes,K);
            CU(cudaGetLastError());
            warp_gemv_mixed_i8<true><<<N/8,256>>>(Aplanes,Ws,Wn,next,N,K);
            CU(cudaGetLastError());
        };
        delta_kernel();gemm_new();CU(cudaDeviceSynchronize());'''
assert src.count(old)==1;src=src.replace(old,new)
old='        float ms=benchmark(iters,delta_kernel);'
new='''        float ms=benchmark(iters,delta_kernel);
        // Dense output computed independently, using 8 bitplanes and ternary weight masks.
        dense_kernel();CU(cudaDeviceSynchronize());
        CU(cudaMemcpy(computed.data(),next,size_t(N)*4,cudaMemcpyDeviceToHost));
        size_t denseErrors=0;
        for(int row=0;row<N;++row)denseErrors+=(computed[row]!=ref[row]);
        if(denseErrors){fprintf(stderr,"DENSE FAIL count=%d errors=%zu\\n",count,denseErrors);return 5;}
        const float densems=benchmark(iters,dense_kernel);
        const char* chosen=(ms<densems?"event_delta":"full_bitweave");'''
assert src.count(old)==1;src=src.replace(old,new)
old='        printf("%d,%d,%d,%.6f,%.6f,%.4f,PASS\\n",N,K,count,ms,fullms,fullms/ms);'
new='''        printf("%d,%d,%d,%.6f,%.6f,%.6f,%s,%.4f,PASS\\n",
               N,K,count,ms,densems,fullms,chosen,densems/ms);'''
assert src.count(old)==1;src=src.replace(old,new)
old='    CU(cudaFree(Ws));CU(cudaFree(Wn));CU(cudaFree(prev));'
new='    CU(cudaFree(Ws));CU(cudaFree(Wn));CU(cudaFree(Aplanes));CU(cudaFree(prev));'
assert src.count(old)==1;src=src.replace(old,new)
out=p/"bitweave_adaptive.cu"
out.write_text(src)
print(f"Wrote {out}: {len(src)}")
