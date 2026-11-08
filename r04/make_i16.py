from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/"sparse_gemm_pack.cu").read_text()
changes=[
 ("int32_t* __restrict__ C,int N,int K)", "int16_t* __restrict__ C,int N,int K)"),
 ("int *indsGPU=nullptr,*dC=nullptr;", "int *indsGPU=nullptr; int16_t *dC=nullptr;"),
 ("CU(cudaMalloc(reinterpret_cast<void**>(&dC),size_t(N)*N*4));","CU(cudaMalloc(reinterpret_cast<void**>(&dC),size_t(N)*N*2));"),
 ("std::vector<int32_t> actual(size_t(N)*N),expected(size_t(N)*N);","std::vector<int16_t> actual(size_t(N)*N);std::vector<int32_t> expected(size_t(N)*N);"),
 ("CU(cudaMemcpy(actual.data(),dC,actual.size()*4,cudaMemcpyDeviceToHost));",
  "CU(cudaMemcpy(actual.data(),dC,actual.size()*2,cudaMemcpyDeviceToHost));")
]
for a,b in changes:
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b,1)
s=s.replace('int wrong=0;','int wrong=0;')
s=s.replace('"N=%d K=%d active=%d/%d (%g%%) validation=%s all_diff=%d cpu_samples=%d\\n"',
            '"INT16_OUTPUT N=%d K=%d active=%d/%d (%g%%) validation=%s all_diff=%d cpu_samples=%d\\n"')
assert "int16_t* __restrict__ C" in s
out=root/"sparse_gemm_pack_i16.cu"
out.write_text(s)
print(str(out),len(s))
