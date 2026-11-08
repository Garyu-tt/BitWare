from pathlib import Path
base=Path(__file__).resolve().parent
src=(base/"bitweave_gemv.cu").read_text()
new=r'''
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
'''
anchor='template<class F> float bench('
assert src.count(anchor)==1
src=src.replace(anchor,new+'\n'+anchor,1)
def replace(old,new):
 global src
 assert src.count(old)==1,(old[:75],src.count(old))
 src=src.replace(old,new,1)
replace('mode<0||mode>1','mode<0||mode>3')
replace('std::uniform_int_distribution<int> bits(0,1),trits(-1,1);',
        'std::uniform_int_distribution<int> bits(0,1),trits(-1,1),activ8(-64,63);')
replace('for(auto &v:a)v=int8_t(bits(rng)*2-1);',
        'for(auto &v:a)v=int8_t(mode>=2?activ8(rng):(bits(rng)*2-1));')
replace('for(auto &v:w)v=int8_t(mode?trits(rng):bits(rng)*2-1);',
        'for(auto &v:w)v=int8_t((mode==1||mode==2)?trits(rng):(bits(rng)*2-1));')
replace('uint32_t *ap=nullptr,*wp=nullptr,*wnz=nullptr;',
        'uint32_t *ap=nullptr,*ap8=nullptr,*wp=nullptr,*wnz=nullptr;')
replace('CU(cudaMalloc(reinterpret_cast<void**>(&ap),size_t(K)/8));',
'''CU(cudaMalloc(reinterpret_cast<void**>(&ap),size_t(K)/8));
    CU(cudaMalloc(reinterpret_cast<void**>(&ap8),size_t(K)));''')
replace('auto packa=[&]{pack<<<dim3((K/32+7)/8,1),256>>>(ad,ap,nullptr,1,K);CU(cudaGetLastError());};',
'''auto packa=[&]{
        if(mode>=2)
            pack_activation_i8<<<(K/32+7)/8,256>>>(ad,ap8,K);
        else
            pack<<<dim3((K/32+7)/8,1),256>>>(ad,ap,nullptr,1,K);
        CU(cudaGetLastError());
    };''')
replace('mode?wnz:nullptr,N,K', '(mode==1||mode==2)?wnz:nullptr,N,K')
replace('if(mode)warp_gemv<true><<<N/8,256>>>(ap,wp,wnz,out,N,K);',
'''if(mode==2)warp_gemv_mixed_i8<true><<<N/8,256>>>(ap8,wp,wnz,out,N,K);
        else if(mode==3)warp_gemv_mixed_i8<false><<<N/8,256>>>(ap8,wp,wnz,out,N,K);
        else if(mode==1)warp_gemv<true><<<N/8,256>>>(ap,wp,wnz,out,N,K);''')
replace('mode?"ternary_weight":"binary_weight"',
        'mode==2?"ternary_weight_i8_activation":mode==3?"binary_weight_i8_activation":mode?"ternary_weight":"binary_weight"')
replace('mode?"ternary":"binary",N,K,name,ms',
        'mode==2?"ternaryXint8":mode==3?"binaryXint8":mode?"ternary":"binary",N,K,name,ms')
replace('CU(cudaFree(wnz));CU(cudaFree(out));', 'CU(cudaFree(wnz));CU(cudaFree(ap8));CU(cudaFree(out));')
dst=base/"bitweave_mixed.cu"
dst.write_text(src)
print(f"{dst}: {len(src)} bytes")
