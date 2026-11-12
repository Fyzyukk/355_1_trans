# B in aiter shuffle_weight(w, layout=(16,16)) layout, PR6696-style chunks.
import sys,shutil,re
src,name=sys.argv[1],sys.argv[2]; d=f'/tmp/exp/{name}'
shutil.rmtree(d,ignore_errors=True); shutil.copytree(src,d,ignore=shutil.ignore_patterns('build','build.log','README.md'))
f=f'{d}/gemm_a4w4_mxfp4_4wave_kernel_template.hpp'; s=open(f).read()
def rep(a,b):
    global s; assert s.count(a)==1,(s.count(a),a[:80]); s=s.replace(a,b)
# make_layout_gb
i=s.index("__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {"); j=s.index("\n}\n",i)+3
s=s[:i]+"""__device__ inline auto make_layout_gb(int lane_id, int wave_id_m, int wave_id_n, int stride_b) {
    // bs: B in aiter shuffle_weight(w, layout=(16, 16)) layout: byte (n, kb) at (n/16)*16*Kb + (kb/32)*512 +
    // ((kb/16)%2)*256 + (n%16)*16 + kb%16, Kb = stride_b.  One 16-row block x one 128 B K tile = 2048 contiguous
    // bytes; its K halves (kk = 0, 1) are 1 KB each.  Wave (P0, P1) issue i = ih * 2 + il copies 16-row block
    // (P0 * 2 + P1) * 2 + ih, K half il, as one contiguous 1 KB, into LDS chunk (P0 * 2 + P1) * 4 + i (as PR 6696):
    // chunk = 2 * block + kk holds [g 4][n 16][16 B].
    static_assert(T::T_M == 2 && T::T_N == 2 && T::W_N == 16 && T::B_K_BYTES == 128);
    constexpr auto shape = opus::make_tuple(
        opus::number<2>{},                                             // p P0 (wave_m)
        opus::number<2>{},                                             // p P1 (wave_n)
        opus::number<2>{},                                             // y ih: 16-row block
        opus::number<2>{},                                             // y il: K half (1 KB)
        opus::number<T::WARP_SIZE>{},                                  // p lane
        opus::number<T::VEC_B>{});                                     // y 16 bytes
    constexpr auto dim = opus::make_tuple(
        opus::make_tuple(opus::p_dim{}, opus::p_dim{}, opus::y_dim{}),
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        shape,
        opus::unfold_x_stride(dim, shape, opus::tuple{T::W_N * stride_b, 1_I}),
        opus::unfold_p_coord(dim, opus::tuple{wave_id_m, wave_id_n, lane_id}));
}
"""+s[j:]
# make_layout_rb
i=s.index("__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {"); j=s.index("\n}\n",i)+3
s=s[:i]+"""__device__ inline auto make_layout_rb(int lane_id, int wave_id_n) {
    // bs: repeat e of wave_n reads 16-row block 2e + wave_n, chunk (2e + wave_n) * 2 + kk (kk -> + kk * chunk stride
    // in load_b): lane reads chunk + 16 * lane = row lane % 16, K bytes [64 kk + 16 g, +16), g = lane / 16.  Each read
    // is one contiguous 1 KB: bank-conflict free.
    static_assert(T::W_N * T::W_K_BYTES / T::WARP_SIZE / T::VEC_B == 1 && T::T_N == 2);
    constexpr int P = T::smem_linear_wave_b + T::smem_padding;
    constexpr auto rb_block_shape = opus::make_tuple(
        opus::number<T::E_N>{},                                        // 4  y N repeat
        opus::number<T::T_N>{},                                        // 2  p wave_n
        opus::number<T::WARP_SIZE>{},                                  // 64 p lane
        opus::number<T::VEC_B>{});                                     // 16 y bytes
    constexpr auto rb_block_dim = opus::make_tuple(
        opus::make_tuple(opus::y_dim{}, opus::p_dim{}),
        opus::make_tuple(opus::p_dim{}, opus::y_dim{}));
    return opus::make_layout<T::VEC_B>(
        rb_block_shape,
        opus::unfold_x_stride(rb_block_dim, rb_block_shape, opus::tuple{opus::number<2 * P>{}, 1_I}),
        opus::unfold_p_coord(rb_block_dim, opus::tuple{wave_id_n, lane_id}));
}
"""+s[j:]
rep("    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * T::B_K_BYTES; };",
    "    auto gb_offset = [&](int half_tile_n, int tile_k) { return half_tile_n * T::HALF_B_N * kargs.stride_b + tile_k * (T::B_K_BYTES * T::W_N); };   // bs: K tile = 2 KB per 16-row block")
rep("                __builtin_bit_cast(fp4x32_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * T::W_K_BYTES));",
    "                __builtin_bit_cast(fp4x32_t, load<T::VEC_B>(sb_, rb_offs[e.value] + sb_offset(0, half) + kk.value * (T::smem_linear_wave_b + T::smem_padding)));   // bs: kk = next chunk")
open(f,'w').write(s)
# host: shuffle B for the kernel, reference keeps raw B
h=f'{d}/gemm_a4w4_mxfp4_4wave_host.cc'; s=open(h).read()
open('/tmp/exp/host_b_lines.txt','w').write('\n'.join(l for l in s.split('\n') if 'd_b' in l or 'h_b' in l or 'b_cnt' in l))
open(h,'w').write(s)
print(name,'kernel ok')
