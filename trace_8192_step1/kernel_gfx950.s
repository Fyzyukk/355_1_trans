	.amdgcn_target "amdgpu9.50-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,"axG",@progbits,_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,comdat
	.protected	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs ; -- Begin function _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
	.globl	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
	.p2align	8
	.type	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,@function
_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs: ; @_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dwordx4 s[20:23], s[0:1], 0x0
	s_load_dwordx8 s[4:11], s[0:1], 0x18
	s_load_dwordx2 s[34:35], s[0:1], 0x38
	s_load_dwordx8 s[12:19], s[0:1], 0x40
	v_readfirstlane_b32 s33, v0
	s_waitcnt lgkmcnt(0)
	s_bfe_u32 s7, s33, 0x10006
	s_mul_i32 s11, s11, s3
	s_lshr_b32 s53, s33, 7
	s_ashr_i32 s24, s11, 31
	s_add_u32 s20, s20, s11
	s_mul_i32 s11, s34, s3
	s_addc_u32 s21, s21, s24
	s_ashr_i32 s24, s11, 31
	s_add_u32 s22, s22, s11
	s_mul_i32 s11, s18, s3
	s_addc_u32 s23, s23, s24
	s_ashr_i32 s18, s11, 31
	s_add_u32 s11, s12, s11
	s_addc_u32 s12, s13, s18
	s_mul_i32 s13, s19, s3
	s_ashr_i32 s18, s13, 31
	s_add_u32 s13, s14, s13
	s_addc_u32 s19, s15, s18
	s_add_i32 s14, s5, 0xff
	s_ashr_i32 s15, s14, 31
	s_lshr_b32 s15, s15, 24
	s_add_i32 s14, s14, s15
	s_ashr_i32 s14, s14, 8
	s_abs_i32 s15, s14
	v_cvt_f32_u32_e32 v1, s15
	s_sub_i32 s25, 0, s15
	s_abs_i32 s24, s2
	s_xor_b32 s18, s2, s14
	v_rcp_f32_e32 v1, v1
	s_ashr_i32 s18, s18, 31
	v_lshlrev_b32_e32 v3, 4, v0
	v_and_b32_e32 v4, 0x70, v3
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_and_b32_e32 v2, 63, v0
	s_mul_i32 s55, s53, 0x840
	s_mov_b32 s38, -1
	v_readfirstlane_b32 s26, v1
	s_mul_i32 s25, s25, s26
	s_mul_hi_u32 s25, s26, s25
	s_add_i32 s26, s26, s25
	s_mul_hi_u32 s25, s24, s26
	s_mul_i32 s26, s25, s15
	s_sub_i32 s24, s24, s26
	s_add_i32 s26, s25, 1
	s_sub_i32 s27, s24, s15
	s_cmp_ge_u32 s24, s15
	s_cselect_b32 s25, s26, s25
	s_cselect_b32 s24, s27, s24
	s_add_i32 s26, s25, 1
	s_cmp_ge_u32 s24, s15
	s_cselect_b32 s15, s26, s25
	s_xor_b32 s15, s15, s18
	s_sub_i32 s15, s15, s18
	s_mul_i32 s14, s15, s14
	s_sub_i32 s2, s2, s14
	s_lshl_b32 s18, s15, 8
	s_lshl_b32 s2, s2, 8
	s_mul_i32 s14, s18, s16
	s_sub_i32 s4, s4, s18
	s_sub_i32 s5, s5, s2
	s_ashr_i32 s15, s14, 31
	s_add_u32 s28, s11, s14
	s_addc_u32 s11, s12, s15
	s_mul_i32 s30, s4, s16
	s_mul_i32 s4, s2, s17
	s_mul_i32 s14, s5, s17
	s_and_b32 s29, s11, 0xffff
	s_ashr_i32 s5, s4, 31
	s_add_u32 s12, s13, s4
	s_addc_u32 s4, s19, s5
	s_and_b32 s13, s4, 0xffff
	s_mul_i32 s4, s18, s8
	s_lshl_b32 s11, s53, 4
	s_ashr_i32 s5, s4, 31
	s_add_u32 s20, s20, s4
	s_addc_u32 s4, s21, s5
	v_lshrrev_b32_e32 v1, 2, v0
	s_and_b32 s21, s4, 0xffff
	s_mul_i32 s4, s2, s9
	v_and_b32_e32 v1, 14, v1
	s_ashr_i32 s5, s4, 31
	v_or_b32_e32 v3, s11, v1
	s_add_u32 s24, s22, s4
	v_or_b32_e32 v3, s7, v3
	s_addc_u32 s4, s23, s5
	s_and_b32 s25, s4, 0xffff
	v_mad_u64_u32 v[10:11], s[4:5], v3, s8, v[4:5]
	s_mul_i32 s4, s8, 0x60
	s_lshl_b32 s56, s7, 10
	v_add_u32_e32 v20, s4, v10
	s_lshl4_add_u32 s4, s7, s53
	v_add_u32_e32 v1, s4, v1
	v_mad_u64_u32 v[12:13], s[4:5], v1, s9, v[4:5]
	s_mul_i32 s4, s9, 0x60
	s_lshl_b32 s5, s7, 6
	v_add_u32_e32 v23, s4, v12
	s_and_b32 s4, s33, 0x80
	s_or_b32 s4, s5, s4
	v_or_b32_e32 v1, s4, v2
	s_lshl_b32 s4, s53, 11
	v_mul_lo_u32 v26, v1, s16
	s_or_b32 s56, s56, s4
	s_lshl_b32 s4, s7, 7
	s_lshl_b32 s5, s53, 6
	s_mul_i32 s16, s7, 0x420
	s_mov_b32 s15, 0x20000
	s_add_i32 s4, s4, s5
	s_add_i32 s16, s16, s55
	s_mov_b32 s36, s20
	s_mov_b32 s37, s21
	s_mov_b32 s39, s15
	v_or_b32_e32 v1, s4, v2
	s_lshl_b32 s4, s53, 10
	s_lshl_b32 s57, s7, 11
	s_mov_b32 m0, s16
	s_add_i32 s40, s16, 0x1080
	v_lshl_add_u32 v18, s8, 5, v10
	s_mul_i32 s54, s7, 0x840
	s_add_i32 s57, s57, s4
	buffer_load_dwordx4 v10, s[36:39], 0 offen lds
	s_mov_b32 m0, s40
	s_add_i32 s41, s16, 0x2100
	s_mul_i32 s4, s53, 0x420
	v_lshl_add_u32 v19, s8, 6, v10
	v_mul_lo_u32 v24, v1, s17
	buffer_load_dwordx4 v18, s[36:39], 0 offen lds
	s_mov_b32 m0, s41
	s_add_i32 s42, s16, 0x3180
	s_add_i32 s17, s54, s4
	buffer_load_dwordx4 v19, s[36:39], 0 offen lds
	s_mov_b32 m0, s42
	s_add_i32 s43, s17, 0x10800
	s_mov_b32 s26, s38
	s_mov_b32 s27, s15
	buffer_load_dwordx4 v20, s[36:39], 0 offen lds
	s_mov_b32 m0, s43
	s_add_i32 s44, s43, 0x1080
	v_lshl_add_u32 v21, s9, 5, v12
	buffer_load_dwordx4 v12, s[24:27], 0 offen lds
	s_mov_b32 m0, s44
	s_add_i32 s45, s43, 0x2100
	v_lshl_add_u32 v22, s9, 6, v12
	buffer_load_dwordx4 v21, s[24:27], 0 offen lds
	s_mov_b32 m0, s45
	s_add_i32 s46, s43, 0x3180
	buffer_load_dwordx4 v22, s[24:27], 0 offen lds
	s_mov_b32 m0, s46
	s_add_i32 s47, s16, 0x4200
	buffer_load_dwordx4 v23, s[24:27], 0 offen lds
	s_lshl_b32 s19, s8, 7
	s_mov_b32 m0, s47
	s_add_i32 s48, s16, 0x5280
	buffer_load_dwordx4 v10, s[36:39], s19 offen lds
	s_mov_b32 m0, s48
	s_add_i32 s49, s16, 0x6300
	buffer_load_dwordx4 v18, s[36:39], s19 offen lds
	s_mov_b32 m0, s49
	s_add_i32 s50, s16, 0x7380
	buffer_load_dwordx4 v19, s[36:39], s19 offen lds
	s_mov_b32 m0, s50
	s_lshl_b32 s33, s9, 7
	buffer_load_dwordx4 v20, s[36:39], s19 offen lds
	s_add_i32 s36, s43, 0x4200
	s_mov_b32 m0, s36
	s_add_i32 s37, s43, 0x5280
	buffer_load_dwordx4 v12, s[24:27], s33 offen lds
	s_mov_b32 m0, s37
	s_add_i32 s39, s43, 0x6300
	buffer_load_dwordx4 v21, s[24:27], s33 offen lds
	s_mov_b32 m0, s39
	s_add_i32 s51, s43, 0x7380
	buffer_load_dwordx4 v22, s[24:27], s33 offen lds
	s_mov_b32 m0, s51
	s_add_i32 s52, s56, 0x21000
	s_mov_b32 s31, s15
	buffer_load_dwordx4 v23, s[24:27], s33 offen lds
	s_mov_b32 m0, s52
	s_add_i32 s34, s57, 0x24000
	buffer_load_dwordx4 v26, s[28:31], 0 offen lds
	s_mov_b32 m0, s34
	s_cmpk_lt_i32 s6, 0x200
	buffer_load_dwordx4 v24, s[12:15], 0 offen lds
	s_cselect_b64 s[8:9], -1, 0
	s_mov_b32 s22, s38
	s_mov_b32 s23, s15
	s_movk_i32 s4, 0x80
	s_and_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_2
; %bb.1:
	s_add_i32 m0, s16, 0x8400
	s_mov_b32 s26, s22
	buffer_load_dwordx4 v10, s[20:23], s4 offen lds
	s_add_i32 m0, s16, 0x9480
	s_mov_b32 s27, s23
	buffer_load_dwordx4 v18, s[20:23], s4 offen lds
	s_add_i32 m0, s16, 0xa500
	s_nop 0
	buffer_load_dwordx4 v19, s[20:23], s4 offen lds
	s_add_i32 m0, s16, 0xb580
	s_nop 0
	buffer_load_dwordx4 v20, s[20:23], s4 offen lds
	s_add_i32 m0, s17, 0x18c00
	s_nop 0
	buffer_load_dwordx4 v12, s[24:27], s4 offen lds
	s_add_i32 m0, s17, 0x19c80
	s_nop 0
	buffer_load_dwordx4 v21, s[24:27], s4 offen lds
	s_add_i32 m0, s17, 0x1ad00
	s_nop 0
	buffer_load_dwordx4 v22, s[24:27], s4 offen lds
	s_add_i32 m0, s17, 0x1bd80
	s_nop 0
	buffer_load_dwordx4 v23, s[24:27], s4 offen lds
	s_add_i32 s4, s19, 0x80
	s_add_i32 m0, s16, 0xc600
	s_nop 0
	buffer_load_dwordx4 v10, s[20:23], s4 offen lds
	s_add_i32 m0, s16, 0xd680
	s_nop 0
	buffer_load_dwordx4 v18, s[20:23], s4 offen lds
	s_add_i32 m0, s16, 0xe700
	s_nop 0
	buffer_load_dwordx4 v19, s[20:23], s4 offen lds
	s_add_i32 m0, s16, 0xf780
	s_nop 0
	buffer_load_dwordx4 v20, s[20:23], s4 offen lds
	s_add_i32 s4, s33, 0x80
	s_add_i32 m0, s17, 0x1ce00
	s_nop 0
	buffer_load_dwordx4 v12, s[24:27], s4 offen lds
	s_add_i32 m0, s17, 0x1de80
	s_nop 0
	buffer_load_dwordx4 v21, s[24:27], s4 offen lds
	s_add_i32 m0, s17, 0x1ef00
	s_nop 0
	buffer_load_dwordx4 v22, s[24:27], s4 offen lds
	s_add_i32 m0, s17, 0x1ff80
	s_nop 0
	buffer_load_dwordx4 v23, s[24:27], s4 offen lds
.LBB0_2:
	s_cmpk_lt_i32 s6, 0x300
	s_mov_b64 s[4:5], -1
	s_cbranch_scc0 .LBB0_8
; %bb.3:
	s_mov_b64 vcc, vcc
	s_cbranch_vccz .LBB0_5
; %bb.4:
	s_waitcnt vmcnt(0)
	s_mov_b64 s[4:5], 0
.LBB0_5:
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB0_7
; %bb.6:
	s_waitcnt vmcnt(16)
.LBB0_7:
	s_mov_b64 s[4:5], 0
.LBB0_8:
	s_load_dwordx2 s[0:1], s[0:1], 0x10
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB0_10
; %bb.9:
	s_add_i32 m0, s56, 0x22000
	s_mov_b32 s31, s15
	buffer_load_dwordx4 v26, s[28:31], 16 offen lds
	s_add_i32 m0, s57, 0x25000
	s_nop 0
	buffer_load_dwordx4 v24, s[12:15], 16 offen lds
	s_waitcnt vmcnt(18)
.LBB0_10:
	s_ashr_i32 s4, s6, 31
	s_lshr_b32 s4, s4, 24
	v_and_b32_e32 v3, 1, v0
	s_add_i32 s4, s6, s4
	v_mov_b32_e32 v4, 0x420
	v_cmp_eq_u32_e32 vcc, 1, v3
	s_ashr_i32 s9, s4, 8
	s_movk_i32 s4, 0x380
	v_cndmask_b32_e32 v3, 0, v4, vcc
	v_lshlrev_b32_e32 v4, 6, v0
	v_and_b32_e32 v5, 48, v0
	v_and_or_b32 v3, v4, s4, v3
	v_and_b32_e32 v6, 15, v0
	v_lshrrev_b32_e32 v1, 4, v2
	s_mov_b32 s8, 0
	v_add3_u32 v14, v3, s54, v5
	v_add3_u32 v17, v3, s55, v5
	v_lshlrev_b32_e32 v3, 9, v1
	s_lshl_b32 s4, s7, 8
	v_lshlrev_b32_e32 v4, 4, v6
	s_lshl_b32 s5, s53, 8
	s_barrier
	; sched_barrier mask(0x00000000)
	v_or3_b32 v5, v3, s4, v4
	v_or_b32_e32 v15, 0x21000, v5
	s_add_i32 s5, s5, 0x24000
	ds_read_b64 v[8:9], v15
	v_add_u32_e32 v3, s5, v3
	v_or_b32_e32 v5, 0x21800, v5
	v_or_b32_e32 v16, v3, v4
	ds_read_b64 v[28:29], v5
	ds_read2st64_b64 v[4:7], v16 offset1:4
	v_and_b32_e32 v11, 16, v2
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v3, v8
	v_mov_b32_e32 v13, 0x6020400
	v_mov_b32_e32 v27, 0x7030501
	v_cmp_ne_u32_e32 vcc, 0, v11
	v_lshrrev_b32_e32 v2, 5, v2
	s_mov_b32 s4, 0x2020202
	v_permlane16_swap_b32_e64 v8, v3 bound_ctrl:1
	v_cndmask_b32_e32 v11, v13, v27, vcc
	v_mul_lo_u32 v2, v2, s4
	v_perm_b32 v3, v3, v8, v11
	v_add_u32_e32 v13, 0x5040100, v2
	v_mov_b32_e32 v2, v4
	v_mov_b32_e32 v8, v3
	s_nop 0
	v_permlane16_swap_b32_e64 v4, v2 bound_ctrl:1
	v_permlane32_swap_b32_e64 v3, v8 bound_ctrl:1
	v_perm_b32 v2, v2, v4, v11
	v_perm_b32 v72, v8, v3, v13
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	v_perm_b32 v74, v3, v2, v13
	v_mov_b32_e32 v2, v9
	s_nop 1
	v_permlane16_swap_b32_e64 v9, v2 bound_ctrl:1
	v_perm_b32 v2, v2, v9, v11
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	ds_read_b128 a[0:3], v14
	v_perm_b32 v71, v3, v2, v13
	v_mov_b32_e32 v2, v5
	s_nop 1
	v_permlane16_swap_b32_e64 v5, v2 bound_ctrl:1
	ds_read_b128 a[16:19], v14 offset:64
	v_perm_b32 v2, v2, v5, v11
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	ds_read_b128 a[4:7], v14 offset:4224
	v_perm_b32 v73, v3, v2, v13
	v_mov_b32_e32 v2, v28
	s_nop 1
	v_permlane16_swap_b32_e64 v28, v2 bound_ctrl:1
	ds_read_b128 a[20:23], v14 offset:4288
	v_perm_b32 v2, v2, v28, v11
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	ds_read_b128 a[8:11], v14 offset:8448
	v_perm_b32 v69, v3, v2, v13
	v_mov_b32_e32 v2, v6
	s_nop 1
	v_permlane16_swap_b32_e64 v6, v2 bound_ctrl:1
	ds_read_b128 a[24:27], v14 offset:8512
	v_perm_b32 v2, v2, v6, v11
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	ds_read_b128 a[12:15], v14 offset:12672
	v_perm_b32 v70, v3, v2, v13
	v_mov_b32_e32 v2, v29
	s_nop 1
	v_permlane16_swap_b32_e64 v29, v2 bound_ctrl:1
	ds_read_b128 a[28:31], v14 offset:12736
	v_perm_b32 v2, v2, v29, v11
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	v_add_u32_e32 v25, 0x10800, v17
	ds_read_b128 a[64:67], v25
	v_perm_b32 v67, v3, v2, v13
	v_mov_b32_e32 v2, v7
	ds_read_b128 a[80:83], v25 offset:64
	ds_read_b128 a[68:71], v25 offset:4224
	ds_read_b128 a[84:87], v25 offset:4288
	ds_read_b128 a[72:75], v25 offset:8448
	ds_read_b128 a[88:91], v25 offset:8512
	v_permlane16_swap_b32_e64 v7, v2 bound_ctrl:1
	ds_read_b128 a[76:79], v25 offset:12672
	ds_read_b128 a[92:95], v25 offset:12736
	v_perm_b32 v2, v2, v7, v11
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	v_perm_b32 v68, v3, v2, v13
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[64:67], a[0:3], 0, v74, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[32:35], v14 offset:16896
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[80:83], a[16:19], a[128:131], v73, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[68:71], a[0:3], 0, v74, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[36:39], v14 offset:21120
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(6)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[84:87], a[16:19], a[132:135], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[72:75], a[0:3], 0, v74, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[40:43], v14 offset:25344
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[88:91], a[16:19], a[136:139], v73, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[76:79], a[0:3], 0, v74, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[44:47], v14 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[92:95], a[16:19], a[140:143], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[64:67], a[4:7], 0, v74, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[48:51], v14 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[80:83], a[20:23], a[144:147], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[68:71], a[4:7], 0, v74, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[52:55], v14 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[84:87], a[20:23], a[148:151], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[72:75], a[4:7], 0, v74, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[56:59], v14 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[88:91], a[20:23], a[152:155], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[76:79], a[4:7], 0, v74, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[60:63], v14 offset:29632
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[92:95], a[20:23], a[156:159], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[64:67], a[8:11], 0, v74, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v25 offset:16896
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[80:83], a[24:27], a[160:163], v73, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[68:71], a[8:11], 0, v74, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v25 offset:21120
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[84:87], a[24:27], a[164:167], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[72:75], a[8:11], 0, v74, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[104:107], v25 offset:25344
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[88:91], a[24:27], a[168:171], v73, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[76:79], a[8:11], 0, v74, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v25 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[92:95], a[24:27], a[172:175], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[64:67], a[12:15], 0, v74, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v25 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[80:83], a[28:31], a[176:179], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[68:71], a[12:15], 0, v74, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v25 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[84:87], a[28:31], a[180:183], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[72:75], a[12:15], 0, v74, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[120:123], v25 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[88:91], a[28:31], a[184:187], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[76:79], a[12:15], 0, v74, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v25 offset:29632
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[92:95], a[28:31], a[188:191], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_cmpk_lt_i32 s6, 0x400
	s_cbranch_scc1 .LBB0_17
; %bb.11:
	s_add_i32 s4, s9, -3
	s_add_i32 s5, s16, 0x8400
	s_add_i32 s38, s16, 0x9480
	s_add_i32 s53, s16, 0xa500
	s_add_i32 s54, s16, 0xb580
	s_add_i32 s55, s17, 0x18c00
	s_add_i32 s56, s17, 0x19c80
	s_add_i32 s57, s17, 0x1ad00
	s_add_i32 s58, s17, 0x1bd80
	s_add_i32 s59, s16, 0xc600
	s_add_i32 s60, s16, 0xd680
	s_add_i32 s61, s16, 0xe700
	s_add_i32 s62, s16, 0xf780
	s_add_i32 s63, s17, 0x1ce00
	s_add_i32 s64, s17, 0x1de80
	s_add_i32 s65, s17, 0x1ef00
	s_add_i32 s66, s17, 0x1ff80
	s_mov_b32 s67, 16
	s_movk_i32 s68, 0x180
	v_mov_b32_e32 v240, 0
	v_mov_b32_e32 v241, 0
	v_mov_b32_e32 v242, 0
	v_mov_b32_e32 v243, 0
	v_mov_b32_e32 v236, 0
	v_mov_b32_e32 v237, 0
	v_mov_b32_e32 v238, 0
	v_mov_b32_e32 v239, 0
	v_mov_b32_e32 v232, 0
	v_mov_b32_e32 v233, 0
	v_mov_b32_e32 v234, 0
	v_mov_b32_e32 v235, 0
	v_mov_b32_e32 v228, 0
	v_mov_b32_e32 v229, 0
	v_mov_b32_e32 v230, 0
	v_mov_b32_e32 v231, 0
	v_mov_b32_e32 v224, 0
	v_mov_b32_e32 v225, 0
	v_mov_b32_e32 v226, 0
	v_mov_b32_e32 v227, 0
	v_mov_b32_e32 v220, 0
	v_mov_b32_e32 v221, 0
	v_mov_b32_e32 v222, 0
	v_mov_b32_e32 v223, 0
	v_mov_b32_e32 v216, 0
	v_mov_b32_e32 v217, 0
	v_mov_b32_e32 v218, 0
	v_mov_b32_e32 v219, 0
	v_mov_b32_e32 v212, 0
	v_mov_b32_e32 v213, 0
	v_mov_b32_e32 v214, 0
	v_mov_b32_e32 v215, 0
	v_mov_b32_e32 v208, 0
	v_mov_b32_e32 v209, 0
	v_mov_b32_e32 v210, 0
	v_mov_b32_e32 v211, 0
	v_mov_b32_e32 v204, 0
	v_mov_b32_e32 v205, 0
	v_mov_b32_e32 v206, 0
	v_mov_b32_e32 v207, 0
	v_mov_b32_e32 v200, 0
	v_mov_b32_e32 v201, 0
	v_mov_b32_e32 v202, 0
	v_mov_b32_e32 v203, 0
	v_mov_b32_e32 v196, 0
	v_mov_b32_e32 v197, 0
	v_mov_b32_e32 v198, 0
	v_mov_b32_e32 v199, 0
	v_mov_b32_e32 v192, 0
	v_mov_b32_e32 v193, 0
	v_mov_b32_e32 v194, 0
	v_mov_b32_e32 v195, 0
	v_accvgpr_write_b32 a252, 0
	v_accvgpr_write_b32 a253, 0
	v_accvgpr_write_b32 a254, 0
	v_accvgpr_write_b32 a255, 0
	v_accvgpr_write_b32 a248, 0
	v_accvgpr_write_b32 a249, 0
	v_accvgpr_write_b32 a250, 0
	v_accvgpr_write_b32 a251, 0
	v_accvgpr_write_b32 a244, 0
	v_accvgpr_write_b32 a245, 0
	v_accvgpr_write_b32 a246, 0
	v_accvgpr_write_b32 a247, 0
	v_accvgpr_write_b32 a240, 0
	v_accvgpr_write_b32 a241, 0
	v_accvgpr_write_b32 a242, 0
	v_accvgpr_write_b32 a243, 0
	v_accvgpr_write_b32 a236, 0
	v_accvgpr_write_b32 a237, 0
	v_accvgpr_write_b32 a238, 0
	v_accvgpr_write_b32 a239, 0
	v_accvgpr_write_b32 a232, 0
	v_accvgpr_write_b32 a233, 0
	v_accvgpr_write_b32 a234, 0
	v_accvgpr_write_b32 a235, 0
	v_accvgpr_write_b32 a228, 0
	v_accvgpr_write_b32 a229, 0
	v_accvgpr_write_b32 a230, 0
	v_accvgpr_write_b32 a231, 0
	v_accvgpr_write_b32 a224, 0
	v_accvgpr_write_b32 a225, 0
	v_accvgpr_write_b32 a226, 0
	v_accvgpr_write_b32 a227, 0
	v_accvgpr_write_b32 a220, 0
	v_accvgpr_write_b32 a221, 0
	v_accvgpr_write_b32 a222, 0
	v_accvgpr_write_b32 a223, 0
	v_accvgpr_write_b32 a216, 0
	v_accvgpr_write_b32 a217, 0
	v_accvgpr_write_b32 a218, 0
	v_accvgpr_write_b32 a219, 0
	v_accvgpr_write_b32 a212, 0
	v_accvgpr_write_b32 a213, 0
	v_accvgpr_write_b32 a214, 0
	v_accvgpr_write_b32 a215, 0
	v_accvgpr_write_b32 a208, 0
	v_accvgpr_write_b32 a209, 0
	v_accvgpr_write_b32 a210, 0
	v_accvgpr_write_b32 a211, 0
	v_accvgpr_write_b32 a204, 0
	v_accvgpr_write_b32 a205, 0
	v_accvgpr_write_b32 a206, 0
	v_accvgpr_write_b32 a207, 0
	v_accvgpr_write_b32 a200, 0
	v_accvgpr_write_b32 a201, 0
	v_accvgpr_write_b32 a202, 0
	v_accvgpr_write_b32 a203, 0
	v_accvgpr_write_b32 a196, 0
	v_accvgpr_write_b32 a197, 0
	v_accvgpr_write_b32 a198, 0
	v_accvgpr_write_b32 a199, 0
	v_accvgpr_write_b32 a192, 0
	v_accvgpr_write_b32 a193, 0
	v_accvgpr_write_b32 a194, 0
	v_accvgpr_write_b32 a195, 0
	v_mov_b32_e32 v188, 0
	v_mov_b32_e32 v189, 0
	v_mov_b32_e32 v190, 0
	v_mov_b32_e32 v191, 0
	v_mov_b32_e32 v184, 0
	v_mov_b32_e32 v185, 0
	v_mov_b32_e32 v186, 0
	v_mov_b32_e32 v187, 0
	v_mov_b32_e32 v180, 0
	v_mov_b32_e32 v181, 0
	v_mov_b32_e32 v182, 0
	v_mov_b32_e32 v183, 0
	v_mov_b32_e32 v176, 0
	v_mov_b32_e32 v177, 0
	v_mov_b32_e32 v178, 0
	v_mov_b32_e32 v179, 0
	v_mov_b32_e32 v172, 0
	v_mov_b32_e32 v173, 0
	v_mov_b32_e32 v174, 0
	v_mov_b32_e32 v175, 0
	v_mov_b32_e32 v168, 0
	v_mov_b32_e32 v169, 0
	v_mov_b32_e32 v170, 0
	v_mov_b32_e32 v171, 0
	v_mov_b32_e32 v164, 0
	v_mov_b32_e32 v165, 0
	v_mov_b32_e32 v166, 0
	v_mov_b32_e32 v167, 0
	v_mov_b32_e32 v160, 0
	v_mov_b32_e32 v161, 0
	v_mov_b32_e32 v162, 0
	v_mov_b32_e32 v163, 0
	v_mov_b32_e32 v156, 0
	v_mov_b32_e32 v157, 0
	v_mov_b32_e32 v158, 0
	v_mov_b32_e32 v159, 0
	v_mov_b32_e32 v152, 0
	v_mov_b32_e32 v153, 0
	v_mov_b32_e32 v154, 0
	v_mov_b32_e32 v155, 0
	v_mov_b32_e32 v148, 0
	v_mov_b32_e32 v149, 0
	v_mov_b32_e32 v150, 0
	v_mov_b32_e32 v151, 0
	v_mov_b32_e32 v144, 0
	v_mov_b32_e32 v145, 0
	v_mov_b32_e32 v146, 0
	v_mov_b32_e32 v147, 0
	v_mov_b32_e32 v140, 0
	v_mov_b32_e32 v141, 0
	v_mov_b32_e32 v142, 0
	v_mov_b32_e32 v143, 0
	v_mov_b32_e32 v136, 0
	v_mov_b32_e32 v137, 0
	v_mov_b32_e32 v138, 0
	v_mov_b32_e32 v139, 0
	v_mov_b32_e32 v132, 0
	v_mov_b32_e32 v133, 0
	v_mov_b32_e32 v134, 0
	v_mov_b32_e32 v135, 0
	v_mov_b32_e32 v128, 0
	v_mov_b32_e32 v129, 0
	v_mov_b32_e32 v130, 0
	v_mov_b32_e32 v131, 0
	v_mov_b32_e32 v244, 0
	v_mov_b32_e32 v245, 0
	v_mov_b32_e32 v246, 0
	v_mov_b32_e32 v247, 0
	v_mov_b32_e32 v248, 0
	v_mov_b32_e32 v249, 0
	v_mov_b32_e32 v250, 0
	v_mov_b32_e32 v251, 0
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, 0
	v_mov_b32_e32 v254, 0
	v_mov_b32_e32 v255, 0
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_mov_b32 s31, s15
	s_mov_b32 s6, 0
.LBB0_12:                               ; =>This Inner Loop Header: Depth=1
	s_mov_b32 s69, s6
	s_add_i32 s6, s6, 1
	s_cmp_lg_u32 s69, 2
	s_cselect_b32 s6, s6, 0
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[64:67], a[32:35], a[192:195], v74, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s16
	s_add_i32 s70, s68, 0xffffff80
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[80:83], a[48:51], a[192:195], v73, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v10, s[20:23], s70 offen lds
	s_mov_b32 m0, s40
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[68:71], a[32:35], a[196:199], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[84:87], a[48:51], a[196:199], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[72:75], a[32:35], a[200:203], v74, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[88:91], a[48:51], a[200:203], v73, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v18, s[20:23], s70 offen lds
	s_mov_b32 m0, s41
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[76:79], a[32:35], a[204:207], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[92:95], a[48:51], a[204:207], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[64:67], a[36:39], a[208:211], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[80:83], a[52:55], a[208:211], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v19, s[20:23], s70 offen lds
	s_mov_b32 m0, s42
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[68:71], a[36:39], a[212:215], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[84:87], a[52:55], a[212:215], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[72:75], a[36:39], a[216:219], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[88:91], a[52:55], a[216:219], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v20, s[20:23], s70 offen lds
	s_mov_b32 m0, s43
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[76:79], a[36:39], a[220:223], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[92:95], a[52:55], a[220:223], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[64:67], a[40:43], a[224:227], v74, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[80:83], a[56:59], a[224:227], v73, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v12, s[24:27], s70 offen lds
	s_mov_b32 m0, s44
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[68:71], a[40:43], a[228:231], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[84:87], a[56:59], a[228:231], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[72:75], a[40:43], a[232:235], v74, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[88:91], a[56:59], a[232:235], v73, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v21, s[24:27], s70 offen lds
	s_mov_b32 m0, s45
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[76:79], a[40:43], a[236:239], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[92:95], a[56:59], a[236:239], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[64:67], a[44:47], a[240:243], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[80:83], a[60:63], a[240:243], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v22, s[24:27], s70 offen lds
	s_mov_b32 m0, s46
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[68:71], a[44:47], a[244:247], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[84:87], a[60:63], a[244:247], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[72:75], a[44:47], a[248:251], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[88:91], a[60:63], a[248:251], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v23, s[24:27], s70 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[76:79], a[44:47], a[252:255], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[92:95], a[60:63], a[252:255], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(8)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[96:99], a[0:3], v[128:131], v70, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_lshl_b32 s70, s69, 12
	v_add_u32_e32 v2, s70, v15
	s_add_i32 s69, s19, s68
	s_mov_b32 m0, s47
	v_add_u32_e32 v2, 8, v2
	s_add_i32 s71, s69, 0xffffff80
	ds_read2st64_b64 v[6:9], v2 offset1:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[112:115], a[16:19], v[128:131], v68, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v10, s[20:23], s71 offen lds
	v_add_u32_e32 v2, s70, v16
	v_add_u32_e32 v2, 8, v2
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[100:103], a[0:3], v[132:135], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read2st64_b64 v[2:5], v2 offset1:4
	s_mov_b32 m0, s48
	s_add_i32 s70, s33, s68
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[116:119], a[16:19], v[132:135], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_i32 s8, s8, 2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[104:107], a[0:3], v[136:139], v70, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[64:67], v25 offset:33792
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[120:123], a[16:19], v[136:139], v68, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v18, s[20:23], s71 offen lds
	s_mov_b32 m0, s49
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[108:111], a[0:3], v[140:143], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[68:71], v25 offset:38016
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[124:127], a[16:19], v[140:143], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[96:99], a[4:7], v[144:147], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[72:75], v25 offset:42240
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[112:115], a[20:23], v[144:147], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v19, s[20:23], s71 offen lds
	s_mov_b32 m0, s50
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[100:103], a[4:7], v[148:151], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[76:79], v25 offset:46464
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[116:119], a[20:23], v[148:151], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[104:107], a[4:7], v[152:155], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[80:83], v25 offset:33856
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[120:123], a[20:23], v[152:155], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v20, s[20:23], s71 offen lds
	s_add_i32 s71, s70, 0xffffff80
	s_mov_b32 m0, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[108:111], a[4:7], v[156:159], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[84:87], v25 offset:38080
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[124:127], a[20:23], v[156:159], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[96:99], a[8:11], v[160:163], v70, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[88:91], v25 offset:42304
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[112:115], a[24:27], v[160:163], v68, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v12, s[24:27], s71 offen lds
	s_mov_b32 m0, s37
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[100:103], a[8:11], v[164:167], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[116:119], a[24:27], v[164:167], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[104:107], a[8:11], v[168:171], v70, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[92:95], v25 offset:46528
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[120:123], a[24:27], v[168:171], v68, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v21, s[24:27], s71 offen lds
	s_mov_b32 m0, s39
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[108:111], a[8:11], v[172:175], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[124:127], a[24:27], v[172:175], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[96:99], a[12:15], v[176:179], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[112:115], a[28:31], v[176:179], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v22, s[24:27], s71 offen lds
	s_mov_b32 m0, s51
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[100:103], a[12:15], v[180:183], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[116:119], a[28:31], v[180:183], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[104:107], a[12:15], v[184:187], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[120:123], a[28:31], v[184:187], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v23, s[24:27], s71 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[108:111], a[12:15], v[188:191], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[124:127], a[28:31], v[188:191], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s71, s6, 12
	s_add_i32 m0, s34, s71
	s_nop 0
	buffer_load_dwordx4 v24, s[12:15], s67 offen lds
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[96:99], a[32:35], v[192:195], v70, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[0:3], v14 offset:33792
	s_waitcnt lgkmcnt(10)
	v_mov_b32_e32 v27, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[112:115], a[48:51], v[192:195], v68, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v6, v27 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[100:103], a[32:35], v[196:199], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_perm_b32 v6, v27, v6, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[116:119], a[48:51], v[196:199], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[104:107], a[32:35], v[200:203], v70, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[4:7], v14 offset:38016
	v_mov_b32_e32 v27, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[120:123], a[48:51], v[200:203], v68, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v6, v27 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[108:111], a[32:35], v[204:207], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_perm_b32 v28, v27, v6, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[124:127], a[48:51], v[204:207], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[96:99], a[36:39], v[208:211], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[8:11], v14 offset:42240
	s_waitcnt lgkmcnt(11)
	v_mov_b32_e32 v6, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[112:115], a[52:55], v[208:211], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v2, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[100:103], a[36:39], v[212:215], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_perm_b32 v2, v6, v2, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[116:119], a[52:55], v[212:215], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[104:107], a[36:39], v[216:219], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[12:15], v14 offset:46464
	v_mov_b32_e32 v6, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[120:123], a[52:55], v[216:219], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[108:111], a[36:39], v[220:223], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_perm_b32 v2, v6, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[124:127], a[52:55], v[220:223], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[96:99], a[40:43], v[224:227], v70, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[16:19], v14 offset:33856
	v_mov_b32_e32 v6, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[112:115], a[56:59], v[224:227], v68, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v7, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[100:103], a[40:43], v[228:231], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v6, v6, v7, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[116:119], a[56:59], v[228:231], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[104:107], a[40:43], v[232:235], v70, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[20:23], v14 offset:38080
	v_mov_b32_e32 v7, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[120:123], a[56:59], v[232:235], v68, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v6, v7 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[108:111], a[40:43], v[236:239], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v27, v7, v6, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[124:127], a[56:59], v[236:239], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[96:99], a[44:47], v[240:243], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[24:27], v14 offset:42304
	v_mov_b32_e32 v6, v3
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[112:115], a[60:63], v[240:243], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v3, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[100:103], a[44:47], v[244:247], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v3, v6, v3, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[116:119], a[60:63], v[244:247], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[104:107], a[44:47], v[248:251], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[28:31], v14 offset:46528
	v_mov_b32_e32 v6, v3
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[120:123], a[60:63], v[248:251], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v3, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[108:111], a[44:47], v[252:255], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v3, v6, v3, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[124:127], a[60:63], v[252:255], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[64:67], a[0:3], a[128:131], v2, v28 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[32:35], v14 offset:50688
	v_mov_b32_e32 v6, v8
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[80:83], a[16:19], a[128:131], v3, v27 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[36:39], v14 offset:54912
	v_permlane16_swap_b32_e64 v8, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[68:71], a[0:3], a[132:135], v2, v28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[40:43], v14 offset:59136
	v_perm_b32 v6, v6, v8, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[84:87], a[16:19], a[132:135], v3, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[44:47], v14 offset:63360
	v_mov_b32_e32 v7, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[72:75], a[0:3], a[136:139], v2, v28 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[48:51], v14 offset:50752
	v_permlane32_swap_b32_e64 v6, v7 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[88:91], a[16:19], a[136:139], v3, v27 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[52:55], v14 offset:54976
	v_perm_b32 v31, v7, v6, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[76:79], a[0:3], a[140:143], v2, v28 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[56:59], v14 offset:59200
	v_mov_b32_e32 v6, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[92:95], a[16:19], a[140:143], v3, v27 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[60:63], v14 offset:63424
	v_permlane16_swap_b32_e64 v4, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[64:67], a[4:7], a[144:147], v2, v28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v25 offset:50688
	v_perm_b32 v4, v6, v4, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[80:83], a[20:23], a[144:147], v3, v27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v25 offset:54912
	v_mov_b32_e32 v6, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[68:71], a[4:7], a[148:151], v2, v28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[104:107], v25 offset:59136
	v_permlane32_swap_b32_e64 v4, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[84:87], a[20:23], a[148:151], v3, v27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v25 offset:63360
	v_perm_b32 v32, v6, v4, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[72:75], a[4:7], a[152:155], v2, v28 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v25 offset:50752
	v_mov_b32_e32 v4, v9
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[88:91], a[20:23], a[152:155], v3, v27 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v25 offset:54976
	v_permlane16_swap_b32_e64 v9, v4 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[76:79], a[4:7], a[156:159], v2, v28 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[120:123], v25 offset:59200
	v_perm_b32 v4, v4, v9, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[92:95], a[20:23], a[156:159], v3, v27 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v25 offset:63424
	v_mov_b32_e32 v6, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[64:67], a[8:11], a[160:163], v2, v28 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v4, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[80:83], a[24:27], a[160:163], v3, v27 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v29, v6, v4, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[68:71], a[8:11], a[164:167], v2, v28 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v4, v5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[84:87], a[24:27], a[164:167], v3, v27 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v5, v4 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[72:75], a[8:11], a[168:171], v2, v28 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v4, v4, v5, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[88:91], a[24:27], a[168:171], v3, v27 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v5, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[76:79], a[8:11], a[172:175], v2, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v4, v5 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[92:95], a[24:27], a[172:175], v3, v27 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v30, v5, v4, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[64:67], a[12:15], a[176:179], v2, v28 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[80:83], a[28:31], a[176:179], v3, v27 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[68:71], a[12:15], a[180:183], v2, v28 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[84:87], a[28:31], a[180:183], v3, v27 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[72:75], a[12:15], a[184:187], v2, v28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[88:91], a[28:31], a[184:187], v3, v27 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[76:79], a[12:15], a[188:191], v2, v28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[92:95], a[28:31], a[188:191], v3, v27 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[64:67], a[32:35], a[192:195], v2, v31 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[80:83], a[48:51], a[192:195], v3, v29 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v10, s[20:23], s68 offen lds
	s_mov_b32 m0, s38
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[68:71], a[32:35], a[196:199], v2, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[84:87], a[48:51], a[196:199], v3, v29 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[72:75], a[32:35], a[200:203], v2, v31 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[88:91], a[48:51], a[200:203], v3, v29 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v18, s[20:23], s68 offen lds
	s_mov_b32 m0, s53
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[76:79], a[32:35], a[204:207], v2, v31 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[92:95], a[48:51], a[204:207], v3, v29 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[64:67], a[36:39], a[208:211], v2, v31 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[80:83], a[52:55], a[208:211], v3, v29 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v19, s[20:23], s68 offen lds
	s_mov_b32 m0, s54
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[68:71], a[36:39], a[212:215], v2, v31 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[84:87], a[52:55], a[212:215], v3, v29 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[72:75], a[36:39], a[216:219], v2, v31 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[88:91], a[52:55], a[216:219], v3, v29 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v20, s[20:23], s68 offen lds
	s_mov_b32 m0, s55
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[76:79], a[36:39], a[220:223], v2, v31 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[92:95], a[52:55], a[220:223], v3, v29 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[64:67], a[40:43], a[224:227], v2, v31 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[80:83], a[56:59], a[224:227], v3, v29 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v12, s[24:27], s68 offen lds
	s_mov_b32 m0, s56
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[68:71], a[40:43], a[228:231], v2, v31 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[84:87], a[56:59], a[228:231], v3, v29 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[72:75], a[40:43], a[232:235], v2, v31 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[88:91], a[56:59], a[232:235], v3, v29 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v21, s[24:27], s68 offen lds
	s_mov_b32 m0, s57
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[76:79], a[40:43], a[236:239], v2, v31 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[92:95], a[56:59], a[236:239], v3, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[64:67], a[44:47], a[240:243], v2, v31 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[80:83], a[60:63], a[240:243], v3, v29 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v22, s[24:27], s68 offen lds
	s_mov_b32 m0, s58
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[68:71], a[44:47], a[244:247], v2, v31 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[84:87], a[60:63], a[244:247], v3, v29 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[72:75], a[44:47], a[248:251], v2, v31 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[88:91], a[60:63], a[248:251], v3, v29 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v23, s[24:27], s68 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[76:79], a[44:47], a[252:255], v2, v31 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[92:95], a[60:63], a[252:255], v3, v29 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(8)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[96:99], a[0:3], v[128:131], v32, v28 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mov_b32 m0, s59
	v_add_u32_e32 v2, s71, v15
	ds_read2st64_b64 v[6:9], v2 offset1:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[112:115], a[16:19], v[128:131], v30, v27 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v10, s[20:23], s69 offen lds
	v_add_u32_e32 v2, s71, v16
	s_mov_b32 m0, s60
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[100:103], a[0:3], v[132:135], v32, v28 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read2st64_b64 v[2:5], v2 offset1:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[116:119], a[16:19], v[132:135], v30, v27 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[104:107], a[0:3], v[136:139], v32, v28 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[64:67], v25
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[120:123], a[16:19], v[136:139], v30, v27 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v18, s[20:23], s69 offen lds
	s_mov_b32 m0, s61
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[108:111], a[0:3], v[140:143], v32, v28 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[68:71], v25 offset:4224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[124:127], a[16:19], v[140:143], v30, v27 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[96:99], a[4:7], v[144:147], v32, v28 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[72:75], v25 offset:8448
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[112:115], a[20:23], v[144:147], v30, v27 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v19, s[20:23], s69 offen lds
	s_mov_b32 m0, s62
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[100:103], a[4:7], v[148:151], v32, v28 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[76:79], v25 offset:12672
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[116:119], a[20:23], v[148:151], v30, v27 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[104:107], a[4:7], v[152:155], v32, v28 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[80:83], v25 offset:64
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[120:123], a[20:23], v[152:155], v30, v27 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v20, s[20:23], s69 offen lds
	s_mov_b32 m0, s63
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[108:111], a[4:7], v[156:159], v32, v28 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[84:87], v25 offset:4288
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[124:127], a[20:23], v[156:159], v30, v27 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[96:99], a[8:11], v[160:163], v32, v28 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[88:91], v25 offset:8512
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[112:115], a[24:27], v[160:163], v30, v27 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v12, s[24:27], s70 offen lds
	s_mov_b32 m0, s64
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[100:103], a[8:11], v[164:167], v32, v28 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[116:119], a[24:27], v[164:167], v30, v27 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[104:107], a[8:11], v[168:171], v32, v28 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[92:95], v25 offset:12736
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[120:123], a[24:27], v[168:171], v30, v27 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v21, s[24:27], s70 offen lds
	s_mov_b32 m0, s65
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[108:111], a[8:11], v[172:175], v32, v28 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[124:127], a[24:27], v[172:175], v30, v27 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[96:99], a[12:15], v[176:179], v32, v28 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[112:115], a[28:31], v[176:179], v30, v27 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v22, s[24:27], s70 offen lds
	s_mov_b32 m0, s66
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[100:103], a[12:15], v[180:183], v32, v28 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[116:119], a[28:31], v[180:183], v30, v27 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[104:107], a[12:15], v[184:187], v32, v28 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[120:123], a[28:31], v[184:187], v30, v27 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v23, s[24:27], s70 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[108:111], a[12:15], v[188:191], v32, v28 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[124:127], a[28:31], v[188:191], v30, v27 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_addk_i32 s71, 0x1000
	s_cmp_lg_u32 s6, 2
	s_cselect_b32 s69, s71, 0
	s_add_i32 s67, s67, 16
	s_add_i32 m0, s69, s52
	s_nop 0
	buffer_load_dwordx4 v26, s[28:31], s67 offen lds
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[96:99], a[32:35], v[192:195], v32, v31 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[0:3], v14
	s_waitcnt lgkmcnt(10)
	v_mov_b32_e32 v27, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[112:115], a[48:51], v[192:195], v30, v29 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v6, v27 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[100:103], a[32:35], v[196:199], v32, v31 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_perm_b32 v6, v27, v6, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[116:119], a[48:51], v[196:199], v30, v29 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[104:107], a[32:35], v[200:203], v32, v31 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[4:7], v14 offset:4224
	v_mov_b32_e32 v27, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[120:123], a[48:51], v[200:203], v30, v29 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v6, v27 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[108:111], a[32:35], v[204:207], v32, v31 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_perm_b32 v72, v27, v6, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[124:127], a[48:51], v[204:207], v30, v29 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[96:99], a[36:39], v[208:211], v32, v31 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[8:11], v14 offset:8448
	s_waitcnt lgkmcnt(11)
	v_mov_b32_e32 v6, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[112:115], a[52:55], v[208:211], v30, v29 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v2, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[100:103], a[36:39], v[212:215], v32, v31 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_perm_b32 v2, v6, v2, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[116:119], a[52:55], v[212:215], v30, v29 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[104:107], a[36:39], v[216:219], v32, v31 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[12:15], v14 offset:12672
	v_mov_b32_e32 v6, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[120:123], a[52:55], v[216:219], v30, v29 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[108:111], a[36:39], v[220:223], v32, v31 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_perm_b32 v74, v6, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[124:127], a[52:55], v[220:223], v30, v29 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[96:99], a[40:43], v[224:227], v32, v31 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[16:19], v14 offset:64
	v_mov_b32_e32 v2, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[112:115], a[56:59], v[224:227], v30, v29 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v7, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[100:103], a[40:43], v[228:231], v32, v31 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v2, v2, v7, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[116:119], a[56:59], v[228:231], v30, v29 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[104:107], a[40:43], v[232:235], v32, v31 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[20:23], v14 offset:4288
	v_mov_b32_e32 v6, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[120:123], a[56:59], v[232:235], v30, v29 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[108:111], a[40:43], v[236:239], v32, v31 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v71, v6, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[124:127], a[56:59], v[236:239], v30, v29 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[96:99], a[44:47], v[240:243], v32, v31 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[24:27], v14 offset:8512
	v_mov_b32_e32 v2, v3
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[112:115], a[60:63], v[240:243], v30, v29 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v3, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[100:103], a[44:47], v[244:247], v32, v31 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v2, v2, v3, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[116:119], a[60:63], v[244:247], v30, v29 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[104:107], a[44:47], v[248:251], v32, v31 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[28:31], v14 offset:12736
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[120:123], a[60:63], v[248:251], v30, v29 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[108:111], a[44:47], v[252:255], v32, v31 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v73, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[124:127], a[60:63], v[252:255], v30, v29 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[64:67], a[0:3], a[128:131], v74, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[32:35], v14 offset:16896
	v_mov_b32_e32 v2, v8
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[80:83], a[16:19], a[128:131], v73, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[36:39], v14 offset:21120
	v_permlane16_swap_b32_e64 v8, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[68:71], a[0:3], a[132:135], v74, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[40:43], v14 offset:25344
	v_perm_b32 v2, v2, v8, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[84:87], a[16:19], a[132:135], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[44:47], v14 offset:29568
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[72:75], a[0:3], a[136:139], v74, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[48:51], v14 offset:16960
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[88:91], a[16:19], a[136:139], v73, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[52:55], v14 offset:21184
	v_perm_b32 v69, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[76:79], a[0:3], a[140:143], v74, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[56:59], v14 offset:25408
	v_mov_b32_e32 v2, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[92:95], a[16:19], a[140:143], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[60:63], v14 offset:29632
	v_permlane16_swap_b32_e64 v4, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[64:67], a[4:7], a[144:147], v74, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v25 offset:16896
	v_perm_b32 v2, v2, v4, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[80:83], a[20:23], a[144:147], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v25 offset:21120
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[68:71], a[4:7], a[148:151], v74, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[104:107], v25 offset:25344
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[84:87], a[20:23], a[148:151], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v25 offset:29568
	v_perm_b32 v70, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[72:75], a[4:7], a[152:155], v74, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v25 offset:16960
	v_mov_b32_e32 v2, v9
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[88:91], a[20:23], a[152:155], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v25 offset:21184
	v_permlane16_swap_b32_e64 v9, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[76:79], a[4:7], a[156:159], v74, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[120:123], v25 offset:25408
	v_perm_b32 v2, v2, v9, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[92:95], a[20:23], a[156:159], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v25 offset:29632
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[64:67], a[8:11], a[160:163], v74, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[80:83], a[24:27], a[160:163], v73, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v67, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[68:71], a[8:11], a[164:167], v74, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v2, v5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[84:87], a[24:27], a[164:167], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v5, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[72:75], a[8:11], a[168:171], v74, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v2, v2, v5, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[88:91], a[24:27], a[168:171], v73, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[76:79], a[8:11], a[172:175], v74, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[92:95], a[24:27], a[172:175], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v68, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[64:67], a[12:15], a[176:179], v74, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[80:83], a[28:31], a[176:179], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[68:71], a[12:15], a[180:183], v74, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[84:87], a[28:31], a[180:183], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[72:75], a[12:15], a[184:187], v74, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[88:91], a[28:31], a[184:187], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[76:79], a[12:15], a[188:191], v74, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[92:95], a[28:31], a[188:191], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_addk_i32 s68, 0x100
	s_cmp_ge_i32 s8, s4
	s_cbranch_scc0 .LBB0_12
; %bb.13:
	s_add_i32 s4, s8, 2
	s_cmp_ge_i32 s4, s9
	s_cbranch_scc0 .LBB0_18
.LBB0_14:
	s_add_i32 s4, s8, 1
	s_cmp_ge_i32 s4, s9
	s_cbranch_scc1 .LBB0_16
.LBB0_15:
	s_waitcnt lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[64:67], a[32:35], a[192:195], v74, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_not_b32 s4, s8
	s_lshl_b32 s4, s4, 1
	s_and_b32 s4, s4, 2
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[80:83], a[48:51], a[192:195], v73, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mulk_i32 s4, 0x4200
	v_add_u32_e32 v10, s4, v17
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[68:71], a[32:35], a[196:199], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[84:87], a[48:51], a[196:199], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[72:75], a[32:35], a[200:203], v74, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[88:91], a[48:51], a[200:203], v73, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[76:79], a[32:35], a[204:207], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[92:95], a[48:51], a[204:207], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[64:67], a[36:39], a[208:211], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(10)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[80:83], a[52:55], a[208:211], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[68:71], a[36:39], a[212:215], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[84:87], a[52:55], a[212:215], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[72:75], a[36:39], a[216:219], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[88:91], a[52:55], a[216:219], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[76:79], a[36:39], a[220:223], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[92:95], a[52:55], a[220:223], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[64:67], a[40:43], a[224:227], v74, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(9)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[80:83], a[56:59], a[224:227], v73, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[68:71], a[40:43], a[228:231], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[84:87], a[56:59], a[228:231], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[72:75], a[40:43], a[232:235], v74, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[88:91], a[56:59], a[232:235], v73, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[76:79], a[40:43], a[236:239], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[92:95], a[56:59], a[236:239], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[64:67], a[44:47], a[240:243], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[80:83], a[60:63], a[240:243], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[68:71], a[44:47], a[244:247], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[84:87], a[60:63], a[244:247], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[72:75], a[44:47], a[248:251], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[88:91], a[60:63], a[248:251], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[76:79], a[44:47], a[252:255], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[92:95], a[60:63], a[252:255], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s5, s6, 12
	s_lshl_b32 s6, s8, 3
	s_and_b32 s6, s6, 8
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[96:99], a[0:3], v[128:131], v70, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_or_b32 s6, s6, s5
	s_xor_b32 s5, s6, 8
	v_add_u32_e32 v2, s5, v15
	ds_read2st64_b64 v[2:5], v2 offset1:4
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[112:115], a[16:19], v[128:131], v68, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v6, s5, v16
	v_add_u32_e32 v10, 0x10800, v10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[100:103], a[0:3], v[132:135], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read2st64_b64 v[6:9], v6 offset1:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[116:119], a[16:19], v[132:135], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[104:107], a[0:3], v[136:139], v70, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[64:67], v10
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[120:123], a[16:19], v[136:139], v68, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[108:111], a[0:3], v[140:143], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[68:71], v10 offset:4224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[124:127], a[16:19], v[140:143], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[96:99], a[4:7], v[144:147], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[72:75], v10 offset:8448
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[112:115], a[20:23], v[144:147], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[100:103], a[4:7], v[148:151], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[76:79], v10 offset:12672
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[116:119], a[20:23], v[148:151], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[104:107], a[4:7], v[152:155], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[80:83], v10 offset:64
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[120:123], a[20:23], v[152:155], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[108:111], a[4:7], v[156:159], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[84:87], v10 offset:4288
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[124:127], a[20:23], v[156:159], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[96:99], a[8:11], v[160:163], v70, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[88:91], v10 offset:8512
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[112:115], a[24:27], v[160:163], v68, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[100:103], a[8:11], v[164:167], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[116:119], a[24:27], v[164:167], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[104:107], a[8:11], v[168:171], v70, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[92:95], v10 offset:12736
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[120:123], a[24:27], v[168:171], v68, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[108:111], a[8:11], v[172:175], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[124:127], a[24:27], v[172:175], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[96:99], a[12:15], v[176:179], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[112:115], a[28:31], v[176:179], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[100:103], a[12:15], v[180:183], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[116:119], a[28:31], v[180:183], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[104:107], a[12:15], v[184:187], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[120:123], a[28:31], v[184:187], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[108:111], a[12:15], v[188:191], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[124:127], a[28:31], v[188:191], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[96:99], a[32:35], v[192:195], v70, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v12, s4, v14
	ds_read_b128 a[24:27], v12 offset:8512
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[112:115], a[48:51], v[192:195], v68, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(10)
	v_mov_b32_e32 v14, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[100:103], a[32:35], v[196:199], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v2, v14 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[116:119], a[48:51], v[196:199], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[104:107], a[32:35], v[200:203], v70, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[28:31], v12 offset:12736
	v_perm_b32 v2, v14, v2, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[120:123], a[48:51], v[200:203], v68, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v14, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[108:111], a[32:35], v[204:207], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v14 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[124:127], a[48:51], v[204:207], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[96:99], a[36:39], v[208:211], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[0:3], v12
	v_perm_b32 v72, v14, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[112:115], a[52:55], v[208:211], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mov_b32_e32 v2, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[100:103], a[36:39], v[212:215], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v6, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[116:119], a[52:55], v[212:215], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[104:107], a[36:39], v[216:219], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[4:7], v12 offset:4224
	v_perm_b32 v2, v2, v6, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[120:123], a[52:55], v[216:219], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v6, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[108:111], a[36:39], v[220:223], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[124:127], a[52:55], v[220:223], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[96:99], a[40:43], v[224:227], v70, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[8:11], v12 offset:8448
	v_perm_b32 v74, v6, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[112:115], a[56:59], v[224:227], v68, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v2, v3
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[100:103], a[40:43], v[228:231], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v3, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[116:119], a[56:59], v[228:231], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[104:107], a[40:43], v[232:235], v70, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[12:15], v12 offset:12672
	v_perm_b32 v2, v2, v3, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[120:123], a[56:59], v[232:235], v68, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[108:111], a[40:43], v[236:239], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[124:127], a[56:59], v[236:239], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[96:99], a[44:47], v[240:243], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[16:19], v12 offset:64
	v_perm_b32 v71, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[112:115], a[60:63], v[240:243], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v2, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[100:103], a[44:47], v[244:247], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v7, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[116:119], a[60:63], v[244:247], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[104:107], a[44:47], v[248:251], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[20:23], v12 offset:4288
	v_perm_b32 v2, v2, v7, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[120:123], a[60:63], v[248:251], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[108:111], a[44:47], v[252:255], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	v_perm_b32 v73, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[124:127], a[60:63], v[252:255], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[64:67], a[0:3], a[128:131], v74, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[32:35], v12 offset:16896
	v_mov_b32_e32 v2, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[80:83], a[16:19], a[128:131], v73, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[36:39], v12 offset:21120
	v_permlane16_swap_b32_e64 v4, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[68:71], a[0:3], a[132:135], v74, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[40:43], v12 offset:25344
	v_perm_b32 v2, v2, v4, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[84:87], a[16:19], a[132:135], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[44:47], v12 offset:29568
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[72:75], a[0:3], a[136:139], v74, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[48:51], v12 offset:16960
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[88:91], a[16:19], a[136:139], v73, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[52:55], v12 offset:21184
	v_perm_b32 v69, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[76:79], a[0:3], a[140:143], v74, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[56:59], v12 offset:25408
	v_mov_b32_e32 v2, v8
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[92:95], a[16:19], a[140:143], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[60:63], v12 offset:29632
	v_permlane16_swap_b32_e64 v8, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[64:67], a[4:7], a[144:147], v74, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v10 offset:16896
	v_perm_b32 v2, v2, v8, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(9)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[80:83], a[20:23], a[144:147], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v10 offset:21120
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[68:71], a[4:7], a[148:151], v74, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[104:107], v10 offset:25344
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[84:87], a[20:23], a[148:151], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v10 offset:29568
	v_perm_b32 v70, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[72:75], a[4:7], a[152:155], v74, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v10 offset:16960
	v_mov_b32_e32 v2, v5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[88:91], a[20:23], a[152:155], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v10 offset:21184
	v_permlane16_swap_b32_e64 v5, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[76:79], a[4:7], a[156:159], v74, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[120:123], v10 offset:25408
	v_perm_b32 v2, v2, v5, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[92:95], a[20:23], a[156:159], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v10 offset:29632
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[64:67], a[8:11], a[160:163], v74, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[80:83], a[24:27], a[160:163], v73, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v67, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[68:71], a[8:11], a[164:167], v74, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v2, v9
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[84:87], a[24:27], a[164:167], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v9, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[72:75], a[8:11], a[168:171], v74, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v2, v2, v9, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[88:91], a[24:27], a[168:171], v73, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[76:79], a[8:11], a[172:175], v74, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[92:95], a[24:27], a[172:175], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v68, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[64:67], a[12:15], a[176:179], v74, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[80:83], a[28:31], a[176:179], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[68:71], a[12:15], a[180:183], v74, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[84:87], a[28:31], a[180:183], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[72:75], a[12:15], a[184:187], v74, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[88:91], a[28:31], a[184:187], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[76:79], a[12:15], a[188:191], v74, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[92:95], a[28:31], a[188:191], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
.LBB0_16:
	s_waitcnt lgkmcnt(14)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[64:67], a[32:35], a[192:195], v74, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mul_i32 s4, s35, s3
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 1
	s_waitcnt lgkmcnt(11)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[80:83], a[48:51], a[192:195], v73, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_u32 s3, s0, s4
	s_mul_i32 s0, s18, s10
	s_addc_u32 s4, s1, s5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[68:71], a[32:35], a[196:199], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s5, s3, s0
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[84:87], a[48:51], a[196:199], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_addc_u32 s4, s4, s1
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[0:1], s[2:3], 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[72:75], a[32:35], a[200:203], v74, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_add_u32 s0, s5, s0
	s_addc_u32 s1, s4, s1
	v_accvgpr_read_b32 v62, a128
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[88:91], a[48:51], a[200:203], v73, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v63, a129
	v_accvgpr_read_b32 v64, a130
	v_accvgpr_read_b32 v65, a131
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[76:79], a[32:35], a[204:207], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v58, a132
	v_accvgpr_read_b32 v59, a133
	v_accvgpr_read_b32 v60, a134
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[92:95], a[48:51], a[204:207], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v61, a135
	v_accvgpr_read_b32 v54, a136
	v_accvgpr_read_b32 v55, a137
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[64:67], a[36:39], a[208:211], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v56, a138
	v_accvgpr_read_b32 v57, a139
	v_accvgpr_read_b32 v50, a140
	s_waitcnt lgkmcnt(10)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[80:83], a[52:55], a[208:211], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v51, a141
	v_accvgpr_read_b32 v52, a142
	v_accvgpr_read_b32 v53, a143
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[68:71], a[36:39], a[212:215], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v46, a144
	v_accvgpr_read_b32 v47, a145
	v_accvgpr_read_b32 v48, a146
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[84:87], a[52:55], a[212:215], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v49, a147
	v_accvgpr_read_b32 v42, a148
	v_accvgpr_read_b32 v43, a149
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[72:75], a[36:39], a[216:219], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v44, a150
	v_accvgpr_read_b32 v45, a151
	v_accvgpr_read_b32 v38, a152
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[88:91], a[52:55], a[216:219], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v39, a153
	v_accvgpr_read_b32 v40, a154
	v_accvgpr_read_b32 v41, a155
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[76:79], a[36:39], a[220:223], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v34, a156
	v_accvgpr_read_b32 v35, a157
	v_accvgpr_read_b32 v36, a158
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[92:95], a[52:55], a[220:223], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v37, a159
	v_accvgpr_read_b32 v30, a160
	v_accvgpr_read_b32 v31, a161
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[64:67], a[40:43], a[224:227], v74, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v32, a162
	v_accvgpr_read_b32 v33, a163
	v_accvgpr_read_b32 v26, a164
	s_waitcnt lgkmcnt(9)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[80:83], a[56:59], a[224:227], v73, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v27, a165
	v_accvgpr_read_b32 v28, a166
	v_accvgpr_read_b32 v29, a167
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[68:71], a[40:43], a[228:231], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v22, a168
	v_accvgpr_read_b32 v23, a169
	v_accvgpr_read_b32 v24, a170
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[84:87], a[56:59], a[228:231], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v25, a171
	v_accvgpr_read_b32 v18, a172
	v_accvgpr_read_b32 v19, a173
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[72:75], a[40:43], a[232:235], v74, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v20, a174
	v_accvgpr_read_b32 v21, a175
	v_accvgpr_read_b32 v14, a176
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[88:91], a[56:59], a[232:235], v73, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v15, a177
	v_accvgpr_read_b32 v16, a178
	v_accvgpr_read_b32 v17, a179
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[76:79], a[40:43], a[236:239], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v10, a180
	v_accvgpr_read_b32 v11, a181
	v_accvgpr_read_b32 v12, a182
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[92:95], a[56:59], a[236:239], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v13, a183
	v_accvgpr_read_b32 v6, a184
	v_accvgpr_read_b32 v7, a185
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[64:67], a[44:47], a[240:243], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v8, a186
	v_accvgpr_read_b32 v9, a187
	v_accvgpr_read_b32 v2, a188
	s_waitcnt lgkmcnt(8)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[80:83], a[60:63], a[240:243], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v3, a189
	v_accvgpr_read_b32 v4, a190
	v_accvgpr_read_b32 v5, a191
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[68:71], a[44:47], a[244:247], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[84:87], a[60:63], a[244:247], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v83, a201
	v_accvgpr_read_b32 v84, a200
	v_accvgpr_read_b32 v85, a203
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[72:75], a[44:47], a[248:251], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v87, a202
	v_accvgpr_read_b32 v86, a205
	v_accvgpr_read_b32 v88, a204
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[88:91], a[60:63], a[248:251], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v89, a207
	v_accvgpr_read_b32 v91, a206
	v_accvgpr_read_b32 v90, a209
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[76:79], a[44:47], a[252:255], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v92, a208
	v_accvgpr_read_b32 v93, a211
	v_accvgpr_read_b32 v94, a210
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[92:95], a[60:63], a[252:255], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v95, a213
	v_accvgpr_read_b32 v96, a212
	v_accvgpr_read_b32 v97, a215
	v_accvgpr_read_b32 v99, a214
	v_accvgpr_read_b32 v98, a217
	v_accvgpr_read_b32 v100, a216
	v_accvgpr_read_b32 v101, a219
	v_accvgpr_read_b32 v103, a218
	v_accvgpr_read_b32 v102, a221
	v_accvgpr_read_b32 v104, a220
	v_accvgpr_read_b32 v105, a223
	v_accvgpr_read_b32 v106, a222
	v_accvgpr_read_b32 v107, a225
	v_accvgpr_read_b32 v108, a224
	v_accvgpr_read_b32 v109, a227
	v_accvgpr_read_b32 v111, a226
	v_accvgpr_read_b32 v110, a229
	v_accvgpr_read_b32 v112, a228
	v_accvgpr_read_b32 v113, a231
	v_accvgpr_read_b32 v115, a230
	v_accvgpr_read_b32 v114, a233
	v_accvgpr_read_b32 v116, a232
	v_accvgpr_read_b32 v117, a235
	v_accvgpr_read_b32 v118, a234
	v_accvgpr_read_b32 v119, a237
	v_accvgpr_read_b32 v120, a236
	v_accvgpr_read_b32 v121, a239
	v_accvgpr_read_b32 v123, a238
	v_accvgpr_read_b32 v122, a241
	v_accvgpr_read_b32 v124, a240
	v_accvgpr_read_b32 v125, a243
	v_accvgpr_read_b32 v126, a242
	v_accvgpr_read_b32 v127, a245
	v_accvgpr_read_b32 v82, a244
	v_accvgpr_read_b32 v81, a247
	v_accvgpr_read_b32 v79, a246
	v_accvgpr_read_b32 v80, a249
	v_accvgpr_read_b32 v78, a248
	v_accvgpr_read_b32 v77, a251
	v_accvgpr_read_b32 v74, a250
	v_accvgpr_read_b32 v76, a253
	v_accvgpr_read_b32 v75, a252
	v_accvgpr_read_b32 v73, a255
	v_accvgpr_read_b32 v66, a254
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[96:99], a[0:3], v[128:131], v70, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(3)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[112:115], a[16:19], v[128:131], v68, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[100:103], a[0:3], v[132:135], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[116:119], a[16:19], v[132:135], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[104:107], a[0:3], v[136:139], v70, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(1)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[120:123], a[16:19], v[136:139], v68, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[108:111], a[0:3], v[140:143], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	s_waitcnt lgkmcnt(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[124:127], a[16:19], v[140:143], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[96:99], a[4:7], v[144:147], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[112:115], a[20:23], v[144:147], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[100:103], a[4:7], v[148:151], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[116:119], a[20:23], v[148:151], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[104:107], a[4:7], v[152:155], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[120:123], a[20:23], v[152:155], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[108:111], a[4:7], v[156:159], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[124:127], a[20:23], v[156:159], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[96:99], a[8:11], v[160:163], v70, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[112:115], a[24:27], v[160:163], v68, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[100:103], a[8:11], v[164:167], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[116:119], a[24:27], v[164:167], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[104:107], a[8:11], v[168:171], v70, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[120:123], a[24:27], v[168:171], v68, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[108:111], a[8:11], v[172:175], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[124:127], a[24:27], v[172:175], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[96:99], a[12:15], v[176:179], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[112:115], a[28:31], v[176:179], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[100:103], a[12:15], v[180:183], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[116:119], a[28:31], v[180:183], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[104:107], a[12:15], v[184:187], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[120:123], a[28:31], v[184:187], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[108:111], a[12:15], v[188:191], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[124:127], a[28:31], v[188:191], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[96:99], a[32:35], v[192:195], v70, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[112:115], a[48:51], v[192:195], v68, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[100:103], a[32:35], v[196:199], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[116:119], a[48:51], v[196:199], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[104:107], a[32:35], v[200:203], v70, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[120:123], a[48:51], v[200:203], v68, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[108:111], a[32:35], v[204:207], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[124:127], a[48:51], v[204:207], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[96:99], a[36:39], v[208:211], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[112:115], a[52:55], v[208:211], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[100:103], a[36:39], v[212:215], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[116:119], a[52:55], v[212:215], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[104:107], a[36:39], v[216:219], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[120:123], a[52:55], v[216:219], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[108:111], a[36:39], v[220:223], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[124:127], a[52:55], v[220:223], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[96:99], a[40:43], v[224:227], v70, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[112:115], a[56:59], v[224:227], v68, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[100:103], a[40:43], v[228:231], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[116:119], a[56:59], v[228:231], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[104:107], a[40:43], v[232:235], v70, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[120:123], a[56:59], v[232:235], v68, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[108:111], a[40:43], v[236:239], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[124:127], a[56:59], v[236:239], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[96:99], a[44:47], v[240:243], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[112:115], a[60:63], v[240:243], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[100:103], a[44:47], v[244:247], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[116:119], a[60:63], v[244:247], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[104:107], a[44:47], v[248:251], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[120:123], a[60:63], v[248:251], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[108:111], a[44:47], v[252:255], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[124:127], a[60:63], v[252:255], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_and_b32_e32 v67, 15, v0
	s_mulk_i32 s7, 0x1080
	v_mul_u32_u24_e32 v67, 0x108, v67
	v_lshlrev_b32_e32 v1, 2, v1
	v_add3_u32 v1, v67, s7, v1
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_cvt_pk_bf16_f32 v65, v64, v65
	v_cvt_pk_bf16_f32 v64, v62, v63
	v_add_lshl_u32 v62, v1, s11, 1
	v_cvt_pk_bf16_f32 v61, v60, v61
	v_cvt_pk_bf16_f32 v60, v58, v59
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	v_cvt_pk_bf16_f32 v53, v52, v53
	v_cvt_pk_bf16_f32 v52, v50, v51
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	v_cvt_pk_bf16_f32 v45, v44, v45
	v_cvt_pk_bf16_f32 v44, v42, v43
	v_add_u32_e32 v42, 0x4000, v62
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	v_cvt_pk_bf16_f32 v37, v36, v37
	v_cvt_pk_bf16_f32 v36, v34, v35
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	v_cvt_pk_bf16_f32 v29, v28, v29
	v_cvt_pk_bf16_f32 v28, v26, v27
	v_add_u32_e32 v26, 0x8000, v62
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	v_cvt_pk_bf16_f32 v21, v20, v21
	v_cvt_pk_bf16_f32 v20, v18, v19
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	v_cvt_pk_bf16_f32 v13, v12, v13
	v_cvt_pk_bf16_f32 v12, v10, v11
	v_add_u32_e32 v14, 0xc000, v62
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_cvt_pk_bf16_f32 v5, v4, v5
	v_cvt_pk_bf16_f32 v4, v2, v3
	ds_write2_b64 v62, v[64:65], v[60:61] offset1:8
	ds_write2_b64 v62, v[56:57], v[52:53] offset0:16 offset1:24
	ds_write2_b64 v42, v[48:49], v[44:45] offset0:64 offset1:72
	ds_write2_b64 v42, v[40:41], v[36:37] offset0:80 offset1:88
	ds_write2_b64 v26, v[32:33], v[28:29] offset0:128 offset1:136
	ds_write2_b64 v26, v[24:25], v[20:21] offset0:144 offset1:152
	ds_write2_b64 v14, v[16:17], v[12:13] offset0:192 offset1:200
	ds_write2_b64 v14, v[8:9], v[4:5] offset0:208 offset1:216
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_lshlrev_b32_e32 v1, 3, v0
	v_lshrrev_b32_e32 v3, 4, v0
	v_and_b32_e32 v2, 0x78, v1
	v_mul_u32_u24_e32 v1, 0x108, v3
	v_add_lshl_u32 v15, v2, v1, 1
	ds_read_b128 v[6:9], v15
	v_mad_u64_u32 v[4:5], s[4:5], s10, v3, v[2:3]
	v_bfe_u32 v3, v0, 4, 25
	v_or_b32_e32 v0, 16, v3
	v_mul_u32_u24_e32 v1, 0x108, v0
	v_add_lshl_u32 v17, v1, v2, 1
	v_mad_u64_u32 v[0:1], s[4:5], s10, v0, v[2:3]
	v_lshlrev_b32_e32 v16, 1, v4
	v_or_b32_e32 v1, 32, v3
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[6:9], v16, s[0:3], 0 offen nt
	ds_read_b128 v[6:9], v17
	v_mul_u32_u24_e32 v5, 0x108, v1
	v_add_lshl_u32 v18, v5, v2, 1
	ds_read_b128 v[20:23], v18
	v_lshlrev_b32_e32 v19, 1, v0
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[6:9], v19, s[0:3], 0 offen nt
	ds_read_b128 v[10:13], v15 offset:33792
	s_lshl_b32 s6, s10, 6
	v_mad_u64_u32 v[6:7], s[4:5], s10, v1, v[2:3]
	v_or_b32_e32 v1, 48, v3
	v_lshlrev_b32_e32 v7, 1, v6
	v_mul_u32_u24_e32 v5, 0x108, v1
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[20:23], v7, s[0:3], 0 offen nt
	v_mad_u64_u32 v[8:9], s[4:5], s10, v1, v[2:3]
	v_lshlrev_b32_e32 v9, 1, v8
	v_add_lshl_u32 v20, v5, v2, 1
	ds_read_b128 v[22:25], v20
	v_or_b32_e32 v1, 0x50, v3
	v_mul_u32_u24_e32 v5, 0x108, v1
	v_add_lshl_u32 v21, v5, v2, 1
	ds_read_b128 v[28:31], v21
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[22:25], v9, s[0:3], 0 offen nt
	s_nop 1
	v_add_u32_e32 v22, s6, v4
	v_lshlrev_b32_e32 v23, 1, v22
	v_mad_u64_u32 v[4:5], s[4:5], s10, v1, v[2:3]
	v_or_b32_e32 v1, 0x60, v3
	buffer_store_dwordx4 v[10:13], v23, s[0:3], 0 offen nt
	v_lshlrev_b32_e32 v5, 1, v4
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[28:31], v5, s[0:3], 0 offen nt
	v_mul_u32_u24_e32 v10, 0x108, v1
	v_add_lshl_u32 v24, v10, v2, 1
	v_mad_u64_u32 v[10:11], s[4:5], s10, v1, v[2:3]
	v_or_b32_e32 v1, 0x70, v3
	v_mul_u32_u24_e32 v3, 0x108, v1
	v_add_lshl_u32 v11, v3, v2, 1
	v_mad_u64_u32 v[2:3], s[4:5], s10, v1, v[2:3]
	v_accvgpr_read_b32 v1, a195
	v_accvgpr_read_b32 v3, a194
	v_cvt_pk_bf16_f32 v13, v3, v1
	v_accvgpr_read_b32 v1, a193
	v_accvgpr_read_b32 v3, a192
	v_cvt_pk_bf16_f32 v12, v3, v1
	v_add_u32_e32 v1, 0x10800, v62
	ds_read_b128 v[28:31], v24
	ds_read_b128 v[32:35], v11
	ds_write_b64 v1, v[12:13]
	v_accvgpr_read_b32 v1, a199
	v_accvgpr_read_b32 v3, a198
	v_cvt_pk_bf16_f32 v13, v3, v1
	v_accvgpr_read_b32 v1, a197
	v_accvgpr_read_b32 v3, a196
	v_cvt_pk_bf16_f32 v12, v3, v1
	v_add_u32_e32 v1, 0x10840, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v87, v85
	v_cvt_pk_bf16_f32 v12, v84, v83
	v_add_u32_e32 v1, 0x10880, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v91, v89
	v_cvt_pk_bf16_f32 v12, v88, v86
	v_add_u32_e32 v1, 0x108c0, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v94, v93
	v_cvt_pk_bf16_f32 v12, v92, v90
	v_add_u32_e32 v1, 0x14a00, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v99, v97
	v_cvt_pk_bf16_f32 v12, v96, v95
	v_add_u32_e32 v1, 0x14a40, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v103, v101
	v_cvt_pk_bf16_f32 v12, v100, v98
	v_add_u32_e32 v1, 0x14a80, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v106, v105
	v_cvt_pk_bf16_f32 v12, v104, v102
	v_add_u32_e32 v1, 0x14ac0, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v111, v109
	v_cvt_pk_bf16_f32 v12, v108, v107
	v_add_u32_e32 v1, 0x18c00, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v115, v113
	v_cvt_pk_bf16_f32 v12, v112, v110
	v_add_u32_e32 v1, 0x18c40, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v118, v117
	v_cvt_pk_bf16_f32 v12, v116, v114
	v_add_u32_e32 v1, 0x18c80, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v123, v121
	v_cvt_pk_bf16_f32 v12, v120, v119
	v_add_u32_e32 v1, 0x18cc0, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v126, v125
	v_cvt_pk_bf16_f32 v12, v124, v122
	v_add_u32_e32 v1, 0x1ce00, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v79, v81
	v_cvt_pk_bf16_f32 v12, v82, v127
	v_add_u32_e32 v1, 0x1ce40, v62
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v74, v77
	v_cvt_pk_bf16_f32 v12, v78, v80
	v_add_u32_e32 v1, 0x1ce80, v62
	v_lshlrev_b32_e32 v25, 1, v10
	v_lshlrev_b32_e32 v27, 1, v2
	ds_write_b64 v1, v[12:13]
	v_cvt_pk_bf16_f32 v13, v66, v73
	v_cvt_pk_bf16_f32 v12, v75, v76
	v_add_u32_e32 v1, 0x1cec0, v62
	s_waitcnt lgkmcnt(14)
	buffer_store_dwordx4 v[28:31], v25, s[0:3], 0 offen nt
	buffer_store_dwordx4 v[32:35], v27, s[0:3], 0 offen nt
	ds_write_b64 v1, v[12:13]
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v1, 0x10800, v15
	ds_read_b128 v[28:31], v1
	v_add_u32_e32 v1, s6, v22
	v_add_u32_e32 v12, 0x10800, v17
	v_lshlrev_b32_e32 v13, 1, v1
	ds_read_b128 v[32:35], v12
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[28:31], v13, s[0:3], 0 offen nt
	s_lshl_b32 s4, s10, 7
	v_add_lshl_u32 v22, v0, s4, 1
	v_add_u32_e32 v28, 0x10800, v18
	ds_read_b128 v[36:39], v28
	v_add_u32_e32 v29, 0x10800, v20
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[32:35], v22, s[0:3], 0 offen nt
	v_add_lshl_u32 v6, v6, s4, 1
	ds_read_b128 v[32:35], v29
	v_add_u32_e32 v30, 0x18c00, v15
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v6, s[0:3], 0 offen nt
	ds_read_b128 v[36:39], v30
	v_add_lshl_u32 v8, v8, s4, 1
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[32:35], v8, s[0:3], 0 offen nt
	v_add_lshl_u32 v31, v1, s6, 1
	v_cvt_pk_bf16_f32 v1, v130, v131
	v_add_u32_e32 v32, 0x10800, v21
	v_add_u32_e32 v33, 0x10800, v24
	v_add_u32_e32 v34, 0x10800, v11
	v_add_lshl_u32 v35, v2, s4, 1
	v_cvt_pk_bf16_f32 v0, v128, v129
	v_cvt_pk_bf16_f32 v3, v134, v135
	v_cvt_pk_bf16_f32 v2, v132, v133
	ds_read_b128 v[44:47], v32
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[36:39], v31, s[0:3], 0 offen nt
	ds_read_b128 v[36:39], v33
	ds_read_b128 v[48:51], v34
	ds_write2_b64 v62, v[0:1], v[2:3] offset0:32 offset1:40
	v_cvt_pk_bf16_f32 v1, v138, v139
	v_cvt_pk_bf16_f32 v0, v136, v137
	v_cvt_pk_bf16_f32 v3, v142, v143
	v_cvt_pk_bf16_f32 v2, v140, v141
	ds_write2_b64 v62, v[0:1], v[2:3] offset0:48 offset1:56
	v_cvt_pk_bf16_f32 v1, v146, v147
	v_cvt_pk_bf16_f32 v0, v144, v145
	v_cvt_pk_bf16_f32 v3, v150, v151
	v_cvt_pk_bf16_f32 v2, v148, v149
	ds_write2_b64 v42, v[0:1], v[2:3] offset0:96 offset1:104
	v_cvt_pk_bf16_f32 v1, v154, v155
	v_cvt_pk_bf16_f32 v0, v152, v153
	v_cvt_pk_bf16_f32 v3, v158, v159
	v_cvt_pk_bf16_f32 v2, v156, v157
	ds_write2_b64 v42, v[0:1], v[2:3] offset0:112 offset1:120
	v_cvt_pk_bf16_f32 v1, v162, v163
	v_cvt_pk_bf16_f32 v0, v160, v161
	v_cvt_pk_bf16_f32 v3, v166, v167
	v_cvt_pk_bf16_f32 v2, v164, v165
	ds_write2_b64 v26, v[0:1], v[2:3] offset0:160 offset1:168
	v_cvt_pk_bf16_f32 v1, v170, v171
	v_cvt_pk_bf16_f32 v0, v168, v169
	v_cvt_pk_bf16_f32 v3, v174, v175
	v_cvt_pk_bf16_f32 v2, v172, v173
	ds_write2_b64 v26, v[0:1], v[2:3] offset0:176 offset1:184
	v_cvt_pk_bf16_f32 v1, v178, v179
	v_cvt_pk_bf16_f32 v0, v176, v177
	v_cvt_pk_bf16_f32 v3, v182, v183
	v_cvt_pk_bf16_f32 v2, v180, v181
	v_add_lshl_u32 v4, v4, s4, 1
	v_add_lshl_u32 v10, v10, s4, 1
	ds_write2_b64 v14, v[0:1], v[2:3] offset0:224 offset1:232
	v_cvt_pk_bf16_f32 v1, v186, v187
	v_cvt_pk_bf16_f32 v0, v184, v185
	v_cvt_pk_bf16_f32 v3, v190, v191
	v_cvt_pk_bf16_f32 v2, v188, v189
	s_waitcnt lgkmcnt(9)
	buffer_store_dwordx4 v[44:47], v4, s[0:3], 0 offen nt
	s_waitcnt lgkmcnt(8)
	buffer_store_dwordx4 v[36:39], v10, s[0:3], 0 offen nt
	s_waitcnt lgkmcnt(7)
	buffer_store_dwordx4 v[48:51], v35, s[0:3], 0 offen nt
	ds_write2_b64 v14, v[0:1], v[2:3] offset0:240 offset1:248
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[0:3], v15 offset:256
	ds_read_b128 v[36:39], v17 offset:256
	ds_read_b128 v[40:43], v18 offset:256
	ds_read_b128 v[44:47], v15 offset:34048
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[0:3], v16, s[0:3], 0 offen offset:256 nt
	ds_read_b128 v[0:3], v20 offset:256
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[36:39], v19, s[0:3], 0 offen offset:256 nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[40:43], v7, s[0:3], 0 offen offset:256 nt
	ds_read_b128 v[16:19], v21 offset:256
	ds_read_b128 v[36:39], v24 offset:256
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[0:3], v9, s[0:3], 0 offen offset:256 nt
	ds_read_b128 v[40:43], v11 offset:256
	buffer_store_dwordx4 v[44:47], v23, s[0:3], 0 offen offset:256 nt
	v_cvt_pk_bf16_f32 v1, v194, v195
	v_cvt_pk_bf16_f32 v0, v192, v193
	v_add_u32_e32 v2, 0x10900, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v198, v199
	v_cvt_pk_bf16_f32 v0, v196, v197
	v_add_u32_e32 v2, 0x10940, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v202, v203
	v_cvt_pk_bf16_f32 v0, v200, v201
	v_add_u32_e32 v2, 0x10980, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v206, v207
	v_cvt_pk_bf16_f32 v0, v204, v205
	v_add_u32_e32 v2, 0x109c0, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v210, v211
	v_cvt_pk_bf16_f32 v0, v208, v209
	v_add_u32_e32 v2, 0x14b00, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v214, v215
	v_cvt_pk_bf16_f32 v0, v212, v213
	v_add_u32_e32 v2, 0x14b40, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v218, v219
	v_cvt_pk_bf16_f32 v0, v216, v217
	v_add_u32_e32 v2, 0x14b80, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v222, v223
	v_cvt_pk_bf16_f32 v0, v220, v221
	v_add_u32_e32 v2, 0x14bc0, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v226, v227
	v_cvt_pk_bf16_f32 v0, v224, v225
	v_add_u32_e32 v2, 0x18d00, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v230, v231
	v_cvt_pk_bf16_f32 v0, v228, v229
	v_add_u32_e32 v2, 0x18d40, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v234, v235
	v_cvt_pk_bf16_f32 v0, v232, v233
	v_add_u32_e32 v2, 0x18d80, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v238, v239
	v_cvt_pk_bf16_f32 v0, v236, v237
	v_add_u32_e32 v2, 0x18dc0, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v242, v243
	v_cvt_pk_bf16_f32 v0, v240, v241
	v_add_u32_e32 v2, 0x1cf00, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v246, v247
	v_cvt_pk_bf16_f32 v0, v244, v245
	v_add_u32_e32 v2, 0x1cf40, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v250, v251
	v_cvt_pk_bf16_f32 v0, v248, v249
	v_add_u32_e32 v2, 0x1cf80, v62
	ds_write_b64 v2, v[0:1]
	v_cvt_pk_bf16_f32 v1, v254, v255
	v_cvt_pk_bf16_f32 v0, v252, v253
	v_add_u32_e32 v2, 0x1cfc0, v62
	s_waitcnt lgkmcnt(14)
	buffer_store_dwordx4 v[16:19], v5, s[0:3], 0 offen offset:256 nt
	buffer_store_dwordx4 v[36:39], v25, s[0:3], 0 offen offset:256 nt
	buffer_store_dwordx4 v[40:43], v27, s[0:3], 0 offen offset:256 nt
	ds_write_b64 v2, v[0:1]
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v0, 0x10900, v15
	ds_read_b128 v[0:3], v0
	ds_read_b128 v[14:17], v12 offset:256
	ds_read_b128 v[18:21], v28 offset:256
	ds_read_b128 v[24:27], v34 offset:256
	ds_read_b128 v[36:39], v32 offset:256
	s_waitcnt lgkmcnt(4)
	buffer_store_dwordx4 v[0:3], v13, s[0:3], 0 offen offset:256 nt
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[14:17], v22, s[0:3], 0 offen offset:256 nt
	ds_read_b128 v[0:3], v29 offset:256
	ds_read_b128 v[12:15], v30 offset:256
	ds_read_b128 v[40:43], v33 offset:256
	s_waitcnt lgkmcnt(5)
	buffer_store_dwordx4 v[18:21], v6, s[0:3], 0 offen offset:256 nt
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[0:3], v8, s[0:3], 0 offen offset:256 nt
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[12:15], v31, s[0:3], 0 offen offset:256 nt
	buffer_store_dwordx4 v[36:39], v4, s[0:3], 0 offen offset:256 nt
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[40:43], v10, s[0:3], 0 offen offset:256 nt
	buffer_store_dwordx4 v[24:27], v35, s[0:3], 0 offen offset:256 nt
	s_endpgm
.LBB0_17:
	v_mov_b32_e32 v252, 0
	v_mov_b32_e32 v253, v252
	v_mov_b32_e32 v254, v252
	v_mov_b32_e32 v255, v252
	v_mov_b64_e32 v[248:249], v[252:253]
	v_mov_b64_e32 v[250:251], v[254:255]
	v_mov_b64_e32 v[244:245], v[252:253]
	v_mov_b64_e32 v[246:247], v[254:255]
	s_mov_b32 s6, 0
	v_mov_b64_e32 v[128:129], v[252:253]
	v_mov_b64_e32 v[130:131], v[254:255]
	v_mov_b64_e32 v[132:133], v[252:253]
	v_mov_b64_e32 v[134:135], v[254:255]
	v_mov_b64_e32 v[136:137], v[252:253]
	v_mov_b64_e32 v[138:139], v[254:255]
	v_mov_b64_e32 v[140:141], v[252:253]
	v_mov_b64_e32 v[142:143], v[254:255]
	v_mov_b64_e32 v[144:145], v[252:253]
	v_mov_b64_e32 v[146:147], v[254:255]
	v_mov_b64_e32 v[148:149], v[252:253]
	v_mov_b64_e32 v[150:151], v[254:255]
	v_mov_b64_e32 v[152:153], v[252:253]
	v_mov_b64_e32 v[154:155], v[254:255]
	v_mov_b64_e32 v[156:157], v[252:253]
	v_mov_b64_e32 v[158:159], v[254:255]
	v_mov_b64_e32 v[160:161], v[252:253]
	v_mov_b64_e32 v[162:163], v[254:255]
	v_mov_b64_e32 v[164:165], v[252:253]
	v_mov_b64_e32 v[166:167], v[254:255]
	v_mov_b64_e32 v[168:169], v[252:253]
	v_mov_b64_e32 v[170:171], v[254:255]
	v_mov_b64_e32 v[172:173], v[252:253]
	v_mov_b64_e32 v[174:175], v[254:255]
	v_mov_b64_e32 v[176:177], v[252:253]
	v_mov_b64_e32 v[178:179], v[254:255]
	v_mov_b64_e32 v[180:181], v[252:253]
	v_mov_b64_e32 v[182:183], v[254:255]
	v_mov_b64_e32 v[184:185], v[252:253]
	v_mov_b64_e32 v[186:187], v[254:255]
	v_mov_b64_e32 v[188:189], v[252:253]
	v_mov_b64_e32 v[190:191], v[254:255]
	v_accvgpr_write_b32 a192, v252
	v_accvgpr_write_b32 a193, v253
	v_accvgpr_write_b32 a194, v254
	v_accvgpr_write_b32 a195, v255
	v_accvgpr_write_b32 a196, v252
	v_accvgpr_write_b32 a197, v253
	v_accvgpr_write_b32 a198, v254
	v_accvgpr_write_b32 a199, v255
	v_accvgpr_write_b32 a200, v252
	v_accvgpr_write_b32 a201, v253
	v_accvgpr_write_b32 a202, v254
	v_accvgpr_write_b32 a203, v255
	v_accvgpr_write_b32 a204, v252
	v_accvgpr_write_b32 a205, v253
	v_accvgpr_write_b32 a206, v254
	v_accvgpr_write_b32 a207, v255
	v_accvgpr_write_b32 a208, v252
	v_accvgpr_write_b32 a209, v253
	v_accvgpr_write_b32 a210, v254
	v_accvgpr_write_b32 a211, v255
	v_accvgpr_write_b32 a212, v252
	v_accvgpr_write_b32 a213, v253
	v_accvgpr_write_b32 a214, v254
	v_accvgpr_write_b32 a215, v255
	v_accvgpr_write_b32 a216, v252
	v_accvgpr_write_b32 a217, v253
	v_accvgpr_write_b32 a218, v254
	v_accvgpr_write_b32 a219, v255
	v_accvgpr_write_b32 a220, v252
	v_accvgpr_write_b32 a221, v253
	v_accvgpr_write_b32 a222, v254
	v_accvgpr_write_b32 a223, v255
	v_accvgpr_write_b32 a224, v252
	v_accvgpr_write_b32 a225, v253
	v_accvgpr_write_b32 a226, v254
	v_accvgpr_write_b32 a227, v255
	v_accvgpr_write_b32 a228, v252
	v_accvgpr_write_b32 a229, v253
	v_accvgpr_write_b32 a230, v254
	v_accvgpr_write_b32 a231, v255
	v_accvgpr_write_b32 a232, v252
	v_accvgpr_write_b32 a233, v253
	v_accvgpr_write_b32 a234, v254
	v_accvgpr_write_b32 a235, v255
	v_accvgpr_write_b32 a236, v252
	v_accvgpr_write_b32 a237, v253
	v_accvgpr_write_b32 a238, v254
	v_accvgpr_write_b32 a239, v255
	v_accvgpr_write_b32 a240, v252
	v_accvgpr_write_b32 a241, v253
	v_accvgpr_write_b32 a242, v254
	v_accvgpr_write_b32 a243, v255
	v_accvgpr_write_b32 a244, v252
	v_accvgpr_write_b32 a245, v253
	v_accvgpr_write_b32 a246, v254
	v_accvgpr_write_b32 a247, v255
	v_accvgpr_write_b32 a248, v252
	v_accvgpr_write_b32 a249, v253
	v_accvgpr_write_b32 a250, v254
	v_accvgpr_write_b32 a251, v255
	v_accvgpr_write_b32 a252, v252
	v_accvgpr_write_b32 a253, v253
	v_accvgpr_write_b32 a254, v254
	v_accvgpr_write_b32 a255, v255
	v_mov_b64_e32 v[192:193], v[252:253]
	v_mov_b64_e32 v[194:195], v[254:255]
	v_mov_b64_e32 v[196:197], v[252:253]
	v_mov_b64_e32 v[198:199], v[254:255]
	v_mov_b64_e32 v[200:201], v[252:253]
	v_mov_b64_e32 v[202:203], v[254:255]
	v_mov_b64_e32 v[204:205], v[252:253]
	v_mov_b64_e32 v[206:207], v[254:255]
	v_mov_b64_e32 v[208:209], v[252:253]
	v_mov_b64_e32 v[210:211], v[254:255]
	v_mov_b64_e32 v[212:213], v[252:253]
	v_mov_b64_e32 v[214:215], v[254:255]
	v_mov_b64_e32 v[216:217], v[252:253]
	v_mov_b64_e32 v[218:219], v[254:255]
	v_mov_b64_e32 v[220:221], v[252:253]
	v_mov_b64_e32 v[222:223], v[254:255]
	v_mov_b64_e32 v[224:225], v[252:253]
	v_mov_b64_e32 v[226:227], v[254:255]
	v_mov_b64_e32 v[228:229], v[252:253]
	v_mov_b64_e32 v[230:231], v[254:255]
	v_mov_b64_e32 v[232:233], v[252:253]
	v_mov_b64_e32 v[234:235], v[254:255]
	v_mov_b64_e32 v[236:237], v[252:253]
	v_mov_b64_e32 v[238:239], v[254:255]
	v_mov_b64_e32 v[240:241], v[252:253]
	v_mov_b64_e32 v[242:243], v[254:255]
	s_add_i32 s4, s8, 2
	s_cmp_ge_i32 s4, s9
	s_cbranch_scc1 .LBB0_14
.LBB0_18:
	s_lshl_b32 s4, s8, 7
	s_add_i32 s28, s9, -2
	s_add_i32 s29, s4, 0x100
	s_lshl_b32 s30, s8, 3
	s_mov_b32 s26, s22
	s_mov_b32 s27, s23
	s_branch .LBB0_20
.LBB0_19:                               ;   in Loop: Header=BB0_20 Depth=1
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[96:99], a[32:35], v[192:195], v70, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v27, s31, v14
	ds_read_b128 a[24:27], v27 offset:8512
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[192:195], a[112:115], a[48:51], v[192:195], v68, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(10)
	v_mov_b32_e32 v28, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[100:103], a[32:35], v[196:199], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v2, v28 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[196:199], a[116:119], a[48:51], v[196:199], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[104:107], a[32:35], v[200:203], v70, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[28:31], v27 offset:12736
	v_perm_b32 v2, v28, v2, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[200:203], a[120:123], a[48:51], v[200:203], v68, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v28, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[108:111], a[32:35], v[204:207], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v28 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[204:207], a[124:127], a[48:51], v[204:207], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[96:99], a[36:39], v[208:211], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[0:3], v27
	v_perm_b32 v72, v28, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[208:211], a[112:115], a[52:55], v[208:211], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(11)
	v_mov_b32_e32 v2, v6
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[100:103], a[36:39], v[212:215], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v6, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[212:215], a[116:119], a[52:55], v[212:215], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[104:107], a[36:39], v[216:219], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[4:7], v27 offset:4224
	v_perm_b32 v2, v2, v6, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[216:219], a[120:123], a[52:55], v[216:219], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v6, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[108:111], a[36:39], v[220:223], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v6 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[220:223], a[124:127], a[52:55], v[220:223], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[96:99], a[40:43], v[224:227], v70, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[8:11], v27 offset:8448
	v_perm_b32 v74, v6, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[224:227], a[112:115], a[56:59], v[224:227], v68, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v2, v3
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[100:103], a[40:43], v[228:231], v70, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v3, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[228:231], a[116:119], a[56:59], v[228:231], v68, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[104:107], a[40:43], v[232:235], v70, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[12:15], v27 offset:12672
	v_perm_b32 v2, v2, v3, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[232:235], a[120:123], a[56:59], v[232:235], v68, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[108:111], a[40:43], v[236:239], v70, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[236:239], a[124:127], a[56:59], v[236:239], v68, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[96:99], a[44:47], v[240:243], v70, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[16:19], v27 offset:64
	v_perm_b32 v71, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[240:243], a[112:115], a[60:63], v[240:243], v68, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v2, v7
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[100:103], a[44:47], v[244:247], v70, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v7, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[244:247], a[116:119], a[60:63], v[244:247], v68, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[104:107], a[44:47], v[248:251], v70, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[20:23], v27 offset:4288
	v_perm_b32 v2, v2, v7, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[248:251], a[120:123], a[60:63], v[248:251], v68, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[108:111], a[44:47], v[252:255], v70, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	v_perm_b32 v73, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[252:255], a[124:127], a[60:63], v[252:255], v68, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(5)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[64:67], a[0:3], a[128:131], v74, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[32:35], v27 offset:16896
	v_mov_b32_e32 v2, v4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(2)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[128:131], a[80:83], a[16:19], a[128:131], v73, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[36:39], v27 offset:21120
	v_permlane16_swap_b32_e64 v4, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[68:71], a[0:3], a[132:135], v74, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[40:43], v27 offset:25344
	v_perm_b32 v2, v2, v4, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[84:87], a[16:19], a[132:135], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[44:47], v27 offset:29568
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[72:75], a[0:3], a[136:139], v74, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[48:51], v27 offset:16960
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[88:91], a[16:19], a[136:139], v73, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[52:55], v27 offset:21184
	v_perm_b32 v69, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[76:79], a[0:3], a[140:143], v74, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[56:59], v27 offset:25408
	v_mov_b32_e32 v2, v8
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[92:95], a[16:19], a[140:143], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[60:63], v27 offset:29632
	v_permlane16_swap_b32_e64 v8, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[64:67], a[4:7], a[144:147], v74, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v26 offset:16896
	v_perm_b32 v2, v2, v8, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	s_waitcnt lgkmcnt(9)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[80:83], a[20:23], a[144:147], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v26 offset:21120
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[68:71], a[4:7], a[148:151], v74, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[104:107], v26 offset:25344
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[84:87], a[20:23], a[148:151], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v26 offset:29568
	v_perm_b32 v70, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[72:75], a[4:7], a[152:155], v74, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v26 offset:16960
	v_mov_b32_e32 v2, v5
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[88:91], a[20:23], a[152:155], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v26 offset:21184
	v_permlane16_swap_b32_e64 v5, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[76:79], a[4:7], a[156:159], v74, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[120:123], v26 offset:25408
	v_perm_b32 v2, v2, v5, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[92:95], a[20:23], a[156:159], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v26 offset:29632
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[64:67], a[8:11], a[160:163], v74, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[80:83], a[24:27], a[160:163], v73, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_perm_b32 v67, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[68:71], a[8:11], a[164:167], v74, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v2, v9
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[84:87], a[24:27], a[164:167], v73, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane16_swap_b32_e64 v9, v2 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[72:75], a[8:11], a[168:171], v74, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v2, v2, v9, v11
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[88:91], a[24:27], a[168:171], v73, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v3, v2
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[76:79], a[8:11], a[172:175], v74, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	s_nop 0
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[92:95], a[24:27], a[172:175], v73, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_perm_b32 v68, v3, v2, v13
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000002) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[64:67], a[12:15], a[176:179], v74, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[80:83], a[28:31], a[176:179], v73, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[68:71], a[12:15], a[180:183], v74, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[84:87], a[28:31], a[180:183], v73, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[72:75], a[12:15], a[184:187], v74, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[88:91], a[28:31], a[184:187], v73, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[76:79], a[12:15], a[188:191], v74, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[92:95], a[28:31], a[188:191], v73, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_add_i32 s8, s8, 1
	s_addk_i32 s29, 0x80
	s_add_i32 s30, s30, 8
	s_cmp_lg_u32 s28, s8
	s_cbranch_scc0 .LBB0_24
.LBB0_20:                               ; =>This Inner Loop Header: Depth=1
	s_and_b32 s4, s8, 1
	s_lshl_b32 s5, s4, 1
	s_xor_b32 s31, s5, 2
	s_mulk_i32 s31, 0x4200
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[64:67], a[32:35], a[192:195], v74, v69 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_mul_i32 s36, s4, 0x8400
	s_add_i32 s5, s16, s36
	s_mov_b32 m0, s5
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[80:83], a[48:51], a[192:195], v73, v67 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v10, s[20:23], s29 offen lds
	s_add_i32 m0, s5, 0x1080
	s_add_i32 s36, s17, s36
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[68:71], a[32:35], a[196:199], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_i32 s36, s36, 0x10800
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[84:87], a[48:51], a[196:199], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[72:75], a[32:35], a[200:203], v74, v69 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[88:91], a[48:51], a[200:203], v73, v67 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v18, s[20:23], s29 offen lds
	s_add_i32 m0, s5, 0x2100
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[76:79], a[32:35], a[204:207], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[92:95], a[48:51], a[204:207], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[64:67], a[36:39], a[208:211], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[80:83], a[52:55], a[208:211], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v19, s[20:23], s29 offen lds
	s_add_i32 m0, s5, 0x3180
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[68:71], a[36:39], a[212:215], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[84:87], a[52:55], a[212:215], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[72:75], a[36:39], a[216:219], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[216:219], a[88:91], a[52:55], a[216:219], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v20, s[20:23], s29 offen lds
	s_mov_b32 m0, s36
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[76:79], a[36:39], a[220:223], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[220:223], a[92:95], a[52:55], a[220:223], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[64:67], a[40:43], a[224:227], v74, v69 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[224:227], a[80:83], a[56:59], a[224:227], v73, v67 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v12, s[24:27], s29 offen lds
	s_add_i32 m0, s36, 0x1080
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[68:71], a[40:43], a[228:231], v74, v69 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[228:231], a[84:87], a[56:59], a[228:231], v73, v67 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[72:75], a[40:43], a[232:235], v74, v69 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[232:235], a[88:91], a[56:59], a[232:235], v73, v67 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v21, s[24:27], s29 offen lds
	s_add_i32 m0, s36, 0x2100
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[76:79], a[40:43], a[236:239], v74, v69 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[236:239], a[92:95], a[56:59], a[236:239], v73, v67 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[64:67], a[44:47], a[240:243], v74, v69 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[240:243], a[80:83], a[60:63], a[240:243], v73, v67 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v22, s[24:27], s29 offen lds
	s_add_i32 m0, s36, 0x3180
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[68:71], a[44:47], a[244:247], v74, v69 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[244:247], a[84:87], a[60:63], a[244:247], v73, v67 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[72:75], a[44:47], a[248:251], v74, v69 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[248:251], a[88:91], a[60:63], a[248:251], v73, v67 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v23, s[24:27], s29 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[76:79], a[44:47], a[252:255], v74, v69 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[252:255], a[92:95], a[60:63], a[252:255], v73, v67 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(8)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_lshl_b32 s37, s6, 12
	s_and_b32 s38, s30, 8
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[96:99], a[0:3], v[128:131], v70, v72 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_or_b32 s38, s38, s37
	s_xor_b32 s37, s38, 8
	v_add_u32_e32 v2, s37, v15
	s_add_i32 s38, s19, s29
	s_add_i32 m0, s5, 0x4200
	ds_read2st64_b64 v[2:5], v2 offset1:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[128:131], a[112:115], a[16:19], v[128:131], v68, v71 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v10, s[20:23], s38 offen lds
	v_add_u32_e32 v6, s37, v16
	v_add_u32_e32 v26, s31, v25
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[100:103], a[0:3], v[132:135], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read2st64_b64 v[6:9], v6 offset1:4
	s_add_i32 m0, s5, 0x5280
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[132:135], a[116:119], a[16:19], v[132:135], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[104:107], a[0:3], v[136:139], v70, v72 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[64:67], v26
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[136:139], a[120:123], a[16:19], v[136:139], v68, v71 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v18, s[20:23], s38 offen lds
	s_add_i32 m0, s5, 0x6300
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[108:111], a[0:3], v[140:143], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[68:71], v26 offset:4224
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[140:143], a[124:127], a[16:19], v[140:143], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[96:99], a[4:7], v[144:147], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[72:75], v26 offset:8448
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[144:147], a[112:115], a[20:23], v[144:147], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v19, s[20:23], s38 offen lds
	s_add_i32 m0, s5, 0x7380
	s_add_i32 s5, s33, s29
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[100:103], a[4:7], v[148:151], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[76:79], v26 offset:12672
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[148:151], a[116:119], a[20:23], v[148:151], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[104:107], a[4:7], v[152:155], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[80:83], v26 offset:64
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[152:155], a[120:123], a[20:23], v[152:155], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v20, s[20:23], s38 offen lds
	s_add_i32 m0, s36, 0x4200
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[108:111], a[4:7], v[156:159], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[84:87], v26 offset:4288
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[156:159], a[124:127], a[20:23], v[156:159], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[96:99], a[8:11], v[160:163], v70, v72 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read_b128 a[88:91], v26 offset:8512
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[160:163], a[112:115], a[24:27], v[160:163], v68, v71 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v12, s[24:27], s5 offen lds
	s_add_i32 m0, s36, 0x5280
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[100:103], a[8:11], v[164:167], v70, v72 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[164:167], a[116:119], a[24:27], v[164:167], v68, v71 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[104:107], a[8:11], v[168:171], v70, v72 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	ds_read_b128 a[92:95], v26 offset:12736
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[168:171], a[120:123], a[24:27], v[168:171], v68, v71 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v21, s[24:27], s5 offen lds
	s_add_i32 m0, s36, 0x6300
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[108:111], a[8:11], v[172:175], v70, v72 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[172:175], a[124:127], a[24:27], v[172:175], v68, v71 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[96:99], a[12:15], v[176:179], v70, v72 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[176:179], a[112:115], a[28:31], v[176:179], v68, v71 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v22, s[24:27], s5 offen lds
	s_add_i32 m0, s36, 0x7380
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[100:103], a[12:15], v[180:183], v70, v72 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[180:183], a[116:119], a[28:31], v[180:183], v68, v71 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[104:107], a[12:15], v[184:187], v70, v72 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[184:187], a[120:123], a[28:31], v[184:187], v68, v71 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v23, s[24:27], s5 offen lds
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000820) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[108:111], a[12:15], v[188:191], v70, v72 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[188:191], a[124:127], a[28:31], v[188:191], v68, v71 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_cmp_eq_u32 s4, 0
	s_mov_b64 s[4:5], -1
	s_cbranch_scc1 .LBB0_22
; %bb.21:                               ;   in Loop: Header=BB0_20 Depth=1
	;;#ASMSTART
	; panel copy: sfa
	;;#ASMEND
	s_mov_b64 s[4:5], 0
.LBB0_22:                               ;   in Loop: Header=BB0_20 Depth=1
	s_and_b64 s[4:5], s[4:5], exec
	s_cselect_b32 s4, 1, 0
	s_cmp_lg_u32 s4, 1
	s_cbranch_scc1 .LBB0_19
; %bb.23:                               ;   in Loop: Header=BB0_20 Depth=1
	s_add_i32 s4, s6, 1
	s_cmp_lg_u32 s6, 2
	s_cselect_b32 s6, s4, 0
	s_lshl_b32 s4, s6, 12
	s_add_i32 s5, s30, 16
	s_add_i32 m0, s34, s4
	s_nop 0
	buffer_load_dwordx4 v24, s[12:15], s5 offen lds
	;;#ASMSTART
	; panel copy: sfb
	;;#ASMEND
	s_branch .LBB0_19
.LBB0_24:
	s_mov_b32 s8, s28
	s_add_i32 s4, s8, 1
	s_cmp_ge_i32 s4, s9
	s_cbranch_scc0 .LBB0_15
	s_branch .LBB0_16
.Lfunc_end0:
	.size	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs, .Lfunc_end0-_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
		.amdhsa_group_segment_fixed_size 159744
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 96
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 512
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.section	.text._Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,"axG",@progbits,_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,comdat
                                        ; -- End function
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.num_vgpr, 256
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.num_agpr, 256
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.numbered_sgpr, 72
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.num_named_barrier, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.private_seg_size, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.uses_vcc, 1
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.uses_flat_scratch, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.has_dyn_sized_stack, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.has_recursion, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 22348
; TotalNumSgprs: 78
; NumVgprs: 256
; NumAgprs: 256
; TotalNumVgprs: 512
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 159744 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 63
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 512
; AccumOffset: 256
; Occupancy: 1
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.set amdgpu.max_num_named_barrier, 0
	.section	.AMDGPU.csdata,"",@progbits
	.type	__hip_cuid_5cdb0af69680e2b7,@object ; @__hip_cuid_5cdb0af69680e2b7
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_5cdb0af69680e2b7
__hip_cuid_5cdb0af69680e2b7:
	.byte	0                               ; 0x0
	.size	__hip_cuid_5cdb0af69680e2b7, 1

	.ident	"clang version 24.0.0git (/root/workspace/llvm-src 49c41889681640665400cb01c9fbb4c0a024cde4)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_5cdb0af69680e2b7
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     256
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .group_segment_fixed_size: 159744
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
    .private_segment_fixed_size: 0
    .sgpr_count:     78
    .sgpr_spill_count: 0
    .symbol:         _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     512
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgpu9.50-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
