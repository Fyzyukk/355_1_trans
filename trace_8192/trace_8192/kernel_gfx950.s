	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,"axG",@progbits,_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,comdat
	.protected	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs ; -- Begin function _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
	.globl	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
	.p2align	8
	.type	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs,@function
_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs: ; @_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
; %bb.0:                                ; %_ZN4opus4impl10packed_vecINS_5fp4_tELi128EEC2Ev.exit
	s_load_dwordx8 s[4:11], s[0:1], 0x18
	s_abs_i32 s25, s2
	s_load_dwordx4 s[20:23], s[0:1], 0x0
	v_lshrrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v3, 14, v2
	s_waitcnt lgkmcnt(0)
	s_add_i32 s7, s5, 0xff
	s_ashr_i32 s12, s7, 31
	s_lshr_b32 s12, s12, 24
	s_add_i32 s7, s7, s12
	s_ashr_i32 s7, s7, 8
	s_abs_i32 s24, s7
	v_cvt_f32_u32_e32 v1, s24
	s_sub_i32 s27, 0, s24
	s_xor_b32 s26, s2, s7
	s_ashr_i32 s26, s26, 31
	v_rcp_iflag_f32_e32 v1, v1
	s_mul_i32 s11, s11, s3
	s_load_dwordx2 s[34:35], s[0:1], 0x38
	s_load_dwordx8 s[12:19], s[0:1], 0x40
	v_lshlrev_b32_e32 v2, 4, v0
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_and_b32_e32 v2, 0x70, v2
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s30, s18, s3
	v_readfirstlane_b32 s28, v1
	s_mul_i32 s27, s27, s28
	s_mul_hi_u32 s27, s28, s27
	s_add_i32 s28, s28, s27
	s_mul_hi_u32 s27, s25, s28
	s_mul_i32 s28, s27, s24
	s_sub_i32 s25, s25, s28
	s_add_i32 s29, s27, 1
	s_sub_i32 s28, s25, s24
	s_cmp_ge_u32 s25, s24
	s_cselect_b32 s27, s29, s27
	s_cselect_b32 s25, s28, s25
	s_add_i32 s28, s27, 1
	s_cmp_ge_u32 s25, s24
	s_cselect_b32 s24, s28, s27
	s_xor_b32 s24, s24, s26
	s_sub_i32 s24, s24, s26
	s_mul_i32 s7, s24, s7
	s_sub_i32 s2, s2, s7
	v_readfirstlane_b32 s25, v0
	s_lshl_b32 s33, s24, 8
	s_lshl_b32 s2, s2, 8
	s_bfe_u32 s7, s25, 0x10006
	s_lshr_b32 s37, s25, 7
	s_ashr_i32 s24, s11, 31
	s_add_u32 s11, s20, s11
	s_addc_u32 s20, s21, s24
	s_mul_i32 s21, s33, s8
	s_ashr_i32 s26, s21, 31
	s_add_u32 s24, s11, s21
	s_mul_i32 s11, s34, s3
	s_addc_u32 s26, s20, s26
	s_ashr_i32 s20, s11, 31
	s_add_u32 s11, s22, s11
	s_mul_i32 s21, s2, s9
	s_addc_u32 s20, s23, s20
	s_ashr_i32 s22, s21, 31
	s_add_u32 s28, s11, s21
	s_addc_u32 s29, s20, s22
	s_sub_i32 s4, s4, s33
	s_mul_i32 s34, s33, s16
	s_mul_i32 s18, s4, s16
	s_mul_i32 s4, s19, s3
	s_ashr_i32 s31, s30, 31
	s_ashr_i32 s36, s34, 31
	s_ashr_i32 s11, s4, 31
	s_add_u32 s4, s14, s4
	s_mul_i32 s14, s2, s17
	s_addc_u32 s11, s15, s11
	s_ashr_i32 s15, s14, 31
	s_add_u32 s20, s4, s14
	s_addc_u32 s4, s11, s15
	s_lshl_b32 s11, s37, 4
	v_or_b32_e32 v4, s11, v3
	s_sub_i32 s5, s5, s2
	v_or_b32_e32 v4, s7, v4
	s_mul_i32 s22, s5, s17
	s_and_b32 s21, s4, 0xffff
	v_mad_u64_u32 v[254:255], s[4:5], v4, s8, v[2:3]
	s_mul_i32 s4, s8, 0x60
	v_and_b32_e32 v1, 63, v0
	v_add_u32_e32 v158, s4, v254
	s_lshl4_add_u32 s4, s7, s37
	v_add_u32_e32 v3, s4, v3
	v_mad_u64_u32 v[118:119], s[4:5], v3, s9, v[2:3]
	s_mul_i32 s4, s9, 0x60
	s_lshl_b32 s5, s7, 6
	v_add_u32_e32 v137, s4, v118
	s_and_b32 s4, s25, 0x80
	s_or_b32 s4, s4, s5
	v_or_b32_e32 v2, s4, v1
	s_lshl_b32 s4, s37, 11
	s_lshl_b32 s41, s7, 11
	s_and_b32 s25, s26, 0xffff
	s_add_u32 s5, s12, s30
	s_addc_u32 s12, s13, s31
	v_mul_lo_u32 v160, v2, s16
	s_add_u32 s16, s5, s34
	s_mul_i32 s39, s37, 0x840
	s_mul_i32 s14, s7, 0x420
	s_mov_b32 s19, 0x20000
	s_addc_u32 s5, s12, s36
	s_add_i32 s14, s14, s39
	s_mov_b32 s26, -1
	s_mov_b32 s27, s19
	s_lshl_b32 s40, s7, 10
	s_mov_b32 m0, s14
	s_mul_i32 s38, s7, 0x840
	v_lshl_add_u32 v146, s8, 5, v254
	s_or_b32 s40, s40, s4
	buffer_load_dwordx4 v254, s[24:27], 0 offen lds
	s_add_i32 m0, s14, 0x1080
	s_mul_i32 s4, s37, 0x420
	v_lshl_add_u32 v148, s8, 6, v254
	buffer_load_dwordx4 v146, s[24:27], 0 offen lds
	s_add_i32 m0, s14, 0x2100
	s_add_i32 s15, s38, s4
	buffer_load_dwordx4 v148, s[24:27], 0 offen lds
	s_add_i32 m0, s14, 0x3180
	s_add_i32 s4, s15, 0x10800
	s_and_b32 s29, s29, 0xffff
	s_mov_b32 s30, s26
	s_mov_b32 s31, s19
	buffer_load_dwordx4 v158, s[24:27], 0 offen lds
	s_mov_b32 m0, s4
	v_lshl_add_u32 v147, s9, 5, v118
	buffer_load_dwordx4 v118, s[28:31], 0 offen lds
	s_add_i32 m0, s4, 0x1080
	v_lshl_add_u32 v149, s9, 6, v118
	buffer_load_dwordx4 v147, s[28:31], 0 offen lds
	s_add_i32 m0, s4, 0x2100
	s_lshl_b32 s34, s8, 7
	buffer_load_dwordx4 v149, s[28:31], 0 offen lds
	s_add_i32 m0, s4, 0x3180
	s_lshl_b32 s36, s9, 7
	buffer_load_dwordx4 v137, s[28:31], 0 offen lds
	s_add_i32 m0, s14, 0x4200
	v_lshl_or_b32 v2, s37, 6, v1
	buffer_load_dwordx4 v254, s[24:27], s34 offen lds
	s_add_i32 m0, s14, 0x5280
	s_lshl_b32 s12, s37, 10
	buffer_load_dwordx4 v146, s[24:27], s34 offen lds
	s_add_i32 m0, s14, 0x6300
	v_lshl_add_u32 v2, s7, 7, v2
	buffer_load_dwordx4 v148, s[24:27], s34 offen lds
	s_add_i32 m0, s14, 0x7380
	s_add_i32 s41, s41, s12
	buffer_load_dwordx4 v158, s[24:27], s34 offen lds
	s_add_i32 m0, s4, 0x4200
	s_add_i32 s13, s40, 0x21000
	buffer_load_dwordx4 v118, s[28:31], s36 offen lds
	s_add_i32 m0, s4, 0x5280
	v_mul_lo_u32 v159, v2, s17
	buffer_load_dwordx4 v147, s[28:31], s36 offen lds
	s_add_i32 m0, s4, 0x6300
	s_and_b32 s17, s5, 0xffff
	buffer_load_dwordx4 v149, s[28:31], s36 offen lds
	s_add_i32 m0, s4, 0x7380
	s_add_i32 s12, s41, 0x23000
	buffer_load_dwordx4 v137, s[28:31], s36 offen lds
	s_mov_b32 m0, s13
	s_mov_b32 s23, s19
	buffer_load_dwordx4 v160, s[16:19], 0 offen lds
	s_mov_b32 m0, s12
	s_cmpk_gt_i32 s6, 0x2ff
	buffer_load_dwordx4 v159, s[20:23], 0 offen lds
	s_cselect_b64 s[4:5], -1, 0
	s_cmpk_lt_i32 s6, 0x300
	s_cbranch_scc0 .LBB0_9
; %bb.1:
	s_mov_b32 s27, s19
	s_cmpk_lt_i32 s6, 0x200
	s_mov_b64 s[8:9], -1
	s_cbranch_scc1 .LBB0_10
.LBB0_2:                                ; %Flow8661
	s_load_dwordx2 s[0:1], s[0:1], 0x10
	s_andn2_b64 vcc, exec, s[8:9]
	s_cbranch_vccnz .LBB0_4
.LBB0_3:
	s_add_i32 m0, s14, 0x8400
	s_movk_i32 s8, 0x80
	buffer_load_dwordx4 v254, s[24:27], s8 offen lds
	s_add_i32 m0, s14, 0x9480
	s_mov_b32 s30, s26
	buffer_load_dwordx4 v146, s[24:27], s8 offen lds
	s_add_i32 m0, s14, 0xa500
	s_mov_b32 s31, s27
	buffer_load_dwordx4 v148, s[24:27], s8 offen lds
	s_add_i32 m0, s14, 0xb580
	s_nop 0
	buffer_load_dwordx4 v158, s[24:27], s8 offen lds
	s_add_i32 m0, s15, 0x18c00
	s_nop 0
	buffer_load_dwordx4 v118, s[28:31], s8 offen lds
	s_add_i32 m0, s15, 0x19c80
	s_nop 0
	buffer_load_dwordx4 v147, s[28:31], s8 offen lds
	s_add_i32 m0, s15, 0x1ad00
	s_nop 0
	buffer_load_dwordx4 v149, s[28:31], s8 offen lds
	s_add_i32 m0, s15, 0x1bd80
	s_nop 0
	buffer_load_dwordx4 v137, s[28:31], s8 offen lds
	s_add_i32 s8, s34, 0x80
	s_add_i32 m0, s14, 0xc600
	s_nop 0
	buffer_load_dwordx4 v254, s[24:27], s8 offen lds
	s_add_i32 m0, s14, 0xd680
	s_nop 0
	buffer_load_dwordx4 v146, s[24:27], s8 offen lds
	s_add_i32 m0, s14, 0xe700
	s_nop 0
	buffer_load_dwordx4 v148, s[24:27], s8 offen lds
	s_add_i32 m0, s14, 0xf780
	s_nop 0
	buffer_load_dwordx4 v158, s[24:27], s8 offen lds
	s_add_i32 s8, s36, 0x80
	s_add_i32 m0, s15, 0x1ce00
	s_nop 0
	buffer_load_dwordx4 v118, s[28:31], s8 offen lds
	s_add_i32 m0, s15, 0x1de80
	s_nop 0
	buffer_load_dwordx4 v147, s[28:31], s8 offen lds
	s_add_i32 m0, s15, 0x1ef00
	s_nop 0
	buffer_load_dwordx4 v149, s[28:31], s8 offen lds
	s_add_i32 m0, s15, 0x1ff80
	s_nop 0
	buffer_load_dwordx4 v137, s[28:31], s8 offen lds
	s_waitcnt vmcnt(16)
.LBB0_4:
	v_and_b32_e32 v2, 1, v0
	s_ashr_i32 s31, s6, 31
	v_mov_b32_e32 v3, 0x420
	v_cmp_eq_u32_e32 vcc, 1, v2
	s_lshr_b32 s31, s31, 24
	v_lshrrev_b32_e32 v5, 4, v1
	v_cndmask_b32_e32 v2, 0, v3, vcc
	v_lshlrev_b32_e32 v3, 6, v0
	v_and_b32_e32 v4, 48, v0
	s_movk_i32 s9, 0x380
	v_accvgpr_write_b32 a128, v0
	v_and_b32_e32 v0, 15, v0
	s_add_i32 s6, s6, s31
	s_mov_b32 s8, 0
	v_and_or_b32 v2, v3, s9, v2
	v_accvgpr_write_b32 a129, v5
	v_accvgpr_write_b32 a130, v0
	s_mov_b32 s9, 24
	s_ashr_i32 s6, s6, 8
	v_add_u32_e32 v2, v2, v4
	v_lshlrev_b32_e32 v4, 9, v5
	s_lshl_b32 s23, s7, 8
	s_lshl_b32 s30, s37, 8
	v_lshlrev_b32_e32 v5, 4, v0
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v164, s38, v2
	v_add_u32_e32 v2, s39, v2
	v_add_u32_e32 v0, 0x10800, v2
	v_or3_b32 v2, v4, s23, v5
	v_or_b32_e32 v161, 0x21000, v2
	v_or_b32_e32 v6, 0x21800, v2
	s_waitcnt vmcnt(0)
	ds_read_b64 v[2:3], v161
	ds_read_b64 v[6:7], v6
	s_add_i32 s30, s30, 0x23000
	v_add_u32_e32 v4, s30, v4
	v_or_b32_e32 v165, v4, v5
	v_and_b32_e32 v5, 16, v1
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v8, 0x6020400
	v_mov_b32_e32 v9, 0x7030501
	v_cmp_ne_u32_e32 vcc, 0, v5
	v_lshrrev_b32_e32 v1, 5, v1
	s_mov_b32 s23, 0x2020202
	v_permlane16_swap_b32_e64 v2, v4 bound_ctrl:1
	v_cndmask_b32_e32 v136, v8, v9, vcc
	v_mul_lo_u32 v1, v1, s23
	v_perm_b32 v2, v4, v2, v136
	v_add_u32_e32 v135, 0x5040100, v1
	v_mov_b32_e32 v1, v3
	v_mov_b32_e32 v4, v2
	s_nop 0
	v_permlane16_swap_b32_e64 v3, v1 bound_ctrl:1
	v_permlane32_swap_b32_e64 v2, v4 bound_ctrl:1
	v_perm_b32 v1, v1, v3, v136
	v_perm_b32 v134, v4, v2, v135
	v_mov_b32_e32 v2, v1
	s_nop 1
	v_permlane32_swap_b32_e64 v1, v2 bound_ctrl:1
	ds_read_b128 a[20:23], v164
	ds_read_b128 a[28:31], v164 offset:64
	v_perm_b32 v129, v2, v1, v135
	v_mov_b32_e32 v1, v6
	s_nop 1
	v_permlane16_swap_b32_e64 v6, v1 bound_ctrl:1
	ds_read2st64_b64 v[2:5], v165 offset1:4
	v_perm_b32 v1, v1, v6, v136
	v_mov_b32_e32 v6, v1
	s_nop 1
	v_permlane32_swap_b32_e64 v1, v6 bound_ctrl:1
	ds_read_b128 a[12:15], v164 offset:4224
	ds_read_b128 a[24:27], v164 offset:4288
	v_perm_b32 v127, v6, v1, v135
	v_mov_b32_e32 v1, v7
	s_nop 1
	v_permlane16_swap_b32_e64 v7, v1 bound_ctrl:1
	ds_read_b128 a[8:11], v164 offset:8448
	ds_read_b128 a[16:19], v164 offset:8512
	v_perm_b32 v1, v1, v7, v136
	v_mov_b32_e32 v6, v1
	s_nop 1
	v_permlane32_swap_b32_e64 v1, v6 bound_ctrl:1
	ds_read_b128 a[0:3], v164 offset:12672
	ds_read_b128 a[4:7], v164 offset:12736
	v_perm_b32 v121, v6, v1, v135
	s_waitcnt lgkmcnt(6)
	v_mov_b32_e32 v1, v2
	s_nop 1
	v_permlane16_swap_b32_e64 v2, v1 bound_ctrl:1
	ds_read_b128 a[52:55], v0
	ds_read_b128 a[60:63], v0 offset:64
	v_perm_b32 v1, v1, v2, v136
	v_mov_b32_e32 v2, v1
	s_nop 1
	v_permlane32_swap_b32_e64 v1, v2 bound_ctrl:1
	ds_read_b128 a[48:51], v0 offset:4224
	ds_read_b128 a[56:59], v0 offset:4288
	v_perm_b32 v1, v2, v1, v135
	v_mov_b32_e32 v2, v3
	s_nop 1
	v_permlane16_swap_b32_e64 v3, v2 bound_ctrl:1
	ds_read_b128 a[36:39], v0 offset:8448
	ds_read_b128 a[44:47], v0 offset:8512
	v_perm_b32 v2, v2, v3, v136
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	ds_read_b128 a[32:35], v0 offset:12672
	ds_read_b128 a[40:43], v0 offset:12736
	v_perm_b32 v255, v3, v2, v135
	v_mov_b32_e32 v2, v4
	s_nop 1
	v_permlane16_swap_b32_e64 v4, v2 bound_ctrl:1
	v_perm_b32 v2, v2, v4, v136
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	v_perm_b32 v128, v3, v2, v135
	v_mov_b32_e32 v2, v5
	s_nop 1
	v_permlane16_swap_b32_e64 v5, v2 bound_ctrl:1
	v_perm_b32 v2, v2, v5, v136
	v_mov_b32_e32 v3, v2
	s_nop 1
	v_permlane32_swap_b32_e64 v2, v3 bound_ctrl:1
	v_perm_b32 v126, v3, v2, v135
	; sched_barrier mask(0x00000000)
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_11
; %bb.5:                                ; %.lr.ph
	v_accvgpr_write_b32 a188, 0
	v_accvgpr_read_b32 v5, a188
	s_max_i32 s4, s6, 3
	v_accvgpr_read_b32 v2, a188
	v_accvgpr_read_b32 v3, a188
	v_accvgpr_read_b32 v4, a188
	v_accvgpr_write_b32 a175, v5
	v_accvgpr_write_b32 a171, v5
	v_accvgpr_write_b32 a167, v5
	v_accvgpr_write_b32 a163, v5
	v_accvgpr_write_b32 a159, v5
	v_accvgpr_write_b32 a155, v5
	v_accvgpr_write_b32 a147, v5
	v_accvgpr_write_b32 a143, v5
	v_accvgpr_write_b32 a139, v5
	v_accvgpr_write_b32 a135, v5
	s_add_i32 s8, s4, -2
	s_lshl_b32 s37, s4, 7
	s_mov_b32 s38, 0
	s_movk_i32 s39, 0x100
	s_movk_i32 s40, 0x1800
	s_mov_b32 s30, s26
	s_mov_b32 s31, s27
	v_accvgpr_mov_b32 a189, a188
	v_accvgpr_mov_b32 a190, a188
	v_accvgpr_mov_b32 a191, a188
	v_accvgpr_mov_b32 a204, a188
	v_accvgpr_mov_b32 a205, a188
	v_accvgpr_mov_b32 a206, a188
	v_accvgpr_mov_b32 a207, a188
	v_accvgpr_mov_b32 a208, a188
	v_accvgpr_mov_b32 a209, a188
	v_accvgpr_mov_b32 a210, a188
	v_accvgpr_mov_b32 a211, a188
	v_accvgpr_mov_b32 a212, a188
	v_accvgpr_mov_b32 a213, a188
	v_accvgpr_mov_b32 a214, a188
	v_accvgpr_mov_b32 a215, a188
	v_accvgpr_write_b32 a174, v4
	v_accvgpr_write_b32 a173, v3
	v_accvgpr_write_b32 a172, v2
	v_accvgpr_write_b32 a170, v4
	v_accvgpr_write_b32 a169, v3
	v_accvgpr_write_b32 a168, v2
	v_accvgpr_write_b32 a166, v4
	v_accvgpr_write_b32 a165, v3
	v_accvgpr_write_b32 a164, v2
	v_accvgpr_write_b32 a162, v4
	v_accvgpr_write_b32 a161, v3
	v_accvgpr_write_b32 a160, v2
	v_accvgpr_write_b32 a158, v4
	v_accvgpr_write_b32 a157, v3
	v_accvgpr_write_b32 a156, v2
	v_accvgpr_write_b32 a154, v4
	v_accvgpr_write_b32 a153, v3
	v_accvgpr_write_b32 a152, v2
	v_accvgpr_mov_b32 a180, a188
	v_accvgpr_mov_b32 a181, a188
	v_accvgpr_mov_b32 a182, a188
	v_accvgpr_mov_b32 a183, a188
	v_accvgpr_write_b32 a146, v4
	v_accvgpr_write_b32 a145, v3
	v_accvgpr_write_b32 a144, v2
	v_accvgpr_write_b32 a142, v4
	v_accvgpr_write_b32 a141, v3
	v_accvgpr_write_b32 a140, v2
	v_accvgpr_write_b32 a138, v4
	v_accvgpr_write_b32 a137, v3
	v_accvgpr_write_b32 a136, v2
	v_accvgpr_write_b32 a134, v4
	v_accvgpr_write_b32 a133, v3
	v_accvgpr_write_b32 a132, v2
	v_accvgpr_mov_b32 a184, a188
	v_accvgpr_mov_b32 a185, a188
	v_accvgpr_mov_b32 a186, a188
	v_accvgpr_mov_b32 a187, a188
	v_accvgpr_read_b32 v58, a188
	v_accvgpr_read_b32 v59, a188
	v_accvgpr_read_b32 v60, a188
	v_accvgpr_read_b32 v61, a188
	v_accvgpr_read_b32 v54, a188
	v_accvgpr_read_b32 v55, a188
	v_accvgpr_read_b32 v56, a188
	v_accvgpr_read_b32 v57, a188
	v_accvgpr_read_b32 v50, a188
	v_accvgpr_read_b32 v51, a188
	v_accvgpr_read_b32 v52, a188
	v_accvgpr_read_b32 v53, a188
	v_accvgpr_read_b32 v46, a188
	v_accvgpr_read_b32 v47, a188
	v_accvgpr_read_b32 v48, a188
	v_accvgpr_read_b32 v49, a188
	v_accvgpr_read_b32 v42, a188
	v_accvgpr_read_b32 v43, a188
	v_accvgpr_read_b32 v44, a188
	v_accvgpr_read_b32 v45, a188
	v_accvgpr_read_b32 v38, a188
	v_accvgpr_read_b32 v39, a188
	v_accvgpr_read_b32 v40, a188
	v_accvgpr_read_b32 v41, a188
	v_accvgpr_read_b32 v34, a188
	v_accvgpr_read_b32 v35, a188
	v_accvgpr_read_b32 v36, a188
	v_accvgpr_read_b32 v37, a188
	v_accvgpr_read_b32 v30, a188
	v_accvgpr_read_b32 v31, a188
	v_accvgpr_read_b32 v32, a188
	v_accvgpr_read_b32 v33, a188
	v_accvgpr_read_b32 v26, a188
	v_accvgpr_read_b32 v27, a188
	v_accvgpr_read_b32 v28, a188
	v_accvgpr_read_b32 v29, a188
	v_accvgpr_read_b32 v22, a188
	v_accvgpr_read_b32 v23, a188
	v_accvgpr_read_b32 v24, a188
	v_accvgpr_read_b32 v25, a188
	v_accvgpr_read_b32 v18, a188
	v_accvgpr_read_b32 v19, a188
	v_accvgpr_read_b32 v20, a188
	v_accvgpr_read_b32 v21, a188
	v_accvgpr_read_b32 v14, a188
	v_accvgpr_read_b32 v15, a188
	v_accvgpr_read_b32 v16, a188
	v_accvgpr_read_b32 v17, a188
	v_accvgpr_read_b32 v10, a188
	v_accvgpr_read_b32 v11, a188
	v_accvgpr_read_b32 v12, a188
	v_accvgpr_read_b32 v13, a188
	v_accvgpr_read_b32 v6, a188
	v_accvgpr_read_b32 v7, a188
	v_accvgpr_read_b32 v8, a188
	v_accvgpr_read_b32 v9, a188
	v_accvgpr_mov_b32 a192, a188
	v_accvgpr_mov_b32 a193, a188
	v_accvgpr_mov_b32 a194, a188
	v_accvgpr_mov_b32 a195, a188
	v_accvgpr_read_b32 v230, a188
	v_accvgpr_read_b32 v231, a188
	v_accvgpr_read_b32 v232, a188
	v_accvgpr_read_b32 v233, a188
	v_accvgpr_read_b32 v226, a188
	v_accvgpr_read_b32 v227, a188
	v_accvgpr_read_b32 v228, a188
	v_accvgpr_read_b32 v229, a188
	v_accvgpr_read_b32 v234, a188
	v_accvgpr_read_b32 v235, a188
	v_accvgpr_read_b32 v236, a188
	v_accvgpr_read_b32 v237, a188
	v_accvgpr_read_b32 v222, a188
	v_accvgpr_read_b32 v223, a188
	v_accvgpr_read_b32 v224, a188
	v_accvgpr_read_b32 v225, a188
	v_accvgpr_read_b32 v218, a188
	v_accvgpr_read_b32 v219, a188
	v_accvgpr_read_b32 v220, a188
	v_accvgpr_read_b32 v221, a188
	v_accvgpr_read_b32 v214, a188
	v_accvgpr_read_b32 v215, a188
	v_accvgpr_read_b32 v216, a188
	v_accvgpr_read_b32 v217, a188
	v_accvgpr_read_b32 v210, a188
	v_accvgpr_read_b32 v211, a188
	v_accvgpr_read_b32 v212, a188
	v_accvgpr_read_b32 v213, a188
	v_accvgpr_read_b32 v202, a188
	v_accvgpr_read_b32 v203, a188
	v_accvgpr_read_b32 v204, a188
	v_accvgpr_read_b32 v205, a188
	v_accvgpr_read_b32 v198, a188
	v_accvgpr_read_b32 v199, a188
	v_accvgpr_read_b32 v200, a188
	v_accvgpr_read_b32 v201, a188
	v_accvgpr_read_b32 v194, a188
	v_accvgpr_read_b32 v195, a188
	v_accvgpr_read_b32 v196, a188
	v_accvgpr_read_b32 v197, a188
	v_accvgpr_read_b32 v186, a188
	v_accvgpr_read_b32 v187, a188
	v_accvgpr_read_b32 v188, a188
	v_accvgpr_read_b32 v189, a188
	v_accvgpr_read_b32 v182, a188
	v_accvgpr_read_b32 v183, a188
	v_accvgpr_read_b32 v184, a188
	v_accvgpr_read_b32 v185, a188
	v_accvgpr_read_b32 v178, a188
	v_accvgpr_read_b32 v179, a188
	v_accvgpr_read_b32 v180, a188
	v_accvgpr_read_b32 v181, a188
	v_accvgpr_read_b32 v174, a188
	v_accvgpr_read_b32 v175, a188
	v_accvgpr_read_b32 v176, a188
	v_accvgpr_read_b32 v177, a188
	v_accvgpr_read_b32 v166, a188
	v_accvgpr_read_b32 v167, a188
	v_accvgpr_read_b32 v168, a188
	v_accvgpr_read_b32 v169, a188
	v_accvgpr_mov_b32 a200, a188
	v_accvgpr_mov_b32 a201, a188
	v_accvgpr_mov_b32 a202, a188
	v_accvgpr_mov_b32 a203, a188
	v_accvgpr_read_b32 v142, a188
	v_accvgpr_read_b32 v143, a188
	v_accvgpr_read_b32 v144, a188
	v_accvgpr_read_b32 v145, a188
	v_accvgpr_read_b32 v138, a188
	v_accvgpr_read_b32 v139, a188
	v_accvgpr_read_b32 v140, a188
	v_accvgpr_read_b32 v141, a188
	v_accvgpr_read_b32 v130, a188
	v_accvgpr_read_b32 v131, a188
	v_accvgpr_read_b32 v132, a188
	v_accvgpr_read_b32 v133, a188
	v_accvgpr_read_b32 v122, a188
	v_accvgpr_read_b32 v123, a188
	v_accvgpr_read_b32 v124, a188
	v_accvgpr_read_b32 v125, a188
	v_accvgpr_read_b32 v114, a188
	v_accvgpr_read_b32 v115, a188
	v_accvgpr_read_b32 v116, a188
	v_accvgpr_read_b32 v117, a188
	v_accvgpr_read_b32 v106, a188
	v_accvgpr_read_b32 v107, a188
	v_accvgpr_read_b32 v108, a188
	v_accvgpr_read_b32 v109, a188
	v_accvgpr_read_b32 v102, a188
	v_accvgpr_read_b32 v103, a188
	v_accvgpr_read_b32 v104, a188
	v_accvgpr_read_b32 v105, a188
	v_accvgpr_read_b32 v98, a188
	v_accvgpr_read_b32 v99, a188
	v_accvgpr_read_b32 v100, a188
	v_accvgpr_read_b32 v101, a188
	v_accvgpr_read_b32 v90, a188
	v_accvgpr_read_b32 v91, a188
	v_accvgpr_read_b32 v92, a188
	v_accvgpr_read_b32 v93, a188
	v_accvgpr_read_b32 v82, a188
	v_accvgpr_read_b32 v83, a188
	v_accvgpr_read_b32 v84, a188
	v_accvgpr_read_b32 v85, a188
	v_accvgpr_read_b32 v78, a188
	v_accvgpr_read_b32 v79, a188
	v_accvgpr_read_b32 v80, a188
	v_accvgpr_read_b32 v81, a188
	v_accvgpr_read_b32 v74, a188
	v_accvgpr_read_b32 v75, a188
	v_accvgpr_read_b32 v76, a188
	v_accvgpr_read_b32 v77, a188
	v_accvgpr_read_b32 v70, a188
	v_accvgpr_read_b32 v71, a188
	v_accvgpr_read_b32 v72, a188
	v_accvgpr_read_b32 v73, a188
	v_accvgpr_read_b32 v66, a188
	v_accvgpr_read_b32 v67, a188
	v_accvgpr_read_b32 v68, a188
	v_accvgpr_read_b32 v69, a188
	v_accvgpr_read_b32 v62, a188
	v_accvgpr_read_b32 v63, a188
	v_accvgpr_read_b32 v64, a188
	v_accvgpr_read_b32 v65, a188
	v_accvgpr_mov_b32 a196, a188
	v_accvgpr_mov_b32 a197, a188
	v_accvgpr_mov_b32 a198, a188
	v_accvgpr_mov_b32 a199, a188
	s_branch .LBB0_7
.LBB0_6:                                ; %_ZZ28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargsENKUliT_T0_E_clIN4opus6numberILi1EEES8_EEDaiS2_S3_.exit
                                        ;   in Loop: Header=BB0_7 Depth=1
	v_perm_b32 v94, v110, v1, v135
	v_perm_b32 v95, v255, v111, v135
	v_perm_b32 v96, v150, v112, v135
	v_perm_b32 v110, v151, v113, v135
	v_perm_b32 v1, v153, v152, v135
	v_perm_b32 v255, v155, v154, v135
	v_perm_b32 v97, v157, v156, v135
	v_perm_b32 v111, v120, v119, v135
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], a[100:103], a[20:23], v[58:61], v128, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_lshl_b32 s4, s41, 1
	s_xor_b32 s4, s4, 2
	s_mulk_i32 s4, 0x4200
	v_add_u32_e32 v112, s4, v0
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], a[108:111], a[28:31], v[58:61], v126, v129 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[52:55], v112
	s_add_i32 s5, s36, s39
	s_add_i32 m0, s42, 0x4200
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], a[88:91], a[20:23], v[54:57], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], a[92:95], a[28:31], v[54:57], v126, v129 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[48:51], v112 offset:4224
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], a[80:83], a[20:23], v[50:53], v128, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], a[84:87], a[28:31], v[50:53], v126, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[36:39], v112 offset:8448
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], a[76:79], a[20:23], v[46:49], v128, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], a[68:71], a[28:31], v[46:49], v126, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[32:35], v112 offset:12672
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], a[100:103], a[12:15], v[42:45], v128, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], a[108:111], a[24:27], v[42:45], v126, v129 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[60:63], v112 offset:64
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], a[88:91], a[12:15], v[38:41], v128, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], a[92:95], a[24:27], v[38:41], v126, v129 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[56:59], v112 offset:4288
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], a[80:83], a[12:15], v[34:37], v128, v134 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], a[84:87], a[24:27], v[34:37], v126, v129 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[44:47], v112 offset:8512
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], a[76:79], a[12:15], v[30:33], v128, v134 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], a[68:71], a[24:27], v[30:33], v126, v129 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[40:43], v112 offset:12736
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], a[100:103], a[8:11], v[26:29], v128, v134 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], a[88:91], a[8:11], v[22:25], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], a[108:111], a[16:19], v[26:29], v126, v129 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], a[92:95], a[16:19], v[22:25], v126, v129 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v118, s[28:31], s5 offen lds
	s_add_i32 m0, s42, 0x5280
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], a[80:83], a[8:11], v[18:21], v128, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], a[76:79], a[8:11], v[14:17], v128, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], a[84:87], a[16:19], v[18:21], v126, v129 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], a[68:71], a[16:19], v[14:17], v126, v129 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v147, s[28:31], s5 offen lds
	s_add_i32 m0, s42, 0x6300
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], a[100:103], a[0:3], v[10:13], v128, v134 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], a[88:91], a[0:3], v[6:9], v128, v134 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], a[108:111], a[4:7], v[10:13], v126, v129 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], a[92:95], a[4:7], v[6:9], v126, v129 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v149, s[28:31], s5 offen lds
	s_add_i32 m0, s42, 0x7380
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], a[80:83], a[0:3], v[2:5], v128, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[76:79], a[0:3], a[192:195], v128, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], a[84:87], a[4:7], v[2:5], v126, v129 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[68:71], a[4:7], a[0:3], v126, v129 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v137, s[28:31], s5 offen lds
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], a[100:103], a[120:123], v[142:145], v128, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v119, s4, v164
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], a[108:111], a[124:127], v[142:145], v126, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[20:23], v119
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], a[88:91], a[120:123], v[138:141], v128, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], a[92:95], a[124:127], v[138:141], v126, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[12:15], v119 offset:4224
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], a[80:83], a[120:123], v[130:133], v128, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], a[84:87], a[124:127], v[130:133], v126, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[8:11], v119 offset:8448
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], a[76:79], a[120:123], v[122:125], v128, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], a[68:71], a[124:127], v[122:125], v126, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[0:3], v119 offset:12672
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[112:115], a[100:103], a[112:115], v[114:117], v128, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], a[108:111], a[116:119], v[112:115], v126, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[28:31], v119 offset:64
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], a[88:91], a[112:115], v[106:109], v128, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], a[92:95], a[116:119], v[106:109], v126, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[24:27], v119 offset:4288
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], a[80:83], a[112:115], v[102:105], v128, v127 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], a[84:87], a[116:119], v[102:105], v126, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[16:19], v119 offset:8512
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], a[76:79], a[112:115], v[98:101], v128, v127 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], a[100:103], a[96:99], v[90:93], v128, v127 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], a[88:91], a[96:99], v[82:85], v128, v127 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], a[80:83], a[96:99], v[78:81], v128, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], a[76:79], a[96:99], v[74:77], v128, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], a[100:103], a[72:75], v[70:73], v128, v127 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], a[88:91], a[72:75], v[66:69], v128, v127 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], a[80:83], a[72:75], v[62:65], v128, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[76:79], a[72:75], a[196:199], v128, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], a[68:71], a[116:119], v[98:101], v126, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[4:7], v119 offset:12736
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], a[108:111], a[104:107], v[90:93], v126, v121 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], a[92:95], a[104:107], v[82:85], v126, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], a[84:87], a[104:107], v[78:81], v126, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], a[68:71], a[104:107], v[74:77], v126, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], a[108:111], a[64:67], v[70:73], v126, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], a[92:95], a[64:67], v[66:69], v126, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], a[84:87], a[64:67], v[62:65], v126, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[68:71], a[64:67], a[176:179], v126, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	s_addk_i32 s39, 0x80
	s_add_i32 s9, s9, 8
	s_addk_i32 s40, 0x800
	s_add_i32 s38, s38, 1
	v_accvgpr_mov_b32 a183, a151
	s_cmp_lg_u32 s37, s39
	v_mov_b32_e32 v121, v110
	v_mov_b32_e32 v127, v96
	v_mov_b32_e32 v129, v95
	v_mov_b32_e32 v134, v94
	v_mov_b32_e32 v126, v111
	v_mov_b32_e32 v128, v97
	v_accvgpr_mov_b32 a182, a150
	v_accvgpr_mov_b32 a181, a149
	v_accvgpr_mov_b32 a180, a148
	s_cbranch_scc0 .LBB0_12
.LBB0_7:                                ; =>This Inner Loop Header: Depth=1
	s_and_b32 s41, s38, 1
	s_add_i32 s23, s38, 3
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[52:55], a[20:23], a[188:191], v1, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_bitcmp1_b32 s38, 0
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s23, s6
	s_mul_i32 s23, s41, 0x8400
	v_add_u32_e32 v94, s23, v164
	ds_read_b128 a[120:123], v94 offset:16896
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[60:63], a[28:31], a[148:151], v255, v129 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v94 offset:21120
	s_cselect_b64 s[42:43], -1, 0
	s_and_b64 s[4:5], s[4:5], s[42:43]
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[48:51], a[20:23], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v94 offset:25344
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[56:59], a[28:31], a[148:151], v255, v129 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[72:75], v94 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[36:39], a[20:23], a[208:211], v1, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v94 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[44:47], a[28:31], a[148:151], v255, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v94 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[32:35], a[20:23], a[212:215], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[104:107], v94 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[40:43], a[28:31], a[148:151], v255, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[64:67], v94 offset:29632
	v_add_u32_e32 v94, s23, v0
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[52:55], a[12:15], a[172:175], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v94 offset:16896
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[60:63], a[24:27], a[172:175], v255, v129 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[88:91], v94 offset:21120
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[48:51], a[12:15], a[168:171], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[80:83], v94 offset:25344
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[56:59], a[24:27], a[168:171], v255, v129 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[76:79], v94 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[36:39], a[12:15], a[164:167], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v94 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[44:47], a[24:27], a[164:167], v255, v129 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[92:95], v94 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[32:35], a[12:15], a[160:163], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[84:87], v94 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[40:43], a[24:27], a[160:163], v255, v129 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[68:71], v94 offset:29632
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_add_i32 s42, s40, 0xfffff000
	s_add_i32 s43, s9, -16
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[52:55], a[8:11], a[156:159], v1, v134 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_and_b32 s42, s42, 0x1000
	s_and_b32 s43, s43, 8
	s_or_b32 s42, s42, s43
	v_add_u32_e32 v110, s42, v161
	v_add_u32_e32 v94, s42, v165
	ds_read2st64_b64 v[110:113], v110 offset1:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[60:63], a[16:19], a[156:159], v255, v129 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	ds_read2st64_b64 v[94:97], v94 offset1:4
	s_add_i32 s43, s14, s23
	s_mov_b32 m0, s43
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[48:51], a[8:11], a[152:155], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[56:59], a[16:19], a[152:155], v255, v129 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[36:39], a[8:11], a[180:183], v1, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v254, s[24:27], s39 offen lds
	s_add_i32 m0, s43, 0x1080
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[32:35], a[8:11], a[144:147], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[44:47], a[16:19], a[148:151], v255, v129 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[40:43], a[16:19], a[144:147], v255, v129 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v146, s[24:27], s39 offen lds
	s_add_i32 m0, s43, 0x2100
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[52:55], a[0:3], a[140:143], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[60:63], a[4:7], a[140:143], v255, v129 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[48:51], a[0:3], a[136:139], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v148, s[24:27], s39 offen lds
	s_add_i32 m0, s43, 0x3180
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[36:39], a[0:3], a[132:135], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[56:59], a[4:7], a[136:139], v255, v129 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[44:47], a[4:7], a[132:135], v255, v129 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v158, s[24:27], s39 offen lds
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[32:35], a[0:3], a[184:187], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[184:187], a[40:43], a[4:7], a[180:183], v255, v129 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[52:55], a[120:123], v[230:233], v1, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_i32 s42, s15, s23
	s_add_i32 s42, s42, 0x10800
	s_mov_b32 m0, s42
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], a[60:63], a[124:127], v[150:153], v255, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_add_i32 s23, s34, s39
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v119, v111
	s_nop 1
	v_permlane16_swap_b32_e64 v111, v119 bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[48:51], a[120:123], v[226:229], v1, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_perm_b32 v111, v119, v111, v136
	v_mov_b32_e32 v119, v112
	s_nop 1
	v_permlane16_swap_b32_e64 v112, v119 bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], a[56:59], a[124:127], v[150:153], v255, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v118, s[28:31], s39 offen lds
	s_add_i32 m0, s42, 0x1080
	v_perm_b32 v112, v119, v112, v136
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[36:39], a[120:123], v[234:237], v1, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v119, v113
	s_nop 1
	v_permlane16_swap_b32_e64 v113, v119 bound_ctrl:1
	v_perm_b32 v113, v119, v113, v136
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], a[32:35], a[120:123], v[222:225], v1, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v119, v94
	s_nop 1
	v_permlane16_swap_b32_e64 v94, v119 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], a[44:47], a[124:127], v[150:153], v255, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], a[40:43], a[124:127], v[154:157], v255, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v147, s[28:31], s39 offen lds
	s_add_i32 m0, s42, 0x2100
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[52:55], a[112:115], v[218:221], v1, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], a[48:51], a[112:115], v[214:217], v1, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], a[60:63], a[116:119], v[150:153], v255, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], a[56:59], a[116:119], v[154:157], v255, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v149, s[28:31], s39 offen lds
	s_add_i32 m0, s42, 0x3180
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[36:39], a[112:115], v[210:213], v1, v127 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], a[32:35], a[112:115], v[202:205], v1, v127 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], a[44:47], a[116:119], v[150:153], v255, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], a[40:43], a[116:119], v[154:157], v255, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v137, s[28:31], s39 offen lds
	s_add_i32 m0, s43, 0x4200
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[52:55], a[96:99], v[198:201], v1, v127 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], a[48:51], a[96:99], v[194:197], v1, v127 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], a[60:63], a[104:107], v[150:153], v255, v121 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], a[56:59], a[104:107], v[154:157], v255, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v254, s[24:27], s23 offen lds
	s_add_i32 m0, s43, 0x5280
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[36:39], a[96:99], v[186:189], v1, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], a[32:35], a[96:99], v[182:185], v1, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], a[44:47], a[104:107], v[150:153], v255, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], a[40:43], a[104:107], v[154:157], v255, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v146, s[24:27], s23 offen lds
	s_add_i32 m0, s43, 0x6300
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[52:55], a[72:75], v[178:181], v1, v127 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], a[48:51], a[72:75], v[174:177], v1, v127 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], a[60:63], a[64:67], v[150:153], v255, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], a[56:59], a[64:67], v[154:157], v255, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v148, s[24:27], s23 offen lds
	s_add_i32 m0, s43, 0x7380
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[36:39], a[72:75], v[166:169], v1, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[32:35], a[32:35], a[72:75], a[200:203], v1, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v1, v110
	s_nop 1
	v_permlane16_swap_b32_e64 v110, v1 bound_ctrl:1
	v_perm_b32 v1, v1, v110, v136
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], a[44:47], a[64:67], v[150:153], v255, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mov_b32_e32 v110, v1
	s_nop 1
	v_permlane32_swap_b32_e64 v1, v110 bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[40:43], a[64:67], a[32:35], v255, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	buffer_load_dwordx4 v158, s[24:27], s23 offen lds
	v_perm_b32 v152, v119, v94, v136
	v_mov_b32_e32 v94, v95
	s_nop 1
	v_permlane16_swap_b32_e64 v95, v94 bound_ctrl:1
	v_perm_b32 v154, v94, v95, v136
	v_mov_b32_e32 v94, v96
	s_nop 1
	v_permlane16_swap_b32_e64 v96, v94 bound_ctrl:1
	v_perm_b32 v156, v94, v96, v136
	v_mov_b32_e32 v94, v97
	s_nop 1
	v_permlane16_swap_b32_e64 v97, v94 bound_ctrl:1
	v_perm_b32 v119, v94, v97, v136
	v_mov_b32_e32 v255, v111
	v_mov_b32_e32 v150, v112
	v_mov_b32_e32 v151, v113
	v_mov_b32_e32 v153, v152
	v_mov_b32_e32 v155, v154
	v_mov_b32_e32 v157, v156
	v_mov_b32_e32 v120, v119
	v_permlane32_swap_b32_e64 v111, v255 bound_ctrl:1
	v_permlane32_swap_b32_e64 v112, v150 bound_ctrl:1
	v_permlane32_swap_b32_e64 v113, v151 bound_ctrl:1
	v_permlane32_swap_b32_e64 v152, v153 bound_ctrl:1
	v_permlane32_swap_b32_e64 v154, v155 bound_ctrl:1
	v_permlane32_swap_b32_e64 v156, v157 bound_ctrl:1
	v_permlane32_swap_b32_e64 v119, v120 bound_ctrl:1
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_6
; %bb.8:                                ;   in Loop: Header=BB0_7 Depth=1
	s_and_b32 s4, s40, 0x1000
	s_add_i32 m0, s13, s4
	s_mov_b32 s23, s19
	buffer_load_dwordx4 v160, s[16:19], s9 offen lds
	s_add_i32 m0, s12, s4
	s_nop 0
	buffer_load_dwordx4 v159, s[20:23], s9 offen lds
	s_branch .LBB0_6
.LBB0_9:
	s_add_i32 s41, s41, 0x24000
	s_add_i32 m0, s40, 0x22000
	s_nop 0
	buffer_load_dwordx4 v160, s[16:19], 16 offen lds
	s_mov_b32 m0, s41
	s_nop 0
	buffer_load_dwordx4 v159, s[20:23], 16 offen lds
	s_mov_b32 s27, s19
	s_cmpk_lt_i32 s6, 0x200
	s_mov_b64 s[8:9], -1
	s_cbranch_scc0 .LBB0_2
.LBB0_10:
	s_waitcnt vmcnt(0)
	s_load_dwordx2 s[0:1], s[0:1], 0x10
	s_cbranch_execz .LBB0_3
	s_branch .LBB0_4
.LBB0_11:
	v_accvgpr_write_b32 a187, 0
	v_accvgpr_read_b32 v5, a187
	v_accvgpr_read_b32 v65, a187
	v_accvgpr_read_b32 v4, a187
	v_accvgpr_read_b32 v3, a187
	v_accvgpr_read_b32 v2, a187
	v_accvgpr_write_b32 a67, v5
	v_accvgpr_write_b32 a155, v5
	v_accvgpr_write_b32 a159, v5
	v_accvgpr_write_b32 a163, v5
	v_accvgpr_write_b32 a167, v5
	v_accvgpr_write_b32 a171, v5
	v_accvgpr_write_b32 a175, v5
	v_accvgpr_write_b32 a151, v5
	v_accvgpr_read_b32 v64, a187
	v_accvgpr_read_b32 v63, a187
	v_accvgpr_read_b32 v62, a187
	v_accvgpr_write_b32 a71, v65
	v_accvgpr_write_b32 a179, v65
	v_accvgpr_mov_b32 a186, a187
	v_accvgpr_mov_b32 a185, a187
	v_accvgpr_mov_b32 a184, a187
	v_accvgpr_write_b32 a66, v4
	v_accvgpr_write_b32 a65, v3
	v_accvgpr_write_b32 a64, v2
	v_accvgpr_read_b32 v113, a187
	v_accvgpr_read_b32 v112, a187
	v_accvgpr_read_b32 v111, a187
	v_accvgpr_read_b32 v110, a187
	v_accvgpr_read_b32 v149, a187
	v_accvgpr_read_b32 v148, a187
	v_accvgpr_read_b32 v147, a187
	v_accvgpr_read_b32 v146, a187
	v_accvgpr_read_b32 v97, a187
	v_accvgpr_read_b32 v96, a187
	v_accvgpr_read_b32 v95, a187
	v_accvgpr_read_b32 v94, a187
	v_accvgpr_read_b32 v153, a187
	v_accvgpr_read_b32 v152, a187
	v_accvgpr_read_b32 v151, a187
	v_accvgpr_read_b32 v150, a187
	v_accvgpr_mov_b32 a183, a187
	v_accvgpr_mov_b32 a182, a187
	v_accvgpr_mov_b32 a181, a187
	v_accvgpr_mov_b32 a180, a187
	v_accvgpr_write_b32 a154, v4
	v_accvgpr_write_b32 a153, v3
	v_accvgpr_write_b32 a152, v2
	v_accvgpr_write_b32 a158, v4
	v_accvgpr_write_b32 a157, v3
	v_accvgpr_write_b32 a156, v2
	v_accvgpr_write_b32 a162, v4
	v_accvgpr_write_b32 a161, v3
	v_accvgpr_write_b32 a160, v2
	v_accvgpr_write_b32 a166, v4
	v_accvgpr_write_b32 a165, v3
	v_accvgpr_write_b32 a164, v2
	v_accvgpr_write_b32 a170, v4
	v_accvgpr_write_b32 a169, v3
	v_accvgpr_write_b32 a168, v2
	v_accvgpr_write_b32 a174, v4
	v_accvgpr_write_b32 a173, v3
	v_accvgpr_write_b32 a172, v2
	v_accvgpr_mov_b32 a215, a187
	v_accvgpr_mov_b32 a214, a187
	v_accvgpr_mov_b32 a213, a187
	v_accvgpr_mov_b32 a212, a187
	v_accvgpr_mov_b32 a211, a187
	v_accvgpr_mov_b32 a210, a187
	v_accvgpr_mov_b32 a209, a187
	v_accvgpr_mov_b32 a208, a187
	v_accvgpr_mov_b32 a207, a187
	v_accvgpr_mov_b32 a206, a187
	v_accvgpr_mov_b32 a205, a187
	v_accvgpr_mov_b32 a204, a187
	v_accvgpr_mov_b32 a191, a187
	v_accvgpr_mov_b32 a190, a187
	v_accvgpr_mov_b32 a189, a187
	v_accvgpr_mov_b32 a188, a187
	v_accvgpr_mov_b32 a195, a187
	v_accvgpr_mov_b32 a194, a187
	v_accvgpr_mov_b32 a193, a187
	v_accvgpr_mov_b32 a192, a187
	v_accvgpr_write_b32 a150, v4
	v_accvgpr_write_b32 a149, v3
	v_accvgpr_write_b32 a148, v2
	v_accvgpr_read_b32 v9, a187
	v_accvgpr_read_b32 v8, a187
	v_accvgpr_read_b32 v7, a187
	v_accvgpr_read_b32 v6, a187
	v_accvgpr_read_b32 v13, a187
	v_accvgpr_read_b32 v12, a187
	v_accvgpr_read_b32 v11, a187
	v_accvgpr_read_b32 v10, a187
	v_accvgpr_read_b32 v17, a187
	v_accvgpr_read_b32 v16, a187
	v_accvgpr_read_b32 v15, a187
	v_accvgpr_read_b32 v14, a187
	v_accvgpr_read_b32 v21, a187
	v_accvgpr_read_b32 v20, a187
	v_accvgpr_read_b32 v19, a187
	v_accvgpr_read_b32 v18, a187
	v_accvgpr_read_b32 v25, a187
	v_accvgpr_read_b32 v24, a187
	v_accvgpr_read_b32 v23, a187
	v_accvgpr_read_b32 v22, a187
	v_accvgpr_read_b32 v29, a187
	v_accvgpr_read_b32 v28, a187
	v_accvgpr_read_b32 v27, a187
	v_accvgpr_read_b32 v26, a187
	v_accvgpr_read_b32 v33, a187
	v_accvgpr_read_b32 v32, a187
	v_accvgpr_read_b32 v31, a187
	v_accvgpr_read_b32 v30, a187
	v_accvgpr_read_b32 v37, a187
	v_accvgpr_read_b32 v36, a187
	v_accvgpr_read_b32 v35, a187
	v_accvgpr_read_b32 v34, a187
	v_accvgpr_read_b32 v41, a187
	v_accvgpr_read_b32 v40, a187
	v_accvgpr_read_b32 v39, a187
	v_accvgpr_read_b32 v38, a187
	v_accvgpr_read_b32 v45, a187
	v_accvgpr_read_b32 v44, a187
	v_accvgpr_read_b32 v43, a187
	v_accvgpr_read_b32 v42, a187
	v_accvgpr_read_b32 v49, a187
	v_accvgpr_read_b32 v48, a187
	v_accvgpr_read_b32 v47, a187
	v_accvgpr_read_b32 v46, a187
	v_accvgpr_read_b32 v53, a187
	v_accvgpr_read_b32 v52, a187
	v_accvgpr_read_b32 v51, a187
	v_accvgpr_read_b32 v50, a187
	v_accvgpr_read_b32 v57, a187
	v_accvgpr_read_b32 v56, a187
	v_accvgpr_read_b32 v55, a187
	v_accvgpr_read_b32 v54, a187
	v_accvgpr_read_b32 v61, a187
	v_accvgpr_read_b32 v60, a187
	v_accvgpr_read_b32 v59, a187
	v_accvgpr_read_b32 v58, a187
	v_accvgpr_mov_b32 a203, a187
	v_accvgpr_mov_b32 a202, a187
	v_accvgpr_mov_b32 a201, a187
	v_accvgpr_mov_b32 a200, a187
	v_accvgpr_write_b32 a70, v64
	v_accvgpr_write_b32 a69, v63
	v_accvgpr_write_b32 a68, v62
	v_accvgpr_read_b32 v169, a187
	v_accvgpr_read_b32 v168, a187
	v_accvgpr_read_b32 v167, a187
	v_accvgpr_read_b32 v166, a187
	v_accvgpr_read_b32 v177, a187
	v_accvgpr_read_b32 v176, a187
	v_accvgpr_read_b32 v175, a187
	v_accvgpr_read_b32 v174, a187
	v_accvgpr_read_b32 v181, a187
	v_accvgpr_read_b32 v180, a187
	v_accvgpr_read_b32 v179, a187
	v_accvgpr_read_b32 v178, a187
	v_accvgpr_read_b32 v185, a187
	v_accvgpr_read_b32 v184, a187
	v_accvgpr_read_b32 v183, a187
	v_accvgpr_read_b32 v182, a187
	v_accvgpr_read_b32 v189, a187
	v_accvgpr_read_b32 v188, a187
	v_accvgpr_read_b32 v187, a187
	v_accvgpr_read_b32 v186, a187
	v_accvgpr_read_b32 v197, a187
	v_accvgpr_read_b32 v196, a187
	v_accvgpr_read_b32 v195, a187
	v_accvgpr_read_b32 v194, a187
	v_accvgpr_read_b32 v201, a187
	v_accvgpr_read_b32 v200, a187
	v_accvgpr_read_b32 v199, a187
	v_accvgpr_read_b32 v198, a187
	v_accvgpr_read_b32 v205, a187
	v_accvgpr_read_b32 v204, a187
	v_accvgpr_read_b32 v203, a187
	v_accvgpr_read_b32 v202, a187
	v_accvgpr_read_b32 v213, a187
	v_accvgpr_read_b32 v212, a187
	v_accvgpr_read_b32 v211, a187
	v_accvgpr_read_b32 v210, a187
	v_accvgpr_read_b32 v217, a187
	v_accvgpr_read_b32 v216, a187
	v_accvgpr_read_b32 v215, a187
	v_accvgpr_read_b32 v214, a187
	v_accvgpr_read_b32 v221, a187
	v_accvgpr_read_b32 v220, a187
	v_accvgpr_read_b32 v219, a187
	v_accvgpr_read_b32 v218, a187
	v_accvgpr_read_b32 v225, a187
	v_accvgpr_read_b32 v224, a187
	v_accvgpr_read_b32 v223, a187
	v_accvgpr_read_b32 v222, a187
	v_accvgpr_read_b32 v237, a187
	v_accvgpr_read_b32 v236, a187
	v_accvgpr_read_b32 v235, a187
	v_accvgpr_read_b32 v234, a187
	v_accvgpr_read_b32 v229, a187
	v_accvgpr_read_b32 v228, a187
	v_accvgpr_read_b32 v227, a187
	v_accvgpr_read_b32 v226, a187
	v_accvgpr_read_b32 v233, a187
	v_accvgpr_read_b32 v232, a187
	v_accvgpr_read_b32 v231, a187
	v_accvgpr_read_b32 v230, a187
	v_accvgpr_mov_b32 a199, a187
	v_accvgpr_mov_b32 a198, a187
	v_accvgpr_mov_b32 a197, a187
	v_accvgpr_mov_b32 a196, a187
	v_accvgpr_write_b32 a178, v64
	v_accvgpr_write_b32 a177, v63
	v_accvgpr_write_b32 a176, v62
	v_accvgpr_read_b32 v69, a187
	v_accvgpr_read_b32 v68, a187
	v_accvgpr_read_b32 v67, a187
	v_accvgpr_read_b32 v66, a187
	v_accvgpr_read_b32 v73, a187
	v_accvgpr_read_b32 v72, a187
	v_accvgpr_read_b32 v71, a187
	v_accvgpr_read_b32 v70, a187
	v_accvgpr_read_b32 v77, a187
	v_accvgpr_read_b32 v76, a187
	v_accvgpr_read_b32 v75, a187
	v_accvgpr_read_b32 v74, a187
	v_accvgpr_read_b32 v81, a187
	v_accvgpr_read_b32 v80, a187
	v_accvgpr_read_b32 v79, a187
	v_accvgpr_read_b32 v78, a187
	v_accvgpr_read_b32 v85, a187
	v_accvgpr_read_b32 v84, a187
	v_accvgpr_read_b32 v83, a187
	v_accvgpr_read_b32 v82, a187
	v_accvgpr_read_b32 v93, a187
	v_accvgpr_read_b32 v92, a187
	v_accvgpr_read_b32 v91, a187
	v_accvgpr_read_b32 v90, a187
	v_accvgpr_read_b32 v101, a187
	v_accvgpr_read_b32 v100, a187
	v_accvgpr_read_b32 v99, a187
	v_accvgpr_read_b32 v98, a187
	v_accvgpr_read_b32 v105, a187
	v_accvgpr_read_b32 v104, a187
	v_accvgpr_read_b32 v103, a187
	v_accvgpr_read_b32 v102, a187
	v_accvgpr_read_b32 v109, a187
	v_accvgpr_read_b32 v108, a187
	v_accvgpr_read_b32 v107, a187
	v_accvgpr_read_b32 v106, a187
	v_accvgpr_read_b32 v117, a187
	v_accvgpr_read_b32 v116, a187
	v_accvgpr_read_b32 v115, a187
	v_accvgpr_read_b32 v114, a187
	v_accvgpr_read_b32 v125, a187
	v_accvgpr_read_b32 v124, a187
	v_accvgpr_read_b32 v123, a187
	v_accvgpr_read_b32 v122, a187
	v_accvgpr_read_b32 v133, a187
	v_accvgpr_read_b32 v132, a187
	v_accvgpr_read_b32 v131, a187
	v_accvgpr_read_b32 v130, a187
	v_accvgpr_read_b32 v141, a187
	v_accvgpr_read_b32 v140, a187
	v_accvgpr_read_b32 v139, a187
	v_accvgpr_read_b32 v138, a187
	v_accvgpr_read_b32 v145, a187
	v_accvgpr_read_b32 v144, a187
	v_accvgpr_read_b32 v143, a187
	v_accvgpr_read_b32 v142, a187
	s_branch .LBB0_13
.LBB0_12:                               ; %Flow
	v_accvgpr_read_b32 v118, a184
	v_accvgpr_read_b32 v121, a187
	v_accvgpr_read_b32 v120, a186
	v_accvgpr_read_b32 v119, a185
	v_accvgpr_write_b32 a64, v118
	v_accvgpr_write_b32 a65, v119
	v_accvgpr_write_b32 a66, v120
	v_accvgpr_write_b32 a67, v121
	v_accvgpr_read_b32 v121, a195
	v_accvgpr_read_b32 v120, a194
	v_accvgpr_read_b32 v119, a193
	v_accvgpr_read_b32 v118, a192
	v_accvgpr_write_b32 a151, v121
	v_accvgpr_write_b32 a150, v120
	v_accvgpr_write_b32 a149, v119
	v_accvgpr_write_b32 a148, v118
	v_accvgpr_read_b32 v118, a200
	v_accvgpr_read_b32 v121, a203
	v_accvgpr_read_b32 v120, a202
	v_accvgpr_read_b32 v119, a201
	v_accvgpr_write_b32 a68, v118
	v_accvgpr_write_b32 a69, v119
	v_accvgpr_write_b32 a70, v120
	v_accvgpr_write_b32 a71, v121
	v_accvgpr_read_b32 v121, a199
	v_accvgpr_read_b32 v120, a198
	v_accvgpr_read_b32 v119, a197
	v_accvgpr_read_b32 v118, a196
	v_accvgpr_write_b32 a179, v121
	v_accvgpr_write_b32 a178, v120
	v_accvgpr_write_b32 a177, v119
	v_accvgpr_write_b32 a176, v118
	v_mov_b32_e32 v121, v110
	v_mov_b32_e32 v127, v96
	v_mov_b32_e32 v129, v95
	v_mov_b32_e32 v134, v94
	v_mov_b32_e32 v126, v111
	v_mov_b32_e32 v128, v97
	v_accvgpr_read_b32 v110, a132
	v_accvgpr_read_b32 v149, a139
	v_accvgpr_read_b32 v94, a140
	v_accvgpr_read_b32 v153, a147
	v_accvgpr_read_b32 v111, a133
	v_accvgpr_read_b32 v112, a134
	v_accvgpr_read_b32 v113, a135
	v_accvgpr_read_b32 v148, a138
	v_accvgpr_read_b32 v147, a137
	v_accvgpr_read_b32 v146, a136
	v_accvgpr_read_b32 v95, a141
	v_accvgpr_read_b32 v96, a142
	v_accvgpr_read_b32 v97, a143
	v_accvgpr_read_b32 v152, a146
	v_accvgpr_read_b32 v151, a145
	v_accvgpr_read_b32 v150, a144
.LBB0_13:                               ; %._crit_edge
	v_accvgpr_mov_b32 a132, a180
	s_add_i32 s4, s8, 1
	v_accvgpr_mov_b32 a133, a181
	v_accvgpr_mov_b32 a134, a182
	v_accvgpr_mov_b32 a135, a183
	v_mov_b32_e32 v158, v0
	v_accvgpr_write_b32 a131, v160
	v_mov_b32_e32 v254, v159
	s_cmp_ge_i32 s4, s6
	s_cbranch_scc1 .LBB0_15
; %bb.14:                               ; %_ZZ28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargsENKUliT_T0_E_clIN4opus6numberILi1EEENS7_ILi0EEEEEDaiS2_S3_.exit
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[52:55], a[20:23], a[188:191], v1, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_and_b32 s5, s8, 1
	s_mul_i32 s8, s5, 0x8400
	v_add_u32_e32 v0, s8, v164
	ds_read_b128 a[104:107], v0 offset:16896
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[188:191], a[60:63], a[28:31], a[180:183], v255, v129 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v0 offset:21120
	v_accvgpr_write_b32 a139, v97
	v_accvgpr_write_b32 a138, v96
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[48:51], a[20:23], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[80:83], v0 offset:25344
	v_accvgpr_write_b32 a137, v95
	v_accvgpr_write_b32 a136, v94
	v_mfma_scale_f32_16x16x128_f8f6f4 a[204:207], a[56:59], a[28:31], a[144:147], v255, v129 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[120:123], v0 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[36:39], a[20:23], a[208:211], v1, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v0 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[208:211], a[44:47], a[28:31], a[140:143], v255, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v0 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[32:35], a[20:23], a[212:215], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[88:91], v0 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[212:215], a[40:43], a[28:31], a[140:143], v255, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v0 offset:29632
	v_add_u32_e32 v0, s8, v158
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_nop 2
	v_accvgpr_mov_b32 a140, a172
	v_accvgpr_mov_b32 a141, a173
	v_accvgpr_mov_b32 a142, a174
	v_accvgpr_mov_b32 a143, a175
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[52:55], a[12:15], a[140:143], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[84:87], v0 offset:16896
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[60:63], a[24:27], a[140:143], v255, v129 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[72:75], v0 offset:21120
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_nop 4
	v_accvgpr_mov_b32 a140, a168
	v_accvgpr_mov_b32 a141, a169
	v_accvgpr_mov_b32 a142, a170
	v_accvgpr_mov_b32 a143, a171
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[48:51], a[12:15], a[140:143], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[64:67], v0 offset:25344
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[172:175], a[56:59], a[24:27], a[140:143], v255, v129 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v0 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_nop 4
	v_accvgpr_mov_b32 a140, a164
	v_accvgpr_mov_b32 a141, a165
	v_accvgpr_mov_b32 a142, a166
	v_accvgpr_mov_b32 a143, a167
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[36:39], a[12:15], a[140:143], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[92:95], v0 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[44:47], a[24:27], a[140:143], v255, v129 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[76:79], v0 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_nop 4
	v_accvgpr_mov_b32 a140, a160
	v_accvgpr_mov_b32 a141, a161
	v_accvgpr_mov_b32 a142, a162
	v_accvgpr_mov_b32 a143, a163
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[32:35], a[12:15], a[140:143], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[68:71], v0 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[40:43], a[24:27], a[140:143], v255, v129 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v0 offset:29632
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt vmcnt(0) lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	s_nop 2
	v_accvgpr_mov_b32 a140, a156
	v_accvgpr_mov_b32 a141, a157
	v_accvgpr_mov_b32 a142, a158
	v_accvgpr_mov_b32 a143, a159
	s_lshl_b32 s8, s4, 11
	s_lshl_b32 s9, s4, 3
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[52:55], a[8:11], a[140:143], v1, v134 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	s_and_b32 s8, s8, 0x1000
	s_and_b32 s9, s9, 8
	s_or_b32 s8, s8, s9
	v_add_u32_e32 v0, s8, v161
	v_accvgpr_write_b32 a163, v113
	v_accvgpr_write_b32 a162, v112
	v_accvgpr_write_b32 a161, v111
	v_accvgpr_write_b32 a160, v110
	ds_read2st64_b64 v[110:113], v0 offset1:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[156:159], a[60:63], a[16:19], a[140:143], v255, v129 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_add_u32_e32 v0, s8, v165
	ds_read2st64_b64 v[94:97], v0 offset1:4
	v_accvgpr_write_b32 a179, v149
	v_accvgpr_mov_b32 a140, a152
	v_accvgpr_mov_b32 a141, a153
	v_accvgpr_mov_b32 a142, a154
	v_accvgpr_mov_b32 a143, a155
	v_accvgpr_write_b32 a178, v148
	v_accvgpr_write_b32 a177, v147
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[48:51], a[8:11], a[140:143], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_write_b32 a176, v146
	v_accvgpr_read_b32 v154, a168
	v_accvgpr_read_b32 v155, a169
	v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[56:59], a[16:19], a[140:143], v255, v129 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v156, a170
	v_accvgpr_read_b32 v157, a171
	v_accvgpr_read_b32 v160, a172
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[36:39], a[8:11], a[132:135], v1, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_mov_b32 a143, a139
	v_accvgpr_mov_b32 a142, a138
	v_accvgpr_mov_b32 a141, a137
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], a[32:35], a[8:11], v[150:153], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_mov_b32 a140, a136
	v_accvgpr_read_b32 v161, a173
	v_accvgpr_read_b32 v162, a174
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[44:47], a[16:19], a[132:135], v255, v129 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v163, a175
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[40:43], a[16:19], v[146:149], v255, v129 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[52:55], a[0:3], a[140:143], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[48:51], a[0:3], a[176:179], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[32:35], a[0:3], a[184:187], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[60:63], a[4:7], a[140:143], v255, v129 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[56:59], a[4:7], a[132:135], v255, v129 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[36:39], a[0:3], a[160:163], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[132:135], a[40:43], a[4:7], a[180:183], v255, v129 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[44:47], a[4:7], a[160:163], v255, v129 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(1)
	v_mov_b32_e32 v0, v110
	s_nop 1
	v_permlane16_swap_b32_e64 v110, v0 bound_ctrl:1
	v_perm_b32 v110, v0, v110, v136
	v_mov_b32_e32 v0, v111
	v_mov_b32_e32 v118, v110
	s_nop 0
	v_permlane16_swap_b32_e64 v111, v0 bound_ctrl:1
	v_perm_b32 v111, v0, v111, v136
	v_mov_b32_e32 v0, v112
	v_mov_b32_e32 v119, v111
	s_nop 0
	v_permlane16_swap_b32_e64 v112, v0 bound_ctrl:1
	v_perm_b32 v112, v0, v112, v136
	v_mov_b32_e32 v0, v113
	v_mov_b32_e32 v120, v112
	s_nop 0
	v_permlane16_swap_b32_e64 v113, v0 bound_ctrl:1
	v_perm_b32 v113, v0, v113, v136
	s_waitcnt lgkmcnt(0)
	v_mov_b32_e32 v0, v94
	v_mov_b32_e32 v137, v113
	s_nop 0
	v_permlane16_swap_b32_e64 v94, v0 bound_ctrl:1
	v_perm_b32 v146, v0, v94, v136
	v_mov_b32_e32 v0, v95
	v_mov_b32_e32 v147, v146
	s_nop 0
	v_permlane16_swap_b32_e64 v95, v0 bound_ctrl:1
	v_perm_b32 v148, v0, v95, v136
	v_mov_b32_e32 v0, v96
	v_mov_b32_e32 v149, v148
	s_nop 0
	v_permlane16_swap_b32_e64 v96, v0 bound_ctrl:1
	v_perm_b32 v250, v0, v96, v136
	v_mov_b32_e32 v0, v97
	v_mov_b32_e32 v251, v250
	s_nop 0
	v_permlane16_swap_b32_e64 v97, v0 bound_ctrl:1
	v_perm_b32 v0, v0, v97, v136
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[52:55], a[104:107], v[230:233], v1, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mov_b32_e32 v136, v0
	v_permlane32_swap_b32_e64 v110, v118 bound_ctrl:1
	v_permlane32_swap_b32_e64 v111, v119 bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], a[60:63], a[108:111], v[94:97], v255, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_permlane32_swap_b32_e64 v112, v120 bound_ctrl:1
	v_permlane32_swap_b32_e64 v113, v137 bound_ctrl:1
	v_permlane32_swap_b32_e64 v146, v147 bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[48:51], a[104:107], v[226:229], v1, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_permlane32_swap_b32_e64 v148, v149 bound_ctrl:1
	v_permlane32_swap_b32_e64 v250, v251 bound_ctrl:1
	v_permlane32_swap_b32_e64 v0, v136 bound_ctrl:1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], a[56:59], a[108:111], v[94:97], v255, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[36:39], a[104:107], v[234:237], v1, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[234:237], a[44:47], a[108:111], v[94:97], v255, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[32:35], a[104:107], v[222:225], v1, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], a[40:43], a[108:111], v[94:97], v255, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[52:55], a[96:99], v[218:221], v1, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], a[60:63], a[100:103], v[94:97], v255, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[48:51], a[96:99], v[214:217], v1, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], a[56:59], a[100:103], v[94:97], v255, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[36:39], a[96:99], v[210:213], v1, v127 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], a[44:47], a[100:103], v[94:97], v255, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[32:35], a[96:99], v[202:205], v1, v127 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], a[40:43], a[100:103], v[94:97], v255, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[52:55], a[80:83], v[198:201], v1, v127 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], a[60:63], a[88:91], v[94:97], v255, v121 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[48:51], a[80:83], v[194:197], v1, v127 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], a[56:59], a[88:91], v[94:97], v255, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[36:39], a[80:83], v[186:189], v1, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], a[44:47], a[88:91], v[94:97], v255, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[32:35], a[80:83], v[182:185], v1, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], a[40:43], a[88:91], v[94:97], v255, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[52:55], a[120:123], v[178:181], v1, v127 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], a[60:63], a[112:115], v[94:97], v255, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[48:51], a[120:123], v[174:177], v1, v127 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], a[56:59], a[112:115], v[94:97], v255, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[36:39], a[120:123], v[166:169], v1, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[32:35], a[120:123], a[200:203], v1, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], a[44:47], a[112:115], v[94:97], v255, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[136:139], a[40:43], a[112:115], a[136:139], v255, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_perm_b32 v87, v251, v250, v135
	v_accvgpr_read_b32 v253, a167
	v_perm_b32 v255, v149, v148, v135
	v_perm_b32 v1, v147, v146, v135
	v_accvgpr_read_b32 v146, a168
	v_accvgpr_read_b32 v252, a166
	v_accvgpr_read_b32 v251, a165
	v_accvgpr_read_b32 v250, a164
	v_accvgpr_read_b32 v147, a169
	v_accvgpr_read_b32 v148, a170
	v_accvgpr_read_b32 v149, a171
	v_perm_b32 v94, v118, v110, v135
	v_perm_b32 v0, v136, v0, v135
	v_perm_b32 v86, v137, v113, v135
	v_perm_b32 v88, v120, v112, v135
	v_perm_b32 v89, v119, v111, v135
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], a[84:87], a[20:23], v[58:61], v128, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_lshl_b32 s5, s5, 1
	s_xor_b32 s5, s5, 2
	s_mulk_i32 s5, 0x4200
	v_add_u32_e32 v95, s5, v158
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], a[92:95], a[28:31], v[58:61], v126, v129 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[52:55], v95
	v_accvgpr_read_b32 v245, a151
	v_accvgpr_read_b32 v244, a150
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], a[72:75], a[20:23], v[54:57], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v243, a149
	v_accvgpr_read_b32 v242, a148
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], a[76:79], a[28:31], v[54:57], v126, v129 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[48:51], v95 offset:4224
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], a[64:67], a[20:23], v[50:53], v128, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], a[68:71], a[28:31], v[50:53], v126, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[36:39], v95 offset:8448
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], a[124:127], a[20:23], v[46:49], v128, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], a[116:119], a[28:31], v[46:49], v126, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[32:35], v95 offset:12672
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], a[84:87], a[12:15], v[42:45], v128, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], a[92:95], a[24:27], v[42:45], v126, v129 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[60:63], v95 offset:64
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], a[72:75], a[12:15], v[38:41], v128, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], a[76:79], a[24:27], v[38:41], v126, v129 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[56:59], v95 offset:4288
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], a[64:67], a[12:15], v[34:37], v128, v134 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], a[68:71], a[24:27], v[34:37], v126, v129 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[44:47], v95 offset:8512
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], a[124:127], a[12:15], v[30:33], v128, v134 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], a[116:119], a[24:27], v[30:33], v126, v129 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[40:43], v95 offset:12736
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], a[84:87], a[8:11], v[26:29], v128, v134 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], a[72:75], a[8:11], v[22:25], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], a[92:95], a[16:19], v[26:29], v126, v129 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], a[76:79], a[16:19], v[22:25], v126, v129 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], a[64:67], a[8:11], v[18:21], v128, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], a[124:127], a[8:11], v[14:17], v128, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], a[68:71], a[16:19], v[18:21], v126, v129 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], a[116:119], a[16:19], v[14:17], v126, v129 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], a[84:87], a[0:3], v[10:13], v128, v134 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], a[72:75], a[0:3], v[6:9], v128, v134 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], a[92:95], a[4:7], v[10:13], v126, v129 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], a[76:79], a[4:7], v[6:9], v126, v129 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], a[64:67], a[0:3], v[2:5], v128, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[0:3], a[124:127], a[0:3], a[192:195], v128, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], a[68:71], a[4:7], v[2:5], v126, v129 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[116:119], a[4:7], a[0:3], v126, v129 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], a[84:87], a[104:107], v[142:145], v128, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_add_u32_e32 v95, s5, v164
	v_accvgpr_mov_b32 a171, a147
	v_accvgpr_mov_b32 a170, a146
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], a[92:95], a[108:111], v[110:113], v126, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[20:23], v95
	v_accvgpr_mov_b32 a169, a145
	v_accvgpr_mov_b32 a168, a144
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], a[72:75], a[104:107], v[138:141], v128, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], a[76:79], a[108:111], v[110:113], v126, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[12:15], v95 offset:4224
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], a[64:67], a[104:107], v[130:133], v128, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], a[68:71], a[108:111], v[110:113], v126, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[8:11], v95 offset:8448
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], a[124:127], a[104:107], v[122:125], v128, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], a[116:119], a[108:111], v[110:113], v126, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[0:3], v95 offset:12672
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], a[84:87], a[96:99], v[114:117], v128, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], a[92:95], a[100:103], v[110:113], v126, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[28:31], v95 offset:64
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], a[72:75], a[96:99], v[106:109], v128, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_nop 4
	v_accvgpr_read_b32 v110, a160
	v_accvgpr_read_b32 v111, a161
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], a[76:79], a[100:103], v[106:109], v126, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[24:27], v95 offset:4288
	v_accvgpr_read_b32 v112, a162
	v_accvgpr_read_b32 v113, a163
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], a[64:67], a[96:99], v[102:105], v128, v127 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], a[68:71], a[100:103], v[102:105], v126, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[16:19], v95 offset:8512
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[96:99], a[124:127], a[96:99], v[98:101], v128, v127 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], a[84:87], a[80:83], v[90:93], v128, v127 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], a[72:75], a[80:83], v[82:85], v128, v127 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], a[64:67], a[80:83], v[78:81], v128, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], a[124:127], a[80:83], v[74:77], v128, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], a[84:87], a[120:123], v[70:73], v128, v127 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], a[72:75], a[120:123], v[66:69], v128, v127 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], a[64:67], a[120:123], v[62:65], v128, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[124:127], a[120:123], a[196:199], v128, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], a[116:119], a[100:103], v[96:99], v126, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[4:7], v95 offset:12736
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], a[92:95], a[88:91], v[90:93], v126, v121 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], a[76:79], a[88:91], v[82:85], v126, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], a[68:71], a[88:91], v[78:81], v126, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], a[116:119], a[88:91], v[74:77], v126, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], a[92:95], a[112:115], v[70:73], v126, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], a[76:79], a[112:115], v[66:69], v126, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], a[68:71], a[112:115], v[62:65], v126, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 a[176:179], a[116:119], a[112:115], a[176:179], v126, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_mov_b32_e32 v134, v94
	v_accvgpr_read_b32 v94, a140
	v_mov_b32_e32 v128, v87
	v_mov_b32_e32 v126, v0
	v_accvgpr_read_b32 v95, a141
	v_accvgpr_read_b32 v96, a142
	v_accvgpr_read_b32 v97, a143
	v_mov_b32_e32 v129, v89
	v_mov_b32_e32 v127, v88
	v_mov_b32_e32 v121, v86
	s_mov_b32 s8, s4
	s_branch .LBB0_16
.LBB0_15:
	v_accvgpr_read_b32 v253, a135
	v_accvgpr_read_b32 v160, a168
	v_accvgpr_read_b32 v252, a134
	v_accvgpr_read_b32 v251, a133
	v_accvgpr_read_b32 v250, a132
	v_accvgpr_read_b32 v245, a163
	v_accvgpr_read_b32 v154, a164
	v_accvgpr_read_b32 v161, a169
	v_accvgpr_read_b32 v162, a170
	v_accvgpr_read_b32 v163, a171
	v_accvgpr_mov_b32 a168, a172
	v_accvgpr_mov_b32 a135, a67
	v_accvgpr_mov_b32 a139, a71
	v_accvgpr_read_b32 v244, a162
	v_accvgpr_read_b32 v243, a161
	v_accvgpr_read_b32 v242, a160
	v_accvgpr_read_b32 v155, a165
	v_accvgpr_read_b32 v156, a166
	v_accvgpr_read_b32 v157, a167
	v_accvgpr_mov_b32 a169, a173
	v_accvgpr_mov_b32 a170, a174
	v_accvgpr_mov_b32 a171, a175
	v_accvgpr_mov_b32 a134, a66
	v_accvgpr_mov_b32 a133, a65
	v_accvgpr_mov_b32 a132, a64
	v_accvgpr_mov_b32 a138, a70
	v_accvgpr_mov_b32 a137, a69
	v_accvgpr_mov_b32 a136, a68
.LBB0_16:
	s_add_i32 s9, s8, 3
	s_and_b32 s4, s8, 0x80000001
	s_cmp_eq_u32 s4, 1
	s_cselect_b64 s[4:5], -1, 0
	s_cmp_lt_i32 s9, s6
	s_cselect_b64 s[14:15], -1, 0
	s_lshl_b32 s6, s8, 1
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[180:183], a[52:55], a[20:23], a[188:191], v1, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	s_and_b32 s6, s6, 2
	s_mulk_i32 s6, 0x4200
	v_add_u32_e32 v0, s6, v164
	ds_read_b128 a[120:123], v0 offset:16896
	s_waitcnt lgkmcnt(4)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[140:143], a[60:63], a[28:31], a[180:183], v255, v129 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[112:115], v0 offset:21120
	s_and_b64 s[4:5], s[4:5], s[14:15]
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[48:51], a[20:23], a[204:207], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[96:99], v0 offset:25344
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[144:147], a[56:59], a[28:31], a[144:147], v255, v129 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[72:75], v0 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[36:39], a[20:23], a[208:211], v1, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[124:127], v0 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[160:163], a[44:47], a[28:31], a[160:163], v255, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[116:119], v0 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[32:35], a[20:23], a[212:215], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[104:107], v0 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_waitcnt lgkmcnt(7)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[164:167], a[40:43], a[28:31], a[164:167], v255, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[64:67], v0 offset:29632
	v_add_u32_e32 v0, s6, v158
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[52:55], a[12:15], a[168:171], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[100:103], v0 offset:16896
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 a[168:171], a[60:63], a[24:27], a[168:171], v255, v129 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[88:91], v0 offset:21120
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[48:51], a[12:15], v[160:163], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[80:83], v0 offset:25344
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[158:161], a[56:59], a[24:27], v[86:89], v255, v129 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	ds_read_b128 a[76:79], v0 offset:29568
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[36:39], a[12:15], v[154:157], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[108:111], v0 offset:16960
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[154:157], a[44:47], a[24:27], v[86:89], v255, v129 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[92:95], v0 offset:21184
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[32:35], a[12:15], v[242:245], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[84:87], v0 offset:25408
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[238:241], a[40:43], a[24:27], v[86:89], v255, v129 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	ds_read_b128 a[68:71], v0 offset:29632
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_waitcnt lgkmcnt(0)
	; sched_barrier mask(0x00000000)
	s_nop 3
	v_accvgpr_read_b32 v86, a156
	v_accvgpr_read_b32 v87, a157
	v_accvgpr_read_b32 v88, a158
	v_accvgpr_read_b32 v89, a159
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[52:55], a[8:11], v[86:89], v1, v134 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[242:245], a[60:63], a[16:19], v[86:89], v255, v129 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(1) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	s_nop 6
	v_accvgpr_read_b32 v86, a152
	v_accvgpr_read_b32 v87, a153
	v_accvgpr_read_b32 v88, a154
	v_accvgpr_read_b32 v89, a155
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[48:51], a[8:11], v[86:89], v1, v134 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[246:249], a[56:59], a[16:19], v[86:89], v255, v129 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[36:39], a[8:11], v[250:253], v1, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[250:253], a[44:47], a[16:19], v[86:89], v255, v129 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[32:35], a[8:11], v[150:153], v1, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[86:89], a[40:43], a[16:19], v[86:89], v255, v129 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[52:55], a[0:3], v[94:97], v1, v134 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[206:209], a[60:63], a[4:7], v[94:97], v255, v129 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[48:51], a[0:3], v[146:149], v1, v134 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[190:193], a[56:59], a[4:7], v[94:97], v255, v129 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[36:39], a[0:3], v[110:113], v1, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[170:173], a[44:47], a[4:7], v[94:97], v255, v129 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(3) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	s_nop 6
	v_accvgpr_read_b32 v94, a132
	v_accvgpr_read_b32 v95, a133
	v_accvgpr_read_b32 v96, a134
	v_accvgpr_read_b32 v97, a135
	s_nop 1
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[32:35], a[0:3], v[94:97], v1, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[162:165], a[40:43], a[4:7], v[94:97], v255, v129 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[52:55], a[120:123], v[230:233], v1, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], a[48:51], a[120:123], v[226:229], v1, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[94:97], a[60:63], a[124:127], v[94:97], v255, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[110:113], a[56:59], a[124:127], v[110:113], v255, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], a[36:39], a[120:123], v[234:237], v1, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[226:229], a[44:47], a[124:127], v[146:149], v255, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], a[32:35], a[120:123], v[222:225], v1, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[222:225], a[40:43], a[124:127], v[146:149], v255, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], a[52:55], a[112:115], v[218:221], v1, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[230:233], a[60:63], a[116:119], v[146:149], v255, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], a[48:51], a[112:115], v[214:217], v1, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[146:149], a[56:59], a[116:119], v[146:149], v255, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[36:39], a[112:115], v[210:213], v1, v127 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], a[32:35], a[112:115], v[202:205], v1, v127 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 1
	v_accvgpr_read_b32 v213, a139
	v_accvgpr_read_b32 v212, a138
	v_accvgpr_read_b32 v211, a137
	v_mfma_scale_f32_16x16x128_f8f6f4 v[150:153], a[44:47], a[116:119], v[150:153], v255, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v210, a136
	v_mfma_scale_f32_16x16x128_f8f6f4 v[202:205], a[40:43], a[116:119], v[202:205], v255, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], a[52:55], a[96:99], v[198:201], v1, v127 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], a[48:51], a[96:99], v[194:197], v1, v127 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[198:201], a[60:63], a[104:107], v[198:201], v255, v121 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[194:197], a[56:59], a[104:107], v[194:197], v255, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], a[36:39], a[96:99], v[186:189], v1, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], a[32:35], a[96:99], v[182:185], v1, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[186:189], a[44:47], a[104:107], v[186:189], v255, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[182:185], a[40:43], a[104:107], v[182:185], v255, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], a[52:55], a[72:75], v[178:181], v1, v127 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], a[48:51], a[72:75], v[174:177], v1, v127 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[178:181], a[60:63], a[64:67], v[178:181], v255, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[174:177], a[56:59], a[64:67], v[174:177], v255, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], a[36:39], a[72:75], v[166:169], v1, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], a[32:35], a[72:75], v[210:213], v1, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[166:169], a[44:47], a[64:67], v[166:169], v255, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[210:213], a[40:43], a[64:67], v[210:213], v255, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	s_andn2_b64 vcc, exec, s[4:5]
	s_cbranch_vccnz .LBB0_18
; %bb.17:
	s_lshl_b32 s4, s9, 11
	s_and_b32 s4, s4, 0x1000
	s_lshl_b32 s5, s9, 3
	s_add_i32 m0, s13, s4
	v_accvgpr_read_b32 v0, a131
	buffer_load_dwordx4 v0, s[16:19], s5 offen lds
	s_add_i32 m0, s12, s4
	s_mov_b32 s23, s19
	buffer_load_dwordx4 v254, s[20:23], s5 offen lds
.LBB0_18:                               ; %_ZZ28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargsENKUliT_T0_E_clIN4opus6numberILi0EEES8_EEDaiS2_S3_.exit
	s_mul_i32 s4, s35, s3
	s_ashr_i32 s5, s4, 31
	s_lshl_b64 s[4:5], s[4:5], 1
	s_add_u32 s3, s0, s4
	s_mul_i32 s0, s33, s10
	s_addc_u32 s4, s1, s5
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[0:1], s[0:1], 1
	s_add_u32 s5, s3, s0
	s_addc_u32 s4, s4, s1
	s_ashr_i32 s3, s2, 31
	s_lshl_b64 s[0:1], s[2:3], 1
	s_add_u32 s0, s5, s0
	s_addc_u32 s1, s4, s1
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s3, 0x20000
	s_mov_b32 s2, -1
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], a[100:103], a[20:23], v[58:61], v128, v134 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v217, a151
	v_accvgpr_read_b32 v216, a150
	v_accvgpr_read_b32 v215, a149
	v_mfma_scale_f32_16x16x128_f8f6f4 v[58:61], a[108:111], a[28:31], v[58:61], v126, v129 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v214, a148
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], a[88:91], a[20:23], v[54:57], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[54:57], a[92:95], a[28:31], v[54:57], v126, v129 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], a[80:83], a[20:23], v[50:53], v128, v134 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[50:53], a[84:87], a[28:31], v[50:53], v126, v129 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], a[76:79], a[20:23], v[46:49], v128, v134 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[46:49], a[68:71], a[28:31], v[46:49], v126, v129 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], a[100:103], a[12:15], v[42:45], v128, v134 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[42:45], a[108:111], a[24:27], v[42:45], v126, v129 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], a[88:91], a[12:15], v[38:41], v128, v134 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[38:41], a[92:95], a[24:27], v[38:41], v126, v129 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], a[80:83], a[12:15], v[34:37], v128, v134 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[34:37], a[84:87], a[24:27], v[34:37], v126, v129 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], a[76:79], a[12:15], v[30:33], v128, v134 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[30:33], a[68:71], a[24:27], v[30:33], v126, v129 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], a[100:103], a[8:11], v[26:29], v128, v134 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], a[88:91], a[8:11], v[22:25], v128, v134 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[26:29], a[108:111], a[16:19], v[26:29], v126, v129 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[22:25], a[92:95], a[16:19], v[22:25], v126, v129 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], a[80:83], a[8:11], v[18:21], v128, v134 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], a[76:79], a[8:11], v[14:17], v128, v134 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[18:21], a[84:87], a[16:19], v[18:21], v126, v129 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[14:17], a[68:71], a[16:19], v[14:17], v126, v129 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], a[100:103], a[0:3], v[10:13], v128, v134 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], a[88:91], a[0:3], v[6:9], v128, v134 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[10:13], a[108:111], a[4:7], v[10:13], v126, v129 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[6:9], a[92:95], a[4:7], v[6:9], v126, v129 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[0:3], a[80:83], a[0:3], v[2:5], v128, v134 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], a[76:79], a[0:3], v[214:217], v128, v134 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[2:5], a[84:87], a[4:7], v[0:3], v126, v129 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[214:217], a[68:71], a[4:7], v[134:137], v126, v129 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(4) SyncID(0)
	; sched_group_barrier mask(0x00000020) size(1) SyncID(0)
	; sched_barrier mask(0x00000000)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], a[100:103], a[120:123], v[142:145], v128, v127 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[142:145], a[108:111], a[124:127], v[134:137], v126, v121 op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], a[88:91], a[120:123], v[138:141], v128, v127 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[138:141], a[92:95], a[124:127], v[134:137], v126, v121 op_sel:[1,0,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], a[80:83], a[120:123], v[130:133], v128, v127 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	s_nop 5
	v_accvgpr_read_b32 v134, a176
	v_accvgpr_read_b32 v135, a177
	v_mfma_scale_f32_16x16x128_f8f6f4 v[130:133], a[84:87], a[124:127], v[130:133], v126, v121 op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_accvgpr_read_b32 v136, a178
	v_accvgpr_read_b32 v137, a179
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], a[76:79], a[120:123], v[122:125], v128, v127 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[122:125], a[68:71], a[124:127], v[122:125], v126, v121 op_sel:[1,0,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], a[100:103], a[112:115], v[114:117], v128, v127 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[114:117], a[108:111], a[116:119], v[114:117], v126, v121 op_sel:[0,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], a[88:91], a[112:115], v[106:109], v128, v127 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[106:109], a[92:95], a[116:119], v[106:109], v126, v121 op_sel:[1,1,0] op_sel_hi:[0,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], a[80:83], a[112:115], v[102:105], v128, v127 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[102:105], a[84:87], a[116:119], v[102:105], v126, v121 op_sel:[0,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], a[76:79], a[112:115], v[98:101], v128, v127 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], a[100:103], a[96:99], v[90:93], v128, v127 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], a[88:91], a[96:99], v[82:85], v128, v127 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], a[80:83], a[96:99], v[78:81], v128, v127 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], a[76:79], a[96:99], v[74:77], v128, v127 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], a[100:103], a[72:75], v[70:73], v128, v127 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], a[88:91], a[72:75], v[66:69], v128, v127 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], a[80:83], a[72:75], v[62:65], v128, v127 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[134:137], a[76:79], a[72:75], v[134:137], v128, v127 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[98:101], a[68:71], a[116:119], v[98:101], v126, v121 op_sel:[1,1,0] op_sel_hi:[1,0,0] cbsz:4 blgp:4
	; sched_group_barrier mask(0x00000008) size(2) SyncID(0)
	; sched_group_barrier mask(0x00000100) size(1) SyncID(0)
	v_mfma_scale_f32_16x16x128_f8f6f4 v[90:93], a[108:111], a[104:107], v[90:93], v126, v121 op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[82:85], a[92:95], a[104:107], v[82:85], v126, v121 op_sel:[1,0,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[78:81], a[84:87], a[104:107], v[78:81], v126, v121 op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[74:77], a[68:71], a[104:107], v[74:77], v126, v121 op_sel:[1,0,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[70:73], a[108:111], a[64:67], v[70:73], v126, v121 op_sel:[0,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[66:69], a[92:95], a[64:67], v[66:69], v126, v121 op_sel:[1,1,0] op_sel_hi:[0,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[62:65], a[84:87], a[64:67], v[62:65], v126, v121 op_sel:[0,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	v_mfma_scale_f32_16x16x128_f8f6f4 v[218:221], a[68:71], a[64:67], v[134:137], v126, v121 op_sel:[1,1,0] op_sel_hi:[1,1,0] cbsz:4 blgp:4
	; sched_barrier mask(0x00000000)
	v_accvgpr_read_b32 v0, a130
	s_mul_i32 s4, s7, 0x880
	v_mul_u32_u24_e32 v0, 0x88, v0
	v_accvgpr_read_b32 v1, a129
	v_lshl_add_u32 v118, v1, 2, v0
	s_add_i32 s4, s4, s11
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_accvgpr_read_b32 v126, a140
	v_accvgpr_read_b32 v134, a144
	v_accvgpr_read_b32 v127, a141
	v_accvgpr_read_b32 v128, a142
	v_accvgpr_read_b32 v129, a143
	v_accvgpr_read_b32 v135, a145
	v_accvgpr_read_b32 v136, a146
	v_accvgpr_read_b32 v137, a147
	v_cvt_pk_bf16_f32 v1, v128, v129
	v_cvt_pk_bf16_f32 v0, v126, v127
	v_add_lshl_u32 v128, s4, v118, 1
	v_cvt_pk_bf16_f32 v119, v136, v137
	v_cvt_pk_bf16_f32 v118, v134, v135
	ds_write2_b64 v128, v[0:1], v[118:119] offset1:8
	v_accvgpr_read_b32 v118, a160
	v_accvgpr_read_b32 v134, a164
	v_accvgpr_read_b32 v119, a161
	v_accvgpr_read_b32 v120, a162
	v_accvgpr_read_b32 v121, a163
	v_accvgpr_read_b32 v135, a165
	v_accvgpr_read_b32 v136, a166
	v_accvgpr_read_b32 v137, a167
	v_cvt_pk_bf16_f32 v1, v120, v121
	v_cvt_pk_bf16_f32 v0, v118, v119
	v_cvt_pk_bf16_f32 v119, v136, v137
	v_cvt_pk_bf16_f32 v118, v134, v135
	ds_write2_b64 v128, v[0:1], v[118:119] offset0:16 offset1:24
	v_accvgpr_read_b32 v118, a168
	v_accvgpr_read_b32 v119, a169
	v_accvgpr_read_b32 v120, a170
	v_accvgpr_read_b32 v121, a171
	v_cvt_pk_bf16_f32 v1, v120, v121
	v_cvt_pk_bf16_f32 v0, v118, v119
	v_cvt_pk_bf16_f32 v119, v160, v161
	v_cvt_pk_bf16_f32 v118, v158, v159
	v_add_u32_e32 v129, 0x2000, v128
	ds_write2_b64 v129, v[0:1], v[118:119] offset0:64 offset1:72
	v_cvt_pk_bf16_f32 v1, v156, v157
	v_cvt_pk_bf16_f32 v0, v154, v155
	v_cvt_pk_bf16_f32 v119, v240, v241
	v_cvt_pk_bf16_f32 v118, v238, v239
	ds_write2_b64 v129, v[0:1], v[118:119] offset0:80 offset1:88
	v_cvt_pk_bf16_f32 v1, v244, v245
	v_cvt_pk_bf16_f32 v0, v242, v243
	v_cvt_pk_bf16_f32 v119, v248, v249
	v_cvt_pk_bf16_f32 v118, v246, v247
	v_add_u32_e32 v134, 0x4000, v128
	ds_write2_b64 v134, v[0:1], v[118:119] offset0:128 offset1:136
	v_cvt_pk_bf16_f32 v1, v252, v253
	v_cvt_pk_bf16_f32 v0, v250, v251
	v_cvt_pk_bf16_f32 v89, v88, v89
	v_cvt_pk_bf16_f32 v88, v86, v87
	ds_write2_b64 v134, v[0:1], v[88:89] offset0:144 offset1:152
	v_cvt_pk_bf16_f32 v1, v208, v209
	v_cvt_pk_bf16_f32 v0, v206, v207
	v_cvt_pk_bf16_f32 v87, v192, v193
	v_cvt_pk_bf16_f32 v86, v190, v191
	v_add_u32_e32 v135, 0x6000, v128
	ds_write2_b64 v135, v[0:1], v[86:87] offset0:192 offset1:200
	v_cvt_pk_bf16_f32 v1, v172, v173
	v_cvt_pk_bf16_f32 v0, v170, v171
	v_cvt_pk_bf16_f32 v87, v164, v165
	v_cvt_pk_bf16_f32 v86, v162, v163
	ds_write2_b64 v135, v[0:1], v[86:87] offset0:208 offset1:216
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_accvgpr_read_b32 v88, a128
	v_lshlrev_b32_e32 v0, 3, v88
	v_lshrrev_b32_e32 v1, 4, v88
	v_and_b32_e32 v0, 0x78, v0
	v_mul_u32_u24_e32 v86, 0x88, v1
	v_add_lshl_u32 v136, v86, v0, 1
	ds_read_b128 v[118:121], v136
	v_mad_u64_u32 v[86:87], s[4:5], s10, v1, v[0:1]
	v_bfe_u32 v1, v88, 4, 25
	v_or_b32_e32 v87, 16, v1
	v_mul_u32_u24_e32 v88, 0x88, v87
	v_lshlrev_b32_e32 v137, 1, v86
	v_add_lshl_u32 v154, v88, v0, 1
	v_mad_u64_u32 v[88:89], s[4:5], s10, v87, v[0:1]
	v_or_b32_e32 v87, 32, v1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[118:121], v137, s[0:3], 0 offen
	ds_read_b128 v[118:121], v154
	v_mul_u32_u24_e32 v89, 0x88, v87
	v_add_lshl_u32 v89, v89, v0, 1
	ds_read_b128 v[156:159], v89
	v_lshlrev_b32_e32 v155, 1, v88
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[118:121], v155, s[0:3], 0 offen
	ds_read_b128 v[160:163], v136 offset:17408
	s_lshl_b32 s6, s10, 6
	v_mad_u64_u32 v[118:119], s[4:5], s10, v87, v[0:1]
	v_or_b32_e32 v87, 48, v1
	v_lshlrev_b32_e32 v119, 1, v118
	v_mul_u32_u24_e32 v120, 0x88, v87
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[156:159], v119, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v97, v96, v97
	v_cvt_pk_bf16_f32 v96, v94, v95
	v_add_lshl_u32 v156, v120, v0, 1
	ds_read_b128 v[170:173], v156
	v_mad_u64_u32 v[120:121], s[4:5], s10, v87, v[0:1]
	v_or_b32_e32 v87, 0x50, v1
	v_mul_u32_u24_e32 v126, 0x88, v87
	v_add_lshl_u32 v157, v126, v0, 1
	v_add_u32_e32 v158, s6, v86
	v_or_b32_e32 v126, 0x60, v1
	v_lshlrev_b32_e32 v121, 1, v120
	v_lshlrev_b32_e32 v159, 1, v158
	v_mul_u32_u24_e32 v127, 0x88, v126
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[170:173], v121, s[0:3], 0 offen
	buffer_store_dwordx4 v[160:163], v159, s[0:3], 0 offen
	ds_read_b128 v[190:193], v157
	v_mad_u64_u32 v[86:87], s[4:5], s10, v87, v[0:1]
	v_add_lshl_u32 v160, v127, v0, 1
	ds_read_b128 v[162:165], v160
	v_mad_u64_u32 v[126:127], s[4:5], s10, v126, v[0:1]
	v_or_b32_e32 v1, 0x70, v1
	v_lshlrev_b32_e32 v87, 1, v86
	v_mul_u32_u24_e32 v127, 0x88, v1
	v_lshlrev_b32_e32 v161, 1, v126
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[190:193], v87, s[0:3], 0 offen
	v_add_lshl_u32 v127, v127, v0, 1
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[162:165], v161, s[0:3], 0 offen
	v_cvt_pk_bf16_f32 v95, v112, v113
	v_cvt_pk_bf16_f32 v94, v110, v111
	v_add_u32_e32 v163, 0x10800, v128
	ds_read_b128 v[170:173], v127
	ds_write2_b64 v163, v[96:97], v[94:95] offset1:8
	v_cvt_pk_bf16_f32 v95, v228, v229
	v_cvt_pk_bf16_f32 v94, v226, v227
	v_cvt_pk_bf16_f32 v97, v224, v225
	v_cvt_pk_bf16_f32 v96, v222, v223
	ds_write2_b64 v163, v[94:95], v[96:97] offset0:16 offset1:24
	v_cvt_pk_bf16_f32 v95, v232, v233
	v_cvt_pk_bf16_f32 v94, v230, v231
	v_cvt_pk_bf16_f32 v97, v148, v149
	v_cvt_pk_bf16_f32 v96, v146, v147
	v_add_u32_e32 v110, 0x2000, v163
	ds_write2_b64 v110, v[94:95], v[96:97] offset0:64 offset1:72
	v_cvt_pk_bf16_f32 v95, v152, v153
	v_cvt_pk_bf16_f32 v94, v150, v151
	v_cvt_pk_bf16_f32 v97, v204, v205
	v_cvt_pk_bf16_f32 v96, v202, v203
	ds_write2_b64 v110, v[94:95], v[96:97] offset0:80 offset1:88
	v_cvt_pk_bf16_f32 v95, v200, v201
	v_cvt_pk_bf16_f32 v94, v198, v199
	v_cvt_pk_bf16_f32 v97, v196, v197
	v_cvt_pk_bf16_f32 v96, v194, v195
	v_add_u32_e32 v111, 0x4000, v163
	ds_write2_b64 v111, v[94:95], v[96:97] offset0:128 offset1:136
	v_cvt_pk_bf16_f32 v95, v188, v189
	v_cvt_pk_bf16_f32 v94, v186, v187
	v_cvt_pk_bf16_f32 v97, v184, v185
	v_cvt_pk_bf16_f32 v96, v182, v183
	v_mad_u64_u32 v[0:1], s[4:5], s10, v1, v[0:1]
	ds_write2_b64 v111, v[94:95], v[96:97] offset0:144 offset1:152
	v_cvt_pk_bf16_f32 v95, v180, v181
	v_cvt_pk_bf16_f32 v94, v178, v179
	v_cvt_pk_bf16_f32 v97, v176, v177
	v_cvt_pk_bf16_f32 v96, v174, v175
	v_add_u32_e32 v112, 0x6000, v163
	v_lshlrev_b32_e32 v162, 1, v0
	ds_write2_b64 v112, v[94:95], v[96:97] offset0:192 offset1:200
	v_cvt_pk_bf16_f32 v95, v168, v169
	v_cvt_pk_bf16_f32 v94, v166, v167
	v_cvt_pk_bf16_f32 v97, v212, v213
	v_cvt_pk_bf16_f32 v96, v210, v211
	s_waitcnt lgkmcnt(7)
	buffer_store_dwordx4 v[170:173], v162, s[0:3], 0 offen
	ds_write2_b64 v112, v[94:95], v[96:97] offset0:208 offset1:216
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	v_add_u32_e32 v94, 0x10800, v136
	ds_read_b128 v[146:149], v94
	v_add_lshl_u32 v95, v158, s6, 1
	v_add_u32_e32 v96, 0x10800, v154
	v_add_u32_e32 v97, 0x10800, v89
	ds_read_b128 v[150:153], v96
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[146:149], v95, s[0:3], 0 offen
	ds_read_b128 v[146:149], v97
	s_lshl_b32 s4, s10, 7
	v_add_lshl_u32 v88, v88, s4, 1
	v_add_lshl_u32 v113, v118, s4, 1
	v_add_u32_e32 v118, 0x10800, v156
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[150:153], v88, s[0:3], 0 offen
	ds_read_b128 v[150:153], v118
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[146:149], v113, s[0:3], 0 offen
	v_add_lshl_u32 v120, v120, s4, 1
	v_cvt_pk_bf16_f32 v1, v60, v61
	v_add_u32_e32 v146, 0x14c00, v136
	ds_read_b128 v[164:167], v146
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[150:153], v120, s[0:3], 0 offen
	v_add_lshl_u32 v147, v158, s4, 1
	v_add_u32_e32 v148, 0x10800, v157
	v_add_u32_e32 v149, 0x10800, v160
	v_add_u32_e32 v150, 0x10800, v127
	v_add_lshl_u32 v151, v0, s4, 1
	v_cvt_pk_bf16_f32 v0, v58, v59
	v_cvt_pk_bf16_f32 v57, v56, v57
	v_cvt_pk_bf16_f32 v56, v54, v55
	ds_read_b128 v[168:171], v148
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[164:167], v147, s[0:3], 0 offen
	ds_read_b128 v[164:167], v149
	ds_read_b128 v[172:175], v150
	ds_write2_b64 v128, v[0:1], v[56:57] offset1:8
	v_cvt_pk_bf16_f32 v1, v52, v53
	v_cvt_pk_bf16_f32 v0, v50, v51
	v_cvt_pk_bf16_f32 v49, v48, v49
	v_cvt_pk_bf16_f32 v48, v46, v47
	ds_write2_b64 v128, v[0:1], v[48:49] offset0:16 offset1:24
	v_cvt_pk_bf16_f32 v1, v44, v45
	v_cvt_pk_bf16_f32 v0, v42, v43
	v_cvt_pk_bf16_f32 v41, v40, v41
	v_cvt_pk_bf16_f32 v40, v38, v39
	ds_write2_b64 v129, v[0:1], v[40:41] offset0:64 offset1:72
	v_cvt_pk_bf16_f32 v1, v36, v37
	v_cvt_pk_bf16_f32 v0, v34, v35
	v_cvt_pk_bf16_f32 v33, v32, v33
	v_cvt_pk_bf16_f32 v32, v30, v31
	ds_write2_b64 v129, v[0:1], v[32:33] offset0:80 offset1:88
	v_cvt_pk_bf16_f32 v1, v28, v29
	v_cvt_pk_bf16_f32 v0, v26, v27
	v_cvt_pk_bf16_f32 v25, v24, v25
	v_cvt_pk_bf16_f32 v24, v22, v23
	ds_write2_b64 v134, v[0:1], v[24:25] offset0:128 offset1:136
	v_cvt_pk_bf16_f32 v1, v20, v21
	v_cvt_pk_bf16_f32 v0, v18, v19
	v_cvt_pk_bf16_f32 v17, v16, v17
	v_cvt_pk_bf16_f32 v16, v14, v15
	ds_write2_b64 v134, v[0:1], v[16:17] offset0:144 offset1:152
	v_cvt_pk_bf16_f32 v1, v12, v13
	v_cvt_pk_bf16_f32 v0, v10, v11
	v_cvt_pk_bf16_f32 v9, v8, v9
	v_cvt_pk_bf16_f32 v8, v6, v7
	v_add_lshl_u32 v86, v86, s4, 1
	v_add_lshl_u32 v126, v126, s4, 1
	ds_write2_b64 v135, v[0:1], v[8:9] offset0:192 offset1:200
	v_cvt_pk_bf16_f32 v1, v4, v5
	v_cvt_pk_bf16_f32 v0, v2, v3
	v_cvt_pk_bf16_f32 v3, v216, v217
	v_cvt_pk_bf16_f32 v2, v214, v215
	s_waitcnt lgkmcnt(9)
	buffer_store_dwordx4 v[168:171], v86, s[0:3], 0 offen
	s_waitcnt lgkmcnt(8)
	buffer_store_dwordx4 v[164:167], v126, s[0:3], 0 offen
	s_waitcnt lgkmcnt(7)
	buffer_store_dwordx4 v[172:175], v151, s[0:3], 0 offen
	ds_write2_b64 v135, v[0:1], v[2:3] offset0:208 offset1:216
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[0:3], v136
	ds_read_b128 v[4:7], v154
	ds_read_b128 v[8:11], v89
	ds_read_b128 v[12:15], v136 offset:17408
	ds_read_b128 v[16:19], v127
	s_waitcnt lgkmcnt(4)
	buffer_store_dwordx4 v[0:3], v137, s[0:3], 0 offen offset:256
	ds_read_b128 v[0:3], v156
	s_waitcnt lgkmcnt(4)
	buffer_store_dwordx4 v[4:7], v155, s[0:3], 0 offen offset:256
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[8:11], v119, s[0:3], 0 offen offset:256
	ds_read_b128 v[4:7], v157
	ds_read_b128 v[8:11], v160
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[0:3], v121, s[0:3], 0 offen offset:256
	buffer_store_dwordx4 v[12:15], v159, s[0:3], 0 offen offset:256
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[4:7], v87, s[0:3], 0 offen offset:256
	v_cvt_pk_bf16_f32 v1, v144, v145
	v_cvt_pk_bf16_f32 v0, v142, v143
	v_cvt_pk_bf16_f32 v3, v140, v141
	v_cvt_pk_bf16_f32 v2, v138, v139
	ds_write2_b64 v163, v[0:1], v[2:3] offset1:8
	v_cvt_pk_bf16_f32 v1, v132, v133
	v_cvt_pk_bf16_f32 v0, v130, v131
	v_cvt_pk_bf16_f32 v3, v124, v125
	v_cvt_pk_bf16_f32 v2, v122, v123
	ds_write2_b64 v163, v[0:1], v[2:3] offset0:16 offset1:24
	v_cvt_pk_bf16_f32 v1, v116, v117
	v_cvt_pk_bf16_f32 v0, v114, v115
	v_cvt_pk_bf16_f32 v3, v108, v109
	v_cvt_pk_bf16_f32 v2, v106, v107
	ds_write2_b64 v110, v[0:1], v[2:3] offset0:64 offset1:72
	v_cvt_pk_bf16_f32 v1, v104, v105
	v_cvt_pk_bf16_f32 v0, v102, v103
	v_cvt_pk_bf16_f32 v3, v100, v101
	v_cvt_pk_bf16_f32 v2, v98, v99
	ds_write2_b64 v110, v[0:1], v[2:3] offset0:80 offset1:88
	v_cvt_pk_bf16_f32 v1, v92, v93
	v_cvt_pk_bf16_f32 v0, v90, v91
	v_cvt_pk_bf16_f32 v3, v84, v85
	v_cvt_pk_bf16_f32 v2, v82, v83
	ds_write2_b64 v111, v[0:1], v[2:3] offset0:128 offset1:136
	v_cvt_pk_bf16_f32 v1, v80, v81
	v_cvt_pk_bf16_f32 v0, v78, v79
	v_cvt_pk_bf16_f32 v3, v76, v77
	v_cvt_pk_bf16_f32 v2, v74, v75
	ds_write2_b64 v111, v[0:1], v[2:3] offset0:144 offset1:152
	v_cvt_pk_bf16_f32 v1, v72, v73
	v_cvt_pk_bf16_f32 v0, v70, v71
	v_cvt_pk_bf16_f32 v3, v68, v69
	v_cvt_pk_bf16_f32 v2, v66, v67
	ds_write2_b64 v112, v[0:1], v[2:3] offset0:192 offset1:200
	v_cvt_pk_bf16_f32 v1, v64, v65
	v_cvt_pk_bf16_f32 v0, v62, v63
	v_cvt_pk_bf16_f32 v3, v220, v221
	v_cvt_pk_bf16_f32 v2, v218, v219
	s_waitcnt lgkmcnt(7)
	buffer_store_dwordx4 v[8:11], v161, s[0:3], 0 offen offset:256
	buffer_store_dwordx4 v[16:19], v162, s[0:3], 0 offen offset:256
	ds_write2_b64 v112, v[0:1], v[2:3] offset0:208 offset1:216
	s_waitcnt lgkmcnt(0)
	s_barrier
	; sched_barrier mask(0x00000000)
	ds_read_b128 v[0:3], v94
	ds_read_b128 v[4:7], v96
	ds_read_b128 v[8:11], v97
	ds_read_b128 v[12:15], v118
	ds_read_b128 v[16:19], v150
	s_waitcnt lgkmcnt(4)
	buffer_store_dwordx4 v[0:3], v95, s[0:3], 0 offen offset:256
	s_waitcnt lgkmcnt(3)
	buffer_store_dwordx4 v[4:7], v88, s[0:3], 0 offen offset:256
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[8:11], v113, s[0:3], 0 offen offset:256
	ds_read_b128 v[0:3], v146
	ds_read_b128 v[4:7], v148
	ds_read_b128 v[8:11], v149
	s_waitcnt lgkmcnt(4)
	buffer_store_dwordx4 v[12:15], v120, s[0:3], 0 offen offset:256
	s_waitcnt lgkmcnt(2)
	buffer_store_dwordx4 v[0:3], v147, s[0:3], 0 offen offset:256
	s_waitcnt lgkmcnt(1)
	buffer_store_dwordx4 v[4:7], v86, s[0:3], 0 offen offset:256
	s_waitcnt lgkmcnt(0)
	buffer_store_dwordx4 v[8:11], v126, s[0:3], 0 offen offset:256
	buffer_store_dwordx4 v[16:19], v151, s[0:3], 0 offen offset:256
	s_endpgm
.Lfunc_end0:
	.size	_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs, .Lfunc_end0-_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
		.amdhsa_group_segment_fixed_size 151552
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
		.amdhsa_next_free_vgpr 472
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
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.num_agpr, 216
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.numbered_sgpr, 44
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.num_named_barrier, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.private_seg_size, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.uses_vcc, 1
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.uses_flat_scratch, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.has_dyn_sized_stack, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.has_recursion, 0
	.set .L_Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 18552
; TotalNumSgprs: 50
; NumVgprs: 256
; NumAgprs: 216
; TotalNumVgprs: 472
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 151552 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 58
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 472
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
	.type	__hip_cuid_ab780f74c407790d,@object ; @__hip_cuid_ab780f74c407790d
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_ab780f74c407790d
__hip_cuid_ab780f74c407790d:
	.byte	0                               ; 0x0
	.size	__hip_cuid_ab780f74c407790d, 1

	.ident	"clang version 23.0.0git (https://github.com/ROCm/llvm-project.git 46fcb339fb61119b337f973c7ca9e710a319fdd0)"
	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_ab780f74c407790d
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     216
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .gfx1250_revision: B0
    .group_segment_fixed_size: 151552
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 256
    .name:           _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs
    .private_segment_fixed_size: 0
    .sgpr_count:     50
    .sgpr_spill_count: 0
    .symbol:         _Z28gemm_a4w4_mxfp4_4wave_kernelI28gemm_a4w4_mxfp4_4wave_traitsEv27opus_gemm_mxfp4_scale_kargs.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     472
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
