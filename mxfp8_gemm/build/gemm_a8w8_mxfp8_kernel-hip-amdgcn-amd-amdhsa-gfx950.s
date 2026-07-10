	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.section	.text._Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs,"axG",@progbits,_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs,comdat
	.protected	_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs ; -- Begin function _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs
	.globl	_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs
	.p2align	8
	.type	_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs,@function
_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs: ; @_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs
; %bb.0:
	s_load_dword s4, s[0:1], 0x1c
	s_load_dword s20, s[0:1], 0x30
	s_load_dword s14, s[0:1], 0x3c
	s_load_dwordx2 s[12:13], s[0:1], 0x10
	v_bfe_u32 v3, v0, 4, 2
	s_waitcnt lgkmcnt(0)
	s_addk_i32 s4, 0xff
	s_ashr_i32 s5, s4, 31
	s_lshr_b32 s5, s5, 24
	s_add_i32 s4, s4, s5
	s_ashr_i32 s15, s4, 8
	s_abs_i32 s16, s15
	v_cvt_f32_u32_e32 v1, s16
	s_sub_i32 s17, 0, s16
	s_load_dwordx8 s[4:11], s[0:1], 0x40
	s_abs_i32 s0, s2
	v_rcp_iflag_f32_e32 v1, v1
	s_xor_b32 s1, s2, s15
	s_ashr_i32 s1, s1, 31
	s_mul_i32 s14, s14, s3
	v_mul_f32_e32 v1, 0x4f7ffffe, v1
	v_cvt_u32_f32_e32 v1, v1
	v_lshlrev_b32_e32 v4, 2, v3
	scratch_store_dwordx2 off, v[4:5], off offset:284 ; 8-byte Folded Spill
	v_and_b32_e32 v2, 3, v0
	v_readfirstlane_b32 s18, v1
	s_mul_i32 s17, s17, s18
	s_mul_hi_u32 s17, s18, s17
	s_add_i32 s18, s18, s17
	s_mul_hi_u32 s17, s0, s18
	s_mul_i32 s18, s17, s16
	s_sub_i32 s0, s0, s18
	s_add_i32 s19, s17, 1
	s_sub_i32 s18, s0, s16
	s_cmp_ge_u32 s0, s16
	s_cselect_b32 s17, s19, s17
	s_cselect_b32 s0, s18, s0
	s_add_i32 s18, s17, 1
	s_cmp_ge_u32 s0, s16
	s_cselect_b32 s0, s18, s17
	s_xor_b32 s0, s0, s1
	s_sub_i32 s0, s0, s1
	s_lshl_b32 s16, s0, 8
	s_mul_i32 s0, s0, s15
	s_sub_i32 s0, s2, s0
	s_ashr_i32 s15, s14, 31
	s_lshl_b32 s0, s0, 8
	s_lshl_b64 s[14:15], s[14:15], 2
	s_add_u32 s1, s12, s14
	s_mul_i32 s12, s16, s20
	s_addc_u32 s2, s13, s15
	s_ashr_i32 s13, s12, 31
	s_lshl_b64 s[12:13], s[12:13], 2
	s_add_u32 s14, s1, s12
	s_addc_u32 s2, s2, s13
	s_ashr_i32 s1, s0, 31
	s_lshl_b64 s[12:13], s[0:1], 2
	s_add_u32 s12, s14, s12
	s_addc_u32 s1, s2, s13
	s_and_b32 s13, s1, 0xffff
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s1, s10, s3
	s_ashr_i32 s2, s1, 31
	s_add_u32 s1, s4, s1
	s_mul_i32 s16, s16, s8
	s_addc_u32 s2, s5, s2
	s_ashr_i32 s4, s16, 31
	s_add_u32 s16, s1, s16
	s_addc_u32 s1, s2, s4
	s_and_b32 s17, s1, 0xffff
	s_mul_i32 s1, s11, s3
	s_ashr_i32 s2, s1, 31
	s_add_u32 s1, s6, s1
	s_mul_i32 s0, s0, s9
	v_lshrrev_b32_e32 v1, 6, v0
	s_addc_u32 s2, s7, s2
	s_ashr_i32 s3, s0, 31
	v_readfirstlane_b32 s21, v1
	s_add_u32 s0, s1, s0
	s_addc_u32 s1, s2, s3
	s_ashr_i32 s4, s21, 31
	s_lshr_b32 s4, s4, 30
	s_add_i32 s5, s21, s4
	s_ashr_i32 s4, s5, 2
	s_and_b32 s5, s5, -4
	s_sub_i32 s5, s21, s5
	v_and_b32_e32 v1, 15, v0
	s_mov_b32 s15, 0x20000
	s_mov_b32 s14, -1
	v_lshl_or_b32 v1, s5, 4, v1
	s_mov_b32 s18, s14
	s_mov_b32 s19, s15
	scratch_store_dword off, v1, off offset:280 ; 4-byte Folded Spill
	v_mul_lo_u32 v7, v1, s8
	v_lshl_or_b32 v1, s4, 4, v4
	v_lshl_add_u32 v9, s8, 6, v7
	v_mul_lo_u32 v13, v1, s9
	buffer_load_ubyte v12, v7, s[16:19], 0 offen
	buffer_load_ubyte v1, v7, s[16:19], 0 offen offset:1
	buffer_load_ubyte v6, v7, s[16:19], 0 offen offset:2
	buffer_load_ubyte v4, v7, s[16:19], 0 offen offset:3
	buffer_load_ubyte v152, v7, s[16:19], 4 offen
	buffer_load_ubyte v160, v7, s[16:19], 4 offen offset:1
	buffer_load_ubyte v63, v7, s[16:19], 4 offen offset:2
	buffer_load_ubyte v5, v7, s[16:19], 4 offen offset:3
	buffer_load_ubyte v11, v9, s[16:19], 0 offen
	s_bfe_u32 s6, s5, 0x10007
	s_add_i32 s6, s5, s6
	s_bfe_i32 s6, s6, 0x80000
	s_sext_i32_i16 s6, s6
	s_lshr_b32 s7, s6, 1
	s_and_b32 s6, s6, 0xfe
	s_sub_i32 s6, s5, s6
	s_lshl_b32 s5, s9, 5
	v_add_u32_e32 v18, s5, v13
	v_add_u32_e32 v15, s9, v13
	v_add_u32_e32 v19, s9, v18
	v_add_u32_e32 v23, s5, v18
	s_and_b32 s1, s1, 0xffff
	s_mov_b32 s2, s14
	s_mov_b32 s3, s15
	v_add_u32_e32 v16, s9, v15
	v_add_u32_e32 v21, s9, v19
	v_add_u32_e32 v24, s9, v23
	v_add_u32_e32 v17, s9, v16
	v_add_u32_e32 v22, s9, v21
	v_add_u32_e32 v25, s9, v24
	v_add_u32_e32 v26, s9, v25
	v_add_u32_e32 v27, s5, v23
	s_sext_i32_i8 s10, s6
	s_lshl_b32 s6, s8, 7
	s_or_b32 s8, s6, 4
	s_lshl_b32 s5, s9, 7
	s_mulk_i32 s7, 0x1080
	v_lshlrev_b32_e32 v0, 5, v0
	v_and_b32_e32 v0, 0x180, v0
	v_lshlrev_b32_e32 v3, 3, v3
	v_add_u32_e32 v10, s9, v27
	v_add_u32_e32 v8, s9, v10
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:1592 ; 4-byte Folded Spill
	buffer_load_ubyte v11, v9, s[16:19], 0 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:1584 ; 4-byte Folded Spill
	buffer_load_ubyte v11, v9, s[16:19], 0 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:1580 ; 4-byte Folded Spill
	buffer_load_ubyte v11, v9, s[16:19], 0 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:1576 ; 4-byte Folded Spill
	buffer_load_ubyte v11, v9, s[16:19], 4 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:860 ; 4-byte Folded Spill
	buffer_load_ubyte v11, v9, s[16:19], 4 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:708 ; 4-byte Folded Spill
	buffer_load_ubyte v11, v9, s[16:19], 4 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:684 ; 4-byte Folded Spill
	buffer_load_ubyte v11, v9, s[16:19], 4 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v11, off offset:668 ; 4-byte Folded Spill
	buffer_load_ubyte v49, v13, s[0:3], 0 offen
	buffer_load_ubyte v14, v13, s[0:3], 0 offen offset:1
	buffer_load_ubyte v36, v13, s[0:3], 0 offen offset:2
	buffer_load_ubyte v29, v13, s[0:3], 0 offen offset:3
	s_nop 0
	buffer_load_ubyte v11, v13, s[0:3], 4 offen
	buffer_load_ubyte v163, v13, s[0:3], 4 offen offset:1
	buffer_load_ubyte v161, v13, s[0:3], 4 offen offset:2
	buffer_load_ubyte v93, v13, s[0:3], 4 offen offset:3
	buffer_load_ubyte v50, v15, s[0:3], 0 offen
	buffer_load_ubyte v45, v15, s[0:3], 0 offen offset:1
	buffer_load_ubyte v39, v15, s[0:3], 0 offen offset:2
	buffer_load_ubyte v30, v15, s[0:3], 0 offen offset:3
	buffer_load_ubyte v28, v15, s[0:3], 4 offen
	buffer_load_ubyte v188, v15, s[0:3], 4 offen offset:1
	buffer_load_ubyte v162, v15, s[0:3], 4 offen offset:2
	buffer_load_ubyte v61, v15, s[0:3], 4 offen offset:3
	buffer_load_ubyte v20, v16, s[0:3], 0 offen
	buffer_load_ubyte v46, v16, s[0:3], 0 offen offset:1
	buffer_load_ubyte v38, v16, s[0:3], 0 offen offset:2
	buffer_load_ubyte v31, v16, s[0:3], 0 offen offset:3
	buffer_load_ubyte v191, v16, s[0:3], 4 offen
	buffer_load_ubyte v189, v16, s[0:3], 4 offen offset:1
	buffer_load_ubyte v43, v16, s[0:3], 4 offen offset:2
	buffer_load_ubyte v60, v16, s[0:3], 4 offen offset:3
	buffer_load_ubyte v48, v17, s[0:3], 0 offen
	buffer_load_ubyte v47, v17, s[0:3], 0 offen offset:1
	buffer_load_ubyte v44, v17, s[0:3], 0 offen offset:2
	buffer_load_ubyte v37, v17, s[0:3], 0 offen offset:3
	buffer_load_ubyte v250, v17, s[0:3], 4 offen
	buffer_load_ubyte v190, v17, s[0:3], 4 offen offset:1
	buffer_load_ubyte v41, v17, s[0:3], 4 offen offset:2
	buffer_load_ubyte v42, v17, s[0:3], 4 offen offset:3
	buffer_load_ubyte v62, v18, s[0:3], 0 offen
	buffer_load_ubyte v84, v18, s[0:3], 0 offen offset:1
	buffer_load_ubyte v251, v18, s[0:3], 0 offen offset:2
	buffer_load_ubyte v194, v18, s[0:3], 0 offen offset:3
	buffer_load_ubyte v197, v18, s[0:3], 4 offen
	buffer_load_ubyte v198, v18, s[0:3], 4 offen offset:1
	buffer_load_ubyte v205, v18, s[0:3], 4 offen offset:2
	buffer_load_ubyte v208, v18, s[0:3], 4 offen offset:3
	buffer_load_ubyte v52, v19, s[0:3], 0 offen
	buffer_load_ubyte v59, v19, s[0:3], 0 offen offset:1
	buffer_load_ubyte v58, v19, s[0:3], 0 offen offset:2
	buffer_load_ubyte v195, v19, s[0:3], 0 offen offset:3
	buffer_load_ubyte v193, v19, s[0:3], 4 offen
	buffer_load_ubyte v199, v19, s[0:3], 4 offen offset:1
	buffer_load_ubyte v203, v19, s[0:3], 4 offen offset:2
	buffer_load_ubyte v207, v19, s[0:3], 4 offen offset:3
	buffer_load_ubyte v71, v21, s[0:3], 0 offen
	buffer_load_ubyte v77, v21, s[0:3], 0 offen offset:1
	buffer_load_ubyte v85, v21, s[0:3], 0 offen offset:2
	buffer_load_ubyte v168, v21, s[0:3], 0 offen offset:3
	buffer_load_ubyte v170, v21, s[0:3], 4 offen
	buffer_load_ubyte v196, v21, s[0:3], 4 offen offset:1
	buffer_load_ubyte v202, v21, s[0:3], 4 offen offset:2
	buffer_load_ubyte v142, v21, s[0:3], 4 offen offset:3
	buffer_load_ubyte v64, v22, s[0:3], 0 offen
	buffer_load_ubyte v70, v22, s[0:3], 0 offen offset:1
	buffer_load_ubyte v76, v22, s[0:3], 0 offen offset:2
	buffer_load_ubyte v192, v22, s[0:3], 0 offen offset:3
	buffer_load_ubyte v169, v22, s[0:3], 4 offen
	buffer_load_ubyte v171, v22, s[0:3], 4 offen offset:1
	buffer_load_ubyte v200, v22, s[0:3], 4 offen offset:2
	buffer_load_ubyte v201, v22, s[0:3], 4 offen offset:3
	buffer_load_ubyte v206, v23, s[0:3], 0 offen
	buffer_load_ubyte v143, v23, s[0:3], 0 offen offset:1
	buffer_load_ubyte v140, v23, s[0:3], 0 offen offset:2
	buffer_load_ubyte v110, v23, s[0:3], 0 offen offset:3
	buffer_load_ubyte v98, v23, s[0:3], 4 offen
	buffer_load_ubyte v95, v23, s[0:3], 4 offen offset:1
	buffer_load_ubyte v92, v23, s[0:3], 4 offen offset:2
	buffer_load_ubyte v86, v23, s[0:3], 4 offen offset:3
	buffer_load_ubyte v204, v24, s[0:3], 0 offen
	buffer_load_ubyte v214, v24, s[0:3], 0 offen offset:1
	buffer_load_ubyte v156, v24, s[0:3], 0 offen offset:2
	buffer_load_ubyte v111, v24, s[0:3], 0 offen offset:3
	buffer_load_ubyte v99, v24, s[0:3], 4 offen
	buffer_load_ubyte v96, v24, s[0:3], 4 offen offset:1
	buffer_load_ubyte v94, v24, s[0:3], 4 offen offset:2
	buffer_load_ubyte v87, v24, s[0:3], 4 offen offset:3
	buffer_load_ubyte v210, v25, s[0:3], 0 offen
	buffer_load_ubyte v212, v25, s[0:3], 0 offen offset:1
	buffer_load_ubyte v215, v25, s[0:3], 0 offen offset:2
	buffer_load_ubyte v141, v25, s[0:3], 0 offen offset:3
	buffer_load_ubyte v32, v25, s[0:3], 4 offen
	s_waitcnt vmcnt(62)
	v_lshlrev_b32_e32 v38, 23, v38
	s_waitcnt vmcnt(47)
	v_lshlrev_b32_e32 v254, 23, v198
	s_waitcnt vmcnt(46)
	v_lshlrev_b32_e32 v246, 23, v205
	s_waitcnt vmcnt(45)
	v_lshlrev_b32_e32 v222, 23, v208
	s_waitcnt vmcnt(39)
	v_lshlrev_b32_e32 v255, 23, v199
	s_waitcnt vmcnt(38)
	v_lshlrev_b32_e32 v247, 23, v203
	s_waitcnt vmcnt(37)
	v_lshlrev_b32_e32 v223, 23, v207
	s_waitcnt vmcnt(31)
	v_lshlrev_b32_e32 v248, 23, v196
	s_waitcnt vmcnt(30)
	v_lshlrev_b32_e32 v224, 23, v202
	s_waitcnt vmcnt(29)
	v_lshlrev_b32_e32 v220, 23, v142
	s_waitcnt vmcnt(23)
	v_lshlrev_b32_e32 v249, 23, v171
	s_waitcnt vmcnt(22)
	v_lshlrev_b32_e32 v225, 23, v200
	s_waitcnt vmcnt(21)
	v_lshlrev_b32_e32 v221, 23, v201
	s_waitcnt vmcnt(20)
	v_lshlrev_b32_e32 v218, 23, v206
	s_waitcnt vmcnt(19)
	v_lshlrev_b32_e32 v166, 23, v143
	s_waitcnt vmcnt(18)
	v_lshlrev_b32_e32 v158, 23, v140
	s_waitcnt vmcnt(17)
	v_lshlrev_b32_e32 v142, 23, v110
	s_waitcnt vmcnt(16)
	v_lshlrev_b32_e32 v110, 23, v98
	s_waitcnt vmcnt(15)
	v_lshlrev_b32_e32 v98, 23, v95
	s_waitcnt vmcnt(13)
	v_lshlrev_b32_e32 v86, 23, v86
	s_waitcnt vmcnt(12)
	v_lshlrev_b32_e32 v219, 23, v204
	s_waitcnt vmcnt(11)
	v_lshlrev_b32_e32 v167, 23, v214
	s_waitcnt vmcnt(10)
	v_lshlrev_b32_e32 v159, 23, v156
	s_waitcnt vmcnt(9)
	v_lshlrev_b32_e32 v143, 23, v111
	s_waitcnt vmcnt(8)
	v_lshlrev_b32_e32 v111, 23, v99
	s_waitcnt vmcnt(7)
	v_lshlrev_b32_e32 v99, 23, v96
	s_waitcnt vmcnt(6)
	v_lshlrev_b32_e32 v95, 23, v94
	v_lshlrev_b32_e32 v94, 23, v92
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v216, 23, v210
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v164, 23, v212
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v156, 23, v215
	v_lshlrev_b32_e32 v87, 23, v87
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1716 ; 4-byte Folded Spill
	buffer_load_ubyte v97, v25, s[0:3], 4 offen offset:1
	s_nop 0
	buffer_load_ubyte v32, v25, s[0:3], 4 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1684 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v25, s[0:3], 4 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:388 ; 4-byte Folded Spill
	buffer_load_ubyte v209, v26, s[0:3], 0 offen
	buffer_load_ubyte v211, v26, s[0:3], 0 offen offset:1
	buffer_load_ubyte v213, v26, s[0:3], 0 offen offset:2
	buffer_load_ubyte v252, v26, s[0:3], 0 offen offset:3
	buffer_load_ubyte v53, v26, s[0:3], 4 offen
	buffer_load_ubyte v32, v26, s[0:3], 4 offen offset:1
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v217, 23, v209
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v165, 23, v211
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v157, 23, v213
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1708 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v26, s[0:3], 4 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1700 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v26, s[0:3], 4 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1692 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 0 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1676 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 0 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1668 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 0 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1660 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 0 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1652 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 4 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1644 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 4 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1636 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 4 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:1628 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v27, s[0:3], 4 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:480 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v7, s[16:19], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:496 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v7, s[16:19], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:460 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v7, s[16:19], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:456 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v7, s[16:19], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:452 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v7, s[16:19], s8 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:436 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v7, s[16:19], s8 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v32, off offset:420 ; 4-byte Folded Spill
	buffer_load_ubyte v32, v7, s[16:19], s8 offen offset:2
	s_nop 0
	buffer_load_ubyte v7, v7, s[16:19], s8 offen offset:3
	s_waitcnt vmcnt(1)
	scratch_store_dword off, v32, off offset:404 ; 4-byte Folded Spill
	s_waitcnt vmcnt(1)
	scratch_store_dword off, v7, off offset:356 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:336 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:332 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:328 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s6 offen offset:3
	s_or_b32 s6, s5, 4
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:324 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s8 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:320 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s8 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:316 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s8 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:312 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v9, s[16:19], s8 offen offset:3
	s_lshl_b32 s8, s10, 9
	s_add_i32 s7, s8, s7
	s_movk_i32 s8, 0x420
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:308 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1548 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1512 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1480 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1448 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1412 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1388 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1360 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v13, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:960 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1556 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1528 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1500 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1464 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1436 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1404 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1368 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v15, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:992 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1564 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1536 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1508 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1476 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1440 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1408 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1380 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v16, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1348 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1624 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1620 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1616 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1612 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1608 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1604 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1600 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v17, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1596 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1320 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1204 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1168 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1148 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1120 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1052 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1236 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v18, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1248 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1328 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1220 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1196 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1160 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1132 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1096 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1036 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v19, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1252 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1288 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1232 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1216 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1176 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1140 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1112 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1068 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v21, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1240 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1296 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1244 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1228 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1212 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1156 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1128 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1104 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v22, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1076 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1008 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:272 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:256 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:264 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:828 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:780 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:756 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v23, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:724 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:1016 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:924 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:900 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:884 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:844 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:804 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:772 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v24, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:740 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:944 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:936 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:916 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:892 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:852 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:820 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:788 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v25, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:764 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:952 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:940 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:932 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:908 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:876 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:836 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:812 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v26, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:796 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:700 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:660 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:652 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:644 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:636 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:628 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:612 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v27, s[0:3], s6 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v7, off offset:596 ; 4-byte Folded Spill
	v_mov_b32_e32 v7, s7
	s_lshl_b32 s7, s4, 9
	v_mad_u32_u24 v7, v2, s8, v7
	s_add_i32 s7, s7, 0x10800
	v_add3_u32 v78, v7, v0, v3
	v_mov_b32_e32 v7, s7
	v_mad_u32_u24 v2, v2, s8, v7
	v_add3_u32 v79, v2, v0, v3
	v_lshlrev_b32_e32 v3, 23, v50
	v_lshlrev_b32_e32 v2, 23, v49
	v_mov_b64_e32 v[50:51], v[2:3]
	v_lshlrev_b32_e32 v3, 23, v48
	v_lshlrev_b32_e32 v2, 23, v20
	v_mov_b64_e32 v[56:57], v[2:3]
	scratch_store_dwordx2 off, v[50:51], off offset:16 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[56:57], off offset:32 ; 8-byte Folded Spill
	buffer_load_ubyte v9, v10, s[0:3], 0 offen
	buffer_load_ubyte v54, v10, s[0:3], 0 offen offset:1
	buffer_load_ubyte v19, v10, s[0:3], 0 offen offset:2
	buffer_load_ubyte v22, v10, s[0:3], 0 offen offset:3
	buffer_load_ubyte v25, v10, s[0:3], 4 offen
	buffer_load_ubyte v33, v10, s[0:3], 4 offen offset:1
	buffer_load_ubyte v35, v10, s[0:3], 4 offen offset:2
	buffer_load_ubyte v40, v10, s[0:3], 4 offen offset:3
	buffer_load_ubyte v0, v10, s[0:3], s5 offen
	v_lshlrev_b32_e32 v20, 23, v12
	v_lshlrev_b32_e32 v12, 23, v6
	v_lshlrev_b32_e32 v6, 23, v4
	v_add_u32_e32 v4, 0x5000, v79
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1588 ; 4-byte Folded Spill
	buffer_load_ubyte v0, v10, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1492 ; 4-byte Folded Spill
	buffer_load_ubyte v0, v10, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1468 ; 4-byte Folded Spill
	buffer_load_ubyte v0, v10, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1456 ; 4-byte Folded Spill
	buffer_load_ubyte v0, v10, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1428 ; 4-byte Folded Spill
	buffer_load_ubyte v0, v10, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1396 ; 4-byte Folded Spill
	buffer_load_ubyte v0, v10, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1376 ; 4-byte Folded Spill
	buffer_load_ubyte v0, v10, s[0:3], s6 offen offset:3
	v_add_u32_e32 v10, 0x7000, v79
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v0, off offset:1356 ; 4-byte Folded Spill
	buffer_load_ubyte v7, v8, s[0:3], 0 offen
	buffer_load_ubyte v16, v8, s[0:3], 0 offen offset:1
	buffer_load_ubyte v55, v8, s[0:3], 0 offen offset:2
	buffer_load_ubyte v18, v8, s[0:3], 0 offen offset:3
	buffer_load_ubyte v23, v8, s[0:3], 4 offen
	buffer_load_ubyte v26, v8, s[0:3], 4 offen offset:1
	buffer_load_ubyte v32, v8, s[0:3], 4 offen offset:2
	buffer_load_ubyte v34, v8, s[0:3], 4 offen offset:3
	buffer_load_ubyte v2, v8, s[0:3], s5 offen
	v_add_u32_e32 v0, s9, v8
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1568 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v8, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1524 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v8, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1496 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v8, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1472 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v8, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1444 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v8, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1424 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v8, s[0:3], s6 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1400 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v8, s[0:3], s6 offen offset:3
	v_add_u32_e32 v8, 0x6000, v79
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1384 ; 4-byte Folded Spill
	buffer_load_ubyte v65, v0, s[0:3], 0 offen
	buffer_load_ubyte v13, v0, s[0:3], 0 offen offset:1
	buffer_load_ubyte v15, v0, s[0:3], 0 offen offset:2
	buffer_load_ubyte v17, v0, s[0:3], 0 offen offset:3
	buffer_load_ubyte v253, v0, s[0:3], 4 offen
	buffer_load_ubyte v21, v0, s[0:3], 4 offen offset:1
	buffer_load_ubyte v24, v0, s[0:3], 4 offen offset:2
	buffer_load_ubyte v27, v0, s[0:3], 4 offen offset:3
	buffer_load_ubyte v2, v0, s[0:3], s5 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1572 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v0, s[0:3], s5 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1544 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v0, s[0:3], s5 offen offset:2
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1540 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v0, s[0:3], s5 offen offset:3
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1520 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v0, s[0:3], s6 offen
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1488 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v0, s[0:3], s6 offen offset:1
	s_waitcnt vmcnt(0)
	scratch_store_dword off, v2, off offset:1460 ; 4-byte Folded Spill
	buffer_load_ubyte v2, v0, s[0:3], s6 offen offset:2
	s_nop 0
	buffer_load_ubyte v0, v0, s[0:3], s6 offen offset:3
	s_getpc_b64 s[0:1]
	s_add_u32 s0, s0, _ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE@rel32@lo+4
	s_addc_u32 s1, s1, _ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE@rel32@hi+12
	s_load_dword s0, s[0:1], 0x0
	s_waitcnt lgkmcnt(0)
	s_mul_i32 s2, s0, s4
	s_waitcnt vmcnt(1)
	scratch_store_dword off, v2, off offset:1432 ; 4-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[20:21], v[50:51] op_sel_hi:[0,1]
	s_waitcnt vmcnt(1)
	scratch_store_dword off, v0, off offset:1420 ; 4-byte Folded Spill
	s_waitcnt vmcnt(0)
	s_barrier
	ds_read2_b64 v[72:75], v79 offset1:4
	ds_read2_b64 v[128:131], v78 offset1:4
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[72:73], v[128:129], 0
	s_nop 7
	v_pk_mul_f32 v[2:3], v[2:3], v[48:49]
	v_pk_mul_f32 v[48:49], v[20:21], v[56:57] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[48:49], v[50:51]
	v_lshlrev_b32_e32 v49, 23, v45
	v_lshlrev_b32_e32 v48, 23, v14
	v_lshlrev_b32_e32 v14, 23, v1
	scratch_store_dwordx2 off, v[48:49], off offset:48 ; 8-byte Folded Spill
	v_pk_mul_f32 v[0:1], v[14:15], v[48:49] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[74:75], v[130:131], 0
	s_nop 7
	v_pk_fma_f32 v[0:1], v[0:1], v[48:49], v[2:3]
	v_lshlrev_b32_e32 v3, 23, v47
	v_lshlrev_b32_e32 v2, 23, v46
	v_lshlrev_b32_e32 v47, 23, v39
	v_lshlrev_b32_e32 v46, 23, v36
	scratch_store_dwordx2 off, v[2:3], off offset:64 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[46:47], off offset:112 ; 8-byte Folded Spill
	ds_read2_b64 v[172:175], v79 offset0:8 offset1:12
	ds_read2_b64 v[148:151], v78 offset0:8 offset1:12
	v_pk_mul_f32 v[2:3], v[14:15], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[2:3], v[50:51], v[56:57]
	v_pk_mul_f32 v[50:51], v[12:13], v[46:47] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[46:49], v[172:173], v[148:149], 0
	v_lshlrev_b32_e32 v39, 23, v44
	scratch_store_dwordx2 off, v[38:39], off offset:120 ; 8-byte Folded Spill
	v_pk_mul_f32 v[38:39], v[12:13], v[38:39] op_sel_hi:[0,1]
	s_nop 4
	v_pk_fma_f32 v[0:1], v[50:51], v[46:47], v[0:1]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[174:175], v[150:151], 0
	v_pk_fma_f32 v[2:3], v[38:39], v[48:49], v[2:3]
	v_lshlrev_b32_e32 v39, 23, v30
	v_lshlrev_b32_e32 v38, 23, v29
	scratch_store_dwordx2 off, v[38:39], off offset:1812 ; 8-byte Folded Spill
	v_pk_mul_f32 v[38:39], v[6:7], v[38:39] op_sel_hi:[0,1]
	s_nop 2
	v_pk_fma_f32 v[104:105], v[38:39], v[44:45], v[0:1]
	v_lshlrev_b32_e32 v1, 23, v37
	v_lshlrev_b32_e32 v0, 23, v31
	scratch_store_dwordx2 off, v[0:1], off offset:128 ; 8-byte Folded Spill
	v_pk_mul_f32 v[0:1], v[6:7], v[0:1] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[0:1], v[46:47], v[2:3]
	v_lshlrev_b32_e32 v1, 23, v28
	v_lshlrev_b32_e32 v0, 23, v11
	v_mov_b64_e32 v[56:57], v[0:1]
	scratch_store_dwordx2 off, v[56:57], off offset:136 ; 8-byte Folded Spill
	v_add_u32_e32 v0, 0x2000, v78
	v_add_u32_e32 v1, 0x1000, v79
	v_add_u32_e32 v2, 0x2000, v79
	v_add_u32_e32 v3, 0x3000, v79
	ds_read2_b64 v[136:139], v0 offset0:32 offset1:36
	ds_read2_b64 v[132:135], v0 offset0:40 offset1:44
	ds_read2_b64 v[184:187], v1 offset0:16 offset1:20
	ds_read2_b64 v[80:83], v1 offset0:24 offset1:28
	ds_read2_b64 v[180:183], v2 offset0:32 offset1:36
	ds_read2_b64 v[88:91], v2 offset0:40 offset1:44
	ds_read2_b64 v[112:115], v3 offset0:48 offset1:52
	ds_read2_b64 v[100:103], v3 offset0:56 offset1:60
	v_add_u32_e32 v0, 0x4000, v79
	v_add_u32_e32 v11, 0x4000, v78
	s_waitcnt lgkmcnt(0)
	ds_read2_b64 v[144:147], v0 offset0:64 offset1:68
	ds_read2_b64 v[230:233], v0 offset0:72 offset1:76
	ds_read2_b64 v[106:109], v4 offset0:80 offset1:84
	ds_read2_b64 v[36:39], v4 offset0:88 offset1:92
	ds_read2_b64 v[226:229], v8 offset0:96 offset1:100
	ds_read2_b64 v[120:123], v8 offset0:104 offset1:108
	ds_read2_b64 v[116:119], v10 offset0:112 offset1:116
	ds_read2_b64 v[238:241], v10 offset0:120 offset1:124
	s_waitcnt lgkmcnt(0)
	v_add_u32_e32 v28, 0x6000, v78
	ds_read2_b64 v[234:237], v11 offset0:64 offset1:68
	ds_read2_b64 v[66:69], v11 offset0:72 offset1:76
	ds_read2_b64 v[124:127], v28 offset0:96 offset1:100
	ds_read2_b64 v[242:245], v28 offset0:104 offset1:108
	ds_read2_b64 v[48:51], v1 offset0:16 offset1:20
	v_add_u32_e32 v11, 0x8000, v79
	v_lshlrev_b32_e32 v47, 23, v17
	v_lshlrev_b32_e32 v46, 23, v18
	v_lshlrev_b32_e32 v45, 23, v25
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:1272 ; 16-byte Folded Spill
	ds_read2_b64 v[48:51], v1 offset0:24 offset1:28
	v_lshlrev_b32_e32 v25, 23, v27
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:1256 ; 16-byte Folded Spill
	ds_read2_b64 v[48:51], v2 offset0:32 offset1:36
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:1180 ; 16-byte Folded Spill
	ds_read2_b64 v[48:51], v2 offset0:40 offset1:44
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:1080 ; 16-byte Folded Spill
	ds_read2_b64 v[48:51], v3 offset0:48 offset1:52
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:1020 ; 16-byte Folded Spill
	ds_read2_b64 v[48:51], v3 offset0:56 offset1:60
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:976 ; 16-byte Folded Spill
	ds_read2_b64 v[48:51], v79 offset1:4
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:1332 ; 16-byte Folded Spill
	ds_read2_b64 v[48:51], v79 offset0:8 offset1:12
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[48:51], off offset:1304 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	ds_read2_b64 v[176:179], v0 offset0:64 offset1:68
	ds_read2_b64 v[0:3], v0 offset0:72 offset1:76
	v_lshlrev_b32_e32 v49, 23, v42
	v_lshlrev_b32_e32 v48, 23, v60
	v_lshlrev_b32_e32 v60, 23, v7
	s_waitcnt lgkmcnt(1)
	scratch_store_dwordx4 off, v[176:179], off offset:580 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[0:3], off offset:564 ; 16-byte Folded Spill
	ds_read2_b64 v[0:3], v4 offset0:80 offset1:84
	v_lshlrev_b32_e32 v51, 23, v15
	v_lshlrev_b32_e32 v50, 23, v55
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[0:3], off offset:548 ; 16-byte Folded Spill
	ds_read2_b64 v[0:3], v4 offset0:88 offset1:92
	v_add_u32_e32 v4, 0x8000, v78
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[0:3], off offset:516 ; 16-byte Folded Spill
	ds_read2_b64 v[0:3], v8 offset0:96 offset1:100
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[0:3], off offset:532 ; 16-byte Folded Spill
	ds_read2_b64 v[0:3], v8 offset0:104 offset1:108
	v_lshlrev_b32_e32 v8, 23, v63
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[0:3], off offset:464 ; 16-byte Folded Spill
	ds_read2_b64 v[0:3], v10 offset0:112 offset1:116
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[0:3], off offset:500 ; 16-byte Folded Spill
	ds_read2_b64 v[0:3], v10 offset0:120 offset1:124
	v_lshlrev_b32_e32 v10, 23, v152
	v_pk_mul_f32 v[28:29], v[10:11], v[56:57] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	scratch_store_dwordx4 off, v[0:3], off offset:372 ; 16-byte Folded Spill
	s_waitcnt lgkmcnt(0)
	ds_read2_b64 v[176:179], v11 offset0:128 offset1:132
	ds_read2_b64 v[152:155], v4 offset0:128 offset1:132
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[176:177], v[152:153], 0
	scratch_store_dword off, v11, off offset:1504 ; 4-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[28:29], v[28:29], v[0:1], v[104:105]
	v_lshlrev_b32_e32 v1, 23, v250
	v_lshlrev_b32_e32 v0, 23, v191
	scratch_store_dwordx2 off, v[0:1], off offset:232 ; 8-byte Folded Spill
	v_pk_mul_f32 v[0:1], v[10:11], v[0:1] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[0:1], v[2:3], v[30:31]
	v_lshlrev_b32_e32 v1, 23, v188
	v_lshlrev_b32_e32 v0, 23, v163
	v_lshlrev_b32_e32 v250, 23, v160
	scratch_store_dwordx2 off, v[0:1], off offset:96 ; 8-byte Folded Spill
	v_pk_mul_f32 v[104:105], v[250:251], v[0:1] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[178:179], v[154:155], 0
	s_nop 7
	v_pk_fma_f32 v[28:29], v[104:105], v[0:1], v[28:29]
	v_lshlrev_b32_e32 v1, 23, v190
	v_lshlrev_b32_e32 v0, 23, v189
	scratch_store_dwordx2 off, v[0:1], off offset:80 ; 8-byte Folded Spill
	v_pk_mul_f32 v[0:1], v[250:251], v[0:1] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[0:1], v[2:3], v[30:31]
	v_lshlrev_b32_e32 v1, 23, v162
	v_lshlrev_b32_e32 v0, 23, v161
	scratch_store_dwordx2 off, v[0:1], off  ; 8-byte Folded Spill
	ds_read2_b64 v[160:163], v4 offset0:136 offset1:140
	ds_read2_b64 v[188:191], v11 offset0:136 offset1:140
	v_pk_mul_f32 v[104:105], v[8:9], v[0:1] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[188:189], v[160:161], 0
	v_add_u32_e32 v11, 0x9000, v79
	scratch_store_dwordx2 off, v[48:49], off offset:160 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[220:221], off offset:1804 ; 8-byte Folded Spill
	s_nop 4
	v_pk_fma_f32 v[28:29], v[104:105], v[0:1], v[28:29]
	v_lshlrev_b32_e32 v1, 23, v41
	v_lshlrev_b32_e32 v0, 23, v43
	scratch_store_dwordx2 off, v[0:1], off offset:144 ; 8-byte Folded Spill
	v_pk_mul_f32 v[0:1], v[8:9], v[0:1] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[0:1], v[2:3], v[30:31]
	v_lshlrev_b32_e32 v3, 23, v61
	v_lshlrev_b32_e32 v2, 23, v93
	v_lshlrev_b32_e32 v0, 23, v5
	scratch_store_dwordx2 off, v[2:3], off offset:152 ; 8-byte Folded Spill
	v_pk_mul_f32 v[42:43], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[190:191], v[162:163], 0
	v_lshlrev_b32_e32 v105, 23, v53
	v_lshlrev_b32_e32 v61, 23, v65
	v_lshlrev_b32_e32 v53, 23, v19
	scratch_store_dwordx2 off, v[218:219], off offset:1796 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[166:167], off offset:1788 ; 8-byte Folded Spill
	s_nop 2
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_lshlrev_b32_e32 v3, 23, v52
	v_lshlrev_b32_e32 v2, 23, v62
	v_lshlrev_b32_e32 v5, 23, v64
	v_lshlrev_b32_e32 v4, 23, v71
	scratch_store_dwordx4 off, v[28:31], off offset:292 ; 16-byte Folded Spill
	scratch_store_dwordx2 off, v[2:3], off offset:168 ; 8-byte Folded Spill
	v_lshlrev_b32_e32 v71, 23, v54
	v_mov_b64_e32 v[30:31], v[4:5]
	v_pk_mul_f32 v[28:29], v[20:21], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[184:185], v[128:129], 0
	scratch_store_dwordx2 off, v[30:31], off offset:176 ; 8-byte Folded Spill
	v_lshlrev_b32_e32 v49, 23, v22
	scratch_store_dwordx2 off, v[164:165], off offset:1780 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[158:159], off offset:1764 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[156:157], off offset:1772 ; 8-byte Folded Spill
	s_nop 2
	v_pk_mul_f32 v[28:29], v[28:29], v[2:3]
	v_pk_mul_f32 v[2:3], v[20:21], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[2:3], v[4:5]
	v_lshlrev_b32_e32 v3, 23, v59
	v_lshlrev_b32_e32 v2, 23, v84
	scratch_store_dwordx2 off, v[2:3], off offset:184 ; 8-byte Folded Spill
	v_pk_mul_f32 v[42:43], v[14:15], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[186:187], v[130:131], 0
	v_lshlrev_b32_e32 v59, 23, v13
	scratch_store_dwordx2 off, v[142:143], off offset:1756 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[110:111], off offset:1740 ; 8-byte Folded Spill
	s_nop 4
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_lshlrev_b32_e32 v3, 23, v70
	v_lshlrev_b32_e32 v2, 23, v77
	scratch_store_dwordx2 off, v[2:3], off offset:192 ; 8-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[14:15], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_lshlrev_b32_e32 v3, 23, v58
	v_lshlrev_b32_e32 v2, 23, v251
	scratch_store_dwordx2 off, v[2:3], off offset:200 ; 8-byte Folded Spill
	v_pk_mul_f32 v[42:43], v[12:13], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[80:81], v[148:149], 0
	v_lshlrev_b32_e32 v77, 23, v9
	v_lshlrev_b32_e32 v58, 23, v16
	s_nop 5
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_lshlrev_b32_e32 v3, 23, v76
	v_lshlrev_b32_e32 v2, 23, v85
	scratch_store_dwordx2 off, v[2:3], off offset:208 ; 8-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[12:13], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_lshlrev_b32_e32 v3, 23, v195
	v_lshlrev_b32_e32 v2, 23, v194
	scratch_store_dwordx2 off, v[2:3], off offset:216 ; 8-byte Folded Spill
	v_pk_mul_f32 v[42:43], v[6:7], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[82:83], v[150:151], 0
	s_nop 7
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_lshlrev_b32_e32 v3, 23, v192
	v_lshlrev_b32_e32 v2, 23, v168
	scratch_store_dwordx2 off, v[2:3], off offset:224 ; 8-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[6:7], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_lshlrev_b32_e32 v3, 23, v193
	ds_read2_b64 v[192:195], v11 offset0:144 offset1:148
	v_lshlrev_b32_e32 v2, 23, v197
	scratch_store_dwordx2 off, v[2:3], off offset:240 ; 8-byte Folded Spill
	v_pk_mul_f32 v[42:43], v[10:11], v[2:3] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[192:193], v[152:153], 0
	ds_read2_b64 v[196:199], v11 offset0:152 offset1:156
	s_nop 6
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_lshlrev_b32_e32 v3, 23, v169
	v_lshlrev_b32_e32 v2, 23, v170
	scratch_store_dwordx2 off, v[2:3], off offset:248 ; 8-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[10:11], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[194:195], v[154:155], 0
	v_pk_mul_f32 v[42:43], v[250:251], v[254:255] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[250:251], v[248:249] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[196:197], v[160:161], 0
	v_pk_mul_f32 v[42:43], v[8:9], v[246:247] op_sel_hi:[0,1]
	v_add_u32_e32 v251, 0xa000, v79
	ds_read2_b64 v[200:203], v251 offset0:160 offset1:164
	ds_read2_b64 v[204:207], v251 offset0:168 offset1:172
	s_nop 3
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[8:9], v[224:225] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[198:199], v[162:163], 0
	v_pk_mul_f32 v[42:43], v[0:1], v[222:223] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[0:1], v[220:221] op_sel_hi:[0,1]
	scratch_load_dword v1, off, off offset:1716 ; 4-byte Folded Reload
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[180:181], v[128:129], 0
	scratch_store_dwordx4 off, v[28:31], off offset:340 ; 16-byte Folded Spill
	v_pk_mul_f32 v[42:43], v[14:15], v[166:167] op_sel_hi:[0,1]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v104, 23, v1
	scratch_load_dword v1, off, off offset:1708 ; 4-byte Folded Reload
	v_pk_mul_f32 v[28:29], v[20:21], v[218:219] op_sel_hi:[0,1]
	s_nop 1
	v_pk_mul_f32 v[28:29], v[28:29], v[2:3]
	v_pk_mul_f32 v[2:3], v[20:21], v[216:217] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[2:3], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[182:183], v[130:131], 0
	scratch_store_dwordx2 off, v[60:61], off offset:1708 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[14:15], v[164:165] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[88:89], v[148:149], 0
	v_pk_mul_f32 v[42:43], v[12:13], v[158:159] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[12:13], v[156:157] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[90:91], v[150:151], 0
	v_pk_mul_f32 v[42:43], v[6:7], v[142:143] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_lshlrev_b32_e32 v43, 23, v252
	v_lshlrev_b32_e32 v42, 23, v141
	v_pk_mul_f32 v[2:3], v[6:7], v[42:43] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[200:201], v[152:153], 0
	v_mov_b64_e32 v[140:141], v[42:43]
	v_pk_mul_f32 v[42:43], v[10:11], v[110:111] op_sel_hi:[0,1]
	scratch_store_dwordx2 off, v[140:141], off offset:1748 ; 8-byte Folded Spill
	s_nop 4
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[10:11], v[104:105] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[202:203], v[154:155], 0
	v_pk_mul_f32 v[42:43], v[250:251], v[98:99] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_lshlrev_b32_e32 v42, 23, v97
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v43, 23, v1
	scratch_load_dword v1, off, off offset:1700 ; 4-byte Folded Reload
	v_pk_mul_f32 v[2:3], v[250:251], v[42:43] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[204:205], v[160:161], 0
	v_mov_b64_e32 v[96:97], v[42:43]
	v_pk_mul_f32 v[42:43], v[8:9], v[94:95] op_sel_hi:[0,1]
	scratch_store_dwordx2 off, v[96:97], off offset:1732 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[58:59], off offset:1700 ; 8-byte Folded Spill
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v93, 23, v1
	scratch_load_dword v1, off, off offset:1684 ; 4-byte Folded Reload
	s_nop 0
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v92, 23, v1
	scratch_load_dword v1, off, off offset:1692 ; 4-byte Folded Reload
	v_pk_mul_f32 v[2:3], v[8:9], v[92:93] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[206:207], v[162:163], 0
	scratch_store_dwordx2 off, v[50:51], off offset:1692 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[92:93], off offset:1724 ; 8-byte Folded Spill
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v85, 23, v1
	scratch_load_dword v1, off, off offset:388 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v84, 23, v1
	v_pk_mul_f32 v[42:43], v[0:1], v[86:87] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[0:1], v[84:85] op_sel_hi:[0,1]
	scratch_load_dword v1, off, off offset:1676 ; 4-byte Folded Reload
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[112:113], v[128:129], 0
	scratch_store_dwordx4 off, v[28:31], off offset:388 ; 16-byte Folded Spill
	scratch_store_dwordx2 off, v[84:85], off offset:1716 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[46:47], off offset:1676 ; 8-byte Folded Spill
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v76, 23, v1
	scratch_load_dword v1, off, off offset:1668 ; 4-byte Folded Reload
	v_pk_mul_f32 v[28:29], v[20:21], v[76:77] op_sel_hi:[0,1]
	s_nop 0
	v_pk_mul_f32 v[28:29], v[28:29], v[2:3]
	v_pk_mul_f32 v[2:3], v[20:21], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[2:3], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[114:115], v[130:131], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v70, 23, v1
	scratch_load_dword v1, off, off offset:1660 ; 4-byte Folded Reload
	v_pk_mul_f32 v[42:43], v[14:15], v[70:71] op_sel_hi:[0,1]
	s_nop 3
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[14:15], v[58:59] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[100:101], v[148:149], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v52, 23, v1
	scratch_load_dword v1, off, off offset:1652 ; 4-byte Folded Reload
	v_pk_mul_f32 v[42:43], v[12:13], v[52:53] op_sel_hi:[0,1]
	s_nop 3
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[12:13], v[50:51] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[102:103], v[150:151], 0
	v_mov_b32_e32 v13, v78
	scratch_store_dwordx2 off, v[52:53], off offset:1684 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v48, 23, v1
	scratch_load_dword v1, off, off offset:1644 ; 4-byte Folded Reload
	v_pk_mul_f32 v[42:43], v[6:7], v[48:49] op_sel_hi:[0,1]
	s_nop 1
	v_pk_fma_f32 v[28:29], v[42:43], v[2:3], v[28:29]
	v_pk_mul_f32 v[2:3], v[6:7], v[46:47] op_sel_hi:[0,1]
	v_pk_fma_f32 v[16:17], v[2:3], v[4:5], v[30:31]
	v_add_u32_e32 v7, 0xa000, v78
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v44, 23, v1
	v_add_u32_e32 v1, 0xb000, v79
	ds_read2_b64 v[208:211], v1 offset0:176 offset1:180
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[208:209], v[152:153], 0
	v_pk_mul_f32 v[18:19], v[10:11], v[44:45] op_sel_hi:[0,1]
	ds_read2_b64 v[212:215], v1 offset0:184 offset1:188
	scratch_store_dwordx2 off, v[44:45], off offset:1668 ; 8-byte Folded Spill
	s_nop 4
	v_pk_fma_f32 v[18:19], v[18:19], v[2:3], v[28:29]
	v_lshlrev_b32_e32 v29, 23, v253
	v_lshlrev_b32_e32 v28, 23, v23
	v_pk_mul_f32 v[2:3], v[10:11], v[28:29] op_sel_hi:[0,1]
	v_pk_fma_f32 v[16:17], v[2:3], v[4:5], v[16:17]
	scratch_load_dword v2, off, off offset:1636 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v3, 23, v33
	v_mov_b64_e32 v[42:43], v[28:29]
	scratch_load_dwordx2 v[28:29], off, off offset:96 ; 8-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[22:23], v[250:251], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[54:55], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[210:211], v[154:155], 0
	s_nop 7
	v_pk_fma_f32 v[18:19], v[22:23], v[2:3], v[18:19]
	v_lshlrev_b32_e32 v23, 23, v21
	v_lshlrev_b32_e32 v22, 23, v26
	v_pk_mul_f32 v[2:3], v[250:251], v[22:23] op_sel_hi:[0,1]
	v_pk_fma_f32 v[16:17], v[2:3], v[4:5], v[16:17]
	scratch_load_dword v2, off, off offset:1628 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v3, 23, v35
	v_mov_b64_e32 v[56:57], v[22:23]
	scratch_store_dwordx2 off, v[56:57], off offset:1660 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[22:23], v[8:9], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[62:63], v[2:3]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[212:213], v[160:161], 0
	scratch_store_dwordx2 off, v[62:63], off offset:1652 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[18:19], v[22:23], v[2:3], v[18:19]
	v_lshlrev_b32_e32 v23, 23, v24
	v_lshlrev_b32_e32 v22, 23, v32
	v_pk_mul_f32 v[2:3], v[8:9], v[22:23] op_sel_hi:[0,1]
	v_pk_fma_f32 v[16:17], v[2:3], v[4:5], v[16:17]
	scratch_load_dword v2, off, off offset:480 ; 4-byte Folded Reload
	v_lshlrev_b32_e32 v3, 23, v40
	v_mov_b64_e32 v[64:65], v[22:23]
	v_lshlrev_b32_e32 v24, 23, v34
	v_mov_b64_e32 v[40:41], v[24:25]
	scratch_store_dwordx2 off, v[40:41], off offset:1628 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[64:65], off offset:1644 ; 8-byte Folded Spill
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[22:23], v[0:1], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[34:35], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[214:215], v[162:163], 0
	scratch_store_dwordx2 off, v[34:35], off offset:1636 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[22:23], v[2:3], v[18:19]
	v_pk_mul_f32 v[2:3], v[0:1], v[24:25] op_sel_hi:[0,1]
	v_pk_fma_f32 v[24:25], v[2:3], v[4:5], v[16:17]
	scratch_load_dword v2, off, off offset:1592 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v252, 23, v2
	scratch_load_dword v2, off, off offset:1624 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v253, 23, v2
	scratch_load_dwordx2 v[2:3], off, off offset:16 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[16:17], v[252:253], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[72:73], v[136:137], 0
	scratch_store_dwordx4 off, v[22:25], off offset:480 ; 16-byte Folded Spill
	scratch_load_dwordx2 v[22:23], off, off offset:48 ; 8-byte Folded Reload
	s_nop 0
	scratch_load_dwordx2 v[72:73], off, off offset:1812 ; 8-byte Folded Reload
	s_nop 3
	v_pk_mul_f32 v[16:17], v[16:17], v[2:3]
	scratch_load_dwordx2 v[2:3], off, off offset:32 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[252:253], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[2:3], v[4:5]
	scratch_load_dword v2, off, off offset:1584 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v168, 23, v2
	scratch_load_dword v2, off, off offset:1620 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v169, 23, v2
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[74:75], v[138:139], 0
	v_pk_mul_f32 v[22:23], v[168:169], v[22:23] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[16:17], v[22:23], v[2:3], v[16:17]
	scratch_load_dwordx2 v[2:3], off, off offset:64 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[168:169], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[18:19], v[2:3], v[4:5], v[18:19]
	scratch_load_dword v2, off, off offset:1580 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v170, 23, v2
	scratch_load_dword v2, off, off offset:1616 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v171, 23, v2
	scratch_load_dwordx2 v[2:3], off, off offset:112 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[22:23], v[170:171], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[172:173], v[132:133], 0
	s_nop 7
	v_pk_fma_f32 v[16:17], v[22:23], v[2:3], v[16:17]
	scratch_load_dwordx2 v[2:3], off, off offset:120 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[170:171], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[18:19], v[2:3], v[4:5], v[18:19]
	scratch_load_dword v2, off, off offset:1576 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v26, 23, v2
	scratch_load_dword v2, off, off offset:1612 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v27, 23, v2
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[174:175], v[134:135], 0
	v_pk_mul_f32 v[22:23], v[26:27], v[72:73] op_sel_hi:[0,1]
	ds_read2_b64 v[172:175], v7 offset0:160 offset1:164
	s_nop 5
	v_pk_fma_f32 v[16:17], v[22:23], v[2:3], v[16:17]
	scratch_load_dwordx2 v[2:3], off, off offset:128 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[26:27], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[18:19], v[2:3], v[4:5], v[18:19]
	scratch_load_dword v2, off, off offset:860 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v24, 23, v2
	scratch_load_dword v2, off, off offset:1608 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v25, 23, v2
	scratch_load_dwordx2 v[2:3], off, off offset:136 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[22:23], v[24:25], v[2:3] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[176:177], v[172:173], 0
	s_nop 7
	v_pk_fma_f32 v[16:17], v[22:23], v[2:3], v[16:17]
	scratch_load_dwordx2 v[2:3], off, off offset:232 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[24:25], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[22:23], v[2:3], v[4:5], v[18:19]
	scratch_load_dword v2, off, off offset:708 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v18, 23, v2
	scratch_load_dword v2, off, off offset:1604 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v19, 23, v2
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[178:179], v[174:175], 0
	v_pk_mul_f32 v[28:29], v[18:19], v[28:29] op_sel_hi:[0,1]
	ds_read2_b64 v[176:179], v7 offset0:168 offset1:172
	s_nop 5
	v_pk_fma_f32 v[28:29], v[28:29], v[2:3], v[16:17]
	scratch_load_dwordx2 v[2:3], off, off offset:80 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[18:19], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[22:23], v[2:3], v[4:5], v[22:23]
	scratch_load_dword v2, off, off offset:684 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v16, 23, v2
	scratch_load_dword v2, off, off offset:1600 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v17, 23, v2
	scratch_load_dwordx2 v[2:3], off, off   ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[30:31], v[16:17], v[2:3] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[188:189], v[176:177], 0
	s_nop 7
	v_pk_fma_f32 v[28:29], v[30:31], v[2:3], v[28:29]
	scratch_load_dwordx2 v[2:3], off, off offset:144 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[30:31], off, off offset:152 ; 8-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[2:3], v[16:17], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[22:23], v[2:3], v[4:5], v[22:23]
	scratch_load_dword v2, off, off offset:668 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v188, 23, v2
	scratch_load_dword v2, off, off offset:1596 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v189, 23, v2
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[190:191], v[178:179], 0
	v_pk_mul_f32 v[30:31], v[188:189], v[30:31] op_sel_hi:[0,1]
	v_mov_b64_e32 v[190:191], v[216:217]
	s_nop 5
	v_pk_fma_f32 v[28:29], v[30:31], v[2:3], v[28:29]
	scratch_load_dwordx2 v[2:3], off, off offset:160 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[188:189], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[22:23]
	scratch_load_dwordx2 v[2:3], off, off offset:168 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[22:23], v[252:253], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[184:185], v[136:137], 0
	scratch_store_dwordx4 off, v[28:31], off offset:668 ; 16-byte Folded Spill
	scratch_load_dwordx2 v[30:31], off, off offset:184 ; 8-byte Folded Reload
	v_mov_b64_e32 v[184:185], v[246:247]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[30:31], v[168:169], v[30:31] op_sel_hi:[0,1]
	s_nop 2
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3]
	scratch_load_dwordx2 v[2:3], off, off offset:176 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[252:253], v[2:3] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[2:3], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[186:187], v[138:139], 0
	v_mov_b64_e32 v[186:187], v[248:249]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dwordx2 v[2:3], off, off offset:192 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[168:169], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dwordx2 v[2:3], off, off offset:200 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[30:31], v[170:171], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[80:81], v[132:133], 0
	s_nop 7
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dwordx2 v[2:3], off, off offset:208 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[30:31], off, off offset:216 ; 8-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[2:3], v[170:171], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[82:83], v[134:135], 0
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[30:31], v[26:27], v[30:31] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dwordx2 v[2:3], off, off offset:224 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[26:27], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dwordx2 v[2:3], off, off offset:240 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[30:31], v[24:25], v[2:3] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[192:193], v[172:173], 0
	v_mov_b64_e32 v[192:193], v[94:95]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dwordx2 v[2:3], off, off offset:248 ; 8-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[18:19], v[254:255] op_sel_hi:[0,1]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[24:25], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[194:195], v[174:175], 0
	v_mov_b64_e32 v[194:195], v[254:255]
	v_mov_b32_e32 v255, v170
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[18:19], v[248:249] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[196:197], v[176:177], 0
	v_pk_mul_f32 v[30:31], v[16:17], v[246:247] op_sel_hi:[0,1]
	v_mov_b32_e32 v247, v26
	v_mov_b64_e32 v[196:197], v[224:225]
	v_mov_b32_e32 v249, v24
	s_nop 3
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[16:17], v[224:225] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[198:199], v[178:179], 0
	v_pk_mul_f32 v[30:31], v[188:189], v[222:223] op_sel_hi:[0,1]
	v_mov_b64_e32 v[198:199], v[222:223]
	v_mov_b64_e32 v[222:223], v[42:43]
	v_mov_b64_e32 v[224:225], v[54:55]
	s_nop 3
	v_pk_fma_f32 v[30:31], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[188:189], v[220:221] op_sel_hi:[0,1]
	v_pk_fma_f32 v[32:33], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[180:181], v[136:137], 0
	v_pk_mul_f32 v[22:23], v[252:253], v[218:219] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[30:33], off offset:684 ; 16-byte Folded Spill
	v_mov_b32_e32 v221, v18
	v_mov_b32_e32 v219, v16
	v_pk_mul_f32 v[30:31], v[168:169], v[166:167] op_sel_hi:[0,1]
	s_nop 2
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3]
	v_pk_mul_f32 v[2:3], v[252:253], v[216:217] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[2:3], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[182:183], v[138:139], 0
	s_nop 7
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[168:169], v[164:165] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[88:89], v[132:133], 0
	v_pk_mul_f32 v[30:31], v[170:171], v[158:159] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[170:171], v[156:157] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[90:91], v[134:135], 0
	v_pk_mul_f32 v[30:31], v[26:27], v[142:143] op_sel_hi:[0,1]
	v_mov_b64_e32 v[142:143], v[198:199]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[26:27], v[140:141] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[200:201], v[172:173], 0
	v_pk_mul_f32 v[30:31], v[24:25], v[110:111] op_sel_hi:[0,1]
	v_mov_b64_e32 v[200:201], v[104:105]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[24:25], v[104:105] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[202:203], v[174:175], 0
	v_pk_mul_f32 v[30:31], v[18:19], v[98:99] op_sel_hi:[0,1]
	v_mov_b64_e32 v[104:105], v[98:99]
	v_mov_b64_e32 v[98:99], v[76:77]
	s_nop 4
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[18:19], v[96:97] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[204:205], v[176:177], 0
	v_pk_mul_f32 v[30:31], v[16:17], v[94:95] op_sel_hi:[0,1]
	v_mov_b64_e32 v[96:97], v[48:49]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[16:17], v[92:93] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[206:207], v[178:179], 0
	v_pk_mul_f32 v[30:31], v[188:189], v[86:87] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[30:31], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[188:189], v[84:85] op_sel_hi:[0,1]
	v_pk_fma_f32 v[32:33], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[112:113], v[136:137], 0
	v_pk_mul_f32 v[22:23], v[252:253], v[76:77] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[30:33], off offset:708 ; 16-byte Folded Spill
	s_nop 1
	v_pk_mul_f32 v[30:31], v[168:169], v[70:71] op_sel_hi:[0,1]
	s_nop 2
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3]
	v_pk_mul_f32 v[2:3], v[252:253], v[60:61] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[2:3], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[114:115], v[138:139], 0
	v_mov_b64_e32 v[114:115], v[190:191]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[168:169], v[58:59] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[100:101], v[132:133], 0
	v_pk_mul_f32 v[30:31], v[170:171], v[52:53] op_sel_hi:[0,1]
	v_mov_b64_e32 v[100:101], v[200:201]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[170:171], v[50:51] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[102:103], v[134:135], 0
	v_pk_mul_f32 v[30:31], v[26:27], v[48:49] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[26:27], v[46:47] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[208:209], v[172:173], 0
	v_pk_mul_f32 v[30:31], v[24:25], v[44:45] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[24:25], v[42:43] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[210:211], v[174:175], 0
	v_pk_mul_f32 v[30:31], v[18:19], v[54:55] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[18:19], v[56:57] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[212:213], v[176:177], 0
	v_pk_mul_f32 v[30:31], v[16:17], v[62:63] op_sel_hi:[0,1]
	v_add_u32_e32 v57, 0xc000, v79
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[16:17], v[64:65] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[214:215], v[178:179], 0
	v_pk_mul_f32 v[30:31], v[188:189], v[34:35] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[30:31], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[188:189], v[40:41] op_sel_hi:[0,1]
	v_pk_fma_f32 v[32:33], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:1556 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1548 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[22:23], v[20:21], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[48:49], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[144:145], v[128:129], 0
	scratch_store_dwordx4 off, v[30:33], off offset:860 ; 16-byte Folded Spill
	scratch_store_dwordx2 off, v[48:49], off offset:1556 ; 8-byte Folded Spill
	s_nop 5
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3]
	scratch_load_dword v2, off, off offset:1564 ; 4-byte Folded Reload
	v_mov_b32_e32 v3, v252
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[28:29], v[2:3], v[252:253]
	v_mov_b32_e32 v3, v253
	scratch_store_dwordx2 off, v[2:3], off offset:1548 ; 8-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[2:3], v[20:21] op_sel_hi:[1,0]
	s_nop 0
	v_pk_mul_f32 v[30:31], v[2:3], v[4:5]
	scratch_load_dword v2, off, off offset:1528 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1512 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[32:33], v[14:15], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[50:51], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[146:147], v[130:131], 0
	scratch_store_dwordx2 off, v[50:51], off offset:1528 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[32:33], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1536 ; 4-byte Folded Reload
	v_mov_b32_e32 v3, v168
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[40:41], v[2:3], v[168:169]
	v_mov_b32_e32 v3, v169
	scratch_store_dwordx2 off, v[2:3], off offset:1512 ; 8-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[2:3], v[14:15] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	scratch_load_dword v2, off, off offset:1500 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1480 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[32:33], v[12:13], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[88:89], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[230:231], v[148:149], 0
	scratch_store_dwordx2 off, v[88:89], off offset:1480 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[32:33], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1508 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v254, 23, v2
	v_pk_mul_f32 v[42:43], v[254:255], v[170:171]
	v_mov_b32_e32 v255, v171
	v_pk_mul_f32 v[2:3], v[254:255], v[12:13] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	scratch_load_dword v2, off, off offset:1464 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1448 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[32:33], v[6:7], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[90:91], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[232:233], v[150:151], 0
	scratch_store_dwordx2 off, v[90:91], off offset:1448 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[32:33], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1476 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v246, 23, v2
	v_pk_mul_f32 v[54:55], v[246:247], v[26:27]
	v_mov_b32_e32 v247, v27
	v_pk_mul_f32 v[2:3], v[246:247], v[6:7] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[30:31]
	scratch_load_dword v2, off, off offset:1436 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1412 ; 4-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	scratch_load_dword v7, off, off offset:1440 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[212:213], off, off offset:120 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[216:217], off, off offset:48 ; 8-byte Folded Reload
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[60:61], v[10:11], v[2:3] op_sel_hi:[0,1]
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v248, 23, v7
	scratch_load_dword v7, off, off offset:1404 ; 4-byte Folded Reload
	v_mov_b64_e32 v[76:77], v[2:3]
	ds_read2_b64 v[2:5], v57 offset0:192 offset1:196
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[2:3], v[152:153], 0
	v_pk_mul_f32 v[74:75], v[248:249], v[24:25]
	v_mov_b32_e32 v249, v25
	scratch_store_dwordx2 off, v[76:77], off offset:1412 ; 8-byte Folded Spill
	s_nop 4
	v_pk_fma_f32 v[22:23], v[60:61], v[32:33], v[22:23]
	v_pk_mul_f32 v[32:33], v[248:249], v[10:11] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[30:31], v[32:33], v[34:35], v[30:31]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:1388 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:1408 ; 4-byte Folded Reload
	v_pk_mul_f32 v[60:61], v[250:251], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[44:45], v[32:33]
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[4:5], v[154:155], 0
	scratch_store_dwordx2 off, v[44:45], off offset:1388 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v220, 23, v7
	scratch_load_dword v7, off, off offset:1368 ; 4-byte Folded Reload
	v_pk_mul_f32 v[80:81], v[220:221], v[18:19]
	v_mov_b32_e32 v221, v19
	s_nop 1
	v_pk_fma_f32 v[22:23], v[60:61], v[32:33], v[22:23]
	v_pk_mul_f32 v[32:33], v[220:221], v[250:251] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[30:31], v[32:33], v[34:35], v[30:31]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:1360 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:1380 ; 4-byte Folded Reload
	v_pk_mul_f32 v[64:65], v[8:9], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[52:53], v[32:33]
	ds_read2_b64 v[32:35], v57 offset0:200 offset1:204
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[60:63], v[32:33], v[160:161], 0
	scratch_store_dwordx2 off, v[52:53], off offset:1368 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[64:65], v[60:61], v[22:23]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v218, 23, v7
	scratch_load_dword v7, off, off offset:992 ; 4-byte Folded Reload
	v_pk_mul_f32 v[82:83], v[218:219], v[16:17]
	v_mov_b32_e32 v219, v17
	v_pk_mul_f32 v[60:61], v[218:219], v[8:9] op_sel_hi:[1,0]
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v59, 23, v7
	scratch_load_dword v7, off, off offset:960 ; 4-byte Folded Reload
	v_pk_fma_f32 v[30:31], v[60:61], v[62:63], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[60:63], v[34:35], v[162:163], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v58, 23, v7
	scratch_load_dword v7, off, off offset:1348 ; 4-byte Folded Reload
	v_pk_mul_f32 v[64:65], v[0:1], v[58:59] op_sel_hi:[0,1]
	v_mov_b64_e32 v[46:47], v[58:59]
	s_nop 2
	v_pk_fma_f32 v[58:59], v[64:65], v[60:61], v[22:23]
	v_mov_b32_e32 v61, v189
	scratch_store_dwordx2 off, v[46:47], off offset:1360 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v60, 23, v7
	v_pk_mul_f32 v[22:23], v[60:61], v[0:1] op_sel_hi:[1,0]
	v_mov_b64_e32 v[84:85], v[60:61]
	v_pk_fma_f32 v[60:61], v[22:23], v[62:63], v[30:31]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[144:145], v[136:137], 0
	v_pk_mul_f32 v[30:31], v[168:169], v[50:51] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[58:61], off offset:960 ; 16-byte Folded Spill
	scratch_store_dwordx2 off, v[84:85], off offset:1348 ; 8-byte Folded Spill
	v_mov_b64_e32 v[144:145], v[194:195]
	s_nop 3
	v_pk_mul_f32 v[22:23], v[28:29], v[64:65]
	v_pk_mul_f32 v[28:29], v[252:253], v[48:49] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[28:29], v[62:63]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[146:147], v[138:139], 0
	s_nop 7
	v_pk_fma_f32 v[22:23], v[40:41], v[64:65], v[22:23]
	v_pk_fma_f32 v[28:29], v[30:31], v[62:63], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[230:231], v[132:133], 0
	v_pk_mul_f32 v[30:31], v[170:171], v[88:89] op_sel_hi:[0,1]
	v_add_u32_e32 v41, 0xd000, v79
	s_nop 5
	v_pk_fma_f32 v[22:23], v[42:43], v[64:65], v[22:23]
	v_pk_fma_f32 v[28:29], v[30:31], v[62:63], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[232:233], v[134:135], 0
	v_pk_mul_f32 v[30:31], v[26:27], v[90:91] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[54:55], v[64:65], v[22:23]
	v_pk_fma_f32 v[28:29], v[30:31], v[62:63], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[2:3], v[172:173], 0
	v_pk_mul_f32 v[2:3], v[24:25], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[188:189], v[46:47] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[28:29], v[2:3], v[62:63], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[4:5], v[174:175], 0
	v_pk_fma_f32 v[22:23], v[74:75], v[64:65], v[22:23]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[80:81], v[4:5], v[22:23]
	v_pk_mul_f32 v[4:5], v[18:19], v[44:45] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[4:5], v[2:3], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[32:33], v[176:177], 0
	s_nop 7
	v_pk_fma_f32 v[22:23], v[82:83], v[4:5], v[22:23]
	v_pk_mul_f32 v[4:5], v[16:17], v[52:53] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[4:5], v[2:3], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[34:35], v[178:179], 0
	s_nop 7
	v_pk_fma_f32 v[28:29], v[30:31], v[2:3], v[28:29]
	v_mov_b32_e32 v2, v84
	v_mov_b32_e32 v3, v188
	v_pk_mul_f32 v[2:3], v[2:3], v[188:189]
	s_nop 0
	v_pk_fma_f32 v[30:31], v[2:3], v[4:5], v[22:23]
	scratch_load_dword v2, off, off offset:1328 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1320 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_mov_b64_e32 v[4:5], v[2:3]
	scratch_load_dword v2, off, off offset:1296 ; 4-byte Folded Reload
	v_pk_mul_f32 v[22:23], v[20:21], v[4:5] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[28:31], off offset:992 ; 16-byte Folded Spill
	v_mov_b64_e32 v[42:43], v[4:5]
	scratch_store_dwordx2 off, v[42:43], off offset:1320 ; 8-byte Folded Spill
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1288 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_mov_b64_e32 v[28:29], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[106:107], v[128:129], 0
	v_mov_b64_e32 v[54:55], v[28:29]
	scratch_store_dwordx2 off, v[54:55], off offset:1296 ; 8-byte Folded Spill
	s_nop 5
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3]
	v_pk_mul_f32 v[2:3], v[20:21], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[2:3], v[4:5]
	scratch_load_dword v2, off, off offset:1220 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1204 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[14:15], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[60:61], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[108:109], v[130:131], 0
	scratch_store_dwordx2 off, v[60:61], off offset:1288 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1244 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v31, 23, v2
	scratch_load_dword v2, off, off offset:1232 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v30, 23, v2
	v_pk_mul_f32 v[2:3], v[14:15], v[30:31] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:1196 ; 4-byte Folded Reload
	v_mov_b64_e32 v[58:59], v[30:31]
	scratch_store_dwordx2 off, v[58:59], off offset:1220 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1168 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[12:13], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[76:77], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[36:37], v[148:149], 0
	scratch_store_dwordx2 off, v[76:77], off offset:1204 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1228 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v31, 23, v2
	scratch_load_dword v2, off, off offset:1216 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v30, 23, v2
	v_pk_mul_f32 v[2:3], v[12:13], v[30:31] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:1160 ; 4-byte Folded Reload
	v_mov_b64_e32 v[80:81], v[30:31]
	scratch_store_dwordx2 off, v[80:81], off offset:1196 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1148 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[6:7], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[74:75], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[38:39], v[150:151], 0
	scratch_store_dwordx2 off, v[74:75], off offset:1168 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1212 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v31, 23, v2
	scratch_load_dword v2, off, off offset:1176 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v30, 23, v2
	v_pk_mul_f32 v[2:3], v[6:7], v[30:31] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:1132 ; 4-byte Folded Reload
	scratch_load_dword v7, off, off offset:1156 ; 4-byte Folded Reload
	v_mov_b64_e32 v[82:83], v[30:31]
	scratch_store_dwordx2 off, v[82:83], off offset:1160 ; 8-byte Folded Spill
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1120 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[10:11], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[88:89], v[2:3]
	ds_read2_b64 v[2:5], v41 offset0:208 offset1:212
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[2:3], v[152:153], 0
	scratch_store_dwordx2 off, v[88:89], off offset:1148 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[32:33], v[22:23]
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:1140 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:1096 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[10:11], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[94:95], v[32:33]
	v_pk_fma_f32 v[28:29], v[30:31], v[34:35], v[28:29]
	scratch_store_dwordx2 off, v[94:95], off offset:1140 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:1052 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:1128 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[250:251], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[90:91], v[32:33]
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[4:5], v[154:155], 0
	scratch_store_dwordx2 off, v[90:91], off offset:1132 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[32:33], v[22:23]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:1112 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:1036 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[250:251], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[92:93], v[32:33]
	v_pk_fma_f32 v[28:29], v[30:31], v[34:35], v[28:29]
	scratch_store_dwordx2 off, v[92:93], off offset:1120 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:1236 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:1104 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[8:9], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[84:85], v[32:33]
	ds_read2_b64 v[32:35], v41 offset0:216 offset1:220
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[32:33], v[160:161], 0
	scratch_store_dwordx2 off, v[84:85], off offset:1112 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[62:63], v[22:23]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v49, 23, v7
	scratch_load_dword v7, off, off offset:1068 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v48, 23, v7
	scratch_load_dword v7, off, off offset:1252 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[8:9], v[48:49] op_sel_hi:[0,1]
	v_mov_b64_e32 v[46:47], v[48:49]
	v_pk_fma_f32 v[28:29], v[30:31], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[34:35], v[162:163], 0
	scratch_store_dwordx2 off, v[46:47], off offset:1096 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v49, 23, v7
	scratch_load_dword v7, off, off offset:1248 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v48, 23, v7
	scratch_load_dword v7, off, off offset:1076 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_mov_b64_e32 v[52:53], v[48:49]
	v_pk_fma_f32 v[48:49], v[30:31], v[62:63], v[22:23]
	v_pk_mul_f32 v[30:31], v[168:169], v[60:61] op_sel_hi:[0,1]
	scratch_store_dwordx2 off, v[52:53], off offset:1104 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v51, 23, v7
	scratch_load_dword v7, off, off offset:1240 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v50, 23, v7
	v_pk_mul_f32 v[22:23], v[0:1], v[50:51] op_sel_hi:[0,1]
	v_mov_b64_e32 v[44:45], v[50:51]
	v_pk_fma_f32 v[50:51], v[22:23], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[106:107], v[136:137], 0
	v_pk_mul_f32 v[22:23], v[252:253], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[252:253], v[54:55] op_sel_hi:[0,1]
	scratch_store_dwordx2 off, v[44:45], off offset:1068 ; 8-byte Folded Spill
	scratch_store_dwordx4 off, v[48:51], off offset:1036 ; 16-byte Folded Spill
	v_mov_b64_e32 v[106:107], v[196:197]
	s_nop 2
	v_pk_mul_f32 v[22:23], v[22:23], v[62:63]
	v_pk_mul_f32 v[28:29], v[28:29], v[64:65]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[108:109], v[138:139], 0
	scratch_load_dwordx4 v[108:111], off, off offset:1332 ; 16-byte Folded Reload
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[62:63], v[22:23]
	v_pk_mul_f32 v[30:31], v[168:169], v[58:59] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[30:31], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[36:37], v[132:133], 0
	v_pk_mul_f32 v[30:31], v[170:171], v[76:77] op_sel_hi:[0,1]
	v_add_u32_e32 v37, 0xe000, v79
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[62:63], v[22:23]
	v_pk_mul_f32 v[30:31], v[170:171], v[80:81] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[30:31], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[38:39], v[134:135], 0
	v_pk_mul_f32 v[30:31], v[26:27], v[74:75] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[62:63], v[22:23]
	v_pk_mul_f32 v[30:31], v[26:27], v[82:83] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[30:31], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[2:3], v[172:173], 0
	v_pk_mul_f32 v[2:3], v[24:25], v[94:95] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[24:25], v[88:89] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[28:29], v[2:3], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[4:5], v[174:175], 0
	v_pk_fma_f32 v[22:23], v[30:31], v[62:63], v[22:23]
	v_pk_mul_f32 v[30:31], v[18:19], v[90:91] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[18:19], v[92:93] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[32:33], v[176:177], 0
	v_pk_mul_f32 v[30:31], v[16:17], v[84:85] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[16:17], v[46:47] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[34:35], v[178:179], 0
	v_pk_mul_f32 v[30:31], v[188:189], v[52:53] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[30:31], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[188:189], v[44:45] op_sel_hi:[0,1]
	v_pk_fma_f32 v[32:33], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:1016 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1008 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_mov_b64_e32 v[4:5], v[2:3]
	scratch_load_dword v2, off, off offset:952 ; 4-byte Folded Reload
	v_pk_mul_f32 v[22:23], v[20:21], v[4:5] op_sel_hi:[0,1]
	v_mov_b64_e32 v[38:39], v[4:5]
	scratch_store_dwordx4 off, v[30:33], off offset:1052 ; 16-byte Folded Spill
	scratch_store_dwordx2 off, v[38:39], off offset:1008 ; 8-byte Folded Spill
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:944 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_mov_b64_e32 v[28:29], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[226:227], v[128:129], 0
	v_mov_b64_e32 v[46:47], v[28:29]
	scratch_store_dwordx2 off, v[46:47], off offset:952 ; 8-byte Folded Spill
	s_nop 5
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3]
	v_pk_mul_f32 v[2:3], v[20:21], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[2:3], v[4:5]
	scratch_load_dword v2, off, off offset:924 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:272 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[14:15], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[76:77], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[228:229], v[130:131], 0
	scratch_store_dwordx2 off, v[76:77], off offset:944 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:940 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v31, 23, v2
	scratch_load_dword v2, off, off offset:936 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v30, 23, v2
	v_pk_mul_f32 v[2:3], v[14:15], v[30:31] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:900 ; 4-byte Folded Reload
	v_mov_b64_e32 v[54:55], v[30:31]
	scratch_store_dwordx2 off, v[54:55], off offset:272 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:256 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[12:13], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[82:83], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[120:121], v[148:149], 0
	scratch_store_dwordx2 off, v[82:83], off offset:924 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:932 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v31, 23, v2
	scratch_load_dword v2, off, off offset:916 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v30, 23, v2
	v_pk_mul_f32 v[2:3], v[12:13], v[30:31] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:884 ; 4-byte Folded Reload
	v_mov_b64_e32 v[42:43], v[30:31]
	scratch_store_dwordx2 off, v[42:43], off offset:256 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:264 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[6:7], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[84:85], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[122:123], v[150:151], 0
	scratch_store_dwordx2 off, v[84:85], off offset:900 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:908 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v31, 23, v2
	scratch_load_dword v2, off, off offset:892 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v30, 23, v2
	v_pk_mul_f32 v[2:3], v[6:7], v[30:31] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:844 ; 4-byte Folded Reload
	scratch_load_dword v7, off, off offset:876 ; 4-byte Folded Reload
	v_mov_b64_e32 v[52:53], v[30:31]
	scratch_store_dwordx2 off, v[52:53], off offset:264 ; 8-byte Folded Spill
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:828 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[30:31], v[10:11], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[88:89], v[2:3]
	ds_read2_b64 v[2:5], v37 offset0:224 offset1:228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[2:3], v[152:153], 0
	scratch_store_dwordx2 off, v[88:89], off offset:884 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[32:33], v[22:23]
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:852 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:804 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[10:11], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[58:59], v[32:33]
	v_pk_fma_f32 v[28:29], v[30:31], v[34:35], v[28:29]
	scratch_store_dwordx2 off, v[58:59], off offset:844 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:780 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:836 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[250:251], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[90:91], v[32:33]
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[4:5], v[154:155], 0
	scratch_store_dwordx2 off, v[90:91], off offset:828 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[32:33], v[22:23]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:820 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:772 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[250:251], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[60:61], v[32:33]
	v_pk_fma_f32 v[28:29], v[30:31], v[34:35], v[28:29]
	scratch_store_dwordx2 off, v[60:61], off offset:804 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v7
	scratch_load_dword v7, off, off offset:756 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v7
	scratch_load_dword v7, off, off offset:812 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[8:9], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[92:93], v[32:33]
	ds_read2_b64 v[32:35], v37 offset0:232 offset1:236
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[32:33], v[160:161], 0
	scratch_store_dwordx2 off, v[92:93], off offset:780 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[62:63], v[22:23]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v45, 23, v7
	scratch_load_dword v7, off, off offset:788 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v44, 23, v7
	scratch_load_dword v7, off, off offset:740 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[8:9], v[44:45] op_sel_hi:[0,1]
	v_mov_b64_e32 v[74:75], v[44:45]
	v_pk_fma_f32 v[28:29], v[30:31], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[34:35], v[162:163], 0
	scratch_store_dwordx2 off, v[74:75], off offset:772 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v49, 23, v7
	scratch_load_dword v7, off, off offset:724 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v48, 23, v7
	scratch_load_dword v7, off, off offset:796 ; 4-byte Folded Reload
	v_pk_mul_f32 v[30:31], v[0:1], v[48:49] op_sel_hi:[0,1]
	v_mov_b64_e32 v[94:95], v[48:49]
	v_pk_fma_f32 v[48:49], v[30:31], v[62:63], v[22:23]
	v_pk_mul_f32 v[30:31], v[168:169], v[76:77] op_sel_hi:[0,1]
	scratch_store_dwordx2 off, v[94:95], off offset:756 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v45, 23, v7
	scratch_load_dword v7, off, off offset:764 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v44, 23, v7
	v_pk_mul_f32 v[22:23], v[0:1], v[44:45] op_sel_hi:[0,1]
	v_pk_fma_f32 v[50:51], v[22:23], v[64:65], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[62:65], v[226:227], v[136:137], 0
	v_mov_b64_e32 v[80:81], v[44:45]
	v_pk_mul_f32 v[28:29], v[252:253], v[46:47] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[252:253], v[38:39] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[228:229], v[138:139], 0
	scratch_store_dwordx4 off, v[48:51], off offset:724 ; 16-byte Folded Spill
	s_nop 2
	v_pk_mul_f32 v[22:23], v[22:23], v[62:63]
	v_pk_mul_f32 v[28:29], v[28:29], v[64:65]
	v_add_u32_e32 v39, 0xf000, v79
	scratch_store_dwordx2 off, v[80:81], off offset:764 ; 8-byte Folded Spill
	v_pk_fma_f32 v[22:23], v[30:31], v[44:45], v[22:23]
	v_pk_mul_f32 v[30:31], v[168:169], v[54:55] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[30:31], v[46:47], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[120:121], v[132:133], 0
	v_pk_mul_f32 v[30:31], v[170:171], v[82:83] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[30:31], v[44:45], v[22:23]
	v_pk_mul_f32 v[30:31], v[170:171], v[42:43] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[30:31], v[46:47], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[122:123], v[134:135], 0
	v_pk_mul_f32 v[30:31], v[26:27], v[84:85] op_sel_hi:[0,1]
	v_mov_b64_e32 v[84:85], v[98:99]
	v_mov_b64_e32 v[98:99], v[70:71]
	s_nop 4
	v_pk_fma_f32 v[22:23], v[30:31], v[44:45], v[22:23]
	v_pk_mul_f32 v[30:31], v[26:27], v[52:53] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[30:31], v[46:47], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[2:3], v[172:173], 0
	v_pk_mul_f32 v[2:3], v[24:25], v[58:59] op_sel_hi:[0,1]
	v_pk_mul_f32 v[30:31], v[24:25], v[88:89] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[28:29], v[2:3], v[46:47], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[4:5], v[174:175], 0
	v_pk_fma_f32 v[22:23], v[30:31], v[44:45], v[22:23]
	v_pk_mul_f32 v[30:31], v[18:19], v[90:91] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[18:19], v[60:61] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[32:33], v[176:177], 0
	v_pk_mul_f32 v[30:31], v[16:17], v[92:93] op_sel_hi:[0,1]
	v_mov_b64_e32 v[92:93], v[192:193]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[16:17], v[74:75] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[4:5], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[34:35], v[178:179], 0
	v_pk_mul_f32 v[30:31], v[188:189], v[94:95] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[30:31], v[30:31], v[2:3], v[22:23]
	v_pk_mul_f32 v[2:3], v[188:189], v[80:81] op_sel_hi:[0,1]
	v_pk_fma_f32 v[32:33], v[2:3], v[4:5], v[28:29]
	scratch_load_dword v2, off, off offset:1588 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:700 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_mov_b64_e32 v[4:5], v[2:3]
	scratch_load_dword v2, off, off offset:1572 ; 4-byte Folded Reload
	v_pk_mul_f32 v[22:23], v[20:21], v[4:5] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[30:33], off offset:740 ; 16-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:1568 ; 4-byte Folded Reload
	v_mov_b64_e32 v[30:31], v[4:5]
	scratch_store_dwordx2 off, v[30:31], off offset:916 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v2, 23, v2
	v_mov_b64_e32 v[28:29], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[116:117], v[128:129], 0
	v_mov_b64_e32 v[42:43], v[28:29]
	scratch_store_dwordx2 off, v[42:43], off offset:908 ; 8-byte Folded Spill
	s_nop 5
	v_pk_mul_f32 v[22:23], v[22:23], v[2:3]
	v_pk_mul_f32 v[2:3], v[20:21], v[28:29] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[2:3], v[4:5]
	scratch_load_dword v2, off, off offset:1492 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v33, 23, v2
	scratch_load_dword v2, off, off offset:660 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v2
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[118:119], v[130:131], 0
	v_pk_mul_f32 v[28:29], v[14:15], v[32:33] op_sel_hi:[0,1]
	v_mov_b64_e32 v[52:53], v[32:33]
	scratch_store_dwordx2 off, v[52:53], off offset:876 ; 8-byte Folded Spill
	s_nop 4
	v_pk_fma_f32 v[22:23], v[28:29], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1544 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v29, 23, v2
	scratch_load_dword v2, off, off offset:1524 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v28, 23, v2
	v_pk_mul_f32 v[2:3], v[14:15], v[28:29] op_sel_hi:[0,1]
	v_pk_fma_f32 v[20:21], v[2:3], v[4:5], v[20:21]
	scratch_load_dword v2, off, off offset:1468 ; 4-byte Folded Reload
	v_mov_b64_e32 v[14:15], v[28:29]
	scratch_store_dwordx2 off, v[14:15], off offset:852 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v3, 23, v2
	scratch_load_dword v2, off, off offset:652 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v2, 23, v2
	v_pk_mul_f32 v[28:29], v[12:13], v[2:3] op_sel_hi:[0,1]
	v_mov_b64_e32 v[58:59], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[238:239], v[148:149], 0
	scratch_store_dwordx2 off, v[58:59], off offset:892 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[22:23], v[28:29], v[2:3], v[22:23]
	scratch_load_dword v2, off, off offset:1540 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v29, 23, v2
	scratch_load_dword v2, off, off offset:1496 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v28, 23, v2
	v_pk_mul_f32 v[2:3], v[12:13], v[28:29] op_sel_hi:[0,1]
	v_pk_fma_f32 v[20:21], v[2:3], v[4:5], v[20:21]
	scratch_load_dword v2, off, off offset:1456 ; 4-byte Folded Reload
	v_mov_b64_e32 v[48:49], v[28:29]
	scratch_store_dwordx2 off, v[48:49], off offset:836 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v33, 23, v2
	scratch_load_dword v2, off, off offset:644 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v32, 23, v2
	v_pk_mul_f32 v[28:29], v[6:7], v[32:33] op_sel_hi:[0,1]
	scratch_load_dword v7, off, off offset:1520 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[240:241], v[150:151], 0
	v_mov_b64_e32 v[64:65], v[32:33]
	scratch_store_dwordx2 off, v[64:65], off offset:820 ; 8-byte Folded Spill
	s_nop 5
	v_pk_fma_f32 v[2:3], v[28:29], v[2:3], v[22:23]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v23, 23, v7
	scratch_load_dword v7, off, off offset:1472 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v22, 23, v7
	v_pk_mul_f32 v[6:7], v[6:7], v[22:23] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[4:5], v[20:21]
	scratch_load_dword v6, off, off offset:1428 ; 4-byte Folded Reload
	v_mov_b64_e32 v[50:51], v[22:23]
	scratch_store_dwordx2 off, v[50:51], off offset:812 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v21, 23, v6
	scratch_load_dword v6, off, off offset:636 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v20, 23, v6
	v_pk_mul_f32 v[6:7], v[10:11], v[20:21] op_sel_hi:[0,1]
	v_mov_b64_e32 v[76:77], v[20:21]
	ds_read2_b64 v[20:23], v39 offset0:240 offset1:244
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[20:21], v[152:153], 0
	scratch_store_dwordx2 off, v[76:77], off offset:796 ; 8-byte Folded Spill
	s_nop 6
	v_pk_fma_f32 v[2:3], v[6:7], v[32:33], v[2:3]
	scratch_load_dword v6, off, off offset:1488 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v29, 23, v6
	scratch_load_dword v6, off, off offset:1444 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v28, 23, v6
	v_pk_mul_f32 v[6:7], v[10:11], v[28:29] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[34:35], v[4:5]
	scratch_load_dword v6, off, off offset:1396 ; 4-byte Folded Reload
	v_mov_b64_e32 v[54:55], v[28:29]
	v_mfma_f32_16x16x32_fp8_fp8 v[32:35], v[22:23], v[154:155], 0
	scratch_load_dword v10, off, off offset:1420 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v29, 23, v6
	scratch_load_dword v6, off, off offset:628 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v63, 23, v10
	scratch_load_dword v10, off, off offset:1384 ; 4-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v28, 23, v6
	v_pk_mul_f32 v[6:7], v[250:251], v[28:29] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[6:7], v[32:33], v[2:3]
	scratch_load_dword v6, off, off offset:1460 ; 4-byte Folded Reload
	v_mov_b64_e32 v[78:79], v[28:29]
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v62, 23, v10
	scratch_store_dwordx2 off, v[62:63], off offset:628 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[78:79], off offset:700 ; 8-byte Folded Spill
	scratch_store_dwordx2 off, v[54:55], off offset:788 ; 8-byte Folded Spill
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v29, 23, v6
	scratch_load_dword v6, off, off offset:1424 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v28, 23, v6
	v_pk_mul_f32 v[6:7], v[250:251], v[28:29] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[34:35], v[4:5]
	scratch_load_dword v6, off, off offset:1376 ; 4-byte Folded Reload
	v_mov_b64_e32 v[60:61], v[28:29]
	ds_read2_b64 v[32:35], v39 offset0:248 offset1:252
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[32:33], v[160:161], 0
	scratch_store_dwordx2 off, v[60:61], off offset:660 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v29, 23, v6
	scratch_load_dword v6, off, off offset:612 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v28, 23, v6
	v_pk_mul_f32 v[6:7], v[8:9], v[28:29] op_sel_hi:[0,1]
	s_nop 0
	v_pk_fma_f32 v[2:3], v[6:7], v[44:45], v[2:3]
	scratch_load_dword v6, off, off offset:1432 ; 4-byte Folded Reload
	v_mov_b64_e32 v[80:81], v[28:29]
	scratch_store_dwordx2 off, v[80:81], off offset:652 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v29, 23, v6
	scratch_load_dword v6, off, off offset:1400 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v28, 23, v6
	v_pk_mul_f32 v[6:7], v[8:9], v[28:29] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[46:47], v[4:5]
	scratch_load_dword v6, off, off offset:1356 ; 4-byte Folded Reload
	v_mov_b64_e32 v[74:75], v[28:29]
	scratch_store_dwordx2 off, v[74:75], off offset:644 ; 8-byte Folded Spill
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v45, 23, v6
	scratch_load_dword v6, off, off offset:596 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v44, 23, v6
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[34:35], v[162:163], 0
	v_pk_mul_f32 v[28:29], v[0:1], v[44:45] op_sel_hi:[0,1]
	v_mov_b64_e32 v[82:83], v[44:45]
	scratch_store_dwordx2 off, v[82:83], off offset:636 ; 8-byte Folded Spill
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[116:117], v[136:137], 0
	v_mov_b64_e32 v[136:137], v[184:185]
	s_nop 2
	v_pk_fma_f32 v[6:7], v[28:29], v[6:7], v[2:3]
	v_pk_mul_f32 v[2:3], v[0:1], v[62:63] op_sel_hi:[0,1]
	scratch_load_dword v0, off, off offset:496 ; 4-byte Folded Reload
	v_pk_fma_f32 v[8:9], v[2:3], v[8:9], v[4:5]
	scratch_store_dwordx4 off, v[6:9], off offset:596 ; 16-byte Folded Spill
	v_pk_mul_f32 v[2:3], v[252:253], v[30:31] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[2:3], v[44:45]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[118:119], v[138:139], 0
	v_pk_mul_f32 v[4:5], v[252:253], v[42:43] op_sel_hi:[0,1]
	v_pk_mul_f32 v[28:29], v[168:169], v[52:53] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[4:5], v[46:47]
	scratch_load_dwordx2 v[44:45], off, off offset:64 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[46:47], off, off offset:112 ; 8-byte Folded Reload
	s_nop 2
	v_pk_fma_f32 v[2:3], v[28:29], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[168:169], v[14:15] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[238:239], v[132:133], 0
	v_pk_mul_f32 v[28:29], v[170:171], v[58:59] op_sel_hi:[0,1]
	v_mov_b64_e32 v[132:133], v[186:187]
	s_waitcnt vmcnt(3)
	v_lshlrev_b32_e32 v214, 23, v0
	scratch_load_dword v0, off, off offset:460 ; 4-byte Folded Reload
	s_nop 2
	v_pk_fma_f32 v[2:3], v[28:29], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[170:171], v[48:49] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[240:241], v[134:135], 0
	v_pk_mul_f32 v[28:29], v[26:27], v[64:65] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[170:171], off, off offset:32 ; 8-byte Folded Reload
	s_nop 5
	v_pk_fma_f32 v[2:3], v[28:29], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[26:27], v[50:51] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[20:21], v[172:173], 0
	v_pk_mul_f32 v[26:27], v[24:25], v[76:77] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[18:19], v[78:79] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[78:79], off, off offset:128 ; 8-byte Folded Reload
	s_nop 4
	v_pk_fma_f32 v[2:3], v[26:27], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[24:25], v[54:55] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[22:23], v[174:175], 0
	s_nop 7
	v_pk_fma_f32 v[2:3], v[20:21], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[18:19], v[60:61] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[32:33], v[176:177], 0
	v_pk_mul_f32 v[18:19], v[16:17], v[80:81] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[80:81], off, off offset:16 ; 8-byte Folded Reload
	v_mov_b64_e32 v[60:61], v[72:73]
	s_nop 4
	v_pk_fma_f32 v[2:3], v[18:19], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[16:17], v[74:75] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[34:35], v[178:179], 0
	v_pk_mul_f32 v[18:19], v[188:189], v[82:83] op_sel_hi:[0,1]
	v_mov_b64_e32 v[34:35], v[222:223]
	s_nop 5
	v_pk_fma_f32 v[6:7], v[18:19], v[6:7], v[2:3]
	v_pk_mul_f32 v[2:3], v[188:189], v[62:63] op_sel_hi:[0,1]
	scratch_load_dwordx4 v[62:65], off, off offset:1304 ; 16-byte Folded Reload
	v_pk_fma_f32 v[8:9], v[2:3], v[8:9], v[4:5]
	scratch_store_dwordx4 off, v[6:9], off offset:612 ; 16-byte Folded Spill
	s_waitcnt vmcnt(5)
	v_lshlrev_b32_e32 v18, 23, v0
	scratch_load_dword v0, off, off offset:456 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[108:109], v[234:235], 0
	v_pk_mul_f32 v[20:21], v[18:19], v[216:217] op_sel_hi:[0,1]
	s_waitcnt vmcnt(5)
	v_pk_mul_f32 v[4:5], v[214:215], v[170:171] op_sel_hi:[0,1]
	s_waitcnt vmcnt(3)
	v_pk_mul_f32 v[2:3], v[214:215], v[80:81] op_sel_hi:[0,1]
	s_nop 2
	v_pk_mul_f32 v[2:3], v[2:3], v[6:7]
	v_pk_mul_f32 v[4:5], v[4:5], v[8:9]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[110:111], v[236:237], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v40, 23, v0
	scratch_load_dword v0, off, off offset:452 ; 4-byte Folded Reload
	s_nop 4
	v_pk_fma_f32 v[2:3], v[20:21], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[18:19], v[44:45] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[62:63], v[66:67], 0
	v_pk_mul_f32 v[20:21], v[40:41], v[46:47] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	scratch_load_dword v10, off, off offset:1504 ; 4-byte Folded Reload
	v_mov_b32_e32 v19, v13
	scratch_load_dwordx2 v[168:169], off, off offset:136 ; 8-byte Folded Reload
	s_nop 2
	v_pk_fma_f32 v[2:3], v[20:21], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[40:41], v[212:213] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[64:65], v[68:69], 0
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v36, 23, v0
	v_pk_mul_f32 v[20:21], v[36:37], v[60:61] op_sel_hi:[0,1]
	s_nop 4
	v_pk_fma_f32 v[2:3], v[20:21], v[6:7], v[2:3]
	v_pk_mul_f32 v[6:7], v[36:37], v[78:79] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	scratch_load_dword v6, off, off offset:436 ; 4-byte Folded Reload
	v_add_u32_e32 v0, 0xc000, v19
	s_waitcnt vmcnt(2)
	ds_read2_b64 v[172:175], v10 offset0:128 offset1:132
	ds_read2_b64 v[160:163], v0 offset0:192 offset1:196
	scratch_load_dwordx2 v[42:43], off, off offset:232 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[210:211], off, off offset:96 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[158:159], off, off offset:80 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[204:205], off, off ; 8-byte Folded Reload
	s_waitcnt vmcnt(4)
	v_lshlrev_b32_e32 v20, 23, v6
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[6:9], v[172:173], v[160:161], 0
	v_pk_mul_f32 v[22:23], v[20:21], v[168:169] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[2:3], v[22:23], v[6:7], v[2:3]
	s_waitcnt vmcnt(3)
	v_pk_mul_f32 v[6:7], v[20:21], v[42:43] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[8:9], v[4:5]
	scratch_load_dword v6, off, off offset:420 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[174:175], v[162:163], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v38, 23, v6
	v_pk_mul_f32 v[6:7], v[38:39], v[210:211] op_sel_hi:[0,1]
	s_nop 4
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[38:39], v[158:159] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	scratch_load_dword v6, off, off offset:404 ; 4-byte Folded Reload
	ds_read2_b64 v[176:179], v10 offset0:136 offset1:140
	ds_read2_b64 v[152:155], v0 offset0:200 offset1:204
	scratch_load_dwordx2 v[74:75], off, off offset:144 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[206:207], off, off offset:152 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[230:231], off, off offset:160 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[232:233], off, off offset:168 ; 8-byte Folded Reload
	scratch_load_dwordx4 v[238:241], off, off offset:1272 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[226:229], off, off offset:1256 ; 16-byte Folded Reload
	scratch_load_dword v0, off, off offset:356 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[130:131], off, off offset:248 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[252:253], off, off offset:176 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[208:209], off, off offset:184 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[164:165], off, off offset:192 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[146:147], off, off offset:200 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[156:157], off, off offset:208 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[166:167], off, off offset:216 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[148:149], off, off offset:224 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[150:151], off, off offset:240 ; 8-byte Folded Reload
	scratch_load_dwordx4 v[50:53], off, off offset:1180 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[30:33], off, off offset:1080 ; 16-byte Folded Reload
	scratch_load_dwordx2 v[102:103], off, off offset:1804 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[112:113], off, off offset:1796 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[140:141], off, off offset:1788 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[118:119], off, off offset:1780 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[120:121], off, off offset:1772 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[116:117], off, off offset:1764 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[134:135], off, off offset:1756 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[122:123], off, off offset:1748 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[128:129], off, off offset:1740 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[94:95], off, off offset:1732 ; 8-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[176:177], v[152:153], 0
	scratch_load_dwordx2 v[138:139], off, off offset:1724 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[76:77], off, off offset:1716 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[88:89], off, off offset:1708 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[90:91], off, off offset:1700 ; 8-byte Folded Reload
	ds_read2_b64 v[180:183], v11 offset0:144 offset1:148
	ds_read2_b64 v[184:187], v11 offset0:152 offset1:156
	scratch_load_dwordx4 v[26:29], off, off offset:1020 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[12:15], off, off offset:976 ; 16-byte Folded Reload
	ds_read2_b64 v[188:191], v251 offset0:160 offset1:164
	scratch_load_dwordx2 v[48:49], off, off offset:1684 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[54:55], off, off offset:1692 ; 8-byte Folded Reload
	ds_read2_b64 v[192:195], v251 offset0:168 offset1:172
	scratch_load_dwordx2 v[70:71], off, off offset:1676 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[72:73], off, off offset:1668 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[16:17], off, off offset:1660 ; 8-byte Folded Reload
	v_mov_b64_e32 v[250:251], v[86:87]
	scratch_load_dwordx2 v[10:11], off, off offset:1652 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[222:223], off, off offset:1636 ; 8-byte Folded Reload
	v_mov_b64_e32 v[86:87], v[224:225]
	scratch_load_dwordx2 v[224:225], off, off offset:1628 ; 8-byte Folded Reload
	ds_read2_b64 v[196:199], v1 offset0:176 offset1:180
	ds_read2_b64 v[200:203], v1 offset0:184 offset1:188
	s_waitcnt vmcnt(42)
	v_lshlrev_b32_e32 v58, 23, v6
	v_pk_mul_f32 v[6:7], v[58:59], v[204:205] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(41)
	v_pk_mul_f32 v[6:7], v[58:59], v[74:75] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[178:179], v[154:155], 0
	s_waitcnt vmcnt(35)
	v_lshlrev_b32_e32 v56, 23, v0
	v_pk_mul_f32 v[6:7], v[56:57], v[206:207] op_sel_hi:[0,1]
	s_nop 4
	v_pk_fma_f32 v[6:7], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[2:3], v[56:57], v[230:231] op_sel_hi:[0,1]
	v_pk_fma_f32 v[8:9], v[2:3], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[238:239], v[234:235], 0
	v_pk_mul_f32 v[2:3], v[214:215], v[232:233] op_sel_hi:[0,1]
	s_waitcnt vmcnt(33)
	v_pk_mul_f32 v[4:5], v[214:215], v[252:253] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[6:9], off offset:356 ; 16-byte Folded Spill
	s_waitcnt vmcnt(33)
	s_nop 0
	v_pk_mul_f32 v[6:7], v[18:19], v[208:209] op_sel_hi:[0,1]
	s_nop 0
	v_pk_mul_f32 v[2:3], v[2:3], v[22:23]
	v_pk_mul_f32 v[4:5], v[4:5], v[24:25]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[240:241], v[236:237], 0
	s_nop 7
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(32)
	v_pk_mul_f32 v[6:7], v[18:19], v[164:165] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[226:227], v[66:67], 0
	s_waitcnt vmcnt(31)
	v_pk_mul_f32 v[6:7], v[40:41], v[146:147] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(30)
	v_pk_mul_f32 v[6:7], v[40:41], v[156:157] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[228:229], v[68:69], 0
	s_waitcnt vmcnt(29)
	v_pk_mul_f32 v[6:7], v[36:37], v[166:167] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(28)
	v_pk_mul_f32 v[6:7], v[36:37], v[148:149] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	s_waitcnt lgkmcnt(5)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[180:181], v[160:161], 0
	s_waitcnt vmcnt(27)
	v_pk_mul_f32 v[6:7], v[20:21], v[150:151] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[20:21], v[130:131] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[182:183], v[162:163], 0
	v_pk_mul_f32 v[6:7], v[38:39], v[144:145] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[38:39], v[132:133] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	s_waitcnt lgkmcnt(4)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[184:185], v[152:153], 0
	v_pk_mul_f32 v[6:7], v[58:59], v[136:137] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[58:59], v[106:107] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[186:187], v[154:155], 0
	v_pk_mul_f32 v[6:7], v[56:57], v[142:143] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[6:7], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(24)
	v_pk_mul_f32 v[2:3], v[56:57], v[102:103] op_sel_hi:[0,1]
	v_pk_fma_f32 v[8:9], v[2:3], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[50:51], v[234:235], 0
	s_waitcnt vmcnt(23)
	v_pk_mul_f32 v[2:3], v[214:215], v[112:113] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[214:215], v[114:115] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[6:9], off offset:404 ; 16-byte Folded Spill
	s_waitcnt vmcnt(23)
	s_nop 0
	v_pk_mul_f32 v[6:7], v[18:19], v[140:141] op_sel_hi:[0,1]
	s_nop 0
	v_pk_mul_f32 v[2:3], v[2:3], v[22:23]
	v_pk_mul_f32 v[4:5], v[4:5], v[24:25]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[52:53], v[236:237], 0
	s_nop 7
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(22)
	v_pk_mul_f32 v[6:7], v[18:19], v[118:119] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[30:31], v[66:67], 0
	s_waitcnt vmcnt(20)
	v_pk_mul_f32 v[6:7], v[40:41], v[116:117] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[40:41], v[120:121] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[32:33], v[68:69], 0
	s_waitcnt vmcnt(19)
	v_pk_mul_f32 v[6:7], v[36:37], v[134:135] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(18)
	v_pk_mul_f32 v[6:7], v[36:37], v[122:123] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	s_waitcnt lgkmcnt(3)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[188:189], v[160:161], 0
	s_waitcnt vmcnt(17)
	v_pk_mul_f32 v[6:7], v[20:21], v[128:129] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[20:21], v[100:101] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[190:191], v[162:163], 0
	v_pk_mul_f32 v[6:7], v[38:39], v[104:105] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(16)
	v_pk_mul_f32 v[6:7], v[38:39], v[94:95] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	s_waitcnt lgkmcnt(2)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[192:193], v[152:153], 0
	v_pk_mul_f32 v[6:7], v[58:59], v[92:93] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(15)
	v_pk_mul_f32 v[6:7], v[58:59], v[138:139] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[194:195], v[154:155], 0
	v_pk_mul_f32 v[6:7], v[56:57], v[250:251] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[6:7], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(14)
	v_pk_mul_f32 v[2:3], v[56:57], v[76:77] op_sel_hi:[0,1]
	v_pk_fma_f32 v[8:9], v[2:3], v[24:25], v[4:5]
	scratch_store_dwordx4 off, v[6:9], off offset:420 ; 16-byte Folded Spill
	scratch_load_dwordx2 v[8:9], off, off offset:1644 ; 8-byte Folded Reload
	s_waitcnt vmcnt(13)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[26:27], v[234:235], 0
	v_pk_mul_f32 v[2:3], v[214:215], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[4:5], v[214:215], v[88:89] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[18:19], v[98:99] op_sel_hi:[0,1]
	s_nop 4
	v_pk_mul_f32 v[2:3], v[2:3], v[22:23]
	v_pk_mul_f32 v[4:5], v[4:5], v[24:25]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[28:29], v[236:237], 0
	s_nop 7
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[18:19], v[90:91] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	s_waitcnt vmcnt(12)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[12:13], v[66:67], 0
	s_waitcnt vmcnt(11)
	v_pk_mul_f32 v[6:7], v[40:41], v[48:49] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(10)
	v_pk_mul_f32 v[6:7], v[40:41], v[54:55] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[14:15], v[68:69], 0
	v_pk_mul_f32 v[6:7], v[36:37], v[96:97] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(9)
	v_pk_mul_f32 v[6:7], v[36:37], v[70:71] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	s_waitcnt lgkmcnt(1)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[196:197], v[160:161], 0
	s_waitcnt vmcnt(8)
	v_pk_mul_f32 v[6:7], v[20:21], v[72:73] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	v_pk_mul_f32 v[6:7], v[20:21], v[34:35] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[198:199], v[162:163], 0
	v_pk_mul_f32 v[6:7], v[38:39], v[86:87] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[2:3], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(7)
	v_pk_mul_f32 v[6:7], v[38:39], v[16:17] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[6:7], v[24:25], v[4:5]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[200:201], v[152:153], 0
	s_waitcnt vmcnt(6)
	v_pk_mul_f32 v[6:7], v[58:59], v[10:11] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[0:1], v[6:7], v[22:23], v[2:3]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[58:59], v[8:9] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[2:3], v[24:25], v[4:5]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[202:203], v[154:155], 0
	v_pk_mul_f32 v[4:5], v[56:57], v[222:223] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[4:5], v[4:5], v[22:23], v[0:1]
	v_pk_mul_f32 v[0:1], v[56:57], v[224:225] op_sel_hi:[0,1]
	v_pk_fma_f32 v[6:7], v[0:1], v[24:25], v[2:3]
	scratch_load_dword v0, off, off offset:336 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[108:109], v[124:125], 0
	scratch_store_dwordx4 off, v[4:7], off offset:436 ; 16-byte Folded Spill
	scratch_load_dword v4, off, off offset:332 ; 4-byte Folded Reload
	s_waitcnt vmcnt(2)
	v_lshlrev_b32_e32 v82, 23, v0
	v_pk_mul_f32 v[0:1], v[82:83], v[80:81] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[82:83], v[170:171] op_sel_hi:[0,1]
	s_nop 1
	v_pk_mul_f32 v[0:1], v[0:1], v[22:23]
	v_pk_mul_f32 v[2:3], v[2:3], v[24:25]
	v_mfma_f32_16x16x32_fp8_fp8 v[22:25], v[110:111], v[126:127], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v80, 23, v4
	v_pk_mul_f32 v[4:5], v[80:81], v[216:217] op_sel_hi:[0,1]
	v_add_u32_e32 v6, 0xe000, v19
	ds_read2_b64 v[108:111], v6 offset0:232 offset1:236
	s_nop 2
	v_pk_fma_f32 v[0:1], v[4:5], v[22:23], v[0:1]
	v_pk_mul_f32 v[4:5], v[80:81], v[44:45] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[24:25], v[2:3]
	scratch_load_dword v4, off, off offset:328 ; 4-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v216, 23, v4
	v_pk_mul_f32 v[4:5], v[216:217], v[46:47] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[62:63], v[242:243], 0
	s_nop 7
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[216:217], v[212:213] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	scratch_load_dword v4, off, off offset:324 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[64:65], v[244:245], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v62, 23, v4
	v_pk_mul_f32 v[4:5], v[62:63], v[60:61] op_sel_hi:[0,1]
	s_nop 4
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[62:63], v[78:79] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	scratch_load_dword v4, off, off offset:320 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[60:61], off, off offset:1448 ; 8-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v78, 23, v4
	v_pk_mul_f32 v[4:5], v[78:79], v[168:169] op_sel_hi:[0,1]
	ds_read2_b64 v[168:171], v6 offset0:224 offset1:228
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[172:173], v[168:169], 0
	s_nop 7
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[78:79], v[42:43] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	scratch_load_dword v4, off, off offset:316 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[174:175], v[170:171], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v42, 23, v4
	v_pk_mul_f32 v[4:5], v[42:43], v[210:211] op_sel_hi:[0,1]
	s_nop 4
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[42:43], v[158:159] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	scratch_load_dword v4, off, off offset:312 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[176:177], v[108:109], 0
	scratch_load_dwordx2 v[210:211], off, off offset:1512 ; 8-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_lshlrev_b32_e32 v158, 23, v4
	v_pk_mul_f32 v[4:5], v[158:159], v[204:205] op_sel_hi:[0,1]
	s_nop 3
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[158:159], v[74:75] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	scratch_load_dword v4, off, off offset:308 ; 4-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[178:179], v[110:111], 0
	s_waitcnt vmcnt(0)
	v_lshlrev_b32_e32 v74, 23, v4
	v_pk_mul_f32 v[4:5], v[74:75], v[206:207] op_sel_hi:[0,1]
	s_nop 4
	v_pk_fma_f32 v[4:5], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[0:1], v[74:75], v[230:231] op_sel_hi:[0,1]
	v_pk_fma_f32 v[6:7], v[0:1], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[238:239], v[124:125], 0
	v_pk_mul_f32 v[0:1], v[82:83], v[232:233] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[82:83], v[252:253] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[4:7], off  ; 16-byte Folded Spill
	s_nop 1
	v_pk_mul_f32 v[4:5], v[80:81], v[208:209] op_sel_hi:[0,1]
	s_nop 1
	v_pk_mul_f32 v[0:1], v[0:1], v[44:45]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[240:241], v[126:127], 0
	scratch_load_dwordx2 v[208:209], off, off offset:1548 ; 8-byte Folded Reload
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[80:81], v[164:165] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[226:227], v[242:243], 0
	v_pk_mul_f32 v[4:5], v[216:217], v[146:147] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[216:217], v[156:157] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[228:229], v[244:245], 0
	v_pk_mul_f32 v[4:5], v[62:63], v[166:167] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[62:63], v[148:149] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[180:181], v[168:169], 0
	v_pk_mul_f32 v[4:5], v[78:79], v[150:151] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[78:79], v[130:131] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[182:183], v[170:171], 0
	v_pk_mul_f32 v[4:5], v[42:43], v[144:145] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[42:43], v[132:133] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[184:185], v[108:109], 0
	v_pk_mul_f32 v[4:5], v[158:159], v[136:137] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[158:159], v[106:107] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[186:187], v[110:111], 0
	v_pk_mul_f32 v[4:5], v[74:75], v[142:143] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[4:5], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[0:1], v[74:75], v[102:103] op_sel_hi:[0,1]
	v_pk_fma_f32 v[6:7], v[0:1], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[50:51], v[124:125], 0
	v_pk_mul_f32 v[0:1], v[82:83], v[112:113] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[82:83], v[114:115] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[4:7], off offset:16 ; 16-byte Folded Spill
	s_nop 1
	v_pk_mul_f32 v[4:5], v[80:81], v[140:141] op_sel_hi:[0,1]
	s_nop 1
	v_pk_mul_f32 v[0:1], v[0:1], v[44:45]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[52:53], v[126:127], 0
	s_nop 7
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[80:81], v[118:119] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[30:31], v[242:243], 0
	v_pk_mul_f32 v[4:5], v[216:217], v[116:117] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[216:217], v[120:121] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[32:33], v[244:245], 0
	v_pk_mul_f32 v[4:5], v[62:63], v[134:135] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[62:63], v[122:123] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[188:189], v[168:169], 0
	v_pk_mul_f32 v[4:5], v[78:79], v[128:129] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[78:79], v[100:101] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[190:191], v[170:171], 0
	v_pk_mul_f32 v[4:5], v[42:43], v[104:105] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[42:43], v[94:95] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[192:193], v[108:109], 0
	v_pk_mul_f32 v[4:5], v[158:159], v[92:93] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[158:159], v[138:139] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[194:195], v[110:111], 0
	v_pk_mul_f32 v[4:5], v[74:75], v[250:251] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[4:5], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[0:1], v[74:75], v[76:77] op_sel_hi:[0,1]
	v_pk_fma_f32 v[6:7], v[0:1], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[26:27], v[124:125], 0
	v_pk_mul_f32 v[0:1], v[82:83], v[84:85] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[82:83], v[88:89] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[4:7], off offset:32 ; 16-byte Folded Spill
	s_nop 1
	v_pk_mul_f32 v[4:5], v[80:81], v[98:99] op_sel_hi:[0,1]
	s_nop 1
	v_pk_mul_f32 v[0:1], v[0:1], v[44:45]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[28:29], v[126:127], 0
	s_nop 7
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[80:81], v[90:91] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[12:13], v[242:243], 0
	v_pk_mul_f32 v[4:5], v[216:217], v[48:49] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[216:217], v[54:55] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[14:15], v[244:245], 0
	v_pk_mul_f32 v[4:5], v[62:63], v[96:97] op_sel_hi:[0,1]
	scratch_load_dwordx4 v[12:15], off, off offset:564 ; 16-byte Folded Reload
	s_nop 5
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[62:63], v[70:71] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[196:197], v[168:169], 0
	v_pk_mul_f32 v[4:5], v[78:79], v[72:73] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[70:71], off, off offset:1556 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[72:73], off, off offset:1528 ; 8-byte Folded Reload
	s_nop 4
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[78:79], v[34:35] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[198:199], v[170:171], 0
	v_pk_mul_f32 v[4:5], v[42:43], v[86:87] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[42:43], v[16:17] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[200:201], v[108:109], 0
	v_pk_mul_f32 v[4:5], v[158:159], v[10:11] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[158:159], v[8:9] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[202:203], v[110:111], 0
	v_pk_mul_f32 v[4:5], v[74:75], v[222:223] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[222:223], off, off offset:1480 ; 8-byte Folded Reload
	s_nop 5
	v_pk_fma_f32 v[4:5], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[0:1], v[74:75], v[224:225] op_sel_hi:[0,1]
	v_pk_fma_f32 v[6:7], v[0:1], v[46:47], v[2:3]
	scratch_store_dwordx4 off, v[4:7], off offset:48 ; 16-byte Folded Spill
	scratch_load_dwordx4 v[6:9], off, off offset:580 ; 16-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	scratch_load_dwordx2 v[226:227], off, off offset:1412 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[230:231], off, off offset:1388 ; 8-byte Folded Reload
	scratch_load_dwordx4 v[100:103], off, off offset:548 ; 16-byte Folded Reload
	scratch_load_dwordx2 v[64:65], off, off offset:660 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[166:167], off, off offset:1368 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[76:77], off, off offset:1360 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[206:207], off, off offset:1348 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[164:165], off, off offset:1320 ; 8-byte Folded Reload
	scratch_load_dwordx4 v[184:187], off, off offset:532 ; 16-byte Folded Reload
	scratch_load_dwordx4 v[24:27], off, off offset:516 ; 16-byte Folded Reload
	scratch_load_dwordx2 v[156:157], off, off offset:1296 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[150:151], off, off offset:1288 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[148:149], off, off offset:1220 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[146:147], off, off offset:1204 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[144:145], off, off offset:1196 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[204:205], off, off offset:1168 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[200:201], off, off offset:1160 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[138:139], off, off offset:1148 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[140:141], off, off offset:1140 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[142:143], off, off offset:1132 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[136:137], off, off offset:1120 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[122:123], off, off offset:1112 ; 8-byte Folded Reload
	s_waitcnt vmcnt(22)
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[6:7], v[234:235], 0
	v_pk_mul_f32 v[0:1], v[214:215], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[208:209], v[214:215] op_sel_hi:[1,0]
	v_pk_mul_f32 v[4:5], v[18:19], v[72:73] op_sel_hi:[0,1]
	s_nop 4
	v_pk_mul_f32 v[0:1], v[0:1], v[44:45]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[8:9], v[236:237], 0
	ds_read2_b64 v[84:87], v57 offset0:192 offset1:196
	ds_read2_b64 v[96:99], v57 offset0:200 offset1:204
	v_mov_b64_e32 v[34:35], v[8:9]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[98:99], v[154:155], 0
	s_nop 2
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[210:211], v[18:19] op_sel_hi:[1,0]
	ds_read2_b64 v[180:183], v41 offset0:208 offset1:212
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[12:13], v[66:67], 0
	v_pk_mul_f32 v[4:5], v[40:41], v[222:223] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[128:129], off, off offset:1096 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[130:131], off, off offset:1104 ; 8-byte Folded Reload
	v_mov_b64_e32 v[32:33], v[6:7]
	s_waitcnt vmcnt(14)
	v_mfma_f32_16x16x32_fp8_fp8 v[52:55], v[24:25], v[66:67], 0
	s_nop 1
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[254:255], v[40:41] op_sel_hi:[1,0]
	scratch_load_dwordx2 v[116:117], off, off offset:1068 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[120:121], off, off offset:1008 ; 8-byte Folded Reload
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[14:15], v[68:69], 0
	v_mov_b64_e32 v[4:5], v[60:61]
	v_pk_mul_f32 v[4:5], v[36:37], v[4:5] op_sel_hi:[0,1]
	s_waitcnt vmcnt(11)
	v_pk_mul_f32 v[22:23], v[40:41], v[144:145] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[174:177], v[182:183], v[162:163], 0
	scratch_load_dwordx2 v[118:119], off, off offset:952 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[252:253], off, off offset:944 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[112:113], off, off offset:924 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[114:115], off, off offset:916 ; 8-byte Folded Reload
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[246:247], v[36:37] op_sel_hi:[1,0]
	s_waitcnt vmcnt(9)
	v_pk_mul_f32 v[30:31], v[38:39], v[136:137] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[84:85], v[160:161], 0
	v_pk_mul_f32 v[4:5], v[20:21], v[226:227] op_sel_hi:[0,1]
	v_mov_b64_e32 v[16:17], v[14:15]
	v_mov_b64_e32 v[14:15], v[12:13]
	scratch_load_dwordx2 v[104:105], off, off offset:900 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[212:213], off, off offset:908 ; 8-byte Folded Reload
	s_nop 2
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[248:249], v[20:21] op_sel_hi:[1,0]
	scratch_load_dwordx2 v[134:135], off, off offset:884 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[92:93], off, off offset:892 ; 8-byte Folded Reload
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[86:87], v[162:163], 0
	v_pk_mul_f32 v[4:5], v[38:39], v[230:231] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[250:251], off, off offset:844 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[224:225], off, off offset:876 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[88:89], off, off offset:828 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[106:107], off, off offset:836 ; 8-byte Folded Reload
	s_nop 2
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[220:221], v[38:39] op_sel_hi:[1,0]
	scratch_load_dwordx2 v[228:229], off, off offset:804 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[94:95], off, off offset:812 ; 8-byte Folded Reload
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[96:97], v[152:153], 0
	v_pk_mul_f32 v[4:5], v[58:59], v[166:167] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[132:133], off, off offset:780 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[238:239], off, off offset:772 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[240:241], off, off offset:764 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[202:203], off, off offset:756 ; 8-byte Folded Reload
	s_nop 2
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[218:219], v[58:59] op_sel_hi:[1,0]
	ds_read2_b64 v[196:199], v37 offset0:224 offset1:228
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[100:101], v[234:235], 0
	v_pk_mul_f32 v[4:5], v[56:57], v[76:77] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[4:5], v[48:49], v[0:1]
	v_pk_mul_f32 v[0:1], v[206:207], v[56:57] op_sel_hi:[1,0]
	v_mfma_f32_16x16x32_fp8_fp8 v[188:191], v[100:101], v[124:125], 0
	v_pk_fma_f32 v[6:7], v[0:1], v[50:51], v[2:3]
	v_pk_mul_f32 v[0:1], v[214:215], v[164:165] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[214:215], v[156:157] op_sel_hi:[0,1]
	scratch_store_dwordx4 off, v[4:7], off offset:64 ; 16-byte Folded Spill
	v_pk_mul_f32 v[0:1], v[0:1], v[44:45]
	v_pk_mul_f32 v[2:3], v[2:3], v[46:47]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[102:103], v[236:237], 0
	scratch_load_dwordx4 v[6:9], off, off offset:500 ; 16-byte Folded Reload
	v_pk_mul_f32 v[4:5], v[18:19], v[150:151] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[232:233], off, off offset:820 ; 8-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[184:185], v[234:235], 0
	scratch_load_dwordx2 v[90:91], off, off offset:788 ; 8-byte Folded Reload
	s_nop 2
	v_pk_fma_f32 v[0:1], v[4:5], v[44:45], v[0:1]
	v_pk_mul_f32 v[4:5], v[18:19], v[148:149] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[4:5], v[46:47], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[26:27], v[68:69], 0
	v_pk_mul_f32 v[4:5], v[40:41], v[146:147] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[4:5], v[52:53], v[0:1]
	v_pk_fma_f32 v[2:3], v[22:23], v[54:55], v[2:3]
	v_mfma_f32_16x16x32_fp8_fp8 v[52:55], v[180:181], v[160:161], 0
	v_pk_mul_f32 v[22:23], v[36:37], v[204:205] op_sel_hi:[0,1]
	s_nop 2
	v_pk_fma_f32 v[0:1], v[22:23], v[44:45], v[0:1]
	v_pk_mul_f32 v[22:23], v[36:37], v[200:201] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[22:23], v[46:47], v[2:3]
	v_pk_mul_f32 v[22:23], v[20:21], v[138:139] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[22:23], v[52:53], v[0:1]
	v_pk_mul_f32 v[22:23], v[20:21], v[140:141] op_sel_hi:[0,1]
	v_pk_fma_f32 v[22:23], v[22:23], v[54:55], v[2:3]
	v_pk_mul_f32 v[2:3], v[38:39], v[142:143] op_sel_hi:[0,1]
	v_pk_fma_f32 v[28:29], v[2:3], v[174:175], v[0:1]
	scratch_load_dwordx2 v[0:1], off, off offset:272 ; 8-byte Folded Reload
	ds_read2_b64 v[172:175], v41 offset0:216 offset1:220
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[52:55], v[172:173], v[152:153], 0
	v_pk_fma_f32 v[22:23], v[30:31], v[176:177], v[22:23]
	s_waitcnt vmcnt(27)
	v_pk_mul_f32 v[30:31], v[58:59], v[122:123] op_sel_hi:[0,1]
	v_pk_mul_f32 v[76:77], v[74:75], v[76:77] op_sel_hi:[0,1]
	s_nop 3
	v_pk_fma_f32 v[28:29], v[30:31], v[52:53], v[28:29]
	s_waitcnt vmcnt(26)
	v_pk_mul_f32 v[30:31], v[58:59], v[128:129] op_sel_hi:[0,1]
	v_pk_fma_f32 v[22:23], v[30:31], v[54:55], v[22:23]
	v_mfma_f32_16x16x32_fp8_fp8 v[52:55], v[174:175], v[154:155], 0
	s_waitcnt vmcnt(25)
	v_pk_mul_f32 v[30:31], v[56:57], v[130:131] op_sel_hi:[0,1]
	s_waitcnt vmcnt(3)
	v_mfma_f32_16x16x32_fp8_fp8 v[10:13], v[6:7], v[234:235], 0
	s_nop 3
	v_pk_fma_f32 v[30:31], v[30:31], v[52:53], v[28:29]
	v_pk_mul_f32 v[28:29], v[56:57], v[116:117] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[186:187], v[236:237], 0
	v_mfma_f32_16x16x32_fp8_fp8 v[2:5], v[8:9], v[236:237], 0
	v_mfma_f32_16x16x32_fp8_fp8 v[234:237], v[32:33], v[124:125], 0
	v_pk_fma_f32 v[32:33], v[28:29], v[54:55], v[22:23]
	scratch_store_dwordx4 off, v[30:33], off offset:80 ; 16-byte Folded Spill
	v_pk_mul_f32 v[22:23], v[214:215], v[120:121] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[48:49]
	v_mfma_f32_16x16x32_fp8_fp8 v[28:31], v[34:35], v[126:127], 0
	scratch_load_dwordx4 v[32:35], off, off offset:464 ; 16-byte Folded Reload
	v_pk_mul_f32 v[48:49], v[214:215], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[52:53], v[48:49], v[50:51]
	v_pk_mul_f32 v[48:49], v[18:19], v[252:253] op_sel_hi:[0,1]
	v_pk_fma_f32 v[22:23], v[48:49], v[44:45], v[22:23]
	s_waitcnt vmcnt(2)
	v_pk_mul_f32 v[44:45], v[18:19], v[0:1] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[0:1], off, off offset:256 ; 8-byte Folded Reload
	s_waitcnt vmcnt(1)
	v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[32:33], v[66:67], 0
	v_pk_fma_f32 v[44:45], v[44:45], v[46:47], v[52:53]
	v_pk_mul_f32 v[46:47], v[40:41], v[112:113] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[22:23], v[46:47], v[48:49], v[22:23]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[46:47], v[40:41], v[0:1] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[0:1], off, off offset:264 ; 8-byte Folded Reload
	v_pk_fma_f32 v[48:49], v[46:47], v[50:51], v[44:45]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[34:35], v[68:69], 0
	v_pk_mul_f32 v[50:51], v[36:37], v[104:105] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[192:195], v[102:103], v[126:127], 0
	v_mov_b64_e32 v[102:103], v[34:35]
	v_mov_b64_e32 v[100:101], v[32:33]
	v_mov_b64_e32 v[32:33], v[184:185]
	s_nop 2
	v_pk_fma_f32 v[22:23], v[50:51], v[44:45], v[22:23]
	v_mov_b64_e32 v[34:35], v[186:187]
	ds_read2_b64 v[184:187], v37 offset0:232 offset1:236
	v_mfma_f32_16x16x32_fp8_fp8 v[176:179], v[32:33], v[124:125], 0
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[44:45], v[36:37], v[0:1] op_sel_hi:[0,1]
	v_pk_fma_f32 v[52:53], v[44:45], v[46:47], v[48:49]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[196:197], v[160:161], 0
	v_pk_mul_f32 v[48:49], v[20:21], v[134:135] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[22:23], v[48:49], v[44:45], v[22:23]
	v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[198:199], v[162:163], 0
	v_pk_mul_f32 v[44:45], v[20:21], v[250:251] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[44:45], v[46:47], v[52:53]
	v_pk_mul_f32 v[44:45], v[38:39], v[88:89] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[52:55], v[34:35], v[126:127], 0
	v_mov_b64_e32 v[34:35], v[8:9]
	s_nop 2
	v_pk_fma_f32 v[22:23], v[44:45], v[48:49], v[22:23]
	v_pk_mul_f32 v[48:49], v[38:39], v[228:229] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[184:185], v[152:153], 0
	v_pk_fma_f32 v[0:1], v[48:49], v[50:51], v[0:1]
	v_pk_mul_f32 v[48:49], v[58:59], v[132:133] op_sel_hi:[0,1]
	v_mov_b64_e32 v[32:33], v[6:7]
	s_nop 4
	v_pk_fma_f32 v[22:23], v[48:49], v[44:45], v[22:23]
	v_pk_mul_f32 v[44:45], v[58:59], v[238:239] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[44:45], v[46:47], v[0:1]
	v_mfma_f32_16x16x32_fp8_fp8 v[44:47], v[186:187], v[154:155], 0
	v_mfma_f32_16x16x32_fp8_fp8 v[48:51], v[32:33], v[124:125], 0
	v_pk_mul_f32 v[124:125], v[56:57], v[202:203] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[6:7], v[124:125], v[44:45], v[22:23]
	v_pk_mul_f32 v[22:23], v[56:57], v[240:241] op_sel_hi:[0,1]
	v_pk_fma_f32 v[8:9], v[22:23], v[46:47], v[0:1]
	scratch_store_dwordx4 off, v[6:9], off offset:96 ; 16-byte Folded Spill
	v_pk_mul_f32 v[0:1], v[214:215], v[114:115] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[0:1], v[10:11]
	v_pk_mul_f32 v[6:7], v[214:215], v[212:213] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[214:215], off, off offset:852 ; 8-byte Folded Reload
	v_pk_mul_f32 v[8:9], v[18:19], v[224:225] op_sel_hi:[0,1]
	v_pk_mul_f32 v[6:7], v[6:7], v[12:13]
	v_pk_fma_f32 v[8:9], v[8:9], v[2:3], v[0:1]
	scratch_load_dwordx2 v[44:45], off, off offset:796 ; 8-byte Folded Reload
	v_pk_mul_f32 v[22:23], v[40:41], v[92:93] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[46:47], off, off offset:644 ; 8-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[124:127], v[34:35], v[126:127], 0
	ds_read2_b64 v[32:35], v39 offset0:240 offset1:244
	v_pk_mul_f32 v[12:13], v[38:39], v[64:65] op_sel_hi:[0,1]
	s_waitcnt vmcnt(2)
	v_pk_mul_f32 v[0:1], v[18:19], v[214:215] op_sel_hi:[0,1]
	v_pk_fma_f32 v[18:19], v[0:1], v[4:5], v[6:7]
	scratch_load_dwordx4 v[4:7], off, off offset:372 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[4:5], v[66:67], 0
	s_nop 7
	v_pk_fma_f32 v[8:9], v[22:23], v[0:1], v[8:9]
	v_pk_mul_f32 v[0:1], v[40:41], v[106:107] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[40:41], off, off offset:700 ; 8-byte Folded Reload
	v_pk_fma_f32 v[10:11], v[0:1], v[2:3], v[18:19]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[6:7], v[68:69], 0
	v_pk_mul_f32 v[18:19], v[36:37], v[232:233] op_sel_hi:[0,1]
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[66:69], v[32:33], v[160:161], 0
	s_nop 4
	v_pk_fma_f32 v[8:9], v[18:19], v[0:1], v[8:9]
	v_pk_mul_f32 v[0:1], v[36:37], v[94:95] op_sel_hi:[0,1]
	v_pk_fma_f32 v[10:11], v[0:1], v[2:3], v[10:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[34:35], v[162:163], 0
	v_pk_mul_f32 v[18:19], v[20:21], v[44:45] op_sel_hi:[0,1]
	v_pk_fma_f32 v[8:9], v[18:19], v[66:67], v[8:9]
	v_pk_mul_f32 v[18:19], v[20:21], v[90:91] op_sel_hi:[0,1]
	v_pk_fma_f32 v[22:23], v[18:19], v[68:69], v[10:11]
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], v[14:15], v[242:243], 0
	s_nop 2
	v_pk_fma_f32 v[12:13], v[12:13], v[2:3], v[22:23]
	v_pk_mul_f32 v[14:15], v[58:59], v[46:47] op_sel_hi:[0,1]
	scratch_load_dwordx2 v[36:37], off, off offset:636 ; 8-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[66:69], v[16:17], v[244:245], 0
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[10:11], v[38:39], v[40:41] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[10:11], v[0:1], v[8:9]
	ds_read2_b64 v[8:11], v39 offset0:248 offset1:252
	scratch_load_dwordx2 v[38:39], off, off offset:652 ; 8-byte Folded Reload
	s_waitcnt lgkmcnt(0)
	v_mfma_f32_16x16x32_fp8_fp8 v[18:21], v[8:9], v[152:153], 0
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[22:23], v[56:57], v[36:37] op_sel_hi:[0,1]
	s_nop 5
	v_pk_fma_f32 v[20:21], v[14:15], v[20:21], v[12:13]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[24:25], v[242:243], 0
	scratch_load_dwordx2 v[24:25], off, off offset:628 ; 8-byte Folded Reload
	s_waitcnt vmcnt(1) lgkmcnt(0)
	v_pk_mul_f32 v[2:3], v[58:59], v[38:39] op_sel_hi:[0,1]
	v_pk_fma_f32 v[18:19], v[2:3], v[18:19], v[0:1]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[10:11], v[154:155], 0
	v_pk_mul_f32 v[58:59], v[42:43], v[142:143] op_sel_hi:[0,1]
	s_nop 6
	v_pk_fma_f32 v[152:153], v[22:23], v[0:1], v[18:19]
	v_mfma_f32_16x16x32_fp8_fp8 v[16:19], v[26:27], v[244:245], 0
	v_pk_mul_f32 v[22:23], v[208:209], v[82:83] op_sel_hi:[1,0]
	v_pk_mul_f32 v[26:27], v[80:81], v[72:73] op_sel_hi:[0,1]
	v_pk_mul_f32 v[22:23], v[22:23], v[236:237]
	v_pk_mul_f32 v[72:73], v[78:79], v[226:227] op_sel_hi:[0,1]
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[0:1], v[56:57], v[24:25] op_sel_hi:[0,1]
	v_pk_fma_f32 v[154:155], v[0:1], v[2:3], v[20:21]
	v_pk_mul_f32 v[0:1], v[82:83], v[70:71] op_sel_hi:[0,1]
	v_pk_mul_f32 v[20:21], v[0:1], v[234:235]
	v_mfma_f32_16x16x32_fp8_fp8 v[0:3], v[100:101], v[242:243], 0
	v_pk_fma_f32 v[26:27], v[26:27], v[28:29], v[20:21]
	v_pk_mul_f32 v[20:21], v[210:211], v[80:81] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[28:29], v[20:21], v[30:31], v[22:23]
	v_pk_mul_f32 v[30:31], v[216:217], v[222:223] op_sel_hi:[0,1]
	v_pk_fma_f32 v[26:27], v[30:31], v[160:161], v[26:27]
	v_pk_mul_f32 v[30:31], v[254:255], v[216:217] op_sel_hi:[1,0]
	v_pk_mul_f32 v[160:161], v[62:63], v[60:61] op_sel_hi:[0,1]
	v_pk_fma_f32 v[56:57], v[30:31], v[162:163], v[28:29]
	v_pk_fma_f32 v[26:27], v[160:161], v[66:67], v[26:27]
	v_pk_mul_f32 v[66:67], v[246:247], v[62:63] op_sel_hi:[1,0]
	v_pk_mul_f32 v[60:61], v[80:81], v[150:151] op_sel_hi:[0,1]
	v_pk_fma_f32 v[56:57], v[66:67], v[68:69], v[56:57]
	v_mfma_f32_16x16x32_fp8_fp8 v[66:69], v[84:85], v[168:169], 0
	v_mfma_f32_16x16x32_fp8_fp8 v[20:23], v[4:5], v[242:243], 0
	v_mfma_f32_16x16x32_fp8_fp8 v[28:31], v[6:7], v[244:245], 0
	s_nop 5
	v_pk_fma_f32 v[26:27], v[72:73], v[66:67], v[26:27]
	v_pk_mul_f32 v[66:67], v[248:249], v[78:79] op_sel_hi:[1,0]
	v_pk_mul_f32 v[72:73], v[42:43], v[230:231] op_sel_hi:[0,1]
	v_pk_fma_f32 v[56:57], v[66:67], v[68:69], v[56:57]
	v_mfma_f32_16x16x32_fp8_fp8 v[66:69], v[86:87], v[170:171], 0
	v_pk_mul_f32 v[6:7], v[62:63], v[204:205] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[84:87], v[182:183], v[170:171], 0
	v_mfma_f32_16x16x32_fp8_fp8 v[234:237], v[102:103], v[244:245], 0
	s_nop 4
	v_pk_fma_f32 v[26:27], v[72:73], v[66:67], v[26:27]
	v_pk_mul_f32 v[66:67], v[220:221], v[42:43] op_sel_hi:[1,0]
	v_mfma_f32_16x16x32_fp8_fp8 v[70:73], v[96:97], v[108:109], 0
	v_pk_fma_f32 v[56:57], v[66:67], v[68:69], v[56:57]
	v_pk_mul_f32 v[66:67], v[158:159], v[166:167] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[160:163], v[198:199], v[170:171], 0
	s_nop 4
	v_pk_fma_f32 v[26:27], v[66:67], v[70:71], v[26:27]
	v_mfma_f32_16x16x32_fp8_fp8 v[66:69], v[98:99], v[110:111], 0
	v_pk_mul_f32 v[70:71], v[218:219], v[158:159] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[56:57], v[70:71], v[72:73], v[56:57]
	v_mfma_f32_16x16x32_fp8_fp8 v[70:73], v[180:181], v[168:169], 0
	v_mfma_f32_16x16x32_fp8_fp8 v[96:99], v[196:197], v[168:169], 0
	s_nop 2
	v_pk_fma_f32 v[66:67], v[76:77], v[66:67], v[26:27]
	v_pk_mul_f32 v[26:27], v[206:207], v[74:75] op_sel_hi:[1,0]
	s_nop 0
	v_pk_fma_f32 v[68:69], v[26:27], v[68:69], v[56:57]
	v_pk_mul_f32 v[26:27], v[82:83], v[164:165] op_sel_hi:[0,1]
	v_pk_mul_f32 v[26:27], v[26:27], v[188:189]
	v_pk_mul_f32 v[56:57], v[82:83], v[156:157] op_sel_hi:[0,1]
	v_pk_mul_f32 v[56:57], v[56:57], v[190:191]
	v_pk_fma_f32 v[26:27], v[60:61], v[192:193], v[26:27]
	v_pk_mul_f32 v[60:61], v[80:81], v[148:149] op_sel_hi:[0,1]
	v_pk_fma_f32 v[56:57], v[60:61], v[194:195], v[56:57]
	v_pk_mul_f32 v[60:61], v[216:217], v[146:147] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[60:61], v[12:13], v[26:27]
	v_pk_mul_f32 v[12:13], v[216:217], v[144:145] op_sel_hi:[0,1]
	v_pk_fma_f32 v[26:27], v[12:13], v[14:15], v[56:57]
	v_pk_fma_f32 v[16:17], v[6:7], v[16:17], v[4:5]
	v_pk_mul_f32 v[4:5], v[62:63], v[200:201] op_sel_hi:[0,1]
	v_pk_fma_f32 v[18:19], v[4:5], v[18:19], v[26:27]
	v_pk_mul_f32 v[26:27], v[78:79], v[138:139] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[4:7], v[172:173], v[108:109], 0
	v_pk_fma_f32 v[26:27], v[26:27], v[70:71], v[16:17]
	v_pk_mul_f32 v[16:17], v[78:79], v[140:141] op_sel_hi:[0,1]
	v_pk_fma_f32 v[56:57], v[16:17], v[72:73], v[18:19]
	v_mfma_f32_16x16x32_fp8_fp8 v[16:19], v[174:175], v[110:111], 0
	v_pk_fma_f32 v[26:27], v[58:59], v[84:85], v[26:27]
	v_pk_mul_f32 v[58:59], v[42:43], v[136:137] op_sel_hi:[0,1]
	v_pk_fma_f32 v[56:57], v[58:59], v[86:87], v[56:57]
	v_pk_mul_f32 v[58:59], v[158:159], v[122:123] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[58:59], v[4:5], v[26:27]
	v_pk_mul_f32 v[26:27], v[158:159], v[128:129] op_sel_hi:[0,1]
	v_pk_fma_f32 v[6:7], v[26:27], v[6:7], v[56:57]
	v_pk_mul_f32 v[26:27], v[74:75], v[130:131] op_sel_hi:[0,1]
	v_pk_fma_f32 v[4:5], v[26:27], v[16:17], v[4:5]
	v_pk_mul_f32 v[16:17], v[82:83], v[120:121] op_sel_hi:[0,1]
	v_pk_mul_f32 v[16:17], v[16:17], v[176:177]
	v_pk_mul_f32 v[26:27], v[80:81], v[252:253] op_sel_hi:[0,1]
	v_pk_fma_f32 v[16:17], v[26:27], v[52:53], v[16:17]
	v_pk_mul_f32 v[26:27], v[216:217], v[112:113] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[70:73], v[184:185], v[108:109], 0
	v_pk_fma_f32 v[0:1], v[26:27], v[0:1], v[16:17]
	v_pk_mul_f32 v[16:17], v[62:63], v[104:105] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[16:17], v[234:235], v[0:1]
	v_mfma_f32_16x16x32_fp8_fp8 v[84:87], v[186:187], v[110:111], 0
	v_pk_mul_f32 v[16:17], v[78:79], v[134:135] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[16:17], v[96:97], v[0:1]
	v_pk_mul_f32 v[16:17], v[42:43], v[88:89] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[16:17], v[160:161], v[0:1]
	v_pk_mul_f32 v[16:17], v[158:159], v[132:133] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[16:17], v[70:71], v[0:1]
	v_pk_mul_f32 v[16:17], v[74:75], v[202:203] op_sel_hi:[0,1]
	s_nop 0
	v_pk_fma_f32 v[52:53], v[16:17], v[84:85], v[0:1]
	scratch_load_dwordx2 v[0:1], off, off offset:272 ; 8-byte Folded Reload
	scratch_load_dwordx2 v[16:17], off, off offset:256 ; 8-byte Folded Reload
	v_mfma_f32_16x16x32_fp8_fp8 v[180:183], v[32:33], v[168:169], 0
	s_waitcnt vmcnt(1)
	v_pk_mul_f32 v[0:1], v[80:81], v[0:1] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[12:15], v[34:35], v[170:171], 0
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[16:17], v[216:217], v[16:17] op_sel_hi:[0,1]
	v_mfma_f32_16x16x32_fp8_fp8 v[168:171], v[8:9], v[108:109], 0
	v_pk_mul_f32 v[8:9], v[74:75], v[116:117] op_sel_hi:[0,1]
	v_pk_fma_f32 v[6:7], v[8:9], v[18:19], v[6:7]
	v_pk_mul_f32 v[18:19], v[82:83], v[118:119] op_sel_hi:[0,1]
	v_pk_mul_f32 v[18:19], v[18:19], v[178:179]
	v_mfma_f32_16x16x32_fp8_fp8 v[8:11], v[10:11], v[110:111], 0
	v_pk_fma_f32 v[0:1], v[0:1], v[54:55], v[18:19]
	s_nop 0
	v_pk_fma_f32 v[0:1], v[16:17], v[2:3], v[0:1]
	scratch_load_dwordx2 v[2:3], off, off offset:264 ; 8-byte Folded Reload
	s_barrier
	s_waitcnt vmcnt(0)
	v_pk_mul_f32 v[2:3], v[62:63], v[2:3] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[236:237], v[0:1]
	v_pk_mul_f32 v[2:3], v[78:79], v[250:251] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[98:99], v[0:1]
	v_pk_mul_f32 v[2:3], v[42:43], v[228:229] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[162:163], v[0:1]
	v_pk_mul_f32 v[2:3], v[158:159], v[238:239] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[72:73], v[0:1]
	v_pk_mul_f32 v[2:3], v[74:75], v[240:241] op_sel_hi:[0,1]
	v_pk_fma_f32 v[54:55], v[2:3], v[86:87], v[0:1]
	v_pk_mul_f32 v[0:1], v[82:83], v[114:115] op_sel_hi:[0,1]
	v_pk_mul_f32 v[0:1], v[0:1], v[48:49]
	v_pk_mul_f32 v[2:3], v[80:81], v[224:225] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[124:125], v[0:1]
	v_pk_mul_f32 v[2:3], v[216:217], v[92:93] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[20:21], v[0:1]
	v_pk_mul_f32 v[2:3], v[62:63], v[232:233] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[28:29], v[0:1]
	v_pk_mul_f32 v[2:3], v[78:79], v[44:45] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[180:181], v[0:1]
	v_pk_mul_f32 v[2:3], v[42:43], v[40:41] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[12:13], v[0:1]
	v_pk_mul_f32 v[2:3], v[158:159], v[38:39] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[168:169], v[0:1]
	v_pk_mul_f32 v[2:3], v[74:75], v[36:37] op_sel_hi:[0,1]
	v_pk_fma_f32 v[0:1], v[2:3], v[8:9], v[0:1]
	v_pk_mul_f32 v[2:3], v[82:83], v[212:213] op_sel_hi:[0,1]
	v_pk_mul_f32 v[2:3], v[2:3], v[50:51]
	v_pk_mul_f32 v[8:9], v[80:81], v[214:215] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[8:9], v[126:127], v[2:3]
	v_pk_mul_f32 v[8:9], v[216:217], v[106:107] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[8:9], v[22:23], v[2:3]
	v_pk_mul_f32 v[8:9], v[62:63], v[94:95] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[8:9], v[30:31], v[2:3]
	v_pk_mul_f32 v[8:9], v[78:79], v[90:91] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[8:9], v[182:183], v[2:3]
	v_pk_mul_f32 v[8:9], v[42:43], v[64:65] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[8:9], v[14:15], v[2:3]
	v_pk_mul_f32 v[8:9], v[158:159], v[46:47] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[8:9], v[170:171], v[2:3]
	v_pk_mul_f32 v[8:9], v[74:75], v[24:25] op_sel_hi:[0,1]
	v_pk_fma_f32 v[2:3], v[8:9], v[10:11], v[2:3]
	scratch_load_dword v8, off, off offset:280 ; 4-byte Folded Reload
	scratch_load_dwordx2 v[10:11], off, off offset:284 ; 8-byte Folded Reload
	s_waitcnt vmcnt(0)
	v_mad_u64_u32 v[8:9], s[0:1], v8, s20, v[10:11]
	scratch_load_dwordx4 v[10:13], off, off offset:292 ; 16-byte Folded Reload
	v_add_lshl_u32 v8, v8, s2, 2
	v_lshl_add_u32 v9, s20, 8, v8
	s_movk_i32 s0, 0x200
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], 0 offen
	scratch_load_dwordx4 v[10:13], off, off offset:340 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], 0 offen offset:128
	scratch_load_dwordx4 v[10:13], off, off offset:388 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], 0 offen offset:256
	scratch_load_dwordx4 v[10:13], off, off offset:480 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], 0 offen offset:384
	scratch_load_dwordx4 v[10:13], off, off offset:668 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], 0 offen
	scratch_load_dwordx4 v[10:13], off, off offset:684 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], 0 offen offset:128
	scratch_load_dwordx4 v[10:13], off, off offset:708 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], 0 offen offset:256
	scratch_load_dwordx4 v[10:13], off, off offset:860 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], 0 offen offset:384
	scratch_load_dwordx4 v[10:13], off, off offset:960 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen
	scratch_load_dwordx4 v[10:13], off, off offset:1036 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:128
	scratch_load_dwordx4 v[10:13], off, off offset:724 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:256
	scratch_load_dwordx4 v[10:13], off, off offset:596 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:384
	scratch_load_dwordx4 v[10:13], off, off offset:992 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen
	scratch_load_dwordx4 v[10:13], off, off offset:1052 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen offset:128
	scratch_load_dwordx4 v[10:13], off, off offset:740 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen offset:256
	scratch_load_dwordx4 v[10:13], off, off offset:612 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen offset:384
	scratch_load_dwordx4 v[10:13], off, off offset:356 ; 16-byte Folded Reload
	s_lshl_b32 s0, s20, 9
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen
	scratch_load_dwordx4 v[10:13], off, off offset:404 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:128
	scratch_load_dwordx4 v[10:13], off, off offset:420 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:256
	scratch_load_dwordx4 v[10:13], off, off offset:436 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:384
	scratch_load_dwordx4 v[10:13], off, off ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen
	scratch_load_dwordx4 v[10:13], off, off offset:16 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen offset:128
	scratch_load_dwordx4 v[10:13], off, off offset:32 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen offset:256
	scratch_load_dwordx4 v[10:13], off, off offset:48 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v9, s[12:15], s0 offen offset:384
	scratch_load_dwordx4 v[10:13], off, off offset:64 ; 16-byte Folded Reload
	s_addk_i32 s0, 0x200
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen
	scratch_load_dwordx4 v[10:13], off, off offset:80 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:128
	scratch_load_dwordx4 v[10:13], off, off offset:96 ; 16-byte Folded Reload
	s_waitcnt vmcnt(0)
	buffer_store_dwordx4 v[10:13], v8, s[12:15], s0 offen offset:256
	buffer_store_dwordx4 v[152:155], v8, s[12:15], s0 offen offset:384
	buffer_store_dwordx4 v[66:69], v9, s[12:15], s0 offen
	buffer_store_dwordx4 v[4:7], v9, s[12:15], s0 offen offset:128
	buffer_store_dwordx4 v[52:55], v9, s[12:15], s0 offen offset:256
	buffer_store_dwordx4 v[0:3], v9, s[12:15], s0 offen offset:384
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs
		.amdhsa_group_segment_fixed_size 135168
		.amdhsa_private_segment_fixed_size 1824
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
		.amdhsa_enable_private_segment 1
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 1
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 256
		.amdhsa_next_free_sgpr 96
		.amdhsa_accum_offset 256
		.amdhsa_reserve_vcc 0
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
	.section	.text._Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs,"axG",@progbits,_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs,comdat
.Lfunc_end0:
	.size	_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs, .Lfunc_end0-_Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs
                                        ; -- End function
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.num_vgpr, 256
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.num_agpr, 0
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.numbered_sgpr, 22
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.private_seg_size, 1824
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.uses_vcc, 0
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.uses_flat_scratch, 0
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.has_dyn_sized_stack, 0
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.has_recursion, 0
	.set _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 24424
; TotalNumSgprs: 28
; NumVgprs: 256
; NumAgprs: 0
; TotalNumVgprs: 256
; ScratchSize: 1824
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 135168 bytes/workgroup (compile time only)
; SGPRBlocks: 12
; VGPRBlocks: 31
; NumSGPRsForWavesPerEU: 102
; NumVGPRsForWavesPerEU: 256
; AccumOffset: 256
; Occupancy: 2
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 1
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 1
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 63
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 0
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 0
	.text
	.protected	_ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE ; @_ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE
	.type	_ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE,@object
	.section	.rodata._ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE,"aG",@progbits,_ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE,comdat
	.weak	_ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE
	.p2align	2, 0x0
_ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE:
	.long	16                              ; 0x10
	.size	_ZN22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EE3W_NE, 4

	.type	__hip_cuid_3d71d078df58e2cf,@object ; @__hip_cuid_3d71d078df58e2cf
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_3d71d078df58e2cf
__hip_cuid_3d71d078df58e2cf:
	.byte	0                               ; 0x0
	.size	__hip_cuid_3d71d078df58e2cf, 1

	.ident	"AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.0 25314 f4087f6b428f0e6f575ebac8a8a724dab123d06e)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __hip_cuid_3d71d078df58e2cf
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .offset:         0
        .size:           96
        .value_kind:     by_value
    .group_segment_fixed_size: 135168
    .kernarg_segment_align: 8
    .kernarg_segment_size: 96
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 512
    .name:           _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs
    .private_segment_fixed_size: 1824
    .sgpr_count:     28
    .sgpr_spill_count: 0
    .symbol:         _Z22gemm_a8w8_mxfp8_kernelI22gemm_a8w8_mxfp8_traitsILi256ELi256ELi128ELi1ELi1ELi32EEEv15opus_gemm_kargs.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     256
    .vgpr_spill_count: 558
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
