| 类别 | 占比 | 执行次数 | 平均 cycle/条 |
|---|---|---|---|
| MFMA | 43.5% | 65536 | 13.8 |
| buffer_load | 18.1% | 8704 | 43.3 |
| VALU | 12.5% | 44000 | 5.9 |
| waitcnt | 9.6% | 6128 | 32.6 |
| SALU | 6.3% | 26400 | 5.0 |
| ds_read | 5.1% | 17936 | 5.9 |
| nop | 2.0% | 5424 | 7.6 |
| buffer_store | 1.6% | 512 | 66.3 |
| barrier | 1.5% | 592 | 52.5 |

主循环里最耗时的指令（每 tile 执行一次，hits >= 400）：

| 地址 | 占比 | cycle/次 | 指令 |
|---|---|---|---|
| 11772 | 6.2% | 268 | `s_waitcnt vmcnt(0) lgkmcnt(0)` |
| 10660 | 4.0% | 176 | `buffer_load_dwordx4 v137, s[28:31], s5 offen lds` |
| 10580 | 1.6% | 67 | `buffer_load_dwordx4 v149, s[28:31], s5 offen lds` |
| 11776 | 1.1% | 50 | `s_barrier` |
| 12732 | 0.7% | 33 | `buffer_load_dwordx4 v146, s[24:27], s23 offen lds` |
| 11888 | 0.6% | 25 | `v_mfma_scale_f32_16x16x128_f8f6f4 a[152:155], a[56:59], a[16` |
| 11904 | 0.6% | 24 | `v_mfma_scale_f32_16x16x128_f8f6f4 a[148:151], a[36:39], a[8:` |
| 10500 | 0.5% | 22 | `buffer_load_dwordx4 v147, s[28:31], s5 offen lds` |

追踪到的 wave 数：16，每个 wave 的 [开始, 结束] cycle 见 att/ui_output_*/filenames.json
