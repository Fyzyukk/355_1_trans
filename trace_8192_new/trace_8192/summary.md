每个 wave 总 cycle：110364（16 个 wave，4 个 block × 4 个 wave）

| 类别 | 占比 | 执行次数 | 平均 cycle/条 |
|---|---|---|---|
| MFMA | 50.8% | 65536 | 13.7 |
| buffer_load | 15.5% | 8736 | 31.3 |
| VALU | 11.6% | 33392 | 6.1 |
| waitcnt | 5.8% | 5424 | 18.9 |
| SALU | 5.3% | 16800 | 5.5 |
| ds_read | 4.4% | 17936 | 4.4 |
| barrier | 3.3% | 1072 | 55.1 |
| nop | 1.3% | 3520 | 6.6 |
| buffer_store | 1.2% | 512 | 41.1 |
| accvgpr | 0.8% | 3072 | 4.5 |

主循环里最耗时的指令（hits >= 200）：

| 地址 | 占比 | cycle/次 | 指令 |
|---|---|---|---|
| 10004 | 1.6% | 114 | `s_barrier` |
| 14320 | 1.0% | 77 | `buffer_load_dwordx4 v26, s[28:31], s67 offen lds` |
| 12976 | 0.6% | 46 | `s_waitcnt lgkmcnt(0)` |
| 13744 | 0.5% | 40 | `buffer_load_dwordx4 v18, s[20:23], s69 offen lds` |
| 12980 | 0.4% | 31 | `s_barrier` |
| 13648 | 0.4% | 31 | `buffer_load_dwordx4 v10, s[20:23], s69 offen lds` |
| 13004 | 0.4% | 29 | `v_mfma_scale_f32_16x16x128_f8f6f4 a[192:195], a[80:83], a[48` |
| 11176 | 0.4% | 28 | `buffer_load_dwordx4 v21, s[24:27], s71 offen lds` |
