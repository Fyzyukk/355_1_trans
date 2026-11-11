# 汇总（trace_8192_step1，GPU6）

| 类别 | . cycle/wave (占比) |
|---|---|
| **合计** | **100893** |
| MFMA | 51685 (51.2%) |
| buffer_load | 13661 (13.5%) |
| VALU | 12344 (12.2%) |
| ds_read | 6217 (6.2%) |
| SALU | 5948 (5.9%) |
| waitcnt | 3415 (3.4%) |
| barrier | 2682 (2.7%) |
| nop | 1651 (1.6%) |
| buffer_store | 1248 (1.2%) |
| ds_write | 1176 (1.2%) |
| accvgpr | 866 (0.9%) |

. 最耗时指令（hits>=200，按 latency）：
    9992  0.9%     63 cyc/次  s_barrier
   14324  0.9%     58 cyc/次  buffer_load_dwordx4 v26, s[28:31], s67 offen lds
   13652  0.4%     28 cyc/次  buffer_load_dwordx4 v10, s[20:23], s69 offen lds
   14260  0.4%     26 cyc/次  buffer_load_dwordx4 v23, s[24:27], s70 offen lds
   10616  0.4%     26 cyc/次  s_barrier
   14108  0.4%     26 cyc/次  buffer_load_dwordx4 v21, s[24:27], s70 offen lds
   14184  0.4%     25 cyc/次  buffer_load_dwordx4 v22, s[24:27], s70 offen lds
   10100  0.4%     25 cyc/次  v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[88:91], a[48:51], a[20
