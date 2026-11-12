| 类别 | t_s1_idle cycle/wave (占比) | t_m3 cycle/wave (占比) | t_m5 cycle/wave (占比) |
|---|---|---|---|
| **合计** | **98912** | **99334** | **99682** |
| MFMA | 51603 (52.2%) | 54260 (54.6%) | 55137 (55.3%) |
| ds_read | 6230 (6.3%) | 9390 (9.5%) | 14267 (14.3%) |
| buffer_load | 12317 (12.5%) | 10188 (10.3%) | 10102 (10.1%) |
| VALU | 12400 (12.5%) | 8686 (8.7%) | 5952 (6.0%) |
| SALU | 5687 (5.7%) | 5343 (5.4%) | 3530 (3.5%) |
| buffer_store | 1229 (1.2%) | 3456 (3.5%) | 3016 (3.0%) |
| barrier | 2390 (2.4%) | 2227 (2.2%) | 2472 (2.5%) |
| ds_write | 1174 (1.2%) | 1978 (2.0%) | 1933 (1.9%) |
| waitcnt | 3366 (3.4%) | 1948 (2.0%) | 1728 (1.7%) |
| accvgpr | 864 (0.9%) | 1058 (1.1%) | 1069 (1.1%) |
| nop | 1651 (1.7%) | 801 (0.8%) | 478 (0.5%) |

t_s1_idle 最耗时指令（hits>=200，按 latency）：
    9992  0.8%     54 cyc/次  s_barrier
   13652  0.5%     35 cyc/次  buffer_load_dwordx4 v10, s[20:23], s69 offen lds
   14324  0.5%     35 cyc/次  buffer_load_dwordx4 v26, s[28:31], s67 offen lds
   13748  0.5%     35 cyc/次  buffer_load_dwordx4 v18, s[20:23], s69 offen lds
   12984  0.4%     29 cyc/次  s_barrier
   14108  0.4%     25 cyc/次  buffer_load_dwordx4 v21, s[24:27], s70 offen lds
   10100  0.4%     25 cyc/次  v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[88:91], a[48:51], a[20
   13024  0.4%     24 cyc/次  buffer_load_dwordx4 v10, s[20:23], s68 offen lds

t_m3 最耗时指令（hits>=200，按 latency）：
   17448  0.6%     38 cyc/次  s_barrier
   20236  0.6%     38 cyc/次  s_barrier
   17540  0.4%     27 cyc/次  v_mfma_scale_f32_16x16x128_f8f6f4 a[200:203], a[72:75], a[32:35], a[20
   18072  0.4%     26 cyc/次  s_barrier
   20284  0.4%     25 cyc/次  buffer_load_dwordx4 v84, s[36:39], s27 offen lds
   17524  0.4%     24 cyc/次  v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[84:87], a[48:51], a[19
   19552  0.3%     21 cyc/次  ds_write_b128 v35, v[36:39]
   22348  0.3%     21 cyc/次  ds_write_b128 v35, v[36:39]

t_m5 最耗时指令（hits>=200，按 latency）：
   21224  2.0%    131 cyc/次  ds_read_b128 a[24:27], v24 offset:8512
   18516  1.8%    120 cyc/次  ds_read_b128 a[24:27], v34 offset:8512
   16600  0.6%     42 cyc/次  s_barrier
   19304  0.6%     40 cyc/次  s_barrier
   20416  0.5%     36 cyc/次  buffer_load_dwordx4 v89, s[40:43], s23 offen offset:1056 lds
   16672  0.5%     31 cyc/次  v_mfma_scale_f32_16x16x128_f8f6f4 a[196:199], a[84:87], a[48:51], a[19
   20336  0.4%     28 cyc/次  buffer_load_dwordx4 v88, s[40:43], s23 offen lds
   17200  0.4%     28 cyc/次  s_barrier
