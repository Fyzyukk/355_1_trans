# Quick Run Checklist

- [ ] Freeze NT baseline
- [ ] Confirm 256x256x128 / 4-wave / 2x2 / 3-buffer parameters
- [ ] Verify compute-cycle model
- [ ] Verify global requested/returned bytes
- [ ] Validate A/B packed index mapping
- [ ] Validate A/B scale ownership
- [ ] Isolate NN B-path delta
- [ ] Isolate TT A-path delta
- [ ] Combine and inspect TN non-additive penalty
- [ ] Measure prefetch lead and wait gates
- [ ] Check LDS bank/segment mapping
- [ ] Track VGPR/AGPR/SGPR/LDS/occupancy
- [ ] Compare 2x2, 1x4, and 4x1 only after correctness
- [ ] Explain final delta versus NT with ATT/PMC evidence
