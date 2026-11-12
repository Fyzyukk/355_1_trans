# MXFP4 x MXFP4 NT, 4-wave, K tile 256 (base: mxfp8 4_wave_no_host_scale).
# opus of the aiter blockscale branch: its tiled mma supports fp4 (packed) operands.
OPUS_INCLUDE_DIR ?= /root/workspace/aiter-blockscale-mxfp8-nn-nt-tt/csrc/include
ROCM_PATH ?= /opt/rocm
# amdgpu_pin_agpr / amdgpu_pin_vgpr need the pin toolchain (clang 24 + pin patches)
CLANG23_ROOT ?= /root/toolchains/yuyzhang512-amdgpu-pin-op-dst-build
CXX       = $(CLANG23_ROOT)/bin/clang++
ARCH      ?= gfx950
FLAGS      = -I. -I$(OPUS_INCLUDE_DIR) -std=c++17 -O3 -Wall -ffast-math --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH) $(EXTRA)
OMP_LIB    = $(ROCM_PATH)/lib/llvm/lib
LDFLAGS    = -fopenmp -L$(OMP_LIB) -lomp -Wl,-rpath,$(OMP_LIB) -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib
DEPS       = gemm_a4w4_mxfp4_4wave_traits.hpp gemm_a4w4_mxfp4_4wave_kernel_template.hpp
BUILD      = build
OUT        = $(BUILD)/gemm_a4w4_mxfp4_4wave.exe
ISA        = $(BUILD)/gemm_a4w4_mxfp4_4wave_kernel-hip-amdgcn-amd-amdhsa-gfx950.s

all: $(OUT)
$(BUILD):
	mkdir -p $@
$(BUILD)/kernel.o: gemm_a4w4_mxfp4_4wave_kernel.cc $(DEPS) | $(BUILD)
	$(CXX) -x hip $< $(FLAGS) -D__HIPCC_RTC__ -save-temps=obj -c -o $@
$(BUILD)/host.o: gemm_a4w4_mxfp4_4wave_host.cc gemm_a4w4_mxfp4_4wave_traits.hpp | $(BUILD)
	$(CXX) -x hip $< $(FLAGS) -fopenmp -I$(ROCM_PATH)/lib/llvm/include -c -o $@
$(OUT): $(BUILD)/kernel.o $(BUILD)/host.o
	$(CXX) $^ --offload-arch=$(ARCH) --rocm-path=$(ROCM_PATH) $(LDFLAGS) -o $@
regs: $(OUT)
	@grep -E "vgpr_count|agpr_count|spill_count|group_segment_fixed_size" $(ISA); \
	 echo "mfma: $$(grep -c v_mfma_scale $(ISA))"
.PHONY: all regs
