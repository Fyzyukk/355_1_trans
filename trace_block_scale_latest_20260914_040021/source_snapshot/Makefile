TOOLCHAIN        ?= /root/toolchains/rocm-llvm23-46fcb339-build
HIPCC            ?= $(TOOLCHAIN)/bin/clang++
ROCM_PATH        ?= /opt/rocm
OPUS_INCLUDE_DIR ?= /root/workspace/aiter/csrc/include
BUILD            ?= build
GPU              ?= 2

FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++17 -O3 -ffast-math -fPIC \
        --offload-arch=gfx950 --rocm-path=$(ROCM_PATH)
OMP_FLAGS = -fopenmp -I$(ROCM_PATH)/lib/llvm/include
LINK_FLAGS = --offload-arch=gfx950 --rocm-path=$(ROCM_PATH) \
             -L$(ROCM_PATH)/lib -lamdhip64 -Wl,-rpath,$(ROCM_PATH)/lib
OMP_LINK = -fopenmp -L$(ROCM_PATH)/lib/llvm/lib -lomp \
           -Wl,-rpath,$(ROCM_PATH)/lib/llvm/lib
EXE = $(BUILD)/gemm_a8w8_blockscale_bpreshuffle.exe
LIB = $(BUILD)/libblockscale_bpreshuffle.so
HEADERS = traits.hpp gemm_a8w8_mxfp8_scale_common.h kernel_dispatch.hpp

.PHONY: all inspect benchmark verify clean
all: $(EXE) $(LIB)

$(BUILD):
	mkdir -p $@

$(BUILD)/kernel.o: gemm_a8w8_mxfp8_scale_kernel.cc tmpl_generic.hpp $(HEADERS) Makefile | $(BUILD)
	$(HIPCC) -x hip $< $(FLAGS) -D__HIPCC_RTC__ -c -o $@

$(BUILD)/host.o: gemm_a8w8_mxfp8_scale_host.cc $(HEADERS) Makefile | $(BUILD)
	$(HIPCC) -x hip $< $(FLAGS) -fno-finite-math-only $(OMP_FLAGS) -c -o $@

$(BUILD)/launch.o: gemm_a8w8_blockscale_bpreshuffle_launch.cc $(HEADERS) Makefile | $(BUILD)
	$(HIPCC) -x hip $< $(FLAGS) -c -o $@

$(EXE): $(BUILD)/kernel.o $(BUILD)/host.o
	$(HIPCC) $^ $(LINK_FLAGS) $(OMP_LINK) -o $@

$(LIB): $(BUILD)/kernel.o $(BUILD)/launch.o
	$(HIPCC) $^ -shared $(LINK_FLAGS) -o $@

inspect: $(EXE)
	$(TOOLCHAIN)/bin/llvm-objcopy --dump-section=.hip_fatbin=$(BUILD)/device.fb $(EXE) $(BUILD)/inspect.exe
	$(TOOLCHAIN)/bin/clang-offload-bundler -type=o -targets=hipv4-amdgcn-amd-amdhsa--gfx950 -input=$(BUILD)/device.fb -output=$(BUILD)/device.co -unbundle
	$(TOOLCHAIN)/bin/llvm-objdump -d --mcpu=gfx950 $(BUILD)/device.co > $(BUILD)/device.isa
	$(TOOLCHAIN)/bin/llvm-readelf --notes $(BUILD)/device.co > $(BUILD)/device.notes

benchmark: $(EXE)
	HIP_VISIBLE_DEVICES=$(GPU) OMP_TOOL=disabled OMP_NUM_THREADS=16 $(EXE) -m 8192 -n 8192 -k 8192 -b 1 -w 200 -i 100 -v 0

verify: $(EXE)
	HIP_VISIBLE_DEVICES=$(GPU) OMP_TOOL=disabled OMP_NUM_THREADS=16 $(EXE) -m 256 -n 256 -k 256 -b 1 -w 0 -i 1 -v 1 --dtype fp32
	HIP_VISIBLE_DEVICES=$(GPU) OMP_TOOL=disabled OMP_NUM_THREADS=16 $(EXE) -m 512 -n 256 -k 1152 -b 2 -w 0 -i 1 -v 1 --dtype bf16

clean:
	rm -rf $(BUILD)
