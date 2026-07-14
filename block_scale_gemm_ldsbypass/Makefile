OPUS_INCLUDE_DIR ?= /path/to/aiter/csrc/include
HIPCC           ?= /opt/rocm/bin/hipcc
ARCH            ?= gfx950

COMMON_FLAGS = -I. -I$(OPUS_INCLUDE_DIR) -std=c++17 -fopenmp -O3 -Wall --offload-arch=$(ARCH) -ffast-math

BUILD = build
OUT   = $(BUILD)/gemm_a8w8_blockscale_ldsbypass.exe

KERNEL_SRCS = gemm_a8w8_blockscale_ldsbypass_kernel.cc
HOST_SRC    = gemm_a8w8_blockscale_ldsbypass_host.cc
KERNEL_OBJS = $(patsubst %.cc,$(BUILD)/%.o,$(KERNEL_SRCS))
HOST_OBJ    = $(BUILD)/gemm_a8w8_blockscale_ldsbypass_host.o

# host-pack variant: same main kernel, B'' built on the CPU (no GPU repack)
HOSTPACK_SRC = gemm_a8w8_blockscale_ldsbypass_hostpack_host.cc
HOSTPACK_OBJ = $(BUILD)/gemm_a8w8_blockscale_ldsbypass_hostpack_host.o
OUT_HOSTPACK = $(BUILD)/gemm_a8w8_blockscale_ldsbypass_hostpack.exe

.PHONY: all clean benchmark hostpack

all: $(OUT) $(OUT_HOSTPACK)

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/%.o: %.cc gemm_a8w8_blockscale_ldsbypass_kernel_template.hpp gemm_a8w8_blockscale_ldsbypass_common.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -D__HIPCC_RTC__ -Rpass-analysis=kernel-resource-usage -save-temps=obj -c -o $@

$(HOST_OBJ): $(HOST_SRC) gemm_a8w8_blockscale_ldsbypass_common.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT): $(KERNEL_OBJS) $(HOST_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

$(HOSTPACK_OBJ): $(HOSTPACK_SRC) gemm_a8w8_blockscale_ldsbypass_common.h | $(BUILD)
	$(HIPCC) $< $(COMMON_FLAGS) -c -o $@

$(OUT_HOSTPACK): $(KERNEL_OBJS) $(HOSTPACK_OBJ)
	$(HIPCC) $^ --offload-arch=$(ARCH) -fopenmp -o $@

hostpack: $(OUT_HOSTPACK)

benchmark: $(OUT)
	$(OUT)

clean:
	rm -rf $(BUILD)
