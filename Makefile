
# Makefile

NVCC ?= nvcc
ARCH ?= sm_80

# Provide SRC on the command line:
#   make build SRC=path/to/file.cu
#   make run   SRC=path/to/file.cu
DEFAULT_SRC := cuda/matmul/01_naive.cu
SRC ?= $(DEFAULT_SRC)

OUTDIR ?= bin
RUN_ARGS ?=

# If BIN isn't provided, derive it from SRC basename (e.g., 01_naive.cu -> 01_naive)
ifeq ($(BIN),)
  ifneq ($(strip $(SRC)),)
    BIN := $(notdir $(basename $(SRC)))
  else
    BIN := a.out
  endif
endif

OUT := $(OUTDIR)/$(BIN)

NVCCFLAGS ?= -O3 -std=c++17 -arch=$(ARCH) -lineinfo

.PHONY: build run

# Allow: make run path/to/file.cu
ifneq ($(filter run build,$(MAKECMDGOALS)),)
  EXTRA_GOALS := $(filter-out run build,$(MAKECMDGOALS))
  ifneq ($(strip $(EXTRA_GOALS)),)
    SRC := $(firstword $(EXTRA_GOALS))
    .PHONY: $(EXTRA_GOALS)
    $(EXTRA_GOALS):
	@:
  endif
endif

build:
	@mkdir -p $(OUTDIR)
	$(NVCC) $(NVCCFLAGS) $(SRC) -o $(OUT)
	@echo "Built: $(OUT)"

run: build
	./$(OUT) $(RUN_ARGS)
