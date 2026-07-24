package placement

import (
	"testing"

	"github.com/rrifftt/ggrun/pkg/detect"
)

func TestFindDraftGPUReturnsErrorWhenNoFit(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 1024, VRAMUsedMB: 0},
		},
	}
	gpu, err := findDraftGPU(caps, &ModelProfile{IsMoE: false, TotalSizeMB: 0}, 2048)
	if err == nil {
		t.Fatalf("expected error when no GPU fits, got gpu=%d", gpu)
	}
	if gpu != -1 {
		t.Fatalf("expected gpu=-1 on error, got gpu=%d", gpu)
	}
}

func TestComputeDraftKVTypeUsesDraftGPUOnly(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, VRAMTotalMB: 2048, VRAMUsedMB: 0},
			{Index: 1, VRAMTotalMB: 12288, VRAMUsedMB: 0},
		},
	}
	// GPU 0 has ~2GB free (no >4GB), GPU 1 has ~12GB free.
	// When draftGPU=0, should return q4_0 (not q8_0 from GPU 1).
	result := computeDraftKVType(caps, 0)
	if result != "q4_0" {
		t.Fatalf("expected q4_0 for draft GPU 0, got %s", result)
	}
}
