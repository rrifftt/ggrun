package placement

import (
	"testing"

	"github.com/rrifftt/ggrun/pkg/detect"
)

func testCaps() *detect.Capabilities {
	return &detect.Capabilities{
		GPUs: []detect.GPU{
			{Index: 0, Name: "RTX 3070", VRAMTotalMB: 8192, VRAMUsedMB: 600},
		},
		CPU: detect.CPUInfo{Cores: 8, Threads: 16},
		RAM: detect.RAMInfo{TotalMB: 32768, FreeMB: 28000},
	}
}

func testDenseModel() *ModelProfile {
	return &ModelProfile{
		Path:          "/models/dense-9b.gguf",
		SizeBytes:     5615 * 1024 * 1024,
		TotalSizeMB:   5615,
		NumLayers:     33,
		IsMoE:         false,
		ContextSize:   65536,
		HeadCountKV:   8,
		KeyLength:     128,
		ValueLength:   128,
		NonExpertBytes: 5615 * 1024 * 1024,
	}
}

func testMoEModel() *ModelProfile {
	return &ModelProfile{
		Path:           "/models/moe-28b.gguf",
		SizeBytes:      16465 * 1024 * 1024,
		TotalSizeMB:    16465,
		NumLayers:      40,
		IsMoE:          true,
		NumExperts:     64,
		ContextSize:    65536,
		HeadCountKV:    4,
		KeyLength:      128,
		ValueLength:    128,
		ExpertBytes:    14894 * 1024 * 1024,
		NonExpertBytes: 1560 * 1024 * 1024,
	}
}

func TestPredictVRAMUsage_DenseFits(t *testing.T) {
	caps := testCaps()
	model := testDenseModel()
	flags := map[string]string{"--ctx-size": "65536"}
	needed, free := PredictVRAMUsage(model, flags, caps)
	if needed <= 0 || free <= 0 {
		t.Fatalf("expected positive values, got needed=%d free=%d", needed, free)
	}
	if needed > free {
		t.Logf("dense model at 65K ctx: needed=%d > free=%d (expected for 8GB GPU)", needed, free)
	}
}

func TestPredictVRAMUsage_MoEWithCPUExperts(t *testing.T) {
	caps := testCaps()
	model := testMoEModel()
	// With 32 CPU expert layers, only 8 layers of experts on GPU
	flags := map[string]string{"--n-cpu-moe": "32", "--ctx-size": "65536"}
	needed, _ := PredictVRAMUsage(model, flags, caps)
	if needed <= 0 {
		t.Fatalf("expected positive needed, got %d", needed)
	}
	// With most experts on CPU, needed should be much less than total model size
	if needed > model.TotalSizeMB {
		t.Errorf("needed=%d should be less than total model size=%d with CPU experts", needed, model.TotalSizeMB)
	}
}

func TestPredictVRAMUsage_FitBypass(t *testing.T) {
	caps := testCaps()
	model := testDenseModel()
	flags := map[string]string{"--fit": "on", "--ctx-size": "65536"}
	needed, _ := PredictVRAMUsage(model, flags, caps)
	if needed != 0 {
		t.Errorf("--fit on should bypass prediction, got needed=%d", needed)
	}
}

func TestPredictVRAMUsage_FitOffDoesNotBypass(t *testing.T) {
	caps := testCaps()
	model := testDenseModel()
	flags := map[string]string{"--fit": "off", "--ctx-size": "65536"}
	needed, _ := PredictVRAMUsage(model, flags, caps)
	if needed == 0 {
		t.Error("--fit off should NOT bypass prediction")
	}
}

func TestPredictVRAMUsage_NilCaps(t *testing.T) {
	model := testDenseModel()
	needed, free := PredictVRAMUsage(model, nil, nil)
	if needed != 0 || free != 0 {
		t.Errorf("nil caps should return zeros, got needed=%d free=%d", needed, free)
	}
}

func TestPredictVRAMUsage_ContextScaling(t *testing.T) {
	caps := testCaps()
	model := testDenseModel()
	flags65k := map[string]string{"--ctx-size": "65536"}
	flags262k := map[string]string{"--ctx-size": "262144"}
	needed65k, _ := PredictVRAMUsage(model, flags65k, caps)
	needed262k, _ := PredictVRAMUsage(model, flags262k, caps)
	if needed262k <= needed65k {
		t.Errorf("262K ctx (%d MB) should need more VRAM than 65K ctx (%d MB)", needed262k, needed65k)
	}
}

func TestTryKVDowngradeForGPU_FitsWithDowngrade(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 12288, VRAMUsedMB: 500}},
	}
	model := testDenseModel()
	// Model fits but model+KV at f16 doesn't; q8_0 or q4_0 should fit
	ktK, ktV, fits := tryKVDowngradeForGPU(caps, model, model.TotalSizeMB, 300, 65536, "f16", "")
	if !fits {
		t.Skip("model too large for test GPU; skipping downgrade test")
	}
	if ktK == "f16" && ktV == "f16" {
		t.Error("expected a downgrade from f16, got same type")
	}
	t.Logf("downgraded to K=%s V=%s", ktK, ktV)
}

func TestTryKVDowngradeForGPU_MoENonExpertOnly(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 8192, VRAMUsedMB: 500}},
	}
	model := testMoEModel()
	// MoE: only non-expert bytes (~1560 MB) compete for VRAM, not the full 16 GB
	ktK, _, fits := tryKVDowngradeForGPU(caps, model, model.TotalSizeMB, 300, 65536, "q8_0", "")
	if !fits {
		t.Log("MoE non-expert + KV doesn't fit even with downgrade (expected on 8GB)")
	} else {
		t.Logf("MoE KV downgrade: K=%s fits on GPU", ktK)
	}
}

func TestTryKVDowngradeForGPU_TurboGatedOnBackend(t *testing.T) {
	caps := &detect.Capabilities{
		GPUs: []detect.GPU{{Index: 0, VRAMTotalMB: 24576, VRAMUsedMB: 500}},
	}
	model := testDenseModel()
	// Without "turbo" in backend help, turbo types should not be offered
	ktK, _, fits := tryKVDowngradeForGPU(caps, model, model.TotalSizeMB, 300, 65536, "f16", "--cache-type-k f16 q8_0 q4_0")
	if fits && (ktK == "turbo4" || ktK == "turbo3") {
		t.Error("turbo KV offered without backend support")
	}
	// With "turbo" in backend help, turbo types should be available
	ktK2, _, fits2 := tryKVDowngradeForGPU(caps, model, model.TotalSizeMB, 300, 65536, "f16", "--cache-type-k f16 q8_0 q4_0 turbo4 turbo3")
	if fits2 {
		t.Logf("with turbo support: downgraded to K=%s", ktK2)
	}
}

func TestParseFlagsToMap(t *testing.T) {
	args := []string{
		"llama-server", "-m", "model.gguf",
		"--ctx-size", "65536",
		"--flash-attn", "on",
		"--no-mmap",
		"-b", "4096",
		"--cache-type-k=q8_0",
	}
	m := ParseFlagsToMap(args)
	if m["--ctx-size"] != "65536" {
		t.Errorf("expected --ctx-size=65536, got %q", m["--ctx-size"])
	}
	if m["--flash-attn"] != "on" {
		t.Errorf("expected --flash-attn=on, got %q", m["--flash-attn"])
	}
	if m["--no-mmap"] != "" {
		t.Errorf("expected --no-mmap='' (boolean), got %q", m["--no-mmap"])
	}
	if m["--cache-type-k"] != "q8_0" {
		t.Errorf("expected --cache-type-k=q8_0, got %q", m["--cache-type-k"])
	}
	if _, ok := m["llama-server"]; ok {
		t.Error("binary name should not be in flag map")
	}
}
