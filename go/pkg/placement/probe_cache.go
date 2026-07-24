package placement

import (

	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/rrifftt/ggrun/pkg/detect"

)

// GPUAssignment describes layers assigned to a GPU.


func LoadPlacementCache(cachePath string, caps *detect.Capabilities, kvTotalMB int) (*CacheEntry, error) {
	data, err := os.ReadFile(cachePath)
	if err != nil {
		return nil, err
	}
	content := string(data)

	// Parse the legacy key=value cache format
	entry := &CacheEntry{
		BatchSize:  1024,
		UBatchSize: 512,
		Parallel:   2,
	}
	hasMMap := false
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.Trim(strings.TrimSpace(parts[1]), `"`)
		switch key {
		case "CACHED_GPU_ASSIGNMENTS":
			entry.GPUAssignments = parseGPUAssignments(val)
		case "CACHED_OT_STRING":
			entry.OTString = val
		case "CACHED_TENSOR_SPLIT":
			entry.TensorSplit = parseTensorSplit(val)
		case "CACHED_SPLIT_MODE":
			entry.SplitMode = val
		case "CACHED_NCPUMOE":
			entry.NCPUMoE, _ = strconv.Atoi(val)
		case "CACHED_BATCH":
			entry.BatchSize, _ = strconv.Atoi(val)
		case "CACHED_UBATCH":
			entry.UBatchSize, _ = strconv.Atoi(val)
		case "CACHED_PARALLEL":
			entry.Parallel, _ = strconv.Atoi(val)
		case "CACHED_KVUNIFIED":
			entry.KVUnified = val == "1"
		case "CACHED_NO_PINNED":
			entry.NoPinned = val == "1"
		case "CACHED_MMAP":
			hasMMap = true
			entry.MMap = val == "1"
		}
	}

	// Validate: each GPU must have enough VRAM for assigned layers + KV share
	for _, assign := range entry.GPUAssignments {
		found := false
		for _, g := range caps.GPUs {
			if g.Index == assign.CUDAIndex {
				found = true
				// We can't validate exact layer MB without model info here,
				// but we can check that the GPU exists
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("cached assignment references unknown GPU %d", assign.CUDAIndex)
		}
	}
	if len(entry.GPUAssignments) == 0 && len(entry.TensorSplit) == 0 && entry.OTString == "" {
		return nil, fmt.Errorf("cache has no MoE placement data")
	}
	if !hasMMap {
		return nil, fmt.Errorf("cache missing CACHED_MMAP")
	}
	if len(entry.GPUAssignments) > 0 && len(entry.TensorSplit) == 0 {
		return nil, fmt.Errorf("cached MoE GPU assignments missing CACHED_TENSOR_SPLIT")
	}

	return entry, nil
}

func SavePlacementCache(cachePath string, entry *CacheEntry) error {
	_ = os.MkdirAll(filepath.Dir(cachePath), 0755)
	var parts []string
	parts = append(parts, fmt.Sprintf("# ggrun placement cache (%s)", time.Now().UTC().Format("2006-01-02T15:04:05Z")))
	if len(entry.GPUAssignments) > 0 {
		var assigns []string
		for _, a := range entry.GPUAssignments {
			assigns = append(assigns, fmt.Sprintf("%d:%d:%d", a.CUDAIndex, a.Start, a.Count))
		}
		parts = append(parts, fmt.Sprintf("CACHED_GPU_ASSIGNMENTS=\"%s\"", strings.Join(assigns, " ")))
	}
	if entry.OTString != "" {
		parts = append(parts, fmt.Sprintf("CACHED_OT_STRING=\"%s\"", entry.OTString))
	}
	if len(entry.TensorSplit) > 0 {
		var split []string
		for _, v := range entry.TensorSplit {
			split = append(split, fmt.Sprintf("%.2f", v))
		}
		parts = append(parts, fmt.Sprintf("CACHED_TENSOR_SPLIT=\"%s\"", strings.Join(split, ",")))
	}
	if entry.SplitMode != "" {
		parts = append(parts, fmt.Sprintf("CACHED_SPLIT_MODE=\"%s\"", entry.SplitMode))
	}
	if entry.NCPUMoE > 0 {
		parts = append(parts, fmt.Sprintf("CACHED_NCPUMOE=\"%d\"", entry.NCPUMoE))
	}
	parts = append(parts, fmt.Sprintf("CACHED_BATCH=\"%d\"", entry.BatchSize))
	parts = append(parts, fmt.Sprintf("CACHED_UBATCH=\"%d\"", entry.UBatchSize))
	parts = append(parts, fmt.Sprintf("CACHED_PARALLEL=\"%d\"", entry.Parallel))
	if entry.KVUnified {
		parts = append(parts, "CACHED_KVUNIFIED=\"1\"")
	}
	if entry.NoPinned {
		parts = append(parts, "CACHED_NO_PINNED=\"1\"")
	}
	if entry.MMap {
		parts = append(parts, "CACHED_MMAP=\"1\"")
	} else {
		parts = append(parts, "CACHED_MMAP=\"0\"")
	}
	return os.WriteFile(cachePath, []byte(strings.Join(parts, "\n")+"\n"), 0644)
}

func StrategyToCacheEntry(s *Strategy) *CacheEntry {
	if s == nil {
		return nil
	}
	return &CacheEntry{
		OTString:    s.OTString,
		TensorSplit: append([]float64(nil), s.TensorSplit...),
		SplitMode:   s.SplitMode,
		NCPUMoE:     s.NCPUMoE,
		BatchSize:   s.BatchSize,
		UBatchSize:  s.UBatchSize,
		Parallel:    s.Parallel,
		MMap:        s.MMap,
		KVUnified:   s.KVPlacement == "gpu",
	}
}

func parseGPUAssignments(s string) []GPUAssignment {
	var out []GPUAssignment
	for _, tok := range strings.Fields(s) {
		parts := strings.Split(tok, ":")
		if len(parts) != 3 {
			continue
		}
		ci, _ := strconv.Atoi(parts[0])
		st, _ := strconv.Atoi(parts[1])
		ct, _ := strconv.Atoi(parts[2])
		out = append(out, GPUAssignment{CUDAIndex: ci, Start: st, Count: ct})
	}
	return out
}

func parseTensorSplit(s string) []float64 {
	fields := strings.FieldsFunc(s, func(r rune) bool { return r == ',' || r == ' ' || r == ':' })
	out := make([]float64, 0, len(fields))
	for _, f := range fields {
		if f == "" {
			continue
		}
		v, err := strconv.ParseFloat(f, 64)
		if err != nil || v < 0 {
			continue
		}
		out = append(out, v)
	}
	return out
}

func loadProbeCache(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) *probeCache {
	path := probeCachePath(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, probeParallelKey(parallel))
	if path == "" {
		return nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}

	content := string(data)
	pc := &probeCache{ComputeBufByGPU: map[int]int{}, RuntimeGraphGrowthByGPU: map[int]int{}}
	schemaVersion := 1
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		k := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(strings.Trim(parts[1], `"`))
		switch {
		case k == "PROBE_CACHE_SCHEMA":
			if v, err := strconv.Atoi(val); err == nil && v > 0 {
				schemaVersion = v
			}
		case k == "PROBED_COMPUTE_BUF_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.ComputeBufMB = v
			}
		case strings.HasPrefix(k, "PROBED_COMPUTE_BUF_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_COMPUTE_BUF_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.ComputeBufByGPU[idx] = v
			}
		case strings.HasPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA"):
			idxRaw := strings.TrimPrefix(k, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				pc.RuntimeGraphGrowthByGPU[idx] = v
			}
		case k == "PROBED_KV_PER_LAYER_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				pc.KVPerLayerMB = v
			}
		}
	}

	if schemaVersion < 2 {
		// Before schema 2, a startup graph-reserve OOM was written once as the
		// measured compute buffer and again as "runtime growth" from the same
		// cudaMalloc line (rounding made the second value exactly compute or
		// compute+1 MiB). Summing both excluded otherwise viable GPUs. Drop only
		// that unambiguous legacy duplicate; genuine post-health growth is kept.
		for idx, growth := range pc.RuntimeGraphGrowthByGPU {
			compute := pc.ComputeBufByGPU[idx]
			if compute > 0 && growth >= compute && growth <= compute+1 {
				delete(pc.RuntimeGraphGrowthByGPU, idx)
			}
		}
	}

	if pc.ComputeBufMB > 0 || len(pc.ComputeBufByGPU) > 0 || len(pc.RuntimeGraphGrowthByGPU) > 0 || pc.KVPerLayerMB > 0 {
		return pc
	}
	return nil
}

func loadSystemProbe(cacheDir string, gpus []detect.GPU) *systemProbe {
	explicitCacheDir := cacheDir != ""
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}

	// Compute GPU signature hash: sort(names+drivers), MD5, take first 12 chars
	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))

	data, err := os.ReadFile(path)
	if err != nil && explicitCacheDir {
		// App-local installs used to read ~/.cache/ggrun before LLM_APP_HOME
		// became authoritative. Preserve those measured CUDA values and migrate
		// them lazily instead of treating the missing new-path file as zero.
		home, _ := os.UserHomeDir()
		legacyPath := filepath.Join(home, ".cache", "ggrun", fmt.Sprintf("system_%s.cache", gpuSig))
		if legacyPath != path {
			if legacyData, legacyErr := os.ReadFile(legacyPath); legacyErr == nil {
				data, err = legacyData, nil
				if mkErr := os.MkdirAll(filepath.Dir(path), 0755); mkErr == nil {
					_ = os.WriteFile(path, legacyData, 0644)
				}
			}
		}
	}
	if err != nil {
		return nil
	}

	sp := &systemProbe{CUDAOverheadByGPU: map[int]int{}}
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.TrimSpace(parts[0])
		val := strings.TrimSpace(strings.Trim(parts[1], `"`))
		switch {
		case key == "SYS_CUDA_OVERHEAD_MB":
			if v, err := strconv.Atoi(val); err == nil && v >= 0 {
				sp.CUDAOverheadMB = v
			}
		case strings.HasPrefix(key, "SYS_CUDA_OVERHEAD_MB_CUDA"):
			idxRaw := strings.TrimPrefix(key, "SYS_CUDA_OVERHEAD_MB_CUDA")
			idx, idxErr := strconv.Atoi(idxRaw)
			v, valErr := strconv.Atoi(val)
			if idxErr == nil && valErr == nil && idx >= 0 && v >= 0 {
				sp.CUDAOverheadByGPU[idx] = v
			}
		}
	}
	if sp.CUDAOverheadMB == 0 && len(sp.CUDAOverheadByGPU) == 0 {
		return nil
	}
	return sp
}

func PlacementCachePathFor(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, tensorSplit string) string {
	if model == nil {
		return ""
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	if kvPlacement == "" {
		kvPlacement = "auto"
	}
	if kvQuality == "" {
		kvQuality = "mid"
	}
	if backendTag == "" {
		backendTag = "llama"
	}

	key := fmt.Sprintf("place:v%d:%s:%d:%d:%d:%d:%d:%d:%s:%s:%s:%s:%d:%s",
		placementPlannerCacheVersion, filepath.Base(model.Path), model.NumLayers, model.NumExperts,
		model.EmbeddingLength, model.FeedForwardLength,
		ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallel, tensorSplit)
	return filepath.Join(cacheDir, md5Hash12(key)+".place")
}

func placementCachePathForSpecMode(path, mode string) string {
	mode = normalizeSpecMode(mode)
	if path == "" || mode == "off" {
		return path
	}
	ext := filepath.Ext(path)
	base := strings.TrimSuffix(path, ext)
	return base + "-spec-" + sanitizeFilename(mode) + ext
}

func SystemCUDAOverheadByGPU(cacheDir string, gpus []detect.GPU) map[int]int {
	sp := loadSystemProbe(cacheDir, gpus)
	if sp == nil {
		return nil
	}
	out := map[int]int{}
	for k, v := range sp.CUDAOverheadByGPU {
		if v > 0 {
			out[k] = v
		}
	}
	if len(out) == 0 && sp.CUDAOverheadMB > 0 {
		for _, g := range gpus {
			out[g.Index] = sp.CUDAOverheadMB
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

