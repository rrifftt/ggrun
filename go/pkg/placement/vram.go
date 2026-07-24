package placement

import (

	"fmt"
	"strconv"
	"strings"

	"github.com/rrifftt/ggrun/pkg/detect"

)

func PredictVRAMUsage(model *ModelProfile, flags map[string]string, caps *detect.Capabilities) (neededMB, freeMB int) {
	if caps == nil || len(caps.GPUs) == 0 {
		return 0, 0
	}
	for _, g := range caps.GPUs {
		freeMB += g.VRAMFreeMB()
	}
	// --fit on lets the backend auto-offload layers to fit KV on GPU.
	// Don't predict OOM for these configs — let the backend handle it.
	if v, ok := flags["--fit"]; ok && v == "on" {
		return 0, freeMB
	}

	// 1. Model weights on GPU
	ncpuMoe := 0
	if v, ok := flags["--n-cpu-moe"]; ok {
		ncpuMoe, _ = strconv.Atoi(v)
	}
	if model.IsMoE && ncpuMoe > 0 && model.ExpertBytes > 0 && model.NumLayers > 0 {
		// MoE: non-expert weights + GPU-resident expert layers
		nonExpertMB := bytesToMiBCeil(model.NonExpertBytes)
		if nonExpertMB <= 0 {
			nonExpertMB = model.TotalSizeMB / 10
		}
		expertPerLayerMB := bytesToMiBCeil(model.ExpertBytes / int64(model.NumLayers))
		gpuExpertLayers := model.NumLayers - ncpuMoe
		if gpuExpertLayers < 0 {
			gpuExpertLayers = 0
		}
		neededMB += nonExpertMB + gpuExpertLayers*expertPerLayerMB
	} else {
		// Dense or full GPU: all weights on GPU
		neededMB += model.TotalSizeMB
	}

	// 2. Context size (needed for compute buffer estimate)
	ctxSize := model.ContextSize
	if v, ok := flags["--ctx-size"]; ok {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			ctxSize = n
		}
	}

	// 3. Compute buffer: empirical estimate from llama.cpp measurements.
	// ~330 MB at 65536 ctx on a 9B model, scales linearly with context.
	computeMB := ctxSize / 200
	if computeMB < 128 {
		computeMB = 128
	}

	// 4. Overhead: CUDA context + internal buffers + fragmentation.
	// Measured ~300-500 MB on NVIDIA GPUs; 400 MB is a safe middle ground.
	const overheadMB = 400

	// Predict the actual allocation that fails: model weights + compute
	// buffer + overhead. KV cache is allocated separately and the backend
	// handles KV pressure via --fit or context reduction. Excluding KV
	// avoids false positives on MoE models where model+KV exceeds free
	// VRAM but the backend still fits both via lazy/mmap allocation.
	neededMB = neededMB + computeMB + overheadMB
	neededMB = neededMB * 105 / 100 // 5% margin for alignment

	return neededMB, freeMB
}

func ParseFlagsToMap(args []string) map[string]string {
	m := make(map[string]string)
	for i := 0; i < len(args); i++ {
		if !strings.HasPrefix(args[i], "-") {
			continue
		}
		key := args[i]
		if eq := strings.Index(key, "="); eq > 0 {
			m[key[:eq]] = key[eq+1:]
			continue
		}
		if i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
			m[key] = args[i+1]
			i++
		} else {
			m[key] = ""
		}
	}
	return m
}

func checkMemoryOrDie(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int, opts Options) error {
	numGPUs := len(caps.GPUs)

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0;
	// preflight/startup OOM recording supplies measured data for later launches.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	if pc != nil {
		computeBufMB = pc.ComputeBufMB
	}

	// Model weights + per-GPU overhead (CUDA context + compute buffer)
	// For single GPU, only count 1 GPU's overhead
	overheadGPUs := numGPUs
	if s.Type == CPUOnly {
		overheadGPUs = 0
	} else if s.Type == SingleGPU {
		overheadGPUs = 1
	}
	modelOverheadMB := totalSizeMB + (cudaOverheadMB+computeBufMB)*overheadGPUs
	gpuKVMB := kvTotalMB
	if s.KVPlacement == "cpu" {
		gpuKVMB = 0
	}
	neededMB := modelOverheadMB + gpuKVMB

	ramOverheadMB := 0
	if s.Type == CPUOnly || s.Type == DenseCPUOffload {
		ramOverheadMB = ramRuntimeOverheadMB(model, s.UBatchSize, totalSizeMB)
		neededMB += ramOverheadMB
	}

	var poolMB int
	var poolLabel string
	switch s.Type {
	case SingleGPU:
		// Best GPU's free VRAM
		bestFree := 0
		for _, g := range caps.GPUs {
			if g.VRAMFreeMB() > bestFree {
				bestFree = g.VRAMFreeMB()
			}
		}
		poolMB = bestFree
		poolLabel = "best GPU"
	case MultiGPUDense:
		// Total free VRAM across all GPUs
		for _, g := range caps.GPUs {
			poolMB += g.VRAMFreeMB()
		}
		poolLabel = "all GPUs"
	case CPUOnly:
		poolMB = caps.RAM.FreeMB
		poolLabel = "RAM"
	case DenseCPUOffload:
		// Total system memory (GPU + RAM) since model is split
		for _, g := range caps.GPUs {
			poolMB += g.VRAMFreeMB()
		}
		poolMB += caps.RAM.FreeMB
		poolLabel = "system memory"
	}

	if neededMB > poolMB {
		// Back-solve max safe context
		maxKVMB := poolMB - modelOverheadMB - ramOverheadMB
		if maxKVMB < 0 {
			maxKVMB = 0
		}
		maxCtx := 0
		if kvTotalMB > 0 {
			maxCtx = maxKVMB * s.ContextSize / kvTotalMB
		}
		msg := fmt.Sprintf(
			"ERROR: model does not fit in %s.\n"+
				"  Model weights:          %dMB\n"+
				"  CUDA overhead (%d GPU): %dMB\n"+
				"  Compute buffers (%d):   %dMB\n"+
				"  KV cache (ctx=%d):      %dMB\n"+
				"  Host runtime buffers:   %dMB\n"+
				"  -----------------------------\n"+
				"  Total needed:          %dMB\n"+
				"  Available (%s):        %dMB\n"+
				"  Shortfall:             %dMB\n",
			poolLabel, totalSizeMB, overheadGPUs, cudaOverheadMB*overheadGPUs,
			overheadGPUs, computeBufMB*overheadGPUs, s.ContextSize, kvTotalMB,
			ramOverheadMB, neededMB, poolLabel, poolMB, neededMB-poolMB)
		if maxCtx > 0 {
			msg += fmt.Sprintf("\n  Max safe context at this memory: --ctx-size %d", maxCtx)
		} else if totalSizeMB > poolMB {
			msg += "\n  Model weights alone exceed available memory."
		} else {
			msg += "\n  Fixed runtime buffers leave no safe space for the requested KV cache."
		}
		msg += "\n  Or use a smaller quantization / model."
		return fmt.Errorf("%s", msg)
	}
	return nil
}

func firstLaunchComputeBufMB(model *ModelProfile, uBatch int) int {
	return firstLaunchComputeBufMBParallel(model, uBatch, 1)
}

func measuredCUDAOverheadMB(sysProbe *systemProbe) int {
	if sysProbe != nil && sysProbe.CUDAOverheadMB > 0 {
		return sysProbe.CUDAOverheadMB
	}
	return 0
}

func firstLaunchComputeBufMBParallel(model *ModelProfile, uBatch, parallel int) int {
	est := uBatch * 4
	if model != nil && model.HiddenSize > 0 && model.NumLayers > 0 {
		coefficient := 42.0
		// DeepSeek4's routed MLA graph is substantially larger than a generic
		// MoE graph. Do not project that architecture-specific coefficient onto
		// Kimi/Mixtral/etc.; their exact backend preflight will calibrate them.
		if strings.EqualFold(model.ModelArch, "deepseek4") && model.ExpertUsedCount > 0 {
			moeCoefficient := float64(model.ExpertUsedCount) * 128.0
			if moeCoefficient > coefficient {
				coefficient = moeCoefficient
			}
		}
		per := float64(model.HiddenSize) * float64(model.NumLayers) * coefficient / 1e6
		est = int(float64(uBatch) * per)
		if parallel > 1 {
			est = ceilDivInt(est, parallel)
		}
	}
	if est < computeFloorMB {
		est = computeFloorMB
	}
	return est
}

