package placement

import (
	"fmt"

	"github.com/rrifftt/ggrun/pkg/detect"
)

func buildCPUOnly(s *Strategy, caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	s.GPULayers = 0
	s.MMap = !opts.NoMMap
	s.BatchSize = 512
	s.UBatchSize = 256
	return s, nil
}

func buildSingleGPU(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	if len(gpuOrder) == 0 {
		return nil, fmt.Errorf("single-GPU strategy requires a GPU")
	}

	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))
	computeBufMB := computeFloorMB
	if pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel); pc != nil {
		computeBufMB = pc.ComputeBufMB
	}

	gpuKVMB := kvTotalMB
	if s.KVPlacement == "cpu" {
		gpuKVMB = 0
	}
	neededMB := totalSizeMB + cudaOverheadMB + computeBufMB + gpuKVMB
	for _, mainIdx := range gpuOrder {
		if caps.GPUs[mainIdx].VRAMFreeMB() >= neededMB {
			s.MainGPU = caps.GPUs[mainIdx].Index
			return s, nil
		}
	}
	return nil, fmt.Errorf("single-GPU strategy no longer fits any selected GPU (need %d MiB)", neededMB)
}

func buildMultiGPUDense(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)

	// Load measured CUDA overhead. Missing probe data is unknown and contributes 0.
	cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

	// Load model probe for compute buffer (same as MoE path)
	probeHit := false
	computeBufMB := computeFloorMB // 1024 default
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	if pc != nil {
		probeHit = true
		computeBufMB = pc.ComputeBufMB
	}

	// Per-layer costs
	weightPerLayerMB := totalSizeMB / model.NumLayers
	if weightPerLayerMB <= 0 {
		weightPerLayerMB = 1
	}
	kvPerLayerMB := kvTotalMB / model.NumLayers
	if kvPerLayerMB < 1 && kvTotalMB > 0 {
		kvPerLayerMB = 1
	}

	// KV-first GPU reserve: VRAM-proportional (KV reads are VRAM-local)
	gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, seqRange(numGPUs), kvPerLayerMB)

	// Tensor-split: proportional to free VRAM only.
	// llama-server distributes BOTH model weights AND KV cache by this ratio.
	// Using effective free (subtracting KV reserve) causes OOM because
	// llama-server puts KV back proportionally to the split anyway.
	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	split := make([]float64, numGPUs)
	totalFree := 0.0
	for _, g := range caps.GPUs {
		totalFree += float64(g.VRAMFreeMB())
	}
	if totalFree > 0 {
		for _, gi := range gpuOrder {
			free := float64(caps.GPUs[gi].VRAMFreeMB())
			if gi == gpuOrder[0] && s.MMProjSizeMB > 0 {
				free -= float64(s.MMProjSizeMB)
				if free < 0 {
					free = 0
				}
			}
			split[gi] = free / totalFree
		}
	}
	s.TensorSplit = normalizeSplit(split)

	// Find smallest GPU subset that fits the model
	// Use effective capacity (free - overhead) not just total VRAM
	gpuOrderBW := orderGPUsByBandwidth(caps.GPUs)
	bestGPUCount := numGPUs
	for n := 2; n <= numGPUs; n++ {
		subsetCapacity := 0
		for j := 0; j < n; j++ {
			gi := gpuOrderBW[j]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			subsetCapacity += effective
		}
		modelWeightMB := totalSizeMB + kvTotalMB/2 // model weights + partial KV overhead
		if modelWeightMB <= subsetCapacity {
			bestGPUCount = n
			break
		}
	}

	// Zero out GPUs not in the selected subset
	if bestGPUCount < numGPUs {
		for idx := bestGPUCount; idx < numGPUs; idx++ {
			gi := gpuOrderBW[idx]
			split[gi] = 0
		}
		s.TensorSplit = normalizeSplit(split)
	}

	// Layer split is the portable default for heterogeneous GPUs. The tensor
	// split path uses NCCL collectives during graph construction and can abort
	// before health on systems without working peer access (observed with an
	// Ampere PCIe x1 + Ada PCIe x4 pair). Row split is also unsafe for some GQA
	// models. Layer split avoids both failure modes and is supported by mainline,
	// ik_llama, and the registered forks.
	s.SplitMode = "layer"

	_ = probeHit // used for logging/debugging
	return s, nil
}

func buildDenseCPUOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	if s.BackendSupportsFit {
		// Unlike the wholly resident paths, dense CPU offload does not have an
		// explicit layer plan. Leave n_gpu_layers and tensor_split unset so
		// llama.cpp can measure real tensor sizes and place the maximum safe
		// number of layers. Supplying -ngl 999 or a tensor split makes its fit
		// pass abort and then OOM during the real load.
		s.TensorSplit = nil
		s.SplitMode = ""
		s.MMap = false
		return s, nil
	}

	numGPUs := len(caps.GPUs)
	if numGPUs > 1 {
		// Load measured CUDA overhead. Missing probe data is unknown and contributes 0.
		cudaOverheadMB := measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs))

		// Load model probe for compute buffer
		computeBufMB := computeFloorMB
		pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
		if pc != nil {
			computeBufMB = pc.ComputeBufMB
		}

		// KV per layer for reserve
		kvPerLayerMB := kvTotalMB / model.NumLayers
		if kvPerLayerMB < 1 && kvTotalMB > 0 {
			kvPerLayerMB = 1
		}

		// KV-first GPU reserve: weighted by VRAM * PCIe bandwidth
		gpuKVReserveMB := kvReserveByBandwidth(kvTotalMB, caps.GPUs, nil, kvPerLayerMB)

		// Tensor split proportional to effective free VRAM * bandwidth
		split := make([]float64, numGPUs)
		gpuOrder := orderGPUsByBandwidth(caps.GPUs)
		totalWeighted := 0.0
		for idx := 0; idx < numGPUs; idx++ {
			gi := gpuOrder[idx]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			bw := float64(g.BandwidthMBps)
			if bw <= 0 {
				bw = 1.0
			}
			totalWeighted += float64(effective) * bw
		}
		for idx := 0; idx < numGPUs; idx++ {
			gi := gpuOrder[idx]
			g := caps.GPUs[gi]
			overhead := cudaOverheadMB + computeBufMB + gpuKVReserveMB[gi]
			effective := g.VRAMFreeMB() - overhead
			if effective < 0 {
				effective = 0
			}
			bw := float64(g.BandwidthMBps)
			if bw <= 0 {
				bw = 1.0
			}
			if totalWeighted > 0 {
				split[gi] = float64(effective) * bw / totalWeighted
			}
		}
		s.TensorSplit = normalizeSplit(split)

		// Keep the same portable heterogeneous-GPU policy as the fully resident
		// dense path. Tensor/row splits can require peer collectives that are not
		// available on many consumer multi-GPU topologies.
		s.SplitMode = "layer"
	}

	s.MMap = false
	return s, nil
}
