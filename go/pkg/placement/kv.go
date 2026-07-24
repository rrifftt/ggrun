package placement

import (
	"strings"

	"github.com/rrifftt/ggrun/pkg/detect"
)

func computeKVTotalMB(model *ModelProfile, ctxSize int, kvType string) int {
	// Prefer the KV size llama.cpp actually allocated on a previous launch (read
	// back from its log) — it is exact for every attention scheme, including the
	// compressed ones (MLA / CSA-HCA / sliding-window) the formula below can't
	// model. Falls through to the per-arch estimate when we have no measurement.
	if r, ok := model.MeasuredKVBytesPerTok[strings.ToLower(kvType)]; ok && r > 0 {
		return int(r*float64(ctxSize)/1048576.0 + 0.5)
	}

	var kvElemsTotal int
	hasMLA := model.KVLoraRank > 0
	hasSSM := model.HasSSM == 1
	hasISWA := model.SlidingWindow > 0

	if hasMLA {
		// MLA: compressed c^{KV} + RoPE'd key once per layer
		kvElemsTotal = model.NumLayers * ctxSize * (model.KVLoraRank + model.RopeDim)
	} else if hasSSM {
		var attnLayers int
		if model.FullAttnInterval > 0 {
			attnLayers = model.NumLayers / model.FullAttnInterval
			if attnLayers < 1 {
				attnLayers = 1
			}
		} else if model.HeadCountKV == 0 {
			attnLayers = 0
		} else {
			attnLayers = (model.NumLayers + 1) / 2
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = attnLayers * ctxSize * kvBytesPerLayerPerToken
	} else if hasISWA {
		swaPeriod := 6
		switch model.ModelArch {
		case "gemma2", "cohere2", "exaone4", "llama4":
			swaPeriod = 4
		case "gemma3":
			swaPeriod = 6
		case "plamo3":
			swaPeriod = 8
		}
		fullLayers := (model.NumLayers + swaPeriod - 1) / swaPeriod
		swaLayers := model.NumLayers - fullLayers
		swaCtx := ctxSize
		if swaCtx > model.SlidingWindow {
			swaCtx = model.SlidingWindow
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = fullLayers*ctxSize*kvBytesPerLayerPerToken + swaLayers*swaCtx*kvBytesPerLayerPerToken
	} else {
		// Standard GQA/MQA
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = model.NumLayers * ctxSize * kvBytesPerLayerPerToken
	}

	bytesPerElem, ok := kvTypeBytesPerElement(kvType)
	if !ok {
		// Compute normally receives a validated type. Preserve the old
		// conservative q8_0 fallback for direct package callers.
		bytesPerElem = 1.0625
	}
	return int(float64(kvElemsTotal) * bytesPerElem / 1024 / 1024)
}

func computeKVTotalMBAsymmetric(model *ModelProfile, ctxSize int, kvTypeK, kvTypeV string) int {
	// 1. MeasuredKVBytesPerTok Override
	combinedKey := strings.ToLower(kvTypeK + "-" + kvTypeV)
	if r, ok := model.MeasuredKVBytesPerTok[combinedKey]; ok && r > 0 {
		return int(r*float64(ctxSize)/1048576.0 + 0.5)
	}
	if rK, okK := model.MeasuredKVBytesPerTok[strings.ToLower(kvTypeK)]; okK && rK > 0 {
		if rV, okV := model.MeasuredKVBytesPerTok[strings.ToLower(kvTypeV)]; okV && rV > 0 {
			totalLen := model.KeyLength + model.ValueLength
			if totalLen > 0 {
				kShare := float64(model.KeyLength) / float64(totalLen)
				vShare := float64(model.ValueLength) / float64(totalLen)
				return int((rK*kShare+rV*vShare)*float64(ctxSize)/1048576.0 + 0.5)
			}
		}
	}

	// 2. Architecture-aware element calculation
	hasMLA := model.KVLoraRank > 0
	hasSSM := model.HasSSM == 1
	hasISWA := model.SlidingWindow > 0

	var kvElemsTotal int
	if hasMLA {
		kvElemsTotal = model.NumLayers * ctxSize * (model.KVLoraRank + model.RopeDim)
	} else if hasSSM {
		var attnLayers int
		if model.FullAttnInterval > 0 {
			attnLayers = model.NumLayers / model.FullAttnInterval
			if attnLayers < 1 {
				attnLayers = 1
			}
		} else if model.HeadCountKV == 0 {
			attnLayers = 0
		} else {
			attnLayers = (model.NumLayers + 1) / 2
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = attnLayers * ctxSize * kvBytesPerLayerPerToken
	} else if hasISWA {
		swaPeriod := 6
		switch model.ModelArch {
		case "gemma2", "cohere2", "exaone4", "llama4":
			swaPeriod = 4
		case "gemma3":
			swaPeriod = 6
		case "plamo3":
			swaPeriod = 8
		}
		fullLayers := (model.NumLayers + swaPeriod - 1) / swaPeriod
		swaLayers := model.NumLayers - fullLayers
		swaCtx := ctxSize
		if swaCtx > model.SlidingWindow {
			swaCtx = model.SlidingWindow
		}
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = fullLayers*ctxSize*kvBytesPerLayerPerToken + swaLayers*swaCtx*kvBytesPerLayerPerToken
	} else {
		kvBytesPerLayerPerToken := model.HeadCountKV * (model.KeyLength + model.ValueLength)
		kvElemsTotal = model.NumLayers * ctxSize * kvBytesPerLayerPerToken
	}

	// 3. Split elements by architecture
	var kvElemsK, kvElemsV int
	if hasMLA {
		kvElemsK = kvElemsTotal / 2
		kvElemsV = kvElemsTotal - kvElemsK
	} else if hasISWA {
		// ISWA has mixed context sizes (full layers use ctxSize, sliding-window
		// layers use swaCtx). Compute the K and V totals directly — do NOT
		// multiply by ctxSize again at the end.
		swaPeriod := 6
		switch model.ModelArch {
		case "gemma2", "cohere2", "exaone4", "llama4":
			swaPeriod = 4
		case "gemma3":
			swaPeriod = 6
		case "plamo3":
			swaPeriod = 8
		}
		fullLayers := (model.NumLayers + swaPeriod - 1) / swaPeriod
		swaLayers := model.NumLayers - fullLayers
		swaCtx := ctxSize
		if swaCtx > model.SlidingWindow {
			swaCtx = model.SlidingWindow
		}
		kvElemsK = (fullLayers*ctxSize + swaLayers*swaCtx) * model.HeadCountKV * model.KeyLength
		kvElemsV = (fullLayers*ctxSize + swaLayers*swaCtx) * model.HeadCountKV * model.ValueLength
	} else {
		var kvElemsPerTokenK, kvElemsPerTokenV int
		if hasSSM {
			var attnLayers int
			if model.FullAttnInterval > 0 {
				attnLayers = model.NumLayers / model.FullAttnInterval
				if attnLayers < 1 {
					attnLayers = 1
				}
			} else if model.HeadCountKV == 0 {
				attnLayers = 0
			} else {
				attnLayers = (model.NumLayers + 1) / 2
			}
			kvElemsPerTokenK = attnLayers * model.HeadCountKV * model.KeyLength
			kvElemsPerTokenV = attnLayers * model.HeadCountKV * model.ValueLength
		} else {
			kvElemsPerTokenK = model.NumLayers * model.HeadCountKV * model.KeyLength
			kvElemsPerTokenV = model.NumLayers * model.HeadCountKV * model.ValueLength
		}
		kvElemsK = kvElemsPerTokenK * ctxSize
		kvElemsV = kvElemsPerTokenV * ctxSize
	}

	bytesPerElemK := kvBytesPerElem(kvTypeK)
	bytesPerElemV := kvBytesPerElem(kvTypeV)
	totalBytes := float64(kvElemsK)*bytesPerElemK + float64(kvElemsV)*bytesPerElemV
	return int(totalBytes/1024/1024 + 0.5)
}

func parseKVType(kvType string) (string, string, string) {
	if strings.Contains(kvType, "-") {
		parts := strings.SplitN(kvType, "-", 2)
		return parts[0], parts[0], parts[1] // base, K, V
	}
	return kvType, kvType, kvType
}

func resolveAutoKVPlacement(caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB, vramOverheadMB int) string {
	freeVRAM := 0
	for _, g := range caps.GPUs {
		freeVRAM += g.VRAMFreeMB()
	}
	// Everything fits in VRAM: KV on GPU (fastest).
	if totalSizeMB+kvTotalMB+vramOverheadMB <= freeVRAM {
		return "gpu"
	}
	if model.IsMoE {
		if strings.EqualFold(model.ModelArch, "deepseek4") {
			// KV on CPU makes llama.cpp auto-disable flash attention, and
			// deepseek4's non-FA graph materializes score tensors that grow
			// with real token position — no load-time reserve can cover that.
			return "gpu"
		}
		// After expert offload, only non-expert weights occupy VRAM.
		// If non-expert + KV + overhead fits, keep KV on GPU for decode speed.
		nonExpertMB := bytesToMiBCeil(model.NonExpertBytes)
		if nonExpertMB <= 0 {
			nonExpertMB = totalSizeMB / 10 // fallback: ~10% is non-expert
		}
		if nonExpertMB+kvTotalMB+vramOverheadMB <= freeVRAM {
			return "gpu"
		}
		return "cpu"
	}
	// Dense model: weights fit in VRAM but weights + KV don't.
	// Put KV on CPU so all layers stay on GPU (fast decode).
	if totalSizeMB+vramOverheadMB <= freeVRAM {
		return "cpu"
	}
	// Model doesn't fit in VRAM at all — KV placement won't save it;
	// keep KV on GPU so the layers that ARE on GPU benefit from it.
	return "gpu"
}

func computeAutoContextSizeKVPlacement(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType, placement string, opts Options) (int, string) {
	freeVRAM := 0
	for _, g := range caps.GPUs {
		freeVRAM += g.VRAMFreeMB()
	}

	// Overhead is derived per-component, not a flat guess: VRAM side = measured
	// CUDA overhead + compute buffer per GPU; RAM side = host/graph/mmap/activation.
	sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
	vramOverheadMB := perGPUVRAMOverheadMB(sysProbe, 0) * len(caps.GPUs)

	var kvBudgetMB int
	if placement == "cpu" {
		// KV lives in RAM, sharing it with the weights that don't fit in VRAM.
		weightsInRAM := totalSizeMB - freeVRAM
		if weightsInRAM < 0 {
			weightsInRAM = 0
		}
		kvBudgetMB = caps.RAM.FreeMB - weightsInRAM - ramRuntimeOverheadMB(model, 0, totalSizeMB)
	} else {
		// KV lives in VRAM alongside the GPU-resident weights. Dense: the whole
		// model. MoE: only the non-expert weights (experts offload to CPU), so the
		// rest of VRAM is free for KV.
		gpuResident := totalSizeMB
		if model.IsMoE {
			if ne := bytesToMiBCeil(model.NonExpertBytes); ne > 0 {
				gpuResident = ne
				// Input embeddings stay in host memory, never in VRAM.
				if te := bytesToMiBCeil(model.TokenEmbdBytes); te > 0 && te < ne {
					gpuResident = ne - te
				}
			}
		}
		kvBudgetMB = freeVRAM - gpuResident - vramOverheadMB
	}
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	kvPairs := expandKVTypes(preferredKVType, opts)
	for _, kvPair := range kvPairs {
		refCtx := 32768
		var refKVMB int
		if kvPair[0] == kvPair[1] {
			refKVMB = computeKVTotalMB(model, refCtx, kvPair[0])
		} else {
			refKVMB = computeKVTotalMBAsymmetric(model, refCtx, kvPair[0], kvPair[1])
		}
		if refKVMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVMB) * 1048576.0 / float64(refCtx)
		maxCtx := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)
		if model.CTXTrain > 0 && model.CTXTrain < maxCtx {
			maxCtx = model.CTXTrain
		}
		best := 0
		for _, c := range []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304} {
			if c <= maxCtx {
				best = c
			}
		}
		if best >= 32768 {
			if kvPair[0] == kvPair[1] {
				return best, kvPair[0]
			}
			return best, kvPair[0] + "-" + kvPair[1]
		}
	}
	return 32768, fallbackKVType(preferredKVType, opts.KVQuality)
}
