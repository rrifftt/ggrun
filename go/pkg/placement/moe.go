package placement

import (

	"fmt"
	"os"
	"strings"

	"github.com/rrifftt/ggrun/pkg/detect"

)

// GPUAssignment describes layers assigned to a GPU.


func buildMoEOffload(s *Strategy, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	numGPUs := len(caps.GPUs)
	if numGPUs == 0 {
		return buildCPUOnly(s, caps, model, opts)
	}
	if model.NumLayers <= 0 {
		return nil, fmt.Errorf("MoE placement requires model layer count")
	}

	gpuOrder := orderGPUsByBandwidth(caps.GPUs)
	s.MainGPU = caps.GPUs[gpuOrder[0]].Index
	if numGPUs > 1 {
		s.SplitMode = "layer"
	}

	moeStartLayer := model.LeadingDense
	if moeStartLayer < 0 || moeStartLayer >= model.NumLayers {
		moeStartLayer = 0
	}
	moeLayerCount := model.NumLayers - moeStartLayer
	if moeLayerCount <= 0 {
		moeLayerCount = model.NumLayers
		moeStartLayer = 0
	}

	// Per-layer costs
	expertTotalMB := bytesToMiBCeil(model.ExpertBytes)
	if expertTotalMB <= 0 {
		expertTotalMB = totalSizeMB * 90 / 100
	}
	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	if nonExpertTotalMB <= 0 {
		nonExpertTotalMB = totalSizeMB - expertTotalMB
	}
	if nonExpertTotalMB < 0 {
		nonExpertTotalMB = 0
	}

	// Input embeddings never leave host memory (llama.cpp keeps the input layer
	// on CPU), so they must not be charged against per-GPU VRAM budgets; they
	// are counted on the RAM side below instead.
	tokenEmbdMB := bytesToMiBCeil(model.TokenEmbdBytes)
	if tokenEmbdMB > 0 && tokenEmbdMB < nonExpertTotalMB {
		nonExpertTotalMB -= tokenEmbdMB
	} else {
		tokenEmbdMB = 0
	}

	// The output head is not pro-rata: llama.cpp assigns it to the device that
	// owns the last of its n_layer+1 split slots. Charging it proportionally
	// once OOM'd the smallest GPU (output.weight ~1GB landed whole on the 4070
	// while placement had budgeted an 0.11 share of it).
	outputMB := bytesToMiBCeil(model.OutputBytes)
	if outputMB > 0 && outputMB < nonExpertTotalMB {
		nonExpertTotalMB -= outputMB
	} else {
		outputMB = 0
	}

	// Shared experts ride with their layer's owning device — the exps=CPU
	// catch-all does not match "_shexp", so even CPU-offloaded layers keep
	// their shared expert in VRAM. GPU whole-layer pins already include shexp
	// in expertPerLayerMB; the CPU side must exclude it.
	shexpTotalMB := bytesToMiBCeil(model.ShexpBytes)
	if shexpTotalMB < 0 || shexpTotalMB >= expertTotalMB {
		shexpTotalMB = 0
	}

	expertPerLayerMB := ceilDivInt(expertTotalMB, moeLayerCount)
	if expertPerLayerMB <= 0 {
		expertPerLayerMB = 1
	}

	// RAM (and gate+up chunk) cost of a CPU-offloaded expert layer: routed
	// experts only, shared expert stays on the owning GPU.
	expertCPUPerLayerMB := ceilDivInt(expertTotalMB-shexpTotalMB, moeLayerCount)
	if expertCPUPerLayerMB <= 0 {
		expertCPUPerLayerMB = expertPerLayerMB
	}

	// nonExpertForGPU is the non-expert weight that must be distributed across
	// GPUs. Token embeddings were already subtracted from nonExpertTotalMB
	// above (they stay on CPU), so no further subtraction is needed here.
	nonExpertForGPU := nonExpertTotalMB
	if nonExpertForGPU < 0 {
		nonExpertForGPU = 0
	}
	nonExpertPerLayerMB := ceilDivInt(nonExpertForGPU, model.NumLayers)
	if nonExpertPerLayerMB <= 0 {
		nonExpertPerLayerMB = 1
	}

	// Load measured CUDA overhead per GPU. Missing probe entries are unknown
	// components, not free VRAM; they contribute 0 to whole-layer fitting but
	// block optional remainder squeeze below until a successful launch measures
	// them.
	sysCUDAOverheadByGPU := SystemCUDAOverheadByGPU(opts.CacheDir, caps.GPUs)

	// Load per-model/runtime probe cache. Until a model has completed one launch
	// with these settings, use a first-launch fallback that keeps the main GPU
	// conservative without charging the full prompt graph to every secondary GPU.
	pc := loadProbeCache(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel)
	fixedPerGPU := make([]int, numGPUs)
	expertOnlyFixedPerGPU := make([]int, numGPUs)
	for i, g := range caps.GPUs {
		computeBufMB := firstLaunchComputeBufMBForGPUParallel(model, s.UBatchSize, s.Parallel, i, gpuOrder)
		runtimeGrowthMB := 0
		if pc != nil {
			// Use the aggregate (primary) compute buffer for split-owner cost
			// accounting. The per-GPU values in the probe cache are measured
			// for the specific placement that ran (e.g. GPU1 expert-only =
			// 299 MB, GPU0 split-owner = 33 GB at ub=256). When the placement
			// changes between measurement and this computation — which is
			// exactly what a re-plan does — the per-GPU values are wrong: a
			// GPU that was expert-only (299 MB) might become a split-owner
			// (needing the full ~33 GB), or vice versa. The aggregate
			// (primary's compute buffer) is the maximum any split-owner would
			// need, so using it for all GPUs is conservative but correct.
			// Expert-only GPUs use expertOnlyComputeReserveMB which caps at
			// the compute floor, so the aggregate doesn't affect them.
			if pc.ComputeBufMB > 0 {
				computeBufMB = pc.ComputeBufMB
			}
			// Real long requests can need more than the load-time graph
			// reserve accounts for — llama-server's context-checkpoint state
			// save (tools/server/server-context.cpp create_checkpoint) lives
			// entirely outside llama_context::sched_reserve(), so no no-alloc
			// measurement (ours or llama.cpp's own --fit) ever sees it.
			// Reproduced 2026-07-08: DeepSeek-V4 crashed needing +1679MB on
			// CUDA2 at 16384 real tokens despite a startup reserve sized for
			// the full max context. Once a canary or a real crash measures
			// this growth (RecordRuntimeGraphGrowth/-FromOOM), reserve it here
			// so the next placement for this exact key packs around it
			// instead of rediscovering the deficit by crashing again.
			runtimeGrowthMB = pc.RuntimeGraphGrowthByGPU[g.Index]
		}
		fixedPerGPU[i] = sysCUDAOverheadByGPU[g.Index] + computeBufMB + runtimeGrowthMB
		expertOnlyFixedPerGPU[i] = sysCUDAOverheadByGPU[g.Index] + expertOnlyComputeReserveMB(computeBufMB) + runtimeGrowthMB
	}

	expertOnlyGPU := expertOnlySlowGPUs(caps.GPUs, fixedPerGPU, expertOnlyFixedPerGPU, expertPerLayerMB, nonExpertPerLayerMB)

	// Use only GPUs that can carry CUDA/compute overhead plus their emitted
	// tensor-split share of all non-expert weights and KV. The split is
	// bandwidth-weighted: under --split-mode layer, the GPU that owns a
	// layer's non-expert weights also computes that layer — including
	// streaming its CPU-resident experts over PCIe. Weighting the split by
	// measured PCIe bandwidth (not just free VRAM) concentrates layer
	// ownership on the fastest-link GPU, so CPU-expert streaming avoids
	// bottlenecking on a slow PCIe link (e.g. a card stuck at x1).
	// When bandwidth is unknown or uniform across GPUs, this degenerates
	// to free-VRAM-proportional (the previous behaviour).

	// Only reserve GPU VRAM for the KV cache when it actually lives on the GPU.
	// With KV on CPU (big-MoE auto-placement, or --kv-placement cpu) the experts
	// must be free to fill that VRAM instead — otherwise every GPU sits half-empty
	// holding room for a KV cache that's in system RAM.
	gpuKVTotalMB := kvTotalMB
	if s.KVPlacement == "cpu" {
		gpuKVTotalMB = 0
	}

	// nonExpertChargeMB is the exact VRAM a GPU pays for non-expert weights
	// under llama.cpp's real slot assignment: its owned repeating layers times
	// the per-layer non-expert bytes, the shared experts of its owned layers,
	// and the whole output head iff it owns the last slot. This replaces the
	// old pro-rata split share, which spread the ~1GB output head across all
	// GPUs and OOM'd whichever small card actually received it.
	nonExpertChargeMB := func(owned []int, outputDev, gi int) int {
		perLayer := float64(nonExpertTotalMB) / float64(model.NumLayers)
		perShexp := 0.0
		if moeLayerCount > 0 {
			perShexp = float64(shexpTotalMB) / float64(moeLayerCount)
		}
		charge := float64(owned[gi]) * (perLayer + perShexp)
		if gi == outputDev {
			charge += float64(outputMB)
		}
		return int(charge + 0.999)
	}

	used := make([]bool, numGPUs)
	for i, g := range caps.GPUs {
		free := g.VRAMFreeMB()
		fixed := fixedPerGPU[i]
		used[i] = !expertOnlyGPU[i] && free > fixed
	}

	var split []float64
	var ownedLayers []int
	outputDev := -1
	for {
		rawSplit := make([]float64, numGPUs)
		totalWeighted := 0.0
		for i, g := range caps.GPUs {
			if used[i] {
				// Weight by VRAM available AFTER the fixed per-GPU CUDA/compute
				// reserve, not raw free VRAM, so high-overhead GPUs are not
				// over-allocated.
				avail := float64(g.VRAMFreeMB() - fixedPerGPU[i])
				totalWeighted += avail * gpuSplitWeight(g)
			}
		}
		if totalWeighted <= 0 {
			return nil, fmt.Errorf("model does not fit on this system: no GPU has free VRAM after CUDA/compute overhead")
		}
		for i, g := range caps.GPUs {
			if used[i] {
				avail := float64(g.VRAMFreeMB() - fixedPerGPU[i])
				rawSplit[i] = avail * gpuSplitWeight(g) / totalWeighted
			}
		}
		split = normalizeSplit(rawSplit)
		ownedLayers, outputDev = layerOwnership(split, model.NumLayers)

		removed := false
		for i, g := range caps.GPUs {
			if !used[i] {
				continue
			}
			kvShareMB := ownedShareMB(gpuKVTotalMB, ownedLayers, model.NumLayers, i)
			if fixedPerGPU[i]+nonExpertChargeMB(ownedLayers, outputDev, i)+kvShareMB > g.VRAMFreeMB() {
				used[i] = false
				removed = true
			}
		}
		if !removed {
			break
		}
	}

	// Post-split retrofit: a GPU eliminated from the split (KV or compute buffer
	// exceeded its share) but not already classified expert-only can still hold
	// whole expert layers. Without this, such a GPU sits idle while its VRAM
	// could offload expert layers from CPU RAM. Reclassify it as expert-only
	// when it can fit at least one complete expert layer under the expert-only
	// reserve, so the expert-capacity loop below picks it up. Capacity always uses
	// current free VRAM: existing workloads and accumulated OOM penalties remain
	// real even if the card changes from split-owner to expert-only.
	for i, g := range caps.GPUs {
		if expertOnlyGPU[i] || used[i] || split[i] > 0 {
			continue
		}
		if i >= len(expertOnlyFixedPerGPU) {
			continue
		}
		if g.VRAMFreeMB()-expertOnlyFixedPerGPU[i] >= expertPerLayerMB {
			expertOnlyGPU[i] = true
		}
	}

	if numGPUs > 1 {
		s.TensorSplit = split
	}

	// Per-GPU expert capacity under the exact emitted split. roomMBPer keeps the
	// exact VRAM budget for experts on each GPU so sub-layer packing can later
	// fill the remainder that whole-layer flooring leaves stranded.
	maxGPULayersPer := make([]int, numGPUs)
	roomMBPer := make([]int, numGPUs)
	maxGPULayers := 0
	for _, gi := range gpuOrder {
		if split[gi] <= 0 && !expertOnlyGPU[gi] {
			continue
		}
		g := caps.GPUs[gi]
		kvShareMB := ownedShareMB(gpuKVTotalMB, ownedLayers, model.NumLayers, gi)
		fixedMB := fixedPerGPU[gi]
		if expertOnlyGPU[gi] {
			fixedMB = expertOnlyFixedPerGPU[gi]
		}
		// Expert-only GPUs carry the non-expert tensors (norms, attention) for
		// every layer pinned to them via -ot, even though ownedLayers==0. Keep
		// the compute-buffer floor: expert-only GPUs run the per-layer forward
		// pass for their pinned layers, so the graph reserve must cover norms,
		// routing, and replicated layer tensors. The probe data shows the
		// actual expert-only compute buffer is small (~299 MB for 3 layers on
		// DeepSeek-V4), well under the 1024 MB floor.
		if expertOnlyGPU[gi] && fixedMB < computeFloorMB {
			fixedMB = computeFloorMB
		}
		nonExpertCharge := nonExpertChargeMB(ownedLayers, outputDev, gi)
		if expertOnlyGPU[gi] {
			nonExpertCharge = 0 // zero here; per-layer cost accounts for it below
		}
		var roomMB int
		if expertOnlyGPU[gi] {
			// Expert-only GPUs don't own dense layers or KV, but other processes,
			// CUDA overhead, and any observed OOM penalty still consume real VRAM.
			roomMB = g.VRAMFreeMB() - fixedMB
		} else {
			roomMB = g.VRAMFreeMB() - fixedMB - nonExpertCharge - kvShareMB
		}
		if roomMB < 0 {
			roomMB = 0
		}
		roomMBPer[gi] = roomMB

		perLayerCost := expertPerLayerMB
		if expertOnlyGPU[gi] {
			nonExpertPerLayer := float64(nonExpertTotalMB) / float64(model.NumLayers)
			perLayerCost = expertPerLayerMB + int(nonExpertPerLayer+0.5)
		}
		capLayers := roomMB / perLayerCost
		if capLayers > moeLayerCount {
			capLayers = moeLayerCount
		}
		maxGPULayersPer[gi] = capLayers
		maxGPULayers += capLayers
	}
	if maxGPULayers > moeLayerCount {
		maxGPULayers = moeLayerCount
	}

	// Hard ceilings: _recompute_cpu_layer_caps
	kvPlacementEffective := s.KVPlacement
	if kvPlacementEffective == "auto" {
		kvPlacementEffective = "gpu"
	}
	cpuKVRAMMB := 0
	if kvPlacementEffective == "cpu" {
		cpuKVRAMMB = kvTotalMB
	}

	// Strict ceiling (--no-mmap path). Uses the real measured-overhead
	// formula (not a hand-copied subset) so this ceiling and the later real
	// RAM check (checkMemoryOrDie -> ramRuntimeOverheadMB) can't disagree —
	// a stale copy here was missing the cpuActMB term, letting this ceiling
	// accept a CPU-layer count the real check would then reject.
	ramOverheadPreMB := ramRuntimeOverheadMB(model, s.UBatchSize, totalSizeMB)
	cpuBudgetStrict := caps.RAM.FreeMB - ramOverheadPreMB - cpuKVRAMMB
	if cpuBudgetStrict < 0 {
		cpuBudgetStrict = 0
	}
	maxCPULayersStrict := 0
	if expertCPUPerLayerMB > 0 {
		maxCPULayersStrict = cpuBudgetStrict / expertCPUPerLayerMB
	}
	if maxCPULayersStrict > moeLayerCount {
		maxCPULayersStrict = moeLayerCount
	}

	// Mmap-aware ceiling.
	//
	// Expert weight bytes live in a read-only, file-backed mmap: the kernel can
	// always evict those clean pages under memory pressure and re-fault them
	// from the model file, so they never have to be simultaneously resident.
	// What CANNOT be reclaimed is the runtime's own anonymous memory — CUDA
	// host staging, graph scratch, the mmap page-table, CPU activation buffers
	// (ramOverheadPreMB), and the KV cache (cpuKVRAMMB, continuously
	// read/written, not file-backed) — if that doesn't fit in free RAM, no
	// amount of expert-page eviction helps, and mmap would thrash from the
	// first token. That is the real (measured, not guessed) floor: it replaces
	// both the old flat "8 layers" floor (too lenient — let a model whose
	// working set vastly exceeds even non-reclaimable RAM through) and the
	// "full footprint must fit" floor (too strict — defeated the entire point
	// of mmap, which is to hold LESS than the full footprint resident).
	preWorkingSetFloor := ramOverheadPreMB + cpuKVRAMMB
	maxCPULayersMMap := 0
	if caps.RAM.FreeMB >= preWorkingSetFloor {
		maxCPULayersMMap = moeLayerCount
	}

	maxCPULayers := maxCPULayersStrict
	ceilCPULabel := "strict --no-mmap"
	if maxGPULayers+maxCPULayersStrict < moeLayerCount &&
		!opts.NoMMap &&
		maxCPULayersMMap > maxCPULayersStrict {
		maxCPULayers = maxCPULayersMMap
		ceilCPULabel = "mmap (page-cache)"
	}

	// Does-not-fit guard
	if maxGPULayers+maxCPULayers < moeLayerCount {
		gap := moeLayerCount - maxGPULayers - maxCPULayers
		gapVRAMMB := gap * expertPerLayerMB
		gapRAMMB := gap * expertPerLayerMB
		return nil, fmt.Errorf(
			"model does not fit on this system.\n"+
				"  Required:    %d MoE layers\n"+
				"  GPU cap:     %d layers across %d GPU(s)\n"+
				"  CPU cap:     %d layers (%s)\n"+
				"  Gap:         %d layers — need ~%dMB more free VRAM or ~%dMB more RAM\n"+
				"\n  Options:\n"+
				"    1. Free VRAM (close other GPU workloads, --gpus to add a card)\n"+
				"    2. Drop --no-mmap so kernel can page experts on demand\n"+
				"    3. Use a smaller quantization or smaller model",
			moeLayerCount, maxGPULayers, numGPUs, maxCPULayers, ceilCPULabel,
			gap, gapVRAMMB, gapRAMMB)
	}

	// Initial layer assignment
	layersPerGPU := make([]int, numGPUs)
	totalGPULayers := 0
	remainingMoELayers := moeLayerCount
	for _, gi := range gpuOrder {
		layers := maxGPULayersPer[gi]
		if layers > remainingMoELayers {
			layers = remainingMoELayers
		}
		layersPerGPU[gi] = layers
		totalGPULayers += layers
		remainingMoELayers -= layers
		if remainingMoELayers == 0 {
			break
		}
	}
	layersCPU := moeLayerCount - totalGPULayers

	// Sub-layer expert packing (GPU squeeze): whole-layer packing floors each
	// GPU's capacity, stranding up to ~expertPerLayerMB of VRAM per card. Fill
	// that remainder with the gate+up projections (2/3 of a layer) of the next
	// CPU-bound layers — down stays on CPU — so stranded VRAM becomes expert
	// residency: more experts on GPU (faster) and that weight leaves system RAM
	// (breathing room). Sizing is exact: gate+up is 2/3 of the file-anchored
	// per-layer ROUTED expert bytes — the shared expert is not part of a chunk
	// (it already lives on the layer's owning GPU).
	gateUpChunkMB := 2 * expertCPUPerLayerMB / 3
	remainderMB := make([]int, numGPUs)
	for gi := range caps.GPUs {
		r := roomMBPer[gi] - layersPerGPU[gi]*expertPerLayerMB
		// Expert-only slow-link GPUs are used for whole expert layers only. A
		// partial gate+up pin on a PCIe x1 device creates cross-device expert
		// traffic without making a full layer self-contained.
		if expertOnlyGPU[gi] {
			r = 0
		}
		if r < 0 {
			r = 0
		}
		remainderMB[gi] = r
	}

	var subPins []subExpertPin
	movedOffCPUMB := 0
	if enableAutomaticSubLayerExpertPins &&
		hasMeasuredCUDAOverheadForActiveGPUs(sysCUDAOverheadByGPU, caps.GPUs, split) {
		subPins, movedOffCPUMB = packGateUpChunks(remainderMB, gpuOrder, gateUpChunkMB, moeStartLayer+totalGPULayers, layersCPU)
	}

	// RAM safety check. Sub-layer pins move gate+up off the CPU, so the resident
	// CPU expert footprint shrinks by exactly the packed bytes. CPU layers cost
	// only their routed experts — shared experts stay in VRAM.
	cpuExpertMB := layersCPU*expertCPUPerLayerMB - movedOffCPUMB
	if cpuExpertMB < 0 {
		cpuExpertMB = 0
	}

	// Non-weight RAM overhead, derived per-component (see ramRuntimeOverheadMB):
	// CUDA host staging + graph scratch + mmap page table + CPU activation.
	ramOverheadMB := ramRuntimeOverheadMB(model, s.UBatchSize, totalSizeMB)
	cpuKVMB := 0
	if kvPlacementEffective == "cpu" {
		cpuKVMB = kvTotalMB
	}
	ramNeeded := cpuExpertMB + cpuKVMB + ramOverheadMB + tokenEmbdMB
	ramAvailMB := caps.RAM.FreeMB

	// Mmap decision for MoE — VRAM-aware, no guessed margin.
	//
	// mmap is a question about RAM, not total model size. Expert layers placed
	// on the GPUs live in VRAM and cost zero system RAM, so what matters is the
	// CPU-resident footprint only: CPU-side experts + CPU-side KV + activation/
	// compute overhead (that is exactly ramNeeded). Compare it against the real
	// available RAM (MemAvailable, already net of any user --ram-headroom) with
	// no fudge factor — a guessed % or fixed-MB cushion breaks at scale (trivial
	// on a 1TB box, wasteful as a percentage on a huge model). If the resident
	// footprint fits, load resident (fast, no SSD paging); otherwise mmap while
	// the working set fits. A deliberate reserve is the user's --ram-headroom.
	//
	// The old test keyed off totalSizeMB, which ignored VRAM entirely: a 146GB
	// model with ~40GB of experts on the GPUs (leaving ~100GB on CPU, well under
	// 122GB RAM) was forced onto mmap and paged from SSD for no reason.
	if opts.NoMMap {
		s.MMap = false
	} else if ramNeeded > ramAvailMB {
		// The CPU-resident expert bytes counted in ramNeeded are clean,
		// file-backed mmap pages — evictable and re-fault-able from the model
		// file, so they don't have to all be resident at once. What must
		// actually fit is the runtime's non-reclaimable anonymous memory: KV
		// cache (continuously read/written) plus compute/activation overhead.
		// (Same reasoning as preWorkingSetFloor above; keep both in sync.)
		workingSetFloor := ramOverheadMB + cpuKVMB
		if ramAvailMB >= workingSetFloor {
			s.MMap = true
		}
	} else {
		s.MMap = false
	}

	// Strict mode: build -ot string with precise layer placement.
	// -ot is only needed for multi-GPU (pinning specific layers to specific
	// GPUs). On a single GPU, --n-cpu-moe alone tells llama.cpp the expert
	// boundary — emitting -ot alongside it is redundant and can interfere
	// with the backend's internal expert scheduling.
	if numGPUs > 1 {
		otString := buildOTStringWithSubPins(layersPerGPU, subPins, caps.GPUs, gpuOrder, moeStartLayer, opts.BackendTag)
		if otString != "" {
			s.OTString = otString
		}
	}
	// VRAM-budget-aware --n-cpu-moe: compute the tightest safe value
	// instead of the conservative default. This eliminates the need for
	// the tune engine to spend 3+ rounds discovering the optimal value.
	if layersCPU > 0 && model.ExpertBytes > 0 && model.NumLayers > 0 {
		expertPerLayerMB := bytesToMiBCeil(model.ExpertBytes / int64(model.NumLayers))
		nonExpertMB := bytesToMiBCeil(model.NonExpertBytes)
		if nonExpertMB <= 0 {
			nonExpertMB = totalSizeMB / 10
		}
		computeMB := firstLaunchComputeBufMB(model, s.UBatchSize)
		cudaMB := 300 // conservative per-GPU CUDA context overhead
		freeVRAM := 0
		for _, g := range caps.GPUs {
			freeVRAM += g.VRAMFreeMB()
		}
		gpuBudgetMB := freeVRAM - nonExpertMB - kvTotalMB - computeMB - cudaMB
		gpuBudgetMB = gpuBudgetMB * 90 / 100 // 10% safety margin
		if gpuBudgetMB > 0 && expertPerLayerMB > 0 {
			maxGPUExperts := gpuBudgetMB / expertPerLayerMB
			budgetLayersCPU := model.NumLayers - maxGPUExperts
			if budgetLayersCPU < 0 {
				budgetLayersCPU = 0
			}
			// Use the tighter of the two estimates (original vs VRAM-budget)
			if budgetLayersCPU < layersCPU {
				layersCPU = budgetLayersCPU
			}
		}
		s.NCPUMoE = layersCPU
	} else if layersCPU > 0 {
		s.NCPUMoE = layersCPU
	}

	// MoE models with CPU-resident experts benefit from --no-mmap.
	// mmap causes page faults on every expert access during decode,
	// which dominates latency. Pre-loading experts into RAM gives 2-3x
	// speedup (measured: 20.8 → 46.1 tok/s on Qwen3.6-28B REAP20).
	// The tune engine's moe-mmap candidate can re-enable it if beneficial.
	if s.NCPUMoE > 0 {
		s.MMap = false
	}

	_ = nonExpertPerLayerMB
	return s, nil
}

func maximizeMoEGPUFitByUBatch(base, s *Strategy, err error, caps *detect.Capabilities, model *ModelProfile, totalSizeMB, kvTotalMB int, opts Options) (*Strategy, error) {
	if base == nil || model == nil {
		return s, err
	}
	_, moeCount := moeLayerRange(model)
	if moeCount <= 0 {
		return s, err
	}
	numGPUs := 0
	if caps != nil {
		numGPUs = len(caps.GPUs)
	}
	var gpus []detect.GPU
	if caps != nil {
		gpus = caps.GPUs
	}

	baseExcluded := numGPUsExcluded(s, gpus)
	if err == nil && s != nil && s.NCPUMoE < moeCount && baseExcluded == 0 {
		// The largest ubatch already has a usable whole-layer plan. More GPU
		// experts do not automatically mean a faster service: live MoE tests
		// showed the smaller ubatch/denser placement can lose prompt and parallel
		// throughput. Preserve the largest proven-fit prefill batch.
		return s, nil
	}

	best, bestErr, bestExcluded := s, err, baseExcluded
	bestNCPUMoE := moeCount + 1
	if err == nil && s != nil {
		bestNCPUMoE = s.NCPUMoE
	}

	for _, ub := range UBatchFitLadder {
		if ub >= base.UBatchSize {
			continue
		}
		cand := *base
		cand.UBatchSize = ub
		if cand.BatchSize < ub {
			cand.BatchSize = ub
		}
		next, cerr := buildMoEOffload(&cand, caps, model, totalSizeMB, kvTotalMB, opts)
		if cerr != nil {
			continue
		}
		nextExcluded := numGPUsExcluded(next, gpus)
		if next.NCPUMoE < moeCount && nextExcluded == 0 {
			fmt.Fprintf(os.Stderr,
				"[placement] ubatch %d did not yield a usable whole-layer MoE plan — using ubatch %d instead (%d expert layer(s) on GPU, %d/%d GPUs used)\n",
				base.UBatchSize, ub, moeCount-next.NCPUMoE, numGPUs, numGPUs)
			return next, nil
		}
		if next.NCPUMoE >= bestNCPUMoE && nextExcluded >= bestExcluded {
			continue // not an improvement over the best rung found so far
		}
		reason := "left no VRAM for MoE experts"
		if baseExcluded > 0 {
			reason = fmt.Sprintf("stranded %d of %d GPUs with zero tensor-split share", baseExcluded, numGPUs)
		}
		fmt.Fprintf(os.Stderr,
			"[placement] ubatch %d %s at this context/KV type — using ubatch %d instead (%d expert layer(s) on GPU, %d/%d GPUs used)\n",
			base.UBatchSize, reason, ub, moeCount-next.NCPUMoE, numGPUs-nextExcluded, numGPUs)
		best, bestErr, bestNCPUMoE, bestExcluded = next, nil, next.NCPUMoE, nextExcluded
	}
	return best, bestErr
}

func buildOTStringFromAssignments(assignments []GPUAssignment, gpus []detect.GPU, numLayers int, backendTag string) string {
	var parts []string

	// Match expert weight tensors (routed *_exps, shared *_shexp) AND the
	// per-layer routing tensors (ffn_gate_inp for routed-gate layers,
	// ffn_gate_tid2eid + ffn_exp_probs_b for hash-routed early layers). The
	// routing tensors must ride with their expert weights on the same CUDA
	// device, otherwise llama.cpp's MoE dispatch cannot send the expert
	// compute to that GPU and the layer silently falls back to CPU/GPU0 —
	// leaving the expert GPU idle (e.g. GPU2 at 0% util with 9GB loaded).
	nextLayer := 0
	for _, assign := range assignments {
		if assign.Count <= 0 {
			continue
		}
		start := assign.Start
		last := start + assign.Count - 1
		var layerParts []string
		for l := start; l <= last; l++ {
			layerParts = append(layerParts, fmt.Sprintf("%d", l))
		}
		layerRange := strings.Join(layerParts, "|")
		parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, layerRange, expertTensorPattern, deviceName(backendTag, assign.CUDAIndex)))
		nextLayer += assign.Count
	}
	parts = append(parts, "exps=CPU")
	return strings.Join(parts, ",")
}

func firstLaunchComputeBufMBForGPUParallel(model *ModelProfile, uBatch, parallel, gpuPos int, order []int) int {
	// Every device that owns regular split layers may need the full graph.
	// llama-fit-params measured 8.9 GiB on CUDA0 and 8.7 GiB on CUDA2 for the
	// same DeepSeek-V4 ub256/parallel4 plan. Discounting secondary owners by 50%
	// overcommitted 12 GiB cards. Expert-only devices are capped separately by
	// expertOnlyComputeReserveMB after their classification is known.
	_ = gpuPos
	_ = order
	return firstLaunchComputeBufMBParallel(model, uBatch, parallel)
}

// perGPUVRAMOverheadMB is the non-weight VRAM a single GPU needs at runtime:
// measured CUDA context/allocator overhead plus the compute buffer. Missing
// CUDA probe data is unknown and contributes 0; no static fallback margin is
// hidden here.
func perGPUVRAMOverheadMB(sysProbe *systemProbe, uBatch int) int {
	return measuredCUDAOverheadMB(sysProbe) + firstLaunchComputeBufMB(nil, uBatch)
}

// measuredCUDAOverheadMB is the measured CUDA context/allocator overhead per GPU.
// Missing probe data contributes 0 so callers cannot accidentally depend on a
