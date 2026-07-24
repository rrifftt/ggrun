package placement

import (
	"github.com/rrifftt/ggrun/pkg/detect"
)

func computeCRAM(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int) (int, int) {
	numGPUs := len(caps.GPUs)

	// Fits on GPU? (model fits entirely in VRAM)
	fitsOnGPU := false
	switch s.Type {
	case SingleGPU, MultiGPUDense:
		fitsOnGPU = true
	}

	// RAM remaining after weights load
	var ramAfterLoad int
	if fitsOnGPU {
		ramAfterLoad = caps.RAM.FreeMB
	} else {
		ramOnCPU := totalSizeMB - caps.TotalVRAM()
		if ramOnCPU < 0 {
			ramOnCPU = 0
		}
		ramAfterLoad = caps.RAM.FreeMB - ramOnCPU
		if ramAfterLoad < 0 {
			ramAfterLoad = 0
		}
	}

	// Single-GPU / CPU-only CRAM
	cram := ramAfterLoad / 10
	if cram > 16384 {
		cram = 16384
	}
	if cram < minCramMB {
		cram = 0
	}

	// -1 means "not computed for this strategy type" (single-GPU/CPU-only):
	// leave the backend's own default alone rather than silently emitting a
	// disable decision nothing actually derived. The multi-GPU branch below
	// always overwrites this with a real, headroom-based value (0 included).
	maxCheckpoints := -1

	// Multi-GPU CRAM
	if numGPUs > 1 && s.Type != CPUOnly {
		// TOTAL_VRAM_MB = sum of FREE VRAM
		totalFreeVRAM := 0
		for _, g := range caps.GPUs {
			totalFreeVRAM += g.VRAMFreeMB()
		}
		modelOnGPUMB := totalSizeMB * vramOverheadPercent / 100
		if modelOnGPUMB > totalFreeVRAM {
			modelOnGPUMB = totalFreeVRAM
		}
		vramHeadroom := totalFreeVRAM - modelOnGPUMB - kvTotalMB - computePerGPUMB*numGPUs
		if vramHeadroom < 0 {
			vramHeadroom = 0
		}
		cacheRAMMB := vramHeadroom / 2
		if cacheRAMMB > 4096 {
			cacheRAMMB = 4096
		}
		if cacheRAMMB < 256 {
			cacheRAMMB = 0
			maxCheckpoints = 0
		} else {
			maxCheckpoints = cacheRAMMB / 200
			if maxCheckpoints < 2 {
				maxCheckpoints = 2
			}
			if maxCheckpoints > 16 {
				maxCheckpoints = 16
			}
		}
		cram = cacheRAMMB
	}

	// A zero checkpoint policy makes every append-only agent turn re-evaluate the
	// complete prompt on hybrid or recurrent models. Unlike host prompt CRAM,
	// context checkpoints are the minimum state needed to restore the recurrent
	// prefix. One rolling checkpoint is sufficient with current llama.cpp, which
	// saves at user-message boundaries, and avoids the unsafe backend default of 32.
	if s.HasSSM {
		slots := s.Parallel
		if slots < 1 {
			slots = 1
		}
		if ramAfterLoad >= slots*hybridCheckpointHeadroomPerSlotMB {
			maxCheckpoints = 1
		} else {
			maxCheckpoints = 0
		}
	}

	return cram, maxCheckpoints
}
