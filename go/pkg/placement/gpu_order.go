package placement

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/rrifftt/ggrun/pkg/detect"
)

func orderGPUsByBandwidth(gpus []detect.GPU) []int {
	indices := make([]int, len(gpus))
	for i := range gpus {
		indices[i] = i
	}
	sort.Slice(indices, func(i, j int) bool {
		gi := gpus[indices[i]]
		gj := gpus[indices[j]]
		// Primary: bandwidth DESC
		if gi.BandwidthMBps != gj.BandwidthMBps {
			return gi.BandwidthMBps > gj.BandwidthMBps
		}
		// Secondary: VRAM total DESC
		if gi.VRAMTotalMB != gj.VRAMTotalMB {
			return gi.VRAMTotalMB > gj.VRAMTotalMB
		}
		// Tertiary: PCI index ASC (lower index = closer to CPU)
		return gi.Index < gj.Index
	})
	return indices
}

func normalizeSplit(split []float64) []float64 {
	var total float64
	for _, v := range split {
		total += v
	}
	if total == 0 {
		return split
	}
	for i := range split {
		split[i] = math.Round(split[i]/total*100) / 100
	}
	return split
}

func splitCompactKey(split []float64) string {
	if len(split) == 0 {
		return "0"
	}
	parts := make([]string, len(split))
	for i, v := range split {
		parts[i] = fmt.Sprintf("%.2f", v)
	}
	return strings.Join(parts, ",")
}

func ceilDivInt(n, d int) int {
	if n <= 0 || d <= 0 {
		return 0
	}
	return (n + d - 1) / d
}

func bytesToMiBCeil(n int64) int {
	if n <= 0 {
		return 0
	}
	return int((n + 1048576 - 1) / 1048576)
}
