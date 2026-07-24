package placement

import (
	"crypto/md5"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/rrifftt/ggrun/pkg/detect"
)
type CacheEntry struct {
	GPUAssignments []GPUAssignment `json:"gpu_assignments"`     // cuda_idx:start:count
	OTString       string          `json:"ot_string,omitempty"` // exact -ot (preserves sub-layer pins)
	TensorSplit    []float64       `json:"tensor_split,omitempty"`
	SplitMode      string          `json:"split_mode,omitempty"`
	NCPUMoE        int             `json:"n_cpu_moe"`
	BatchSize      int             `json:"batch_size"`
	UBatchSize     int             `json:"ubatch_size"`
	Parallel       int             `json:"parallel"`
	KVUnified      bool            `json:"kv_unified"`
	NoPinned       bool            `json:"no_pinned"`
	MMap           bool            `json:"mmap"`
}

// GPUAssignment describes layers assigned to a GPU.
type GPUAssignment struct {
	CUDAIndex int `json:"cuda_index"`
	Start     int `json:"start"`
	Count     int `json:"count"`
}


// VRAM and compute sizing constants
const (
	vramOverheadPercent = 130  // model size * this / 100 = estimated VRAM needed
	computePerGPUMB     = 512  // legacy; non-MoE single-GPU sizing only
	computeFloorMB      = 1024 // cited llama.cpp compute floor; CUDA overhead measured separately
	minCramMB           = 512

	// Hybrid and recurrent prompt restoration needs a context checkpoint because
	// its state cannot be shifted like an ordinary transformer KV cache. Keep the
	// policy bounded to one checkpoint per slot and require generous host headroom;
	// measured checkpoints were about 63 MiB for Qwen3.5 and 107 MiB for DeepSeek
	// V4, so 512 MiB per slot leaves room for architecture and allocator variance.
	hybridCheckpointHeadroomPerSlotMB = 512

	// Cards below this fraction of the fastest PCIe link are too slow to own
	// regular layer slots in MoE layer-split mode, but can still be useful as
	// expert-only VRAM when one or more whole expert layers fit.
	// Cards at one-third or less of the fastest link are better used as whole-
	// expert storage. On the measured x16/x4/x1 host, letting the x4 card own a
	// tiny dense split consumed an 8.7 GiB graph and left no complete-expert
	// capacity; expert-only placement produced the fastest stable service.
	expertOnlyMaxBandwidthRatio = 0.33
)

// StrategyType selects how the model is placed.
type StrategyType string

const (
	CPUOnly         StrategyType = "cpu_only"
	SingleGPU       StrategyType = "single_gpu"
	MultiGPUDense   StrategyType = "multi_gpu_dense"
	DenseCPUOffload StrategyType = "dense_cpu_offload"
	MoEOffload      StrategyType = "moe_offload"
)

// Strategy represents the computed placement for a model on this hardware.
type Strategy struct {
	Type        StrategyType `json:"type"`
	ContextSize int          `json:"context_size"`
	GPULayers   int          `json:"gpu_layers"` // always 999; llama-server decides

	TensorSplit []float64 `json:"tensor_split,omitempty"`
	SplitMode   string    `json:"split_mode,omitempty"` // graph, layer, row
	MainGPU     int       `json:"main_gpu,omitempty"`

	KVPlacement string `json:"kv_placement"`        // gpu, cpu, auto
	KVQuality   string `json:"kv_quality"`          // high, mid, low
	KVType      string `json:"kv_type"`             // f16, q8_0, q4_0
	KVTypeK     string `json:"kv_type_k,omitempty"` // asymmetric key cache type
	KVTypeV     string `json:"kv_type_v,omitempty"` // asymmetric value cache type

	NCPUMoE          int    `json:"n_cpu_moe,omitempty"` // for MoE offload
	OTString         string `json:"ot_string,omitempty"` // -ot override-tensor flags
	UseNativeOffload bool   `json:"-"`                   // forces -ot omission if true

	// PlacementCachePath is the keyed file where this exact placement (model +
	// ctx + ubatch + kv placement + backend + GPU set) is persisted, so a load
	// that lands right — or is corrected by OOM-recovery — is reused next launch
	// instead of re-predicted. Runtime-only; not part of the serialized strategy.
	PlacementCachePath string `json:"-"`

	// BackendSupportsFit is true when the backend's --help lists -fit/--fit.
	// ggrun leaves --fit to the backend's native logic, and does not override
	// it with --fit off.
	BackendSupportsFit bool `json:"-"`

	MMap           bool `json:"mmap"`
	MLock          bool `json:"mlock"`
	FlashAttention bool `json:"flash_attention"`

	Threads      int    `json:"threads"`
	BatchSize    int    `json:"batch_size"`
	UBatchSize   int    `json:"ubatch_size"`
	BackendTag   string `json:"backend_tag,omitempty"` // "llama" or "ik_llama"
	IsMoE        bool   `json:"is_moe"`
	ReasoningOff bool   `json:"reasoning_off"` // default off for OpenAI compat

	ThreadsBatch int `json:"threads_batch"` // batch threads (logical cores)
	Parallel     int `json:"parallel,omitempty"`
	CRAM         int `json:"cram,omitempty"` // prompt cache MB

	MaxCheckpoints int  `json:"max_checkpoints,omitempty"`
	UseCUDAGraphs  bool `json:"use_cuda_graphs,omitempty"`

	Host   string       `json:"host,omitempty"`    // listen address
	HasSSM bool         `json:"has_ssm,omitempty"` // SSM/Mamba hybrid flag
	Draft  *DraftConfig `json:"draft,omitempty"`   // speculative decoding config

	MMProjPath   string `json:"mmproj_path,omitempty"` // vision projector GGUF
	MMProjSizeMB int    `json:"-"`                     // mmproj VRAM on primary GPU
}

// ModelProfile describes the GGUF model.
type ModelProfile struct {
	Path              string `json:"path"`
	Name              string `json:"name,omitempty"`         // GGUF metadata: model name
	Basename          string `json:"basename,omitempty"`     // GGUF metadata: model basename
	QuantizedBy       string `json:"quantized_by,omitempty"` // GGUF metadata: quantizer (e.g. "unsloth")
	SizeBytes         int64  `json:"size_bytes"`
	TotalSizeMB       int    `json:"total_size_mb"` // includes multi-part shards
	NumLayers         int    `json:"num_layers"`
	NumParams         int64  `json:"num_params"`
	IsMoE             bool   `json:"is_moe"`
	NumExperts        int    `json:"num_experts,omitempty"`
	ContextSize       int    `json:"context_size"`
	HiddenSize        int    `json:"hidden_size"`
	HeadCount         int    `json:"head_count"`
	HeadCountKV       int    `json:"head_count_kv"`
	KeyLength         int    `json:"key_length"`
	ValueLength       int    `json:"value_length"`
	VocabSize         int    `json:"vocab_size"`
	TokenizerModel    string `json:"tokenizer_model,omitempty"`
	TokenizerPre      string `json:"tokenizer_pre,omitempty"`
	TokenizerHash     string `json:"tokenizer_hash,omitempty"`
	QuantType         string `json:"quant_type"`
	ExpertBytes       int64  `json:"expert_bytes"`
	NonExpertBytes    int64  `json:"non_expert_bytes"`
	TokenEmbdBytes    int64  `json:"token_embd_bytes,omitempty"` // subset of NonExpertBytes; always host-resident
	OutputBytes       int64  `json:"output_bytes,omitempty"`     // subset of NonExpertBytes; whole on the last split device
	ShexpBytes        int64  `json:"shexp_bytes,omitempty"`      // subset of ExpertBytes; stays on the layer's device
	Fused             int    `json:"fused"`
	EmbeddingLength   int    `json:"embedding_length"`
	FeedForwardLength int    `json:"feed_forward_length"`
	KVLoraRank        int    `json:"kv_lora_rank"`
	QLoraRank         int    `json:"q_lora_rank"`
	RopeDim           int    `json:"rope_dim"`
	KeyLengthMLA      int    `json:"key_length_mla"`
	ValueLengthMLA    int    `json:"value_length_mla"`
	HasSSM            int    `json:"has_ssm"`
	SlidingWindow     int    `json:"sliding_window"`
	FullAttnInterval  int    `json:"full_attn_interval"`
	HasShexp          bool   `json:"has_shexp"`
	CTXTrain          int    `json:"ctx_train"`
	ModelArch         string `json:"model_arch"`
	ExpertUsedCount   int    `json:"expert_used_count,omitempty"`

	// MeasuredKVBytesPerTok maps a KV cache type (e.g. "q8_0") to the KV cache
	// bytes-per-token that llama.cpp ACTUALLY allocated on a previous launch of
	// this model, read back from the backend log. It is the ground truth for
	// compressed-attention models (MLA/CSA-HCA/SWA) where the GGUF formula is
	// unreliable; computeKVTotalMB prefers it over the formula when present.
	MeasuredKVBytesPerTok map[string]float64 `json:"-"`

	ExpertFF           int `json:"expert_ff,omitempty"`
	ExpertSharedFF     int `json:"expert_shared_ff,omitempty"`
	LeadingDense       int `json:"leading_dense,omitempty"`
	NextNPredictLayers int `json:"nextn_predict_layers,omitempty"`
}

// Options allows user overrides.
type Options struct {
	ContextSize     int
	KVPlacement     string // auto, gpu, cpu
	KVQuality       string // high, mid, low
	GPUs            []int  // restrict to specific GPUs
	CPUMode         bool
	RamBudgetMB     int
	VRAMHeadroomMB  int // hold back this much total VRAM as a safety margin
	RAMHeadroomMB   int // hold back this much system RAM as a safety margin
	BackendTag      string
	BackendHelp     string // backend --help output; gates turbo KV types // "llama" or "ik_llama"
	BackendCacheTag string // backend identity for probe/cache isolation; defaults to BackendTag
	BackendIdentity string // exact backend build/commit identity for speculative performance profiles
	SamplingProfile string // default, greedy, recommended, or a hash of explicit sampling overrides
	NoMMap          bool
	Parallel        int
	CacheFile       string // path to placement cache for MoE recovery
	CacheDir        string // path to ggrun cache dir (for probes)
	Host            string // listen address (default 127.0.0.1)
	VisionAuto      bool   // auto-detect mmproj for vision
	MMProjPath      string // explicit vision projector GGUF
	SpecMode        string // off, auto, draft, eagle3, dflash, ngram, ngram-mod, ngram-k4v, mtp

	// SpecCandidateValidator asks the selected backend to load a proposed
	// companion without allocating model buffers. GGUF metadata establishes
	// target compatibility; this hook establishes runtime compatibility for
	// backend-specific architectures and quant types before ggrun serves them.
	SpecCandidateValidator func(path string) error
	ForceSpecMoE           bool // allow speculative decoding on MoE despite default gate
	ReasoningOff           bool // emit `--reasoning off` (benchmark/tune only; normal serving keeps the model's thinking)

	// SkipPlacementCache disables loading the keyed .place cache for this Compute.
	// Set during a corrective OOM re-plan so it derives fresh from the penalized
	// VRAM instead of reloading the placement that just OOM'd.
	SkipPlacementCache bool
}

func backendCacheTag(opts Options) string {
	if tag := strings.TrimSpace(opts.BackendCacheTag); tag != "" {
		return tag
	}
	return opts.BackendTag
}

func applyRAMBudget(caps *detect.Capabilities, budgetMB int) *detect.Capabilities {
	if budgetMB <= 0 || caps == nil {
		return caps
	}
	capped := *caps
	capped.RAM.FreeMB = budgetMB
	capped.RAM.TotalMB = budgetMB
	return &capped
}

// parseKVType splits a combined KV string (e.g. "f16-q8_0") into base, K, and V types.
// Compute builds a Strategy from hardware capabilities and model profile.
func Compute(caps *detect.Capabilities, model *ModelProfile, opts Options) (*Strategy, error) {
	var err error
	caps, err = restrictGPUs(caps, opts.GPUs)
	if err != nil {
		return nil, err
	}
	caps = applyRAMBudget(caps, opts.RamBudgetMB)
	caps = detect.ApplyVRAMHeadroom(caps, opts.VRAMHeadroomMB)
	caps = detect.ApplyRAMHeadroom(caps, opts.RAMHeadroomMB)

	// Load any KV cache size llama.cpp measured for this model on a prior launch,
	// so context sizing uses measured truth (exact for compressed attention).
	if model.MeasuredKVBytesPerTok == nil {
		if rates := loadMeasuredKVRates(opts.CacheDir, model); rates != nil {
			model.MeasuredKVBytesPerTok = rates
		}
	}

	s := &Strategy{
		ContextSize:    opts.ContextSize,
		KVPlacement:    opts.KVPlacement,
		KVQuality:      opts.KVQuality,
		MMap:           !opts.NoMMap,
		MLock:          false,
		Threads:        caps.CPU.Cores, // physical cores
		ThreadsBatch:   caps.CPU.Cores, // physical cores
		BackendTag:     opts.BackendTag,
		IsMoE:          model.IsMoE,
		GPULayers:      999,
		FlashAttention: true, // finalized against the resolved KV placement before each return
		// Thinking stays ON for normal serving (backend default `--reasoning auto`);
		// only benchmark/tune opt in to `--reasoning off` for clean, fast measurement.
		ReasoningOff: opts.ReasoningOff,
		// DeepSeek4 uses non-shiftable recurrent memory even though current GGUFs
		// do not expose the generic SSM metadata bit. Treat it like other hybrid
		// models for context shifting and checkpoint restoration.
		HasSSM: model.HasSSM == 1 || strings.EqualFold(model.ModelArch, "deepseek4"),
		Host:   opts.Host,
		// ggrun sets explicit placement (-ngl/-ot/--tensor-split), so the backend's
		// own auto memory-fitting (-fit) is redundant with this explicit plan.
		// Disable it when the backend supports the flag.
		BackendSupportsFit: backendHelpSupports(opts.BackendHelp, "-fit"),
	}

	if s.ContextSize <= 0 {
		s.ContextSize = defaultContextSize(model, caps)
	}
	if opts.KVPlacement == "" {
		s.KVPlacement = "auto"
	}
	if opts.KVQuality == "" {
		// q8_0 KV cache: near-lossless, preserves model quality. The fitting
		// logic falls back to q4_0 only when VRAM genuinely can't hold q8_0.
		s.KVQuality = "mid"
	}
	if opts.Parallel > 0 {
		s.Parallel = opts.Parallel
	}

	// Vision: use an explicit projector, or auto-detect one when --vision is set.
	if opts.MMProjPath != "" {
		if err := validateMMProj(opts.MMProjPath, model.Name, model.Basename); err != nil {
			return nil, err
		}
		s.MMProjPath = opts.MMProjPath
		if fi, err := os.Stat(opts.MMProjPath); err == nil {
			s.MMProjSizeMB = int(fi.Size() / 1024 / 1024)
		}
	} else if opts.VisionAuto && model.Path != "" {
		if path, err := findOrDownloadMMProj(model.Path, opts.CacheDir, model.Name, model.Basename, model.QuantizedBy); err == nil {
			s.MMProjPath = path
			if fi, err := os.Stat(path); err == nil {
				s.MMProjSizeMB = int(fi.Size() / 1024 / 1024)
			}
		} else {
			fmt.Fprintf(os.Stderr, "[vision] %v\n", err)
		}
	}

	// Total size MB (model + mmproj if vision)
	totalSizeMB := model.TotalSizeMB + s.MMProjSizeMB
	if totalSizeMB <= 0 {
		totalSizeMB = int(model.SizeBytes / 1024 / 1024)
	}

	// KV cache type selection. Besides the friendly high/mid/low presets,
	// users can select one of llama.cpp's supported cache types directly (for
	// example q5_1). That exact type must drive both the memory plan and the
	// emitted server flags; treating it as q8_0 here can make a plan needlessly
	// small, while treating an unknown type optimistically can cause an OOM.
	var kvErr error
	s.KVType, kvErr = NormalizeKVType(s.KVQuality)
	if kvErr != nil {
		return nil, fmt.Errorf("KV cache type: %w", kvErr)
	}

	// Resolve KV placement "auto" → concrete value up-front so every caller and
	// every explicit-context retry sees the same placement policy.
	if s.KVPlacement == "auto" || s.KVPlacement == "" {
		if opts.CPUMode || len(caps.GPUs) == 0 {
			s.KVPlacement = "cpu"
		} else {
			sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
			perGPUOH := perGPUVRAMOverheadMB(sysProbe, 0)
			kvNeedMB := computeKVTotalMB(model, s.ContextSize, s.KVType)
			s.KVPlacement = resolveAutoKVPlacement(caps, model, totalSizeMB, kvNeedMB, perGPUOH*len(caps.GPUs))
			// Before accepting CPU KV, try progressively smaller KV types
			// (quality-first: gentlest reduction that fits). KV on GPU is
			// always faster than KV on CPU (~10x bandwidth advantage).
			if s.KVPlacement == "cpu" {
				if ktK, ktV, fits := tryKVDowngradeForGPU(caps, model, totalSizeMB, perGPUOH*len(caps.GPUs), s.ContextSize, s.KVType, opts.BackendHelp); fits {
					s.KVType = ktK
					s.KVTypeK = ktK
					s.KVTypeV = ktV
					s.KVPlacement = "gpu"
				}
			}
		}
	}

	// Auto-fit context: compute both single-GPU and multi-GPU, pick the larger.
	if opts.ContextSize <= 0 {
		if opts.CPUMode || len(caps.GPUs) == 0 {
			cpuCaps := *caps
			cpuCaps.GPUs = nil
			ctx, kvStr := computeAutoContextSize(&cpuCaps, model, totalSizeMB, s.KVType, opts)
			s.ContextSize = ctx
			s.KVType, s.KVTypeK, s.KVTypeV = parseKVType(kvStr)
		} else {
			sysProbe := loadSystemProbe(opts.CacheDir, caps.GPUs)
			// Per-GPU non-weight VRAM overhead, derived per-component (measured
			// CUDA overhead + compute buffer) — no flat guess, no extra margin.
			perGPUOH := perGPUVRAMOverheadMB(sysProbe, 0)
			bestFree := 0
			for _, g := range caps.GPUs {
				if g.VRAMFreeMB() > bestFree {
					bestFree = g.VRAMFreeMB()
				}
			}
			// Single-GPU estimate
			singleCtx, singleKVStr := computeAutoContextSizeSingleGPU(caps, model, totalSizeMB, s.KVType, opts)
			singleKVType, _, _ := parseKVType(singleKVStr)
			singleKVM := computeKVTotalMB(model, singleCtx, singleKVType)
			singleFits := (totalSizeMB+perGPUOH+singleKVM) <= bestFree && singleCtx >= 32768
			// Multi-GPU estimate
			multiCtx, multiKVStr := computeAutoContextSize(caps, model, totalSizeMB, s.KVType, opts)
			multiKVType, _, _ := parseKVType(multiKVStr)
			multiKVM := computeKVTotalMB(model, multiCtx, multiKVType)
			multiFree := 0
			for _, g := range caps.GPUs {
				multiFree += g.VRAMFreeMB()
			}
			multiFits := (totalSizeMB+perGPUOH*len(caps.GPUs)+multiKVM) <= multiFree && multiCtx >= 32768

			if multiFits && multiCtx > singleCtx {
				s.ContextSize = multiCtx
				s.KVType, s.KVTypeK, s.KVTypeV = parseKVType(multiKVStr)
			} else if singleFits {
				s.ContextSize = singleCtx
				s.KVType, s.KVTypeK, s.KVTypeV = parseKVType(singleKVStr)
			} else if multiFits {
				s.ContextSize = multiCtx
				s.KVType, s.KVTypeK, s.KVTypeV = parseKVType(multiKVStr)
			} else {
				// The model doesn't fit wholly in VRAM (a big MoE offloading experts
				// to CPU). Don't collapse to the 32768/q4_0 floor — size the context
				// by where its KV will actually live. --kv-placement drives it:
				// gpu → VRAM-bounded (safe, experts offload); cpu → RAM-bounded
				// (large window); auto → gpu if it fits, else cpu for a big MoE.
				placement := s.KVPlacement
				if placement == "auto" || placement == "" {
					placement = resolveAutoKVPlacement(caps, model, totalSizeMB, computeKVTotalMB(model, s.ContextSize, s.KVType), perGPUOH*len(caps.GPUs))
				}
				s.KVPlacement = placement
				ctx, kvStr := computeAutoContextSizeKVPlacement(caps, model, totalSizeMB, s.KVType, placement, opts)
				s.ContextSize = ctx
				s.KVType, s.KVTypeK, s.KVTypeV = parseKVType(kvStr)
			}
		}
	}

	// Compute KV cache size
	var kvTotalMB int
	if s.KVTypeK != "" && s.KVTypeK != s.KVTypeV {
		kvTotalMB = computeKVTotalMBAsymmetric(model, s.ContextSize, s.KVTypeK, s.KVTypeV)
	} else {
		kvTotalMB = computeKVTotalMB(model, s.ContextSize, s.KVType)
	}

	// Batch sizes based on fit
	bestGPUFree := 0
	for _, g := range caps.GPUs {
		if g.VRAMFreeMB() > bestGPUFree {
			bestGPUFree = g.VRAMFreeMB()
		}
	}
	// Batch tier by exact fit on the best single GPU: model + measured CUDA
	// overhead + KV + that tier's actual compute buffer must fit in VRAM. No
	// percentage guess, no fixed headroom — each tier's compute buffer is the
	// real cost of running that (u)batch (firstLaunchComputeBufMB).
	batchBaseMB := totalSizeMB + measuredCUDAOverheadMB(loadSystemProbe(opts.CacheDir, caps.GPUs)) + kvTotalMB
	switch {
	case batchBaseMB+firstLaunchComputeBufMB(model, 1024) <= bestGPUFree:
		s.BatchSize, s.UBatchSize = 8192, 1024
	case batchBaseMB+firstLaunchComputeBufMB(model, 512) <= bestGPUFree:
		s.BatchSize, s.UBatchSize = 4096, 512
	default:
		s.BatchSize, s.UBatchSize = 2048, 512
	}

	// Persist/reuse this exact placement under a key that includes kv placement,
	// ctx, ubatch, backend, and the GPU set — computed from the now-resolved
	// strategy so the launcher (save) and this load agree byte-for-byte.
	s.PlacementCachePath = PlacementCachePathFor(opts.CacheDir, model, s.ContextSize, s.UBatchSize, s.KVQuality, s.KVPlacement, backendCacheTag(opts), caps.GPUs, s.Parallel, splitCompactKey(s.TensorSplit))
	s.PlacementCachePath = placementCachePathForSpecMode(s.PlacementCachePath, opts.SpecMode)

	// Try cached placement first (MoE only). Prefer the keyed placement cache
	// (remembers a load that landed right or was OOM-corrected); fall back to a
	// tune-cache file if one was selected.
	placeFile := opts.CacheFile
	if s.PlacementCachePath != "" {
		if _, statErr := os.Stat(s.PlacementCachePath); statErr == nil {
			placeFile = s.PlacementCachePath
		}
	}
	if opts.SkipPlacementCache {
		placeFile = ""
	}

	if placeFile != "" && model.IsMoE {
		cache, err := LoadPlacementCache(placeFile, caps, kvTotalMB)
		if err == nil && cache != nil {
			s.Type = MoEOffload
			s.BatchSize = cache.BatchSize
			s.UBatchSize = cache.UBatchSize
			s.Parallel = cache.Parallel
			// The caller's requested slot count wins over what happened to be
			// cached: parallel shapes slots and compute buffers, not the weight
			// layout the cache exists to remember. Without this, a placement
			// cached by a --parallel 1 CLI run would silently strip a multi-slot
			// launch down to a single slot.
			if opts.Parallel > 0 && opts.Parallel != cache.Parallel {
				s.Parallel = opts.Parallel
			}
			s.NCPUMoE = cache.NCPUMoE
			// Restore the cached mmap decision, don't reset it: the entry was
			// saved from a load that landed right, and CACHED_MMAP=0 means it
			// loaded resident (--no-mmap). Resetting to mmap-allowed here made
			// a cache hit silently drop --no-mmap and page 100GB+ of experts
			// from disk. An explicit user --no-mmap still always wins.
			s.MMap = cache.MMap && !opts.NoMMap
			if cache.KVUnified {
				s.KVPlacement = "gpu"
			}
			if len(cache.TensorSplit) > 0 {
				s.TensorSplit = normalizeSplit(cache.TensorSplit)
				s.SplitMode = cache.SplitMode
				if s.SplitMode == "" {
					s.SplitMode = "layer"
				}
			}
			// Prefer the exact cached -ot (preserves sub-layer gate+up pins);
			// fall back to rebuilding from GPU assignments for legacy caches.
			if cache.OTString != "" {
				s.OTString = cache.OTString
			} else if len(cache.GPUAssignments) > 0 {
				otString := buildOTStringFromAssignments(cache.GPUAssignments, caps.GPUs, model.NumLayers, opts.BackendTag)
				if otString != "" {
					s.OTString = otString
				}
			}
			s.FlashAttention = defaultFlashAttention(model, opts, s.KVPlacement)
			// A placement cache stores only target-model placement. Speculative
			// mode is a launch choice and its companion may have appeared since
			// the target cache was written, so resolve it on cache hits too.
			if opts.SpecMode != "" && opts.SpecMode != "off" {
				draftOpts := opts
				draftOpts.ContextSize = s.ContextSize
				s.Draft = ComputeDraft(model, caps, draftOpts)
			}
			// Placement caches intentionally persist only weight placement. Runtime
			// cache policy depends on current free RAM, slot count, and architecture,
			// so recompute it on every launch instead of inheriting the zero-value
			// checkpoint policy from the early cache-hit return.
			s.CRAM, s.MaxCheckpoints = computeCRAM(caps, model, s, totalSizeMB, kvTotalMB)
			if s.Host == "" {
				s.Host = "127.0.0.1"
			}
			return s, nil
		}
	}

	// Strategy selection
	strategy := chooseStrategy(caps, model, s, totalSizeMB, kvTotalMB, opts)
	s.Type = strategy

	// Vision override: mmproj needs extra VRAM — force multi-GPU
	if s.MMProjPath != "" && strategy == SingleGPU && len(caps.GPUs) > 1 {
		if model.IsMoE {
			strategy = MoEOffload
			s.Type = MoEOffload
		} else {
			strategy = MultiGPUDense
			s.Type = MultiGPUDense
		}
	}

	switch strategy {
	case CPUOnly:
		s, err = buildCPUOnly(s, caps, model, opts)
	case SingleGPU:
		s, err = buildSingleGPU(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case MultiGPUDense:
		s, err = buildMultiGPUDense(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case DenseCPUOffload:
		s, err = buildDenseCPUOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
	case MoEOffload:
		preUBatch := *s // buildMoEOffload returns (nil, err) on hard failure, losing s.UBatchSize
		s, err = buildMoEOffload(s, caps, model, totalSizeMB, kvTotalMB, opts)
		s, err = maximizeMoEGPUFitByUBatch(&preUBatch, s, err, caps, model, totalSizeMB, kvTotalMB, opts)
	}
	if err != nil {
		return nil, err
	}

	// OOM guard: refuse if model+KV+compute don't fit (non-MoE only)
	if strategy != MoEOffload {
		if err := checkMemoryOrDie(caps, model, s, totalSizeMB, kvTotalMB, opts); err != nil {
			return nil, err
		}
	}

	if opts.SpecMode != "" && opts.SpecMode != "off" {
		draftOpts := opts
		draftOpts.ContextSize = s.ContextSize
		s.Draft = ComputeDraft(model, caps, draftOpts)
	}

	// Compute CRAM (prompt cache)
	cram, maxCheckpoints := computeCRAM(caps, model, s, totalSizeMB, kvTotalMB)
	s.CRAM = cram
	s.MaxCheckpoints = maxCheckpoints

	// Default host
	if s.Host == "" {
		s.Host = "127.0.0.1"
	}

	s.FlashAttention = defaultFlashAttention(model, opts, s.KVPlacement)
	return s, nil
}

// Target placement differs when a separate speculative model reserves VRAM.
// Keep those successful placements away from the faster spec-off cache (and
// vice versa); otherwise launching DFlash once could permanently CPU-offload
// extra target experts even on later non-speculative launches.
// restrictGPUs filters caps.GPUs to the user-selected device indices (--gpus).
// Devices are renumbered from 0 because the launcher restricts visibility via
// CUDA_VISIBLE_DEVICES / GGML_VK_VISIBLE_DEVICES, so the backend enumerates
// only the selected devices starting at index 0.
func restrictGPUs(caps *detect.Capabilities, want []int) (*detect.Capabilities, error) {
	if caps == nil || len(want) == 0 || len(caps.GPUs) == 0 {
		return caps, nil
	}
	wanted := make(map[int]bool, len(want))
	for _, idx := range want {
		wanted[idx] = true
	}
	filtered := *caps
	filtered.GPUs = nil
	for _, g := range caps.GPUs {
		if wanted[g.Index] {
			gg := g
			gg.Index = len(filtered.GPUs)
			filtered.GPUs = append(filtered.GPUs, gg)
		}
	}
	if len(filtered.GPUs) == 0 {
		return nil, fmt.Errorf("--gpus %v matches no detected GPU (have %d GPUs)", want, len(caps.GPUs))
	}
	return &filtered, nil
}

// chooseStrategy selects the placement strategy from hardware and model size.
func chooseStrategy(caps *detect.Capabilities, model *ModelProfile, s *Strategy, totalSizeMB, kvTotalMB int, opts Options) StrategyType {
	numGPUs := len(caps.GPUs)
	if opts.CPUMode || numGPUs == 0 {
		return CPUOnly
	}

	// Single GPU: model + overhead fits in best GPU
	// Use FREE VRAM (desktop/compositor uses some VRAM)
	bestFreeVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMFreeMB() > bestFreeVRAM {
			bestFreeVRAM = g.VRAMFreeMB()
		}
	}
	// Use measured overhead: model weights + CUDA overhead + compute buffer + KV.
	// Do not subtract a static reserve from free VRAM here; free VRAM is already
	// the measured allocator-visible capacity after resident desktop/process usage.
	// When KV is on CPU, it doesn't consume VRAM — exclude it from the fit check
	// so a model whose weights fit (but weights+KV don't) still gets SingleGPU
	// with all layers on GPU instead of falling through to DenseCPUOffload.
	gpuKVMB := kvTotalMB
	if s.KVPlacement == "cpu" {
		gpuKVMB = 0
	}
	neededMB, _ := EstimateVRAMNeed(model, s.ContextSize, s.UBatchSize, gpuKVMB, caps, opts.CacheDir)
	if neededMB <= bestFreeVRAM {
		return SingleGPU
	}

	// Multi-GPU dense: model fits across ALL GPUs (sum of FREE VRAM)
	if !model.IsMoE {
		totalFreeVRAM := 0
		for _, g := range caps.GPUs {
			totalFreeVRAM += g.VRAMFreeMB()
		}
		neededMBMulti, _ := EstimateVRAMNeed(model, s.ContextSize, s.UBatchSize, gpuKVMB, caps, opts.CacheDir)
		if neededMBMulti <= totalFreeVRAM {
			return MultiGPUDense
		}
	}

	// MoE expert offload
	if model.IsMoE {
		return MoEOffload
	}

	// Dense model with CPU spill
	return DenseCPUOffload
}
// buildMoEOffload computes a fully specified multi-GPU MoE plan: tensor split
// for non-expert/KV tensors plus override-tensor pins for expert tensors.
// UBatchFitLadder are ubatch sizes tried, largest first, when the default
// ubatch's compute buffer leaves literally zero VRAM for any MoE expert
// layer. Flash-attention's compute buffer scales with ubatch at large
// context — measured on DeepSeek-V4 at ctx 1048576, f16 KV, 3-way tensor
// split: ~17-20GB/GPU at ubatch 512 vs ~2.5-2.8GB/GPU at ubatch 64. A
// smaller ubatch trades prefill batch size for GPU-resident experts, which
// is the better trade once every decode step would otherwise stream all
// experts from CPU RAM. Exported so the launch-time preflight (cmd/ggrun)
// can pre-measure every rung via the fit-params oracle before this ladder
// runs, instead of the retry silently falling back to the first-launch
// heuristic for ubatch values that were never actually measured.
var UBatchFitLadder = []int{256, 128, 64}

// Automatic partial expert projection pins are intentionally disabled. Live
// parallel-4 benchmarks showed that gate+up-only packing raised serial decode
// 2.4% but reduced prompt-heavy aggregate throughput 17-19% because every
// partial layer adds a CPU/GPU boundary. Keep the parser/cache support for
// explicit or legacy plans, but the general planner emits complete expert
// layers only until a topology-aware benchmark can prove partial pins faster.
const enableAutomaticSubLayerExpertPins = false

// numGPUsExcluded counts GPUs that got no tensor-split share and no explicit
// expert pins in a multi-GPU MoE placement. A zero split is acceptable for a
// deliberately expert-only slow PCIe GPU; it is only broken when the GPU is
// completely unused.
func numGPUsExcluded(s *Strategy, gpus []detect.GPU) int {
	numGPUs := len(gpus)
	if s == nil || numGPUs <= 1 || len(s.TensorSplit) != numGPUs {
		return 0
	}
	excluded := 0
	for i, v := range s.TensorSplit {
		if v <= 0 && !otStringUsesDevice(s.OTString, gpus[i].Index) {
			excluded++
		}
	}
	return excluded
}

// maximizeMoEGPUFitByUBatch retries buildMoEOffload at a smaller ubatch when
// the default ubatch produces a placement that wastes GPU capacity: (a)
// starves every expert layer off every GPU (NCPUMoE == total MoE layers),
// (b) fails to fit at all — flash-attention's compute buffer can be large
// enough at big contexts that compute+KV alone exceed a GPU's VRAM before any
// expert weight is even considered, surfacing as a hard "does not fit" error
// rather than a soft zero-experts result — or (c) excludes a whole GPU from
// the tensor split even though other GPUs are in use. base carries the
// pre-call strategy fields (ContextSize, KVPlacement, UBatchSize, ...) since
// a hard failure returns (nil, err) and loses them. Each retry reuses
// buildMoEOffload's own measured-or-heuristic compute accounting for that
// ubatch — no new margin is invented, only a different, real ubatch is
// tried — and stops at the largest ladder rung that measurably improves the
// placement, preserving as much prefill batching as the VRAM allows.

// layerOwnership mirrors llama.cpp's tensor-split slot assignment exactly
// (llama-model.cpp): n_layer+1 slots — the repeating layers plus the output
// head — are distributed by upper_bound over the cumulative normalized split.
// Returns the owned repeating-layer count per device and the index of the
// device that owns the output slot (-1 if no device has a share). The input
// layer (token embeddings) always stays on the CPU and owns no slot.
func layerOwnership(split []float64, numLayers int) (owned []int, outputDev int) {
	owned = make([]int, len(split))
	layerDevices, outputDev := layerDeviceAssignments(split, numLayers)
	for _, dev := range layerDevices {
		if dev >= 0 && dev < len(owned) {
			owned[dev]++
		}
	}
	return owned, outputDev
}

// layerDeviceAssignments returns the exact device index for every repeating
// layer plus the output slot. Keeping the per-layer ownership is needed when a
// cost applies only to the MoE suffix (for example shared experts).
func layerDeviceAssignments(split []float64, numLayers int) (layerDevices []int, outputDev int) {
	layerDevices = make([]int, numLayers)
	for i := range layerDevices {
		layerDevices[i] = -1
	}
	outputDev = -1

	sum := 0.0
	for _, v := range split {
		if v > 0 {
			sum += v
		}
	}
	if sum <= 0 || numLayers <= 0 {
		return
	}

	cum := make([]float64, len(split))
	c := 0.0
	for i, v := range split {
		if v > 0 {
			c += v
		}
		cum[i] = c / sum
	}

	slots := numLayers + 1
	for slot := 0; slot < slots; slot++ {
		f := float64(slot) / float64(slots)
		dev := -1
		for i := range cum {
			if cum[i] > f {
				dev = i
				break
			}
		}
		if dev < 0 {
			dev = len(split) - 1
		}
		if slot == numLayers {
			outputDev = dev
		} else {
			layerDevices[slot] = dev
		}
	}
	return layerDevices, outputDev
}

// ownedShareMB charges a device its owned-layer fraction of a per-layer total
// (e.g. the KV cache, which llama.cpp allocates on each layer's device).
func ownedShareMB(totalMB int, owned []int, numLayers, idx int) int {
	if totalMB <= 0 || numLayers <= 0 || idx < 0 || idx >= len(owned) || owned[idx] <= 0 {
		return 0
	}
	return int(math.Ceil(float64(totalMB) * float64(owned[idx]) / float64(numLayers)))
}

// hasMeasuredCUDAOverheadForActiveGPUs gates optional sub-layer squeeze.
// VERIFICATION: cold-cache placement must not treat an unmeasured CUDA context
// as free remainder; no percentage/static reserve is hidden here.
func hasMeasuredCUDAOverheadForActiveGPUs(overheadByGPU map[int]int, gpus []detect.GPU, split []float64) bool {
	if len(gpus) == 0 || len(split) == 0 {
		return false
	}
	for i, g := range gpus {
		if i >= len(split) || split[i] <= 0 {
			continue
		}
		if overheadByGPU[g.Index] <= 0 {
			return false
		}
	}
	return true
}

// ReplanAfterOOM recomputes the full placement after a cudaMalloc OOM, with the
// failed device(s) penalized by how much they overshot. Because it re-runs the
// real packer, the correction is fill-preserving: it refits the failed card
// tightly (partial gate+up chunks, not whole layers) AND reclaims stranded VRAM
// on the other cards via the sub-pin squeeze — so experts move off system RAM
// instead of a blind whole-layer drop that over-corrects and erases the squeeze.
// penaltyMB is keyed by GPU Index and accumulates across retries. Returns the new
// strategy, or an error if there's nothing to replan or it no longer fits.
func ReplanAfterOOM(caps *detect.Capabilities, model *ModelProfile, opts Options, penaltyMB map[int]int) (*Strategy, error) {
	if caps == nil || model == nil || len(penaltyMB) == 0 {
		return nil, fmt.Errorf("replan: nothing to do")
	}
	c2 := *caps
	c2.GPUs = append([]detect.GPU(nil), caps.GPUs...)
	any := false
	for i := range c2.GPUs {
		if p := penaltyMB[c2.GPUs[i].Index]; p > 0 {
			c2.GPUs[i].VRAMUsedMB += p // shrink usable VRAM by the overshoot
			any = true
		}
	}
	if !any {
		return nil, fmt.Errorf("replan: no matching device")
	}
	o := opts
	o.SkipPlacementCache = true // derive fresh, don't reload the placement that OOM'd
	o.CacheFile = ""
	return Compute(&c2, model, o)
}

// currentUBatch reads the launch args' current -ub/--ubatch-size value, or 0
// if unset/unparseable.
func currentUBatch(args []string) int {
	idx := argIndex(args, "-ub", "--ubatch-size")
	if idx < 0 || idx+1 >= len(args) {
		return 0
	}
	v, _ := strconv.Atoi(args[idx+1])
	return v
}

// nextUBatchDown returns the next smaller rung on the same fit ladder
// maximizeMoEGPUFitByUBatch uses at placement time, or ok=false if current is
// already at or below the ladder's floor.
func nextUBatchDown(current int) (int, bool) {
	for _, rung := range UBatchFitLadder {
		if rung < current {
			return rung, true
		}
	}
	return 0, false
}

// DerateCUDAOOMArgs recovers from a cudaMalloc load failure. isComputeBuffer
// distinguishes the two failure classes the caller can observe in the
// backend log: a graph_reserve /gallocr (compute-buffer) OOM scales with
// ubatch, not expert-layer placement, so shrinking ubatch one rung down the
// same ladder used at placement time is tried first; a model-weight
// allocation failure (isComputeBuffer=false) goes straight to moving expert
// layers from the failed device back to CPU, since ubatch has no bearing on
// weight tensor size.
func DerateCUDAOOMArgs(args []string, model *ModelProfile, caps *detect.Capabilities, device, allocMB int, isComputeBuffer bool) ([]string, *CacheEntry, bool) {
	if model == nil || model.NumLayers <= 0 || allocMB <= 0 {
		return nil, nil, false
	}

	if isComputeBuffer {
		if next, ok := nextUBatchDown(currentUBatch(args)); ok {
			newArgs := append([]string(nil), args...)
			setOrAppendArg(&newArgs, "-ub", strconv.Itoa(next))
			// Keep the in-memory Strategy's UBatchSize in sync with serverArgs —
			// applyDeratedPlacementEntry applies this to strategy, which is what
			// the success path persists to the .place cache. Without it, a cache
			// hit later would resurrect the OOM'd, too-large ubatch.
			return newArgs, &CacheEntry{UBatchSize: next}, true
		}
	}

	_, moeLayers := moeLayerRange(model)
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), moeLayers)
	if expertPerLayerMB <= 0 {
		return nil, nil, false
	}

	overshootMB := allocMB
	if caps != nil {
		for _, g := range caps.GPUs {
			if g.Index == device && allocMB > g.VRAMFreeMB() {
				overshootMB = allocMB - g.VRAMFreeMB()
				break
			}
		}
	}
	dropLayers := ceilDivInt(overshootMB, expertPerLayerMB)
	if dropLayers <= 0 {
		dropLayers = 1
	}

	otIdx := argIndex(args, "-ot", "--override-tensor")
	if otIdx < 0 || otIdx+1 >= len(args) {
		return nil, nil, false
	}
	assignments := parseOTAssignments(args[otIdx+1])
	if len(assignments) == 0 {
		return nil, nil, false
	}

	remainingDrop := dropLayers
	actualDrop := 0
	for i := range assignments {
		if assignments[i].CUDAIndex != device || assignments[i].Count <= 0 {
			continue
		}
		drop := remainingDrop
		if drop > assignments[i].Count {
			drop = assignments[i].Count
		}
		assignments[i].Count -= drop
		actualDrop += drop
		remainingDrop -= drop
		if remainingDrop == 0 {
			break
		}
	}
	if actualDrop == 0 {
		return nil, nil, false
	}

	newArgs := append([]string(nil), args...)
	newArgs[otIdx+1] = buildOTStringFromAssignments(assignments, nil, model.NumLayers, "")
	setOrAppendArg(&newArgs, "--n-cpu-moe", strconv.Itoa(currentNCPUMoE(args)+actualDrop))
	entry := cacheEntryFromArgs(newArgs, assignments)
	return newArgs, entry, true
}

func moeLayerRange(model *ModelProfile) (int, int) {
	if model == nil || model.NumLayers <= 0 {
		return 0, 0
	}
	start := model.LeadingDense
	if start < 0 || start >= model.NumLayers {
		start = 0
	}
	count := model.NumLayers - start
	if count <= 0 {
		return 0, model.NumLayers
	}
	return start, count
}

var otAssignmentPattern = regexp.MustCompile(`blk\\\.\(([^)]*)\).*=(?:CUDA|Vulkan)(\d+)`)

func parseOTAssignments(ot string) []GPUAssignment {
	var out []GPUAssignment
	for _, part := range strings.Split(ot, ",") {
		m := otAssignmentPattern.FindStringSubmatch(part)
		if m == nil {
			continue
		}
		device, err := strconv.Atoi(m[2])
		if err != nil {
			continue
		}
		layers := strings.Split(m[1], "|")
		if len(layers) == 0 {
			continue
		}
		start, err := strconv.Atoi(layers[0])
		if err != nil {
			continue
		}
		out = append(out, GPUAssignment{CUDAIndex: device, Start: start, Count: len(layers)})
	}
	return out
}

func argIndex(args []string, names ...string) int {
	for i, arg := range args {
		for _, name := range names {
			if arg == name {
				return i
			}
		}
	}
	return -1
}

func setOrAppendArg(args *[]string, name, value string) {
	if idx := argIndex(*args, name); idx >= 0 {
		if idx+1 < len(*args) {
			(*args)[idx+1] = value
			return
		}
	}
	*args = append(*args, name, value)
}

func currentNCPUMoE(args []string) int {
	idx := argIndex(args, "--n-cpu-moe")
	if idx < 0 || idx+1 >= len(args) {
		return 0
	}
	v, _ := strconv.Atoi(args[idx+1])
	return v
}

func cacheEntryFromArgs(args []string, assignments []GPUAssignment) *CacheEntry {
	entry := &CacheEntry{GPUAssignments: positiveAssignments(assignments)}
	if idx := argIndex(args, "--tensor-split"); idx >= 0 && idx+1 < len(args) {
		entry.TensorSplit = parseTensorSplit(args[idx+1])
	}
	if idx := argIndex(args, "--split-mode"); idx >= 0 && idx+1 < len(args) {
		entry.SplitMode = args[idx+1]
	}
	if idx := argIndex(args, "--n-cpu-moe"); idx >= 0 && idx+1 < len(args) {
		entry.NCPUMoE, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "-b", "--batch-size"); idx >= 0 && idx+1 < len(args) {
		entry.BatchSize, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "-ub", "--ubatch-size"); idx >= 0 && idx+1 < len(args) {
		entry.UBatchSize, _ = strconv.Atoi(args[idx+1])
	}
	if idx := argIndex(args, "--parallel", "-np"); idx >= 0 && idx+1 < len(args) {
		entry.Parallel, _ = strconv.Atoi(args[idx+1])
	}
	entry.MMap = argIndex(args, "--no-mmap") < 0
	// Persist the resolved KV placement so a cache hit re-applies it:
	// without this, no .place cache carries CACHED_KVUNIFIED and the load-side
	// check at placement.go:397 never fires.
	if argIndex(args, "--kv-offload") >= 0 {
		entry.KVUnified = true
	} else if argIndex(args, "--no-kv-offload") >= 0 {
		entry.KVUnified = false
	}
	return entry
}

func positiveAssignments(assignments []GPUAssignment) []GPUAssignment {
	out := make([]GPUAssignment, 0, len(assignments))
	for _, a := range assignments {
		if a.Count > 0 {
			out = append(out, a)
		}
	}
	return out
}

// buildOTString builds the -ot override-tensor string for MoE.
// Builds the -ot override-tensor string: explicit layer list with escaped dots.
const expertTensorPattern = `ffn_((gate_up|up_gate|gate|up|down)_(ch|)exps|(gate_inp|gate|up|down)_shexp|gate_inp|gate_tid2eid|exp_probs_b)`

func buildOTStringFromStart(layersPerGPU []int, gpus []detect.GPU, gpuOrder []int, startLayer int, backendTag string) string {
	var parts []string
	nextLayer := startLayer
	for _, gi := range gpuOrder {
		count := layersPerGPU[gi]
		if count > 0 {
			start := nextLayer
			last := start + count - 1
			cudaIdx := gpus[gi].Index
			// Build explicit layer list, e.g. 0|1|2|...|31
			var layerParts []string
			for l := start; l <= last; l++ {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			layerRange := strings.Join(layerParts, "|")
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, layerRange, expertTensorPattern, deviceName(backendTag, cudaIdx)))
			nextLayer += count
		}
	}
	parts = append(parts, "exps=CPU")
	return strings.Join(parts, ",")
}

// subExpertPin pins one MoE layer's gate+up expert projections (2/3 of the
// layer's expert weight) to a specific GPU. The layer's down projection is left
// unpinned so the exps=CPU catch-all keeps it in system RAM.
type subExpertPin struct {
	Layer int // absolute layer index
	GI    int // position in caps.GPUs
}

// packGateUpChunks fills the VRAM that whole-layer packing floors away. Whole
// layers cost ~expertPerLayerMB each, so each GPU strands up to that much VRAM
// (its remainder). A layer's gate+up projections are 2/3 of its expert weight;
// pinning them onto a GPU with leftover room — down stays on CPU — turns stranded
// VRAM into expert residency: more experts on the GPU, that much less in system
// RAM. Greedy by remainder: each CPU-bound layer goes to the GPU (bandwidth
// order for ties) with the most room that can still hold a gate+up chunk.
// remainderMB is indexed by caps.GPUs position; gpuOrder gives the fill order.
// Returns the pins and the total expert MB moved off the CPU.
func packGateUpChunks(remainderMB []int, gpuOrder []int, gateUpChunkMB, cpuStartLayer, cpuLayerCount int) ([]subExpertPin, int) {
	if gateUpChunkMB <= 0 || cpuLayerCount <= 0 {
		return nil, 0
	}
	rem := make([]int, len(remainderMB))
	copy(rem, remainderMB)
	var pins []subExpertPin
	movedMB := 0
	for i := 0; i < cpuLayerCount; i++ {
		best := -1
		for _, gi := range gpuOrder {
			if gi < 0 || gi >= len(rem) {
				continue
			}
			if rem[gi] >= gateUpChunkMB && (best < 0 || rem[gi] > rem[best]) {
				best = gi
			}
		}
		if best < 0 {
			break // no GPU can hold another gate+up chunk
		}
		pins = append(pins, subExpertPin{Layer: cpuStartLayer + i, GI: best})
		rem[best] -= gateUpChunkMB
		movedMB += gateUpChunkMB
	}
	return pins, movedMB
}

// buildOTStringWithSubPins is buildOTStringFromStart plus optional sub-layer
// gate+up pins. Whole-layer pins come first, then the gate+up pins, then the
// exps=CPU catch-all — first-match-wins (see llama.cpp arg.cpp) keeps each
// partial layer's gate+up on its GPU while down falls through to CPU. With no
// sub-pins the output is identical to buildOTStringFromStart.
func buildOTStringWithSubPins(layersPerGPU []int, subPins []subExpertPin, gpus []detect.GPU, gpuOrder []int, startLayer int, backendTag string) string {
	var parts []string

	// Match expert weight tensors (routed *_exps, shared *_shexp) AND the
	// per-layer routing tensors (ffn_gate_inp for routed-gate layers,
	// ffn_gate_tid2eid + ffn_exp_probs_b for hash-routed early layers). The
	// routing tensors must ride with their expert weights on the same CUDA
	// device, otherwise llama.cpp's MoE dispatch cannot send the expert
	// compute to that GPU and the layer silently falls back to CPU/GPU0 —
	// leaving the expert GPU idle (e.g. GPU2 at 0% util with 9GB loaded).
	gateUpPattern := `ffn_(gate_up|up_gate|gate|up)_(ch|)exps`

	nextLayer := startLayer
	for _, gi := range gpuOrder {
		count := layersPerGPU[gi]
		if count > 0 {
			start := nextLayer
			last := start + count - 1
			var layerParts []string
			for l := start; l <= last; l++ {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, strings.Join(layerParts, "|"), expertTensorPattern, deviceName(backendTag, gpus[gi].Index)))
			nextLayer += count
		}
	}

	// Group sub-pins by GPU, preserving first-seen order, one compact rule per
	// GPU, emitted before the exps=CPU catch-all.
	if len(subPins) > 0 {
		byGPU := map[int][]int{}
		var order []int
		for _, p := range subPins {
			if _, ok := byGPU[p.GI]; !ok {
				order = append(order, p.GI)
			}
			byGPU[p.GI] = append(byGPU[p.GI], p.Layer)
		}
		for _, gi := range order {
			var layerParts []string
			for _, l := range byGPU[gi] {
				layerParts = append(layerParts, fmt.Sprintf("%d", l))
			}
			parts = append(parts, fmt.Sprintf(`blk\.(%s)\.%s.*=%s`, strings.Join(layerParts, "|"), gateUpPattern, deviceName(backendTag, gpus[gi].Index)))
		}
	}

	parts = append(parts, "exps=CPU")
	return strings.Join(parts, ",")
}
func otStringUsesDevice(ot string, index int) bool {
	return strings.Contains(ot, fmt.Sprintf("=CUDA%d", index)) ||
		strings.Contains(ot, fmt.Sprintf("=Vulkan%d", index))
}

func deviceName(backendTag string, index int) string {
	if strings.EqualFold(backendTag, "vulkan") {
		return fmt.Sprintf("Vulkan%d", index)
	}
	return fmt.Sprintf("CUDA%d", index)
}

// computeKVTotalMB calculates exact KV cache size.
func kvBytesPerElem(kvType string) float64 {
	switch kvType {
	case "q4_0":
		return 0.5625
	case "q8_0":
		return 1.0625
	case "f16":
		return 2.0
	default:
		return 1.0625
	}
}
// NormalizeKVType resolves ggrun's quality presets and the cache types accepted
// by llama.cpp's --cache-type-k/--cache-type-v flags. The returned spelling is
// safe to pass straight to llama-server and to use as the probe/cache key.
func NormalizeKVType(value string) (string, error) {
	typeName := strings.ToLower(strings.TrimSpace(value))
	typeName = strings.TrimPrefix(typeName, "ggml_")
	switch typeName {
	case "", "mid":
		return "q8_0", nil
	case "high":
		return "f16", nil
	case "low":
		return "q4_0", nil
	case "fp16":
		return "f16", nil
	case "fp32":
		return "f32", nil
	case "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1",
		"turbo4", "turbo3", "turbo2", "turbo1":
		return typeName, nil
	default:
		return "", fmt.Errorf("unsupported type %q (use high, mid, low, f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, turbo4, turbo3, turbo2, or turbo1)", value)
	}
}

func exactKVTypeRequested(quality string) bool {
	switch strings.ToLower(strings.TrimSpace(quality)) {
	case "", "high", "mid", "low":
		return false
	default:
		return true
	}
}

func fallbackKVType(preferred, quality string) string {
	if exactKVTypeRequested(quality) {
		return preferred
	}
	return "q4_0"
}

func kvTypeBytesPerElement(kvType string) (float64, bool) {
	typeName, err := NormalizeKVType(kvType)
	if err != nil {
		return 0, false
	}
	switch typeName {
	case "f32":
		return 4, true
	case "f16", "bf16":
		return 2, true
	case "q8_0":
		return 1.0625, true // 34-byte block / 32 values
	case "q5_1":
		return 0.75, true // 24-byte block / 32 values
	case "q5_0":
		return 0.6875, true // 22-byte block / 32 values
	case "q4_1":
		return 0.625, true // 20-byte block / 32 values
	case "q4_0", "iq4_nl":
		return 0.5625, true // 18-byte block / 32 values
	case "turbo4":
		return 0.5, true // turboquant 4-bit
	case "turbo3":
		return 0.375, true // turboquant 3-bit
	case "turbo2":
		return 0.25, true // turboquant 2-bit
	case "turbo1":
		return 0.125, true // turboquant 1-bit
	default:
		return 0, false
	}
}

func expandKVTypes(preferredKVType string, opts Options) [][2]string {
	quality := opts.KVQuality
	if quality == "" {
		quality = "mid"
	}
	seen := make(map[string]bool)
	var result [][2]string
	addIfNew := func(k, v string) {
		key := k + "/" + v
		if !seen[key] {
			seen[key] = true
			result = append(result, [2]string{k, v})
		}
	}
	addIfNew(preferredKVType, preferredKVType)
	addIfNew("q8_0", "q8_0")
	addIfNew("q4_0", "q4_0")
	switch quality {
	case "mid":
		addIfNew("f16", "q8_0")
	case "low":
		addIfNew("q8_0", "q4_0")
		addIfNew("f16", "q8_0")
	}
	return result
}

func (s *Strategy) kvTypeKOrDefault() string {
	if s.KVTypeK != "" {
		return s.KVTypeK
	}
	return s.KVType
}

func (s *Strategy) kvTypeVOrDefault() string {
	if s.KVTypeV != "" {
		return s.KVTypeV
	}
	return s.KVType
}
// kvReserveByBandwidth distributes KV cache proportionally to free VRAM.
// KV reads are VRAM-local — PCIe bandwidth does not affect KV access speed.
func kvReserveByBandwidth(kvTotalMB int, gpus []detect.GPU, order []int, kvPerLayerMB int) []int {
	reserve := make([]int, len(gpus))
	totalFree := 0
	for _, g := range gpus {
		totalFree += g.VRAMFreeMB()
	}
	if kvTotalMB <= 0 || totalFree <= 0 {
		return reserve
	}
	useOrder := order
	if len(useOrder) == 0 {
		useOrder = seqRange(len(gpus))
	}
	for _, gi := range useOrder {
		share := (kvTotalMB*gpus[gi].VRAMFreeMB() + totalFree - 1) / totalFree
		if kvPerLayerMB > 0 {
			share = ((share + kvPerLayerMB - 1) / kvPerLayerMB) * kvPerLayerMB
		}
		reserve[gi] = share
	}
	return reserve
}

func seqRange(n int) []int {
	r := make([]int, n)
	for i := range r {
		r[i] = i
	}
	return r
}
// gpuSplitWeight returns the weight applied to a GPU's free VRAM when
// computing the tensor-split for MoE offload. Under --split-mode layer the
// GPU that owns a layer's non-expert weights computes that layer, so
// CPU-resident experts stream over that GPU's PCIe link. Weighting by PCIe
// bandwidth concentrates ownership on fast-link GPUs. Returns 1.0 when
// bandwidth is unknown so the split degenerates to free-VRAM-proportional.
func gpuSplitWeight(g detect.GPU) float64 {
	if g.BandwidthMBps <= 0 {
		return 1.0
	}
	return float64(g.BandwidthMBps)
}

// expertOnlySlowGPUs classifies GPUs as expert-only devices: they must not own
// regular layer slots, but their VRAM can still hold whole expert layers. A GPU
// is an expert-only candidate when EITHER (a) its PCIe link is very slow
// relative to the fastest GPU (bandwidth ratio <= expertOnlyMaxBandwidthRatio),
// so owning dense layers would bottleneck the layer pipeline, OR (b) it cannot
// fit the split-owner compute reserve plus one dense layer's non-expert weight
// (capacity trigger), so a dense split slot would be wasted on it. The
// classification uses the normal split-owner reserve to make sure at least one
// true layer owner remains, and the expert-only reserve to decide whether a
// candidate card can carry at least one complete expert layer.
func expertOnlySlowGPUs(gpus []detect.GPU, splitFixedPerGPU, expertOnlyFixedPerGPU []int, expertPerLayerMB, nonExpertPerLayerMB int) []bool {
	expertOnly := make([]bool, len(gpus))
	if len(gpus) <= 1 || expertPerLayerMB <= 0 {
		return expertOnly
	}

	maxBandwidth := 0
	for _, g := range gpus {
		if g.BandwidthMBps > maxBandwidth {
			maxBandwidth = g.BandwidthMBps
		}
	}
	if maxBandwidth <= 0 {
		return expertOnly
	}

	splitCandidates := 0
	for i, g := range gpus {
		if i < len(splitFixedPerGPU) && g.VRAMFreeMB() > splitFixedPerGPU[i] {
			splitCandidates++
		}
	}

	candidates := make([]int, 0, len(gpus))
	for i, g := range gpus {
		if i >= len(splitFixedPerGPU) || i >= len(expertOnlyFixedPerGPU) || g.BandwidthMBps <= 0 {
			continue
		}
		slowLink := float64(g.BandwidthMBps)/float64(maxBandwidth) <= expertOnlyMaxBandwidthRatio
		cantFitDense := nonExpertPerLayerMB > 0 && g.VRAMFreeMB()-splitFixedPerGPU[i] < nonExpertPerLayerMB
		if !slowLink && !cantFitDense {
			continue
		}
		if g.VRAMFreeMB()-expertOnlyFixedPerGPU[i] < expertPerLayerMB {
			continue
		}
		candidates = append(candidates, i)
	}

	sort.Slice(candidates, func(i, j int) bool {
		gi := gpus[candidates[i]]
		gj := gpus[candidates[j]]
		if gi.BandwidthMBps != gj.BandwidthMBps {
			return gi.BandwidthMBps < gj.BandwidthMBps
		}
		if gi.VRAMFreeMB() != gj.VRAMFreeMB() {
			return gi.VRAMFreeMB() > gj.VRAMFreeMB()
		}
		return gi.Index < gj.Index
	})

	for _, i := range candidates {
		if splitCandidates <= 1 {
			break
		}
		expertOnly[i] = true
		splitCandidates--
	}
	return expertOnly
}

// expertOnlyComputeReserveMB is the compute margin for a GPU that will only run
// explicitly pinned experts. It keeps measured/recorded CUDA and runtime growth
// outside this helper and caps the prompt-graph reserve at llama.cpp's compute
// floor, because an expert-only GPU does not own regular prompt-processing
// layer slots or KV. The probe cache's per-GPU compute buffer is measured for
// the placement that actually ran (split-owner with dense layers), which is
// dramatically larger than what an expert-only GPU needs (e.g. DeepSeek-V4:
// ~9.8 GB split-owner vs ~299 MB expert-only on the same card). Using the
// split-owner value for an expert-only GPU blocks it from receiving any expert
// layers at all. Cap at the compute floor: it is a conservative reserve for
// the small expert-projection graph, and the preflight gate catches any real
// overflow before the load.
func expertOnlyComputeReserveMB(splitOwnerComputeMB int) int {
	return computeFloorMB
}

// modelAwareHeadroom estimates the non-weight VRAM/RAM the runtime needs beyond
// the model weights (prompt-graph compute buffer + a small runtime-growth
// margin). Replaces the flat 8 GiB guess previously hard-coded in the auto
// context-size paths so large models reserve enough and small models don't
// waste context capacity.
// firstLaunchComputeBufMB is a conservative compute-buffer reservation used until
// the post-launch probe measures the real value for this model + settings. The
// prompt-processing graph scales with ubatch AND model shape: activation working
// set per token is roughly proportional to hidden_size * num_layers. The old flat
// ~4 MiB/ubatch estimate (ub 512 -> 2048 MiB) under-reserved large MoE graphs by
// ~4.4x — once expert-packing filled the GPU, llama.cpp's compute-buffer alloc
// OOM'd ("failed to create context", V4 needs ~9020 MiB at ub 512). We now size
// from the model: bytes per (token·hidden·layer) ≈ 42, calibrated so V4 reserves
// ~9412 MiB (covers the measured ~9020). Over-estimating is safe — the probe cache
// overrides this with the measured value after the first launch; under-estimating
// is fatal (OOM crash). A nil model falls back to the old per-ubatch heuristic.
// firstLaunchComputeBufMBParallel estimates the graph reserve for a cold-cache
// launch. MoE routed-expert activation grows with experts selected per token;
// parallel slots divide the physical ubatch graph approximately evenly. The
// 128-byte fan-out coefficient is calibrated against llama-fit-params for
// DeepSeek-V4 (6 active experts: 33.9 GiB at ub256/parallel1 and 8.9 GiB at
// ub256/parallel4). Dense/unknown models keep the prior 42-byte coefficient.
// perGPUVRAMOverheadMB is the non-weight VRAM a single GPU needs at runtime:
// measured CUDA context/allocator overhead plus the compute buffer. Missing
// CUDA probe data is unknown and contributes 0; no static fallback margin is
// hidden here.

// measuredCUDAOverheadMB is the measured CUDA context/allocator overhead per GPU.
// Missing probe data contributes 0 so callers cannot accidentally depend on a
// fabricated reserve.
// ramRuntimeOverheadMB is the non-weight system RAM the runtime needs: CUDA host
// pinned staging, compute-graph scratch, the mmap page table, and CPU activation
// buffers. Each term is derived from model dims / file size rather than a flat
// guessed reserve. (mmapPT is exact: a 4KB-page table is fileSize/4096*8B ≈
// fileSize/512. The host/graph terms are the last constants still pending a
// measurement probe — kept here as the single place to swap in measured values.)
func ramRuntimeOverheadMB(model *ModelProfile, uBatch, totalSizeMB int) int {
	const cudaHostMB = 1024
	const graphScratchMB = 2048
	mmapPTMB := totalSizeMB / 500

	actFFN := model.FeedForwardLength
	if model.NumExperts > 0 && model.ExpertUsedCount > 0 && model.ExpertFF > 0 {
		actFFN = model.ExpertUsedCount * model.ExpertFF
		if model.ExpertSharedFF > 0 {
			actFFN += model.ExpertSharedFF
		}
	}
	if model.KVLoraRank > 0 {
		actFFN += model.KVLoraRank + model.QLoraRank
	}
	cpuActMB := uBatch * (model.EmbeddingLength + actFFN) * 4 * 2 / 1048576
	if cpuActMB < 64 {
		cpuActMB = 64
	}
	return cudaHostMB + graphScratchMB + mmapPTMB + cpuActMB
}

// checkMemoryOrDie refuses to launch when model + KV + compute buffers exceed the pool.
// OOM guard: refuse to launch if model+KV+compute don't fit.
// computeCRAM calculates prompt cache size from remaining memory after load.
// defaultFlashAttention decides whether ggrun forces `--flash-attn on`. The
// decision must follow the resolved KV placement because the CUDA FA kernel
// requires GPU-resident KV.
func defaultFlashAttention(model *ModelProfile, opts Options, kvPlacement string) bool {
	// llama.cpp auto-disables flash attention whenever the KV cache isn't
	// GPU-resident — the FA CUDA kernel
	// needs its KV tensor on the same device doing the attention compute.
	// Claiming FlashAttention here when kvPlacement=="cpu" would emit a
	// self-contradicting `--flash-attn on --no-kv-offload` command; for
	// deepseek4 specifically that also silently re-opens the unbounded
	// compute-buffer growth this flag exists to prevent (see [[Task #10]]).
	return kvPlacement != "cpu"
}

func defaultContextSize(model *ModelProfile, caps *detect.Capabilities) int {
	if model.ContextSize > 0 && model.ContextSize < 32768 {
		return model.ContextSize
	}
	return 32768
}

// computeAutoContextSizeSingleGPU computes the largest context that fits on
// a SINGLE GPU (the best one). Used to prefer single-GPU mode (faster).
func computeAutoContextSizeSingleGPU(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Find best single GPU by free VRAM (accounts for desktop/compositor usage)
	bestVRAM := 0
	for _, g := range caps.GPUs {
		if g.VRAMFreeMB() > bestVRAM {
			bestVRAM = g.VRAMFreeMB()
		}
	}

	// Total hardware for single GPU: best GPU VRAM + up to 4GB RAM (not entire system)
	// Single GPU context shouldn't use entire system RAM — the model must fit on ONE GPU.
	totalHWMB := bestVRAM + 4096

	// Fixed overhead: model weights + dynamic VRAM headroom.
	dynamicHeadroom := totalSizeMB / 5
	if dynamicHeadroom < 4096 {
		dynamicHeadroom = 4096
	}
	if dynamicHeadroom > 8192 {
		dynamicHeadroom = 8192
	}
	fixedOverheadMB := totalSizeMB + dynamicHeadroom

	// If model doesn't fit at all, return minimum
	if totalHWMB <= fixedOverheadMB {
		return 32768, preferredKVType
	}

	// KV budget = total hardware - model - headroom
	kvBudgetMB := totalHWMB - fixedOverheadMB
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	kvPairs := expandKVTypes(preferredKVType, opts)
	for _, kvPair := range kvPairs {
		refCtx := 32768
		var refKVTotalMB int
		if kvPair[0] == kvPair[1] {
			refKVTotalMB = computeKVTotalMB(model, refCtx, kvPair[0])
		} else {
			refKVTotalMB = computeKVTotalMBAsymmetric(model, refCtx, kvPair[0], kvPair[1])
		}
		if refKVTotalMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVTotalMB) * 1048576.0 / float64(refCtx)
		maxCtxRaw := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)
		hwCapCtx := maxCtxRaw
		if model.CTXTrain > 0 && model.CTXTrain < hwCapCtx {
			hwCapCtx = model.CTXTrain
		}
		powerOfTwoValues := []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}
		suggestedCtx := 32768
		for _, c := range powerOfTwoValues {
			if c <= hwCapCtx {
				suggestedCtx = c
			}
		}
		if suggestedCtx >= 32768 {
			if kvPair[0] == kvPair[1] {
				return suggestedCtx, kvPair[0]
			}
			return suggestedCtx, kvPair[0] + "-" + kvPair[1]
		}
	}
	return 32768, fallbackKVType(preferredKVType, opts.KVQuality)
}

// resolveAutoKVPlacement decides gpu vs cpu for the KV cache when --kv-placement
// is "auto". A model that fits in VRAM keeps its KV on GPU (fast, VRAM to spare).
// A big MoE whose experts must offload to CPU puts KV on CPU instead: that frees
// VRAM for more expert layers (the decode-bandwidth bottleneck) and unlocks a much
// larger context. A dense model bigger than VRAM keeps KV on GPU (its only spot).
// tryKVDowngradeForGPU finds the highest-quality KV cache type combination
// that keeps KV on GPU when the current type doesn't fit. KV on GPU is
// always faster than KV on CPU (~10x bandwidth advantage), but aggressive
// quantization degrades model output quality. This function tries the
// gentlest reduction first (preserving K quality, then V) and only goes
// more aggressive if needed.
//
// For MoE models, only non-expert weights (attention, norms, embeddings)
// are checked against the VRAM budget — expert FFN weights live on CPU
// via --n-cpu-moe and don't compete for VRAM.
//
// Returns (kvTypeK, kvTypeV, fitsOnGPU). kvTypeV empty means symmetric.
func tryKVDowngradeForGPU(caps *detect.Capabilities, model *ModelProfile, totalSizeMB, vramOverheadMB, ctxSize int, currentKVType, backendHelp string) (string, string, bool) {
	freeVRAM := 0
	for _, g := range caps.GPUs {
		freeVRAM += g.VRAMFreeMB()
	}
	if freeVRAM <= 0 {
		return currentKVType, "", false
	}
	fitOnGPU := int(float64(freeVRAM-vramOverheadMB) / 0.90)

	// VRAM the model occupies on GPU. For MoE, only non-expert weights
	// (attention, norms, embeddings) are GPU-resident; expert FFN weights
	// are on CPU via --n-cpu-moe.
	modelOnGPU := totalSizeMB
	if model.IsMoE && model.NonExpertBytes > 0 {
		modelOnGPU = bytesToMiBCeil(model.NonExpertBytes)
	}
	if modelOnGPU > fitOnGPU {
		return currentKVType, "", false
	}

	// Quality-first ladder: gentlest reduction first. Keys are more
	// quantization-sensitive than values, so we keep K at higher quality
	// and compress V more aggressively before touching K.
	type kvCombo struct{ k, v string }
	combos := []kvCombo{
		{"q8_0", "q5_1"}, // K lossless, gentle V reduction    (~15% savings)
		{"q8_0", "q4_1"}, // K lossless, moderate V reduction  (~21%)
		{"q8_0", "q4_0"}, // K lossless, aggressive V          (~24%)
		{"q5_1", "q4_1"}, // gentle K, moderate V              (~35%)
		{"q5_0", "q4_0"}, // moderate both                     (~41%)
		{"q4_0", "q4_0"}, // aggressive both, smallest universal (~47%)
	}
	// Turbo types are gated on backend support (turboquant forks only).
	// Placed last: they save the most VRAM but quality is backend-dependent.
	if strings.Contains(backendHelp, "turbo") {
		combos = append(combos,
			kvCombo{"turbo4", "turbo4"}, // symmetric turbo  (~53%)
			kvCombo{"turbo4", "turbo3"}, // asymmetric turbo (~59%)
		)
	}
	for _, c := range combos {
		// Skip if this combo isn't a downgrade from the current type
		if c.k == currentKVType && c.v == currentKVType {
			continue
		}
		tryKVMB := computeKVTotalMBAsymmetric(model, ctxSize, c.k, c.v)
		if modelOnGPU+tryKVMB <= fitOnGPU {
			return c.k, c.v, true
		}
	}
	return currentKVType, "", false
}
// computeAutoContextSizeKVPlacement computes the largest context whose KV cache
// fits in the memory implied by placement: VRAM for "gpu", system RAM for "cpu".
// For a MoE, "gpu" keeps only the non-expert weights on GPU and reserves the rest
// of VRAM for KV (experts offload to CPU), while "cpu" leaves VRAM for experts and
// puts the (large) KV in RAM. This is what makes --kv-placement drive the context
// ceiling instead of a fixed VRAM+RAM budget that can overflow a GPU-pinned KV.
// computeAutoContextSize computes the largest context that fits in available
// hardware memory.
// Uses TOTAL_VRAM + RAM_AVAIL.
func computeAutoContextSize(caps *detect.Capabilities, model *ModelProfile, totalSizeMB int, preferredKVType string, opts Options) (int, string) {
	// Total hardware = all GPU VRAM + free RAM
	totalVRAM := 0
	for _, g := range caps.GPUs {
		totalVRAM += g.VRAMTotalMB
	}
	totalHWMB := totalVRAM + caps.RAM.FreeMB

	// Fixed overhead: model weights + dynamic VRAM headroom.
	dynamicHeadroom := totalSizeMB / 5
	if dynamicHeadroom < 4096 {
		dynamicHeadroom = 4096
	}
	if dynamicHeadroom > 8192 {
		dynamicHeadroom = 8192
	}
	fixedOverheadMB := totalSizeMB + dynamicHeadroom

	// If model doesn't fit at all, return minimum
	if totalHWMB <= fixedOverheadMB {
		return 32768, preferredKVType
	}

	// KV budget = total hardware - model - headroom
	kvBudgetMB := totalHWMB - fixedOverheadMB
	if kvBudgetMB <= 0 {
		return 32768, preferredKVType
	}

	kvPairs := expandKVTypes(preferredKVType, opts)
	for _, kvPair := range kvPairs {
		refCtx := 32768
		var refKVTotalMB int
		if kvPair[0] == kvPair[1] {
			refKVTotalMB = computeKVTotalMB(model, refCtx, kvPair[0])
		} else {
			refKVTotalMB = computeKVTotalMBAsymmetric(model, refCtx, kvPair[0], kvPair[1])
		}
		if refKVTotalMB <= 0 {
			continue
		}
		kvBytesPerToken := float64(refKVTotalMB) * 1048576.0 / float64(refCtx)
		maxCtxRaw := int(float64(kvBudgetMB) * 1048576.0 / kvBytesPerToken)
		hwCapCtx := maxCtxRaw
		if model.CTXTrain > 0 && model.CTXTrain < hwCapCtx {
			hwCapCtx = model.CTXTrain
		}
		powerOfTwoValues := []int{32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304}
		suggestedCtx := 32768
		for _, c := range powerOfTwoValues {
			if c <= hwCapCtx {
				suggestedCtx = c
			}
		}
		if suggestedCtx >= 32768 {
			if kvPair[0] == kvPair[1] {
				return suggestedCtx, kvPair[0]
			}
			return suggestedCtx, kvPair[0] + "-" + kvPair[1]
		}
	}

	// Preset qualities may fall back to the compact type. An exact llama.cpp
	// type is user-owned, so preserve it and lower context instead.
	return 32768, fallbackKVType(preferredKVType, opts.KVQuality)
}

// Args converts a Strategy into llama-server command-line arguments.
func (s *Strategy) Args(modelPath string, port int) []string {
	host := s.Host
	if host == "" {
		host = "127.0.0.1"
	}
	args := []string{
		"-m", modelPath,
		"--host", host,
		"--port", fmt.Sprintf("%d", port),
		"--ctx-size", fmt.Sprintf("%d", s.ContextSize),
	}

	// NEW — emit whenever KV is GPU-resident, regardless of the stored bool:
	if s.FlashAttention || s.KVPlacement == "gpu" {
		args = append(args, "--flash-attn", "on")
	}

	args = append(args,
		"-b", fmt.Sprintf("%d", s.BatchSize),
		"-ub", fmt.Sprintf("%d", s.UBatchSize),
		"--cache-type-k", s.kvTypeKOrDefault(),
		"--cache-type-v", s.kvTypeVOrDefault(),
		"--jinja",
		"--threads", fmt.Sprintf("%d", s.Threads),
		"--threads-batch", fmt.Sprintf("%d", s.ThreadsBatch),
	)

	if s.KVPlacement == "cpu" {
		args = append(args, "--no-kv-offload")
	} else if s.KVPlacement == "gpu" {
		args = append(args, "--kv-offload")
	}

	if s.ReasoningOff {
		args = append(args, "--reasoning", "off")
	}

	// SSM/Mamba models need --no-context-shift
	if s.HasSSM {
		args = append(args, "--no-context-shift")
	}

	if s.Parallel > 0 {
		args = append(args, "--parallel", fmt.Sprintf("%d", s.Parallel))
	} else {
		args = append(args, "--parallel", "1")
	}

	// Vision support: auto-detected mmproj
	if s.MMProjPath != "" {
		args = append(args, "--mmproj", s.MMProjPath)
	}

	// GPU offloading. CPU-only still
	// prints -ngl 0 so compatibility tests and user scripts can see the mode.
	if s.Type == CPUOnly {
		args = append(args, "-ngl", "0")
	} else if s.Type == DenseCPUOffload && s.BackendSupportsFit {
		// Keep n_gpu_layers and tensor_split unset. Backend fit uses exact GGUF
		// tensor sizes to choose a safe GPU/CPU layer boundary at the requested
		// context. Explicit values make llama.cpp fit abort without changing them.
		args = append(args, "--fit", "on")
	} else if len(s.TensorSplit) > 0 || s.Type != CPUOnly {
		args = append(args, "-ngl", "999")
		// ggrun manages placement explicitly (-ngl/-ot/--tensor-split). Disable
		// the backend's auto-fit so it doesn't abort on the user-set -ngl or
		// silently override ggrun's MoE expert placement.
		if s.BackendSupportsFit {
			args = append(args, "--fit", "off")
		}
		// Metal has exactly one logical device — device-routing flags are
		// CUDA/Vulkan concepts and llama-server rejects unknown device names.
		if s.MainGPU >= 0 && len(s.TensorSplit) == 0 && !strings.EqualFold(s.BackendTag, "metal") {
			args = append(args, "-mg", fmt.Sprintf("%d", s.MainGPU))
			if s.Type == SingleGPU {
				args = append(args, "--device", deviceName(s.BackendTag, s.MainGPU))
			}
		}
	}

	if len(s.TensorSplit) > 0 {
		var splitStr string
		for i, v := range s.TensorSplit {
			if i > 0 {
				splitStr += ","
			}
			splitStr += fmt.Sprintf("%.2f", v)
		}
		args = append(args, "--tensor-split", splitStr)
	}

	if s.SplitMode != "" {
		args = append(args, "--split-mode", s.SplitMode)
	}

	// Force Native Offload: strip the -ot regex if UseNativeOffload is true.
	if s.UseNativeOffload {
		s.OTString = ""
	}

	if s.OTString != "" {
		args = append(args, "-ot", s.OTString)
	}

	if s.NCPUMoE > 0 {
		args = append(args, "--n-cpu-moe", fmt.Sprintf("%d", s.NCPUMoE))
	}

	if !s.MMap {
		args = append(args, "--no-mmap")
	}
	if s.MLock {
		args = append(args, "--mlock")
	}

	// CRAM is always a real, derived decision (never "not applicable") — 0
	// must reach the backend as an explicit "-cram 0" (disable), not silence
	// that lets llama-server fall back to its own 8192 MiB default. Same for
	// MaxCheckpoints when computeCRAM actually evaluated it (>= 0): nesting
	// this inside "CRAM > 0" used to mean a correctly-computed "0, disable
	// checkpoints — VRAM is too tight" was silently dropped, leaving
	// llama-server's default of 32 checkpoints active. That default's context
	// checkpoint save (tools/server/server-context.cpp create_checkpoint)
	// needs backend memory sched_reserve() never accounts for, and is exactly
	// what crashed DeepSeek-V4 mid-request on 2026-07-08 despite a placement
	// that had loaded clean and passed health check.
	args = append(args, "-cram", fmt.Sprintf("%d", s.CRAM))
	if s.MaxCheckpoints >= 0 {
		args = append(args, "--ctx-checkpoints", fmt.Sprintf("%d", s.MaxCheckpoints))
	}

	// ik_llama.cpp fork specific flags
	if s.BackendTag == "ik_llama" {
		args = append(args, "--run-time-repack")
		args = append(args, "-khad")
		args = append(args, "--defrag-thold", "0.1")
		if s.IsMoE {
			args = append(args, "-muge")
			args = append(args, "-ger")
		}
		if len(s.TensorSplit) > 0 || s.Type != CPUOnly {
			args = append(args, "-mqkv")
		}
	}

	// Speculative decoding flags (MTP, EAGLE-3, draft model, or explicit ngram)
	if s.Draft != nil && s.Draft.Type != DraftNone {
		args = append(args, DraftFlags(s.Draft)...)
	}

	return args
}

// loadSystemProbe tries to load measured CUDA overhead from cache.
// Keys the probe cache by a GPU-signature hash.

// SystemCUDAOverheadMB returns the legacy measured CUDA context overhead. New
// placement/preflight code must use SystemCUDAOverheadByGPU so headroom remains
// per-device and measured-only; this helper is kept for old call sites/tests.
func SystemCUDAOverheadMB(cacheDir string, gpus []detect.GPU) int {
	if p := loadSystemProbe(cacheDir, gpus); p != nil && p.CUDAOverheadMB > 0 {
		return p.CUDAOverheadMB
	}
	return 0
}

// SystemCUDAOverheadByGPU returns measured CUDA context/allocator overhead keyed
// by CUDA device index. Missing entries are intentionally absent: callers must
// not fill them with static margins.


// gpuSignatureHash computes MD5 hash of sorted GPU name+driver pairs.
// GPU signature: nvidia-smi --query-gpu=name,driver_version | sort | md5sum | cut -c1-12
func gpuSignatureHash(gpus []detect.GPU) string {
	var parts []string
	for _, g := range gpus {
		parts = append(parts, fmt.Sprintf("%s,%s", g.Name, g.Driver))
	}
	sort.Strings(parts)
	input := strings.Join(parts, "\n") + "\n"
	h := md5.New()
	h.Write([]byte(input))
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// RunPostLaunchProbe measures actual CUDA overhead after a successful server launch.
// It reads current VRAM usage from nvidia-smi, parses buffer sizes from the
// server's captured stderr log, and caches the result for future launches.
// Parses the server log after launch to record measured overhead.

// kvCachePath is the per-model cache of measured KV bytes-per-token. Keyed by
// model basename + byte size so requantizations/different models never collide.
func kvCachePath(cacheDir string, model *ModelProfile) string {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}
	base := model.Basename
	if base == "" {
		base = filepath.Base(model.Path)
	}
	return filepath.Join(cacheDir, fmt.Sprintf("kv_%s_%d.cache", base, model.SizeBytes))
}

// loadMeasuredKVRates reads the per-model KV cache into a kvType→bytes/token map.
func loadMeasuredKVRates(cacheDir string, model *ModelProfile) map[string]float64 {
	path := kvCachePath(cacheDir, model)
	data, err := os.ReadFile(path)
	if err != nil && cacheDir != "" {
		// See loadSystemProbe: older installs wrote exact KV measurements to the
		// user cache even when the current app uses an app-local cache directory.
		// A compressed-attention model can be overestimated by many GiB without
		// this value, so migrate the measurement rather than reverting to formula.
		legacyPath := kvCachePath("", model)
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

	out := map[string]float64{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		// format: KV_BYTES_PER_TOK_<kvtype>=<float>
		const pfx = "KV_BYTES_PER_TOK_"
		if !strings.HasPrefix(line, pfx) {
			continue
		}
		kv := strings.SplitN(strings.TrimPrefix(line, pfx), "=", 2)
		if len(kv) != 2 {
			continue
		}
		if v, err := strconv.ParseFloat(strings.TrimSpace(kv[1]), 64); err == nil && v > 0 {
			out[strings.ToLower(strings.TrimSpace(kv[0]))] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// parseKVBufferTotalMB extracts the model's TOTAL KV cache allocation (MiB) at the
// launched context from a backend log. llama.cpp's wording varies across versions
// and backends, so match all known forms: an aggregate "KV self size = X MiB" /
// "KV cache size = X MiB" line (already the total — take it directly), otherwise
// SUM the per-device "... KV buffer size = X MiB" lines across CUDA devices + CPU.
// Returns 0 when the log carries no KV line (caller falls back to the formula or
// the VRAM-delta probe).
func parseKVBufferTotalMB(log string) float64 {
	var aggregate, bufSum float64
	for _, line := range strings.Split(log, "\n") {
		low := strings.ToLower(line)
		if !strings.Contains(low, "=") {
			continue
		}
		switch {
		case strings.Contains(low, "kv self size"), strings.Contains(low, "kv cache size"):
			if v := parseMiB(line); v > aggregate {
				aggregate = v // aggregate line: the total, printed once
			}
		case strings.Contains(low, "kv buffer size"):
			bufSum += parseMiB(line) // per-device: sum across GPUs + CPU
		}
	}
	if aggregate > 0 {
		return aggregate
	}
	return bufSum
}

// kvBytesPerTokenFromVRAMDelta derives KV bytes-per-token from two launches that
// differ ONLY in context size. Weights, compute buffers, and CUDA overhead are
// identical across the two, so the VRAM difference is pure KV cache — exact for
// every architecture and independent of whether the backend logs its KV size at
// all. Returns 0 if the samples are unusable.
func kvBytesPerTokenFromVRAMDelta(ctxA, vramA_MB, ctxB, vramB_MB int) float64 {
	dCtx := ctxB - ctxA
	dVRAM := vramB_MB - vramA_MB
	if dCtx < 0 {
		dCtx, dVRAM = -dCtx, -dVRAM
	}
	if dCtx == 0 || dVRAM <= 0 {
		return 0
	}
	return float64(dVRAM) * 1048576.0 / float64(dCtx)
}

// setCtxSizeArg returns a copy of args with --ctx-size set to ctx (adding it if
// absent). Used by the VRAM-delta probe to launch the same placement twice at
// different contexts.
func setCtxSizeArg(args []string, ctx int) []string {
	out := make([]string, len(args))
	copy(out, args)
	for i := 0; i < len(out)-1; i++ {
		if out[i] == "--ctx-size" || out[i] == "-c" {
			out[i+1] = strconv.Itoa(ctx)
			return out
		}
	}
	return append(out, "--ctx-size", strconv.Itoa(ctx))
}

// measureLoadedVRAM launches backendPath with args, waits for VRAM to plateau
// (the model + KV finished allocating), returns total VRAM used across gpus (MiB),
// then kills the process. Log-independent — it reads nvidia-smi, not stderr.
func measureLoadedVRAM(backendPath string, args []string, gpus []detect.GPU, timeout time.Duration) int {
	cmd := exec.Command(backendPath, args...)
	cmd.Stdout = nil
	cmd.Stderr = nil
	if err := cmd.Start(); err != nil {
		return 0
	}
	defer func() {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
	}()

	deadline := time.Now().Add(timeout)
	prev, stable := -1, 0
	for time.Now().Before(deadline) {
		time.Sleep(1500 * time.Millisecond)
		total := 0
		for _, g := range gpus {
			total += QueryVRAMUsed(g.Index)
		}
		// Plateau = two consecutive readings within 64 MiB, above a floor.
		if total > 512 && prev > 512 && absInt(total-prev) <= 64 {
			stable++
			if stable >= 2 {
				return total
			}
		} else {
			stable = 0
		}
		prev = total
	}
	if prev > 512 {
		return prev
	}
	return 0
}

func absInt(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// ProbeKVViaVRAMDelta measures a model's KV bytes-per-token by launching the same
// placement (baseArgs, minus the binary) twice — at a small and a larger context —
// and attributing the VRAM difference entirely to KV. It is the log-independent
// fallback for backends that don't print their KV buffer size. Requires roughly
// idle GPUs for a clean delta. Writes the per-model KV cache and returns true on
// success. Best-effort: returns false if either launch fails to allocate.
//
// NOTE: the two-launch flow needs live validation on real hardware before it is
// auto-invoked; the arithmetic (kvBytesPerTokenFromVRAMDelta) and cache round-trip
// are unit-tested.
func ProbeKVViaVRAMDelta(backendPath string, baseArgs []string, gpus []detect.GPU, cacheDir string, model *ModelProfile, kvType string) bool {
	if backendPath == "" || model == nil || len(gpus) == 0 {
		return false
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	const ctxA, ctxB = 8192, 65536
	loadTimeout := 15 * time.Minute

	vramA := measureLoadedVRAM(backendPath, setCtxSizeArg(baseArgs, ctxA), gpus, loadTimeout)
	if vramA <= 0 {
		return false
	}
	vramB := measureLoadedVRAM(backendPath, setCtxSizeArg(baseArgs, ctxB), gpus, loadTimeout)
	if vramB <= 0 {
		return false
	}

	rate := kvBytesPerTokenFromVRAMDelta(ctxA, vramA, ctxB, vramB)
	if rate <= 0 {
		return false
	}
	writeMeasuredKVRate(cacheDir, model, strings.ToLower(kvType), rate,
		fmt.Sprintf("VRAM-delta probe: ctx %d=%dMB, ctx %d=%dMB", ctxA, vramA, ctxB, vramB))
	return true
}

// RunPostLaunchKVProbe reads the KV cache size llama.cpp actually allocated at
// ctxSize from the backend log and caches it as bytes-per-token for this model +
// kvType, so future launches size the context from measured truth instead of the
// per-arch GGUF formula. A successful launch deliberately refreshes a preflight
// estimate because the final buffer log is the most precise measurement.
func RunPostLaunchKVProbe(cacheDir string, model *ModelProfile, ctxSize int, kvType, serverLog string) {
	if model == nil || ctxSize <= 0 || serverLog == "" {
		return
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	kvType = strings.ToLower(kvType)

	totalKVMB := parseKVBufferTotalMB(serverLog)
	if totalKVMB <= 0 {
		return
	}
	bytesPerTok := totalKVMB * 1048576.0 / float64(ctxSize)
	if bytesPerTok <= 0 {
		return
	}
	writeMeasuredKVRate(cacheDir, model, kvType, bytesPerTok,
		fmt.Sprintf("launch log: ctx=%d total_kv=%.0fMB", ctxSize, totalKVMB))
}

// RecordMeasuredContextMB records the backend-authoritative total context
// allocation reported by llama-fit-params. For placement purposes this is the
// exact quantity computeKVTotalMB must reserve: summing every device row keeps
// GPU-, CPU-, and mixed-KV placements on the same accounting path. The model is
// updated in memory as well as on disk so an immediate preflight re-plan sees
// the measurement without waiting for another process or launch.
func RecordMeasuredContextMB(cacheDir string, model *ModelProfile, ctxSize int, kvType string, totalContextMB int) {
	if model == nil || ctxSize <= 0 || totalContextMB <= 0 {
		return
	}
	if kvType == "" {
		kvType = "q8_0"
	}
	kvType = strings.ToLower(kvType)
	bytesPerTok := float64(totalContextMB) * 1048576.0 / float64(ctxSize)
	if bytesPerTok <= 0 {
		return
	}
	writeMeasuredKVRate(cacheDir, model, kvType, bytesPerTok,
		fmt.Sprintf("fit-params preflight: ctx=%d total_context=%dMB", ctxSize, totalContextMB))
}

// RecordMeasuredComputeBuffers records per-GPU compute-buffer MiB measured by
// the no-alloc fit-params preflight (cmd/ggrun/preflight.go) for the exact
// ctx/ubatch/KV/backend/GPU-set key, merging with (not clobbering) any
// runtime-growth or KV-per-layer data already cached for that same key. This
// lets the FIRST launch attempt for a shape use real numbers instead of the
// first-launch heuristic (ubatch*4, clamped 1024-4096) — which measurably
// under-estimates flash-attention's compute buffer at large context (e.g.
// ~17-20GB actual vs a 4096MB clamp for DeepSeek-V4 at ctx 1M, f16 KV).
func RecordMeasuredComputeBuffers(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, computeByGPU map[int]int) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || len(computeByGPU) == 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	growth := map[int]int{}
	kvPerLayerMB := 0
	if pc != nil {
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			growth[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, growth, kvPerLayerMB)
}

// RunPostLaunchModelProbe records measured compute-buffer data for the exact
// model/runtime placement that just loaded. Future placement can use this instead
// of the first-launch compute estimate.

// RuntimeGraphGrowthByGPU returns measured post-health graph growth keyed by CUDA
// device. These values are populated only from observed runtime allocation growth
// or exact cudaMalloc failures for the same runtime signature; missing means
// unknown, not zero-margin proof.
func RuntimeGraphGrowthByGPU(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) map[int]int {
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if pc == nil || len(pc.RuntimeGraphGrowthByGPU) == 0 {
		return nil
	}
	out := map[int]int{}
	for k, v := range pc.RuntimeGraphGrowthByGPU {
		if v > 0 {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

// HasRuntimeGraphGrowthProbe reports whether all active GPUs have measured
// runtime-growth data for this exact runtime signature. This is a verification
// marker for future agents: no static fallback is hidden behind this predicate.
func HasRuntimeGraphGrowthProbe(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) bool {
	growth := RuntimeGraphGrowthByGPU(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	if len(gpus) == 0 || len(growth) == 0 {
		return false
	}
	for _, g := range gpus {
		if _, ok := growth[g.Index]; !ok {
			return false
		}
	}
	return true
}

// RecordRuntimeGraphGrowth stores exact per-device runtime graph growth for the
// current runtime signature. The values must come from measurement: VRAM delta
// after a canary or an exact cudaMalloc allocation request parsed from backend
// logs. It intentionally never adds static margins.
func RecordRuntimeGraphGrowth(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, growthByGPU map[int]int) error {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || len(growthByGPU) == 0 {
		return nil
	}
	pc := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel)
	computeByGPU := map[int]int{}
	kvPerLayerMB := 0
	mergedGrowth := map[int]int{}
	if pc != nil {
		for k, v := range pc.ComputeBufByGPU {
			computeByGPU[k] = v
		}
		for k, v := range pc.RuntimeGraphGrowthByGPU {
			mergedGrowth[k] = v
		}
		kvPerLayerMB = pc.KVPerLayerMB
	}
	for idx, v := range growthByGPU {
		if v > mergedGrowth[idx] {
			mergedGrowth[idx] = v
		}
	}
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, mergedGrowth, kvPerLayerMB)
}

// RecordRuntimeGraphGrowthFromOOM records a runtime graph allocation observed in
// a cudaMalloc OOM line. allocMB is already the backend-requested size, so using
// it as growth is measured accounting, not a reserve margin.
func RecordRuntimeGraphGrowthFromOOM(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel, device, allocMB int) error {
	if device < 0 || allocMB <= 0 {
		return nil
	}
	return RecordRuntimeGraphGrowth(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, map[int]int{device: allocMB})
}

// RunPostLaunchModelProbeVRAMDelta writes per-GPU compute-buffer probe cache from
// nvidia-smi VRAM delta (current - baseline) instead of log parsing. Log-independent;
// works even when the server binary suppresses LLAMA_LOG_INFO. It estimates model
// weight per GPU from the placement OT assignments, subtracts that from the VRAM
// delta, and stores the remainder as the measured compute-buffer value for that GPU.
func computeBuffersFromVRAMDelta(
	model *ModelProfile, strategy *Strategy, gpus []detect.GPU,
	baselineVRAMByGPU, usedVRAMByGPU, overheadByGPU map[int]int,
) map[int]int {
	if model == nil || strategy == nil || model.NumLayers <= 0 || len(gpus) == 0 {
		return nil
	}
	// Partial gate+up pins do not expose their byte size in the OT string. A
	// guessed model subtraction would turn this fallback into poisoned compute
	// data, so rely on fit/log probes for those placements.
	if strings.Contains(strategy.OTString, `ffn_(gate_up|up_gate|gate|up)_(ch|)exps`) {
		return nil
	}

	assignments := parseOTAssignments(strategy.OTString)
	expertLayersByGPU := map[int]int{}
	for _, a := range assignments {
		expertLayersByGPU[a.CUDAIndex] += a.Count
	}

	moeLayers := model.NumLayers - model.LeadingDense
	if moeLayers <= 0 {
		moeLayers = model.NumLayers
	}
	expertPerLayerMB := ceilDivInt(bytesToMiBCeil(model.ExpertBytes), moeLayers)

	nonExpertTotalMB := bytesToMiBCeil(model.NonExpertBytes)
	if tokenMB := bytesToMiBCeil(model.TokenEmbdBytes); tokenMB > 0 && tokenMB < nonExpertTotalMB {
		nonExpertTotalMB -= tokenMB
	}
	outputMB := bytesToMiBCeil(model.OutputBytes)
	if outputMB > 0 && outputMB < nonExpertTotalMB {
		nonExpertTotalMB -= outputMB
	} else {
		outputMB = 0
	}
	shexpTotalMB := bytesToMiBCeil(model.ShexpBytes)
	if shexpTotalMB < 0 || shexpTotalMB >= bytesToMiBCeil(model.ExpertBytes) {
		shexpTotalMB = 0
	}

	owned, outputDev := layerOwnership(strategy.TensorSplit, model.NumLayers)
	layerDevices, _ := layerDeviceAssignments(strategy.TensorSplit, model.NumLayers)
	moeOwned := make([]int, len(strategy.TensorSplit))
	moeStart := model.LeadingDense
	if moeStart < 0 || moeStart >= model.NumLayers {
		moeStart = 0
	}
	for layer := moeStart; layer < len(layerDevices); layer++ {
		if dev := layerDevices[layer]; dev >= 0 && dev < len(moeOwned) {
			moeOwned[dev]++
		}
	}

	kvTotalMB := 0
	if strings.EqualFold(strategy.KVPlacement, "gpu") && strategy.ContextSize > 0 {
		kvTotalMB = computeKVTotalMB(model, strategy.ContextSize, strategy.KVType)
	}

	computeByGPU := map[int]int{}
	for gi, g := range gpus {
		overheadMB, measured := overheadByGPU[g.Index]
		if !measured {
			continue
		}
		usedMB := usedVRAMByGPU[g.Index]
		baselineMB := baselineVRAMByGPU[g.Index]
		if usedMB <= baselineMB {
			continue
		}
		modelMB := expertLayersByGPU[g.Index] * expertPerLayerMB
		if gi < len(owned) && owned[gi] > 0 {
			modelMB += ownedShareMB(nonExpertTotalMB, owned, model.NumLayers, gi)
		}
		if gi < len(moeOwned) && moeOwned[gi] > 0 {
			modelMB += int(math.Ceil(float64(shexpTotalMB) * float64(moeOwned[gi]) / float64(moeLayers)))
		}
		if gi == outputDev {
			modelMB += outputMB
		}
		kvShareMB := ownedShareMB(kvTotalMB, owned, model.NumLayers, gi)
		bufMB := usedMB - baselineMB - overheadMB - modelMB - kvShareMB
		if bufMB > 0 {
			computeByGPU[g.Index] = bufMB
		}
	}
	if len(computeByGPU) == 0 {
		return nil
	}
	return computeByGPU
}

func RunPostLaunchModelProbeVRAMDelta(
	cacheDir string, model *ModelProfile, strategy *Strategy,
	backendTag string, gpus []detect.GPU, baselineVRAMByGPU map[int]int,
) bool {
	if model == nil || strategy == nil || len(gpus) == 0 || len(baselineVRAMByGPU) == 0 {
		return false
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}

	// This is strictly a fallback for backends that expose neither fit-params nor
	// compute-buffer log rows. Never replace an authoritative probe already
	// recorded by preflight for this exact runtime signature.
	existing := loadProbeCache(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel)
	if existing != nil && len(existing.ComputeBufByGPU) > 0 {
		return false
	}

	usedVRAMByGPU := map[int]int{}
	for _, g := range gpus {
		usedMB := QueryVRAMUsed(g.Index)
		if usedMB > 0 {
			usedVRAMByGPU[g.Index] = usedMB
		}
	}

	computeByGPU := computeBuffersFromVRAMDelta(model, strategy, gpus, baselineVRAMByGPU, usedVRAMByGPU, SystemCUDAOverheadByGPU(cacheDir, gpus))
	if len(computeByGPU) == 0 {
		return false
	}

	// Preserve any runtime-growth history from a previous OOM so the probe
	// cache does not silently erase it (audit cross-check #3).
	var mergedGrowth map[int]int
	if existing != nil {
		mergedGrowth = existing.RuntimeGraphGrowthByGPU
	}

	if err := writeProbeCacheForModel(cacheDir, model, strategy.ContextSize, strategy.UBatchSize,
		strategy.KVQuality, strategy.KVPlacement, backendTag, gpus, strategy.Parallel, computeByGPU, mergedGrowth, 0); err == nil {
		indices := make([]int, 0, len(computeByGPU))
		for idx := range computeByGPU {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		parts := make([]string, 0, len(indices))
		for _, idx := range indices {
			parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, computeByGPU[idx]))
		}
		fmt.Fprintf(os.Stderr, "  VRAM probe: compute_buf %s\n", strings.Join(parts, ", "))
		return true
	}
	return false
}

func RunPostLaunchModelProbe(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, serverLog string) bool {
	if model == nil || ctxSize <= 0 || ubatch <= 0 || serverLog == "" {
		return false
	}
	computeByGPU := ParseComputeBuffersByGPU(serverLog)
	_, kvPerLayerMB := ParseLogForProbe(serverLog)
	if len(computeByGPU) == 0 && kvPerLayerMB <= 0 {
		return false
	}
	if err := writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallel, computeByGPU, nil, kvPerLayerMB); err == nil {
		if len(computeByGPU) == 0 {
			return true
		}
		parts := make([]string, 0, len(computeByGPU))
		indices := make([]int, 0, len(computeByGPU))
		for idx := range computeByGPU {
			indices = append(indices, idx)
		}
		sort.Ints(indices)
		for _, idx := range indices {
			parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, computeByGPU[idx]))
		}
		fmt.Fprintf(os.Stderr, "  Model probe written: compute_buf %s\n", strings.Join(parts, ", "))
		return true
	}
	return false
}

// writeMeasuredKVRate records a measured KV bytes-per-token for model+kvType,
// merging with any rates already cached for other kvTypes.
func writeMeasuredKVRate(cacheDir string, model *ModelProfile, kvType string, bytesPerTok float64, note string) {
	kvType = strings.ToLower(kvType)
	rates := loadMeasuredKVRates(cacheDir, model)
	if rates == nil {
		rates = map[string]float64{}
	}
	for k, v := range model.MeasuredKVBytesPerTok {
		if v > 0 && rates[k] <= 0 {
			rates[k] = v
		}
	}
	rates[kvType] = bytesPerTok
	model.MeasuredKVBytesPerTok = rates

	path := kvCachePath(cacheDir, model)
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Measured KV cache for %s (%s)\n", model.Basename, note)
	for k, v := range rates {
		fmt.Fprintf(&b, "KV_BYTES_PER_TOK_%s=%.4f\n", k, v)
	}
	if err := os.WriteFile(path, []byte(b.String()), 0644); err == nil {
		fmt.Fprintf(os.Stderr, "  KV probe: %s = %.0f bytes/token (%s)\n", kvType, bytesPerTok, note)
	}
}

func RunPostLaunchProbe(cacheDir string, gpus []detect.GPU, serverLog string) {
	if len(gpus) == 0 || serverLog == "" {
		return
	}

	// Only write if per-device measured values already exist. Legacy global-only
	// files are upgraded the next time a launch provides per-device accounting.
	sp := loadSystemProbe(cacheDir, gpus)
	if sp != nil && len(sp.CUDAOverheadByGPU) > 0 {
		return
	}

	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun")
	}

	overheadByGPU := map[int]int{}
	for _, gpu := range gpus {
		usedMB := QueryVRAMUsed(gpu.Index)
		if usedMB <= 0 {
			continue
		}
		modelBufMB, kvBufMB, computeBufMB := parseBuffersFromLog(serverLog, gpu.Index)
		accounted := modelBufMB + kvBufMB + computeBufMB
		if accounted <= 0 {
			continue
		}
		cudaOverhead := usedMB - accounted
		if cudaOverhead <= 0 || cudaOverhead >= usedMB {
			continue
		}
		overheadByGPU[gpu.Index] = cudaOverhead
	}
	if len(overheadByGPU) == 0 {
		return
	}

	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return
	}

	gpuSig := gpuSignatureHash(gpus)
	path := filepath.Join(cacheDir, fmt.Sprintf("system_%s.cache", gpuSig))

	var b strings.Builder
	fmt.Fprintf(&b, "# System probe (post-launch per-device measurement)\n")
	fmt.Fprintf(&b, "# Generated: %s\n", time.Now().Format(time.RFC3339))
	indices := make([]int, 0, len(overheadByGPU))
	for idx := range overheadByGPU {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	legacyMax := 0
	parts := make([]string, 0, len(indices))
	for _, idx := range indices {
		v := overheadByGPU[idx]
		if v > legacyMax {
			legacyMax = v
		}
		fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB_CUDA%d=%d\n", idx, v)
		parts = append(parts, fmt.Sprintf("CUDA%d=%dMB", idx, v))
	}
	// Compatibility for older readers. This is still measured data, not a margin.
	fmt.Fprintf(&b, "SYS_CUDA_OVERHEAD_MB=%d\n", legacyMax)
	if err := os.WriteFile(path, []byte(b.String()), 0644); err == nil {
		fmt.Fprintf(os.Stderr, "  System probe written: cuda_overhead %s\n", strings.Join(parts, ", "))
	}
}

// QueryVRAMUsed returns current nvidia-smi memory.used for a given GPU index.
func QueryVRAMUsed(gpuIndex int) int {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=memory.used", "--format=csv,noheader,nounits",
		"-i", fmt.Sprintf("%d", gpuIndex),
	).Output()
	if err != nil {
		return 0
	}
	s := strings.TrimSpace(string(out))
	v, err := strconv.Atoi(s)
	if err != nil {
		return 0
	}
	return v
}

// parseBuffersFromLog parses llama-server log for CUDA buffer sizes on a specific GPU.
// Returns modelBufMB, kvBufMB, computeBufMB.
// Handles both mainline ("CUDAN model buffer size = X MiB") and
// ik_llama ("CUDAN buffer size = X MiB") formats.
func parseBuffersFromLog(log string, gpuIndex int) (modelBufMB, kvBufMB, computeBufMB int) {
	cudaTag := fmt.Sprintf("CUDA%d", gpuIndex)
	var maxModelBuf, maxComputeBuf float64
	var totalKVBuf float64
	var kvCount int

	lines := strings.Split(log, "\n")
	for _, line := range lines {
		if !strings.Contains(line, cudaTag) {
			continue
		}
		// Model buffer: "CUDA0 model buffer size = X MiB" or "CUDA0 buffer size = X MiB"
		if strings.Contains(line, "buffer size =") && !strings.Contains(line, "KV") && !strings.Contains(line, "compute") {
			if v := parseMiB(line); v > maxModelBuf {
				maxModelBuf = v
			}
		}
		// KV buffer: "CUDA0 KV buffer size = X MiB"
		if strings.Contains(line, "KV buffer size =") {
			if v := parseMiB(line); v > 0 {
				totalKVBuf += v
				kvCount++
			}
		}
		// Compute buffer: "CUDA0 compute buffer size = X MiB"
		if strings.Contains(line, "compute buffer size =") {
			if v := parseMiB(line); v > maxComputeBuf {
				maxComputeBuf = v
			}
		}
	}

	modelBufMB = int(maxModelBuf + 0.5)
	computeBufMB = int(maxComputeBuf + 0.5)
	if totalKVBuf > 0 && kvCount > 0 {
		kvBufMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}
	return
}

// parseMiB extracts a floating-point MiB value from a log line containing "X MiB".
func parseMiB(line string) float64 {
	idx := strings.LastIndex(line, "=")
	if idx < 0 {
		return 0
	}
	// Take the number between "=" and the FIRST "MiB" after it, so lines with
	// trailing detail (e.g. "KV self size = X MiB, K (f16): Y MiB, ...") parse the
	// aggregate X and not the per-component values.
	rest := line[idx+1:]
	mib := strings.Index(rest, "MiB")
	if mib < 0 {
		return 0
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(rest[:mib]), 64)
	if err != nil {
		return 0
	}
	return v
}

// loadProbeCache tries to load per-model/runtime probe data.
// Keys the probe cache file by an MD5 of the model + placement runtime signature.
func probeCachePath(cacheDir string, model *ModelProfile, ctxSize int, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int) string {
	if model == nil {
		return ""
	}
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun", "probes")
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

	// MD5 hash key over model/runtime/placement. Compute buffers differ with KV
	// placement, backend, GPU set, and parallel slots. Tensor split is derived
	// after probes are loaded, so it cannot honestly participate in this key.
	modelName := filepath.Base(model.Path)
	// Keep the historical trailing separator so serial (parallel key 0) cache
	// paths remain compatible with measurements from before slot isolation.
	key := fmt.Sprintf("probe:v%d:%s:%d:%d:%d:%d:%d:%d:%s:%s:%s:%s:%d:",
		placementPlannerCacheVersion, modelName, model.NumLayers, model.NumExperts,
		model.EmbeddingLength, model.FeedForwardLength,
		ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallel)
	hash := md5Hash12(key)
	return filepath.Join(cacheDir, hash+".probe")
}

// PlacementCachePathFor returns the keyed file that persists the validated MoE
// placement for this exact model + runtime + hardware. Keyed identically to the
// probe cache (kv placement, ctx, ubatch, backend, GPU set all part of the key)
// so kv=gpu vs kv=cpu — or two context sizes — never share a cache entry. Both
// the fit (load) and OOM-recovery / success (save) use this same path, so a
// placement that loads cleanly is remembered instead of re-predicted.

// Increment this when planner semantics or emitted tensor override patterns
// change. A validated placement is only reusable under the exact semantics that
// produced it; otherwise an old .place file can silently restore stale routing.
const placementPlannerCacheVersion = 2
// md5Hash12 computes first 12 chars of MD5 hash of input string.
func md5Hash12(input string) string {
	h := md5.New()
	h.Write([]byte(input))
	return fmt.Sprintf("%x", h.Sum(nil))[:12]
}

// internal types for probe loading
type systemProbe struct {
	CUDAOverheadMB    int
	CUDAOverheadByGPU map[int]int
}

type probeCache struct {
	ComputeBufMB            int
	ComputeBufByGPU         map[int]int
	RuntimeGraphGrowthByGPU map[int]int
	KVPerLayerMB            int
}

const probeCacheSchema = 2

// probeParallelKey preserves the legacy serial key (0) for normal --parallel 1
// launches while isolating multi-slot graph measurements (e.g. --parallel 4).
// Graph allocation changes with slot count; sharing those probes
// was another form of compute-buffer double accounting across launch modes.
func probeParallelKey(parallel int) int {
	if parallel <= 1 {
		return 0
	}
	return parallel
}

// splitCompactKey returns a compact string for a tensor-split vector, suitable
// for cache-key inclusion so placements with different split ratios don't share
// cached compute-buffer measurements (audit cross-check #6).
// WriteProbeCache writes a legacy probe cache. Prefer WriteProbeCacheForModel,
// which writes to the same runtime-signature key that loadProbeCache reads.
func WriteProbeCache(cacheDir, modelName string, computeBufMB, kvPerLayerMB int) error {
	if cacheDir == "" {
		home, _ := os.UserHomeDir()
		cacheDir = filepath.Join(home, ".cache", "ggrun", "probes")
	}
	if err := os.MkdirAll(cacheDir, 0755); err != nil {
		return err
	}
	// Sanitize model name for filename
	safeName := strings.ReplaceAll(modelName, "/", "_")
	path := filepath.Join(cacheDir, safeName+".probe")
	content := fmt.Sprintf(
		"# Probe cache for %s\n"+
			"# Generated: %s\n"+
			"PROBED_COMPUTE_BUF_MB=%d\n"+
			"PROBED_KV_PER_LAYER_MB=%d\n",
		modelName, time.Now().Format(time.RFC3339), computeBufMB, kvPerLayerMB,
	)
	return os.WriteFile(path, []byte(content), 0644)
}

// WriteProbeCacheForModel writes measured compute-buffer and KV sizes to the
// per-model/runtime cache consumed by placement. computeByGPU is keyed by CUDA
// device index as emitted in the backend log.
func WriteProbeCacheForModel(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, computeByGPU map[int]int, kvPerLayerMB int) error {
	return writeProbeCacheForModel(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, 0, computeByGPU, nil, kvPerLayerMB)
}

func writeProbeCacheForModel(cacheDir string, model *ModelProfile, ctxSize, ubatch int, kvQuality, kvPlacement, backendTag string, gpus []detect.GPU, parallel int, computeByGPU map[int]int, runtimeGrowthByGPU map[int]int, kvPerLayerMB int) error {
	parallelKey := probeParallelKey(parallel)
	path := probeCachePath(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallelKey)
	if path == "" {
		return nil
	}
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	// One runtime signature can produce several tensor placements while the
	// preflight/recovery loop converges. Graph sizes are placement-dependent, so
	// retain the maximum ever observed per device instead of letting a later,
	// smaller graph erase the reserve that another valid placement required.
	mergedCompute := map[int]int{}
	for idx, v := range computeByGPU {
		mergedCompute[idx] = v
	}
	mergedGrowth := map[int]int{}
	for idx, v := range runtimeGrowthByGPU {
		mergedGrowth[idx] = v
	}
	if existing := loadProbeCache(cacheDir, model, ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpus, parallelKey); existing != nil {
		for idx, v := range existing.ComputeBufByGPU {
			if v > mergedCompute[idx] {
				mergedCompute[idx] = v
			}
		}
		for idx, v := range existing.RuntimeGraphGrowthByGPU {
			if v > mergedGrowth[idx] {
				mergedGrowth[idx] = v
			}
		}
		if kvPerLayerMB <= 0 {
			kvPerLayerMB = existing.KVPerLayerMB
		}
	}

	maxCompute := 0
	for _, v := range mergedCompute {
		if v > maxCompute {
			maxCompute = v
		}
	}

	var b strings.Builder
	fmt.Fprintf(&b, "# Probe cache for %s\n", filepath.Base(model.Path))
	fmt.Fprintf(&b, "# Generated: %s\n", time.Now().Format(time.RFC3339))
	fmt.Fprintf(&b, "# ctx=%d ubatch=%d kv_quality=%s kv_placement=%s backend=%s gpu_sig=%s parallel=%d\n", ctxSize, ubatch, kvQuality, kvPlacement, backendTag, gpuSignatureHash(gpus), parallelKey)
	fmt.Fprintf(&b, "PROBE_CACHE_SCHEMA=%d\n", probeCacheSchema)
	if maxCompute > 0 {
		fmt.Fprintf(&b, "PROBED_COMPUTE_BUF_MB=%d\n", maxCompute)
	}
	indices := make([]int, 0, len(mergedCompute))
	for idx := range mergedCompute {
		indices = append(indices, idx)
	}
	sort.Ints(indices)
	for _, idx := range indices {
		if mergedCompute[idx] > 0 {
			fmt.Fprintf(&b, "PROBED_COMPUTE_BUF_MB_CUDA%d=%d\n", idx, mergedCompute[idx])
		}
	}
	growthIndices := make([]int, 0, len(mergedGrowth))
	for idx := range mergedGrowth {
		growthIndices = append(growthIndices, idx)
	}
	sort.Ints(growthIndices)
	for _, idx := range growthIndices {
		if mergedGrowth[idx] > 0 {
			fmt.Fprintf(&b, "PROBED_RUNTIME_GRAPH_GROWTH_MB_CUDA%d=%d\n", idx, mergedGrowth[idx])
		}
	}
	if kvPerLayerMB > 0 {
		fmt.Fprintf(&b, "PROBED_KV_PER_LAYER_MB=%d\n", kvPerLayerMB)
	}
	return os.WriteFile(path, []byte(b.String()), 0644)
}

var memoryBreakdownComputePattern = regexp.MustCompile(`CUDA([0-9]+).*?\(\s*-?[0-9]+\s*=\s*-?[0-9]+\s*\+\s*-?[0-9]+\s*\+\s*([0-9]+)\s*\)`)

// ParseLogForProbe extracts compute_buf and kv_per_layer from server log output.
// Looks for lines like: "CUDA0 compute buffer size = 1410.12 MiB" and fit
// breakdown rows like: "CUDA0 ... ( self = model + context + compute )".
func ParseLogForProbe(logData string) (computeBufMB, kvPerLayerMB int) {
	computeByGPU := ParseComputeBuffersByGPU(logData)
	for _, v := range computeByGPU {
		if v > computeBufMB {
			computeBufMB = v
		}
	}

	// Sum KV buffer sizes across CUDA devices
	var totalKVBuf float64
	var kvCount int
	lines := strings.Split(logData, "\n")
	for _, line := range lines {
		if idx := strings.Index(line, "KV buffer size ="); idx >= 0 {
			rest := line[idx+len("KV buffer size ="):]
			rest = strings.TrimSpace(rest)
			rest = strings.TrimSuffix(rest, " MiB")
			if v, err := strconv.ParseFloat(rest, 64); err == nil {
				totalKVBuf += v
				kvCount++
			}
		}
	}
	if totalKVBuf > 0 && kvCount > 0 {
		// Approximate per-device KV share by kvCount (devices holding KV)
		kvPerLayerMB = int(totalKVBuf/float64(kvCount) + 0.5)
	}
	return
}

// ParseComputeBuffersByGPU extracts measured/planned compute-buffer MiB by CUDA
// index. It accepts both final llama.cpp buffer lines and the memory breakdown
// table printed during fit.
func ParseComputeBuffersByGPU(logData string) map[int]int {
	out := map[int]int{}
	for _, line := range strings.Split(logData, "\n") {
		if idx := strings.Index(line, "compute buffer size = "); idx >= 0 {
			cudaIdx := cudaIndexFromLine(line)
			if cudaIdx < 0 {
				continue
			}
			rest := line[idx+len("compute buffer size = "):]
			rest = strings.TrimSpace(rest)
			rest = strings.TrimSuffix(rest, " MiB")
			if v, err := strconv.ParseFloat(rest, 64); err == nil && v > 0 {
				mb := int(v + 0.5)
				if mb > out[cudaIdx] {
					out[cudaIdx] = mb
				}
			}
			continue
		}
		if m := memoryBreakdownComputePattern.FindStringSubmatch(line); m != nil {
			cudaIdx, idxErr := strconv.Atoi(m[1])
			mb, mbErr := strconv.Atoi(m[2])
			if idxErr == nil && mbErr == nil && mb > out[cudaIdx] {
				out[cudaIdx] = mb
			}
		}
	}
	return out
}

func cudaIndexFromLine(line string) int {
	idx := strings.Index(line, "CUDA")
	if idx < 0 {
		return -1
	}
	start := idx + len("CUDA")
	end := start
	for end < len(line) && line[end] >= '0' && line[end] <= '9' {
		end++
	}
	if end == start {
		return -1
	}
	n, err := strconv.Atoi(line[start:end])
	if err != nil {
		return -1
	}
	return n
}

// PredictVRAMUsage estimates the VRAM needed for a given flag combination
// without launching the server. Returns (neededMB, freeMB). The tune engine
// uses this to skip candidates that are mathematically guaranteed to OOM,
// saving 30-60s per skipped candidate.
// ParseFlagsToMap converts a llama-server argv slice into a flag->value map
// for use with PredictVRAMUsage. Boolean flags map to "".