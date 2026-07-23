#!/usr/bin/env python3
"""Fix Phase 2: correct field names and add missing engine.go pre-launch check.
Run from go/ module root: python fix_phase2.py"""

import os

def read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write(path, content):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)

def replace_once(src, old, new, label, path):
    if new.strip() and new.strip() in src:
        print(f"  [skip] {label} — already applied")
        return src, False
    if old not in src:
        print(f"  [FAIL] {label} — pattern not found")
        print(f"         Looking for: {repr(old[:120])}...")
        return src, False
    if src.count(old) > 1:
        print(f"  [FAIL] {label} — found {src.count(old)} times")
        return src, False
    src = src.replace(old, new, 1)
    print(f"  [ok]   {label}")
    return src, True

changes = 0

# ============================================================================
# 1. placement.go — fix the broken smarter --n-cpu-moe block
# ============================================================================
print("\n=== placement.go ===")
plc_path = os.path.join("pkg", "placement", "placement.go")
plc = read(plc_path)

# 1a. Replace the broken VRAM-budget block with corrected field names
old_broken = """\t// VRAM-budget-aware --n-cpu-moe: compute the tightest safe value
\t// instead of the conservative default. This eliminates the need for
\t// the tune engine to spend 3+ rounds discovering the optimal value.
\tif layersCPU > 0 && model.ExpertBytesPerLayer > 0 {
\t\texpertPerLayerMB := bytesToMiBCeil(model.ExpertBytesPerLayer)
\t\tnonExpertMB := bytesToMiBCeil(model.NonExpertBytes)
\t\tif nonExpertMB <= 0 {
\t\t\tnonExpertMB = totalSizeMB / 10
\t\t}
\t\tkvMB := estimateKVCacheMB(model, opts.ContextSize, kvType)
\t\tcomputeMB := estimateComputeBufferMB(s.UBatchSize, model)
\t\tcudaMB := 300 // conservative per-GPU CUDA context overhead
\t\tgpuBudgetMB := freeVRAM - nonExpertMB - kvMB - computeMB - cudaMB
\t\tgpuBudgetMB = gpuBudgetMB * 90 / 100 // 10% safety margin
\t\tif gpuBudgetMB > 0 && expertPerLayerMB > 0 {
\t\t\tmaxGPUExperts := gpuBudgetMB / expertPerLayerMB
\t\t\tbudgetLayersCPU := model.ExpertLayers - maxGPUExperts
\t\t\tif budgetLayersCPU < 0 {
\t\t\t\tbudgetLayersCPU = 0
\t\t\t}
\t\t\t// Use the tighter of the two estimates (original vs VRAM-budget)
\t\t\tif budgetLayersCPU < layersCPU {
\t\t\t\tlayersCPU = budgetLayersCPU
\t\t\t}
\t\t}
\t\ts.NCPUMoE = layersCPU
\t} else if layersCPU > 0 {
\t\ts.NCPUMoE = layersCPU
\t}"""

new_fixed = """\t// VRAM-budget-aware --n-cpu-moe: compute the tightest safe value
\t// instead of the conservative default. This eliminates the need for
\t// the tune engine to spend 3+ rounds discovering the optimal value.
\tif layersCPU > 0 && model.ExpertBytes > 0 && model.NumLayers > 0 {
\t\texpertPerLayerMB := bytesToMiBCeil(model.ExpertBytes / int64(model.NumLayers))
\t\tnonExpertMB := bytesToMiBCeil(model.NonExpertBytes)
\t\tif nonExpertMB <= 0 {
\t\t\tnonExpertMB = totalSizeMB / 10
\t\t}
\t\tcomputeMB := firstLaunchComputeBufMB(model, s.UBatchSize)
\t\tcudaMB := 300 // conservative per-GPU CUDA context overhead
\t\tfreeVRAM := 0
\t\tfor _, g := range caps.GPUs {
\t\t\tfreeVRAM += g.VRAMFreeMB()
\t\t}
\t\tgpuBudgetMB := freeVRAM - nonExpertMB - kvTotalMB - computeMB - cudaMB
\t\tgpuBudgetMB = gpuBudgetMB * 90 / 100 // 10% safety margin
\t\tif gpuBudgetMB > 0 && expertPerLayerMB > 0 {
\t\t\tmaxGPUExperts := gpuBudgetMB / expertPerLayerMB
\t\t\tbudgetLayersCPU := model.NumLayers - maxGPUExperts
\t\t\tif budgetLayersCPU < 0 {
\t\t\t\tbudgetLayersCPU = 0
\t\t\t}
\t\t\t// Use the tighter of the two estimates (original vs VRAM-budget)
\t\t\tif budgetLayersCPU < layersCPU {
\t\t\t\tlayersCPU = budgetLayersCPU
\t\t\t}
\t\t}
\t\ts.NCPUMoE = layersCPU
\t} else if layersCPU > 0 {
\t\ts.NCPUMoE = layersCPU
\t}"""
plc, ok = replace_once(plc, old_broken, new_fixed, "fix smarter --n-cpu-moe field names", plc_path)
changes += ok

# 1b. Replace the broken PredictVRAMUsage with corrected version
old_predict_start = "// PredictVRAMUsage estimates the VRAM needed"
old_predict_end = "\treturn m\n}\n"
# Find and remove the entire broken block
idx_start = plc.find(old_predict_start)
if idx_start >= 0:
    idx_end = plc.find(old_predict_end, idx_start)
    if idx_end >= 0:
        idx_end += len(old_predict_end)
        plc = plc[:idx_start] + plc[idx_end:]
        print("  [ok]   removed broken PredictVRAMUsage/ParseFlagsToMap")
        changes += 1
    else:
        print("  [FAIL] could not find end of broken PredictVRAMUsage")
else:
    print("  [skip] broken PredictVRAMUsage not found (already removed?)")

# 1c. Append corrected PredictVRAMUsage and ParseFlagsToMap
corrected_funcs = '''
// PredictVRAMUsage estimates the VRAM needed for a given flag combination
// without launching the server. Returns (neededMB, freeMB). The tune engine
// uses this to skip candidates that are mathematically guaranteed to OOM,
// saving 30-60s per skipped candidate.
func PredictVRAMUsage(model *ModelProfile, flags map[string]string, caps *detect.Capabilities) (neededMB, freeMB int) {
\tif caps == nil || len(caps.GPUs) == 0 {
\t\treturn 0, 0
\t}
\tfor _, g := range caps.GPUs {
\t\tfreeMB += g.VRAMFreeMB()
\t}

\t// 1. Model weights on GPU
\tncpuMoe := 0
\tif v, ok := flags["--n-cpu-moe"]; ok {
\t\tncpuMoe, _ = strconv.Atoi(v)
\t}
\tif model.IsMoE && ncpuMoe > 0 && model.ExpertBytes > 0 && model.NumLayers > 0 {
\t\t// MoE: non-expert weights + GPU-resident expert layers
\t\tnonExpertMB := bytesToMiBCeil(model.NonExpertBytes)
\t\tif nonExpertMB <= 0 {
\t\t\tnonExpertMB = model.TotalSizeMB / 10
\t\t}
\t\texpertPerLayerMB := bytesToMiBCeil(model.ExpertBytes / int64(model.NumLayers))
\t\tgpuExpertLayers := model.NumLayers - ncpuMoe
\t\tif gpuExpertLayers < 0 {
\t\t\tgpuExpertLayers = 0
\t\t}
\t\tneededMB += nonExpertMB + gpuExpertLayers*expertPerLayerMB
\t} else {
\t\t// Dense or full GPU: all weights on GPU
\t\tneededMB += model.TotalSizeMB
\t}

\t// 2. KV cache on GPU
\tkvOnGPU := true
\tif _, ok := flags["--no-kv-offload"]; ok {
\t\tkvOnGPU = false
\t}
\tif kvOnGPU {
\t\tctxSize := model.ContextSize
\t\tif v, ok := flags["--ctx-size"]; ok {
\t\t\tif n, err := strconv.Atoi(v); err == nil && n > 0 {
\t\t\t\tctxSize = n
\t\t\t}
\t\t}
\t\tkvType := "f16"
\t\tif v, ok := flags["--cache-type-k"]; ok && v != "" {
\t\t\tkvType = v
\t\t}
\t\tneededMB += computeKVTotalMB(model, ctxSize, kvType)
\t}

\t// 3. Compute buffer (scales with ubatch)
\tubatch := 512
\tif v, ok := flags["-ub"]; ok {
\t\tif n, err := strconv.Atoi(v); err == nil && n > 0 {
\t\t\tubatch = n
\t\t}
\t}
\tneededMB += firstLaunchComputeBufMB(model, ubatch)

\t// 4. CUDA overhead (per-GPU context, ~200-500 MB)
\tneededMB += 300

\t// 5. Safety margin (10% for fragmentation and unexpected allocations)
\tneededMB = neededMB * 110 / 100

\treturn neededMB, freeMB
}

// ParseFlagsToMap converts a llama-server argv slice into a flag->value map
// for use with PredictVRAMUsage. Boolean flags map to "".
func ParseFlagsToMap(args []string) map[string]string {
\tm := make(map[string]string)
\tfor i := 0; i < len(args); i++ {
\t\tif !strings.HasPrefix(args[i], "-") {
\t\t\tcontinue
\t\t}
\t\tkey := args[i]
\t\tif eq := strings.Index(key, "="); eq > 0 {
\t\t\tm[key[:eq]] = key[eq+1:]
\t\t\tcontinue
\t\t}
\t\tif i+1 < len(args) && !strings.HasPrefix(args[i+1], "-") {
\t\t\tm[key] = args[i+1]
\t\t\ti++
\t\t} else {
\t\t\tm[key] = ""
\t\t}
\t}
\treturn m
}
'''
plc = plc.rstrip() + "\n" + corrected_funcs
print("  [ok]   appended corrected PredictVRAMUsage and ParseFlagsToMap")
changes += 1

write(plc_path, plc)

# ============================================================================
# 2. engine.go — add the missing pre-launch OOM check in the main tune loop
# ============================================================================
print("\n=== engine.go ===")
eng_path = os.path.join("pkg", "tune", "engine.go")
eng = read(eng_path)

old_launch = """\t\t// Candidate flags need their own backend process; otherwise every round
\t\t// measures the same already-running baseline.
\t\tstopBaseline()
\t\tcandidate, err := e.round(round, modelPath, candidateFlags)"""

new_launch = """\t\t// Pre-launch OOM prediction: skip candidates that are mathematically
\t\t// guaranteed to OOM, saving 30-60s per skipped candidate.
\t\tif e.PredictOOM != nil && e.PredictOOM(candidateFlags) {
\t\t\tif e.OnProgress != nil {
\t\t\t\te.OnProgress(fmt.Sprintf("AI-tune: skipping candidate %q (predicted OOM)", suggestion.Name))
\t\t\t}
\t\t\tcrashed := Entry{
\t\t\t\tTimestamp:     Now(),
\t\t\t\tModelPath:     modelPath,
\t\t\t\tModelName:     e.Model,
\t\t\t\tHardwareHash:  e.hardwareHash(),
\t\t\t\tBackend:       e.Backend,
\t\t\t\tVision:        e.Vision,
\t\t\t\tRound:         round,
\t\t\t\tName:          suggestion.Name,
\t\t\t\tFlags:         flagMap(candidateFlags),
\t\t\t\tOverrideFlags: overrides,
\t\t\t\tStatus:        "predicted-oom",
\t\t\t}
\t\t\te.addCache(&crashed)
\t\t\tentries = append(entries, crashed)
\t\t\tcrashedFlagSets = append(crashedFlagSets, overrides)
\t\t\te.saveTuneProgress(modelPath, baseline, best, entries, minImprovementPct, false)
\t\t\tround++
\t\t\tcontinue
\t\t}

\t\t// Candidate flags need their own backend process; otherwise every round
\t\t// measures the same already-running baseline.
\t\tstopBaseline()
\t\tcandidate, err := e.round(round, modelPath, candidateFlags)"""

eng, ok = replace_once(eng, old_launch, new_launch, "add pre-launch OOM prediction in main tune loop", eng_path)
changes += ok

write(eng_path, eng)

# ============================================================================
print(f"\n{'='*60}")
print(f"Applied {changes} fix(es). Rebuild with:")
print(f'  $env:CGO_ENABLED="0"; go build ./...')
print(f"{'='*60}")