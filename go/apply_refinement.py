#!/usr/bin/env python3
"""Add 2nd-pass refinement to the tune engine.
After the main tune finds the best placement/KV config, run a second pass
testing compute knobs (b/ub/t/tb) on top of the winning config.

Run from the go/ module root: python apply_refinement.py"""

import sys, os

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
        print(f"  [FAIL] {label} — pattern not found in {path}")
        print(f"         Looking for: {repr(old[:100])}...")
        return src, False
    count = src.count(old)
    if count > 1:
        print(f"  [FAIL] {label} — pattern found {count} times (expected 1)")
        return src, False
    src = src.replace(old, new, 1)
    print(f"  [ok]   {label}")
    return src, True

changes = 0

# ============================================================================
# 1. engine.go
# ============================================================================
print("\n=== engine.go ===")
eng_path = os.path.join("pkg", "tune", "engine.go")
eng = read(eng_path)

# 1a. Add RefinementRounds to Engine struct
old = """\tBenchmarkTimeout     time.Duration
\tServerStartupTimeout time.Duration
\tBackendHelp          string"""
new = """\tBenchmarkTimeout     time.Duration
\tServerStartupTimeout time.Duration
\tBackendHelp          string
\tRefinementRounds     int // 2nd-pass: compute knobs (b/ub/t/tb) on top of the winner"""
eng, ok = replace_once(eng, old, new, "add RefinementRounds to Engine struct", eng_path)
changes += ok

# 1b. Add refinement loop after confirmation pass, before final "done" message
old = """\tif e.OnProgress != nil {
\t\te.OnProgress(fmt.Sprintf("AI-tune: done. Best result: %.1f tok/s", best.Result.GenTPS))
\t}"""
new = """\t// =================================================================
\t// Refinement pass: test compute knobs (b/ub/t/tb) on top of the
\t// winning placement/KV config. The main loop finds the best data
\t// placement; this pass finds the best compute settings for that
\t// placement. Each candidate is applied on top of the winner's flags,
\t// not the original baseline.
\t// =================================================================
\tif e.RefinementRounds > 0 && best != nil && best != baseline && len(best.OverrideFlags) > 0 {
\t\tif e.OnProgress != nil {
\t\t\te.OnProgress(fmt.Sprintf("AI-tune: refinement pass (%d rounds) on top of %s...", e.RefinementRounds, best.Name))
\t\t}
\t\twinnerFlags := ApplyOverrides(initialFlags, best.OverrideFlags, protected)
\t\trefPlan := refinementPlan(winnerFlags, e.Backend, e.Caps, e.BackendHelp)
\t\tfor refIdx := 0; refIdx < len(refPlan) && refIdx < e.RefinementRounds; refIdx++ {
\t\t\tc := refPlan[refIdx]
\t\t\to := sanitizeFlagValues(c.FlagValues, protected)
\t\t\tcandidateFlags := ApplyOverrides(winnerFlags, o, protected)
\t\t\tif equalFlags(winnerFlags, candidateFlags) {
\t\t\t\tif e.OnProgress != nil {
\t\t\t\t\te.OnProgress(fmt.Sprintf("AI-tune: refinement candidate %q made no effective flag changes", c.Name))
\t\t\t\t}
\t\t\t\tcontinue
\t\t\t}
\t\t\tstopBaseline()
\t\t\trefRound := e.Rounds + 3 + refIdx
\t\t\tcandidate, err := e.round(refRound, modelPath, candidateFlags)
\t\t\tif err != nil {
\t\t\t\tcrashed := Entry{
\t\t\t\t\tTimestamp:     Now(),
\t\t\t\t\tModelPath:     modelPath,
\t\t\t\t\tModelName:     e.Model,
\t\t\t\t\tHardwareHash:  e.hardwareHash(),
\t\t\t\t\tBackend:       e.Backend,
\t\t\t\t\tVision:        e.Vision,
\t\t\t\t\tRound:         refRound,
\t\t\t\t\tName:          c.Name,
\t\t\t\t\tFlags:         flagMap(candidateFlags),
\t\t\t\t\tOverrideFlags: mergeOverrides(best.OverrideFlags, o),
\t\t\t\t\tStatus:        "crashed",
\t\t\t\t}
\t\t\t\te.addCache(&crashed)
\t\t\t\tentries = append(entries, crashed)
\t\t\t\tif e.OnProgress != nil {
\t\t\t\t\te.OnProgress(fmt.Sprintf("AI-tune: refinement candidate failed: %v", err))
\t\t\t\t}
\t\t\t\tcontinue
\t\t\t}
\t\t\tcandidate.Name = c.Name
\t\t\tcandidate.OverrideFlags = mergeOverrides(best.OverrideFlags, o)
\t\t\tcandidate.Status = "ok"
\t\t\tcandidate.Backend = e.Backend
\t\t\tcandidate.Vision = e.Vision
\t\t\te.addCache(candidate)
\t\t\tentries = append(entries, *candidate)
\t\t\tif meaningfulImprovement(candidate.Result.GenTPS, best.Result.GenTPS, minImprovementPct) {
\t\t\t\tbest = candidate
\t\t\t\twinnerFlags = ApplyOverrides(initialFlags, best.OverrideFlags, protected)
\t\t\t\tif e.OnProgress != nil {
\t\t\t\t\te.OnProgress(fmt.Sprintf("AI-tune: refinement new best %.1f tok/s (%s)", best.Result.GenTPS, c.Name))
\t\t\t\t}
\t\t\t}
\t\t\te.saveTuneProgress(modelPath, baseline, best, entries, minImprovementPct, false)
\t\t}
\t}

\tif e.OnProgress != nil {
\t\te.OnProgress(fmt.Sprintf("AI-tune: done. Best result: %.1f tok/s", best.Result.GenTPS))
\t}"""
eng, ok = replace_once(eng, old, new, "add refinement loop after confirmation pass", eng_path)
changes += ok

# 1c. Add refinementPlan and mergeOverrides before guardRiskyMoEOverrides
old = """func guardRiskyMoEOverrides(overrides map[string]interface{}, baseFlags []string) map[string]interface{} {"""
new = """// refinementPlan generates compute-knob candidates (batch, ubatch, threads)
// for the 2nd-pass refinement. Tested on top of the winning placement/KV
// config, not the original baseline.
func refinementPlan(baseFlags []string, backend string, caps *detect.Capabilities, backendHelp string) []Suggestion {
\tbase := flagMap(baseFlags)
\tbatch := atoiDefault(base["-b"], 4096)
\tubatch := atoiDefault(base["-ub"], 512)
\tisMoEOffload := isMoEOffloadFlags(base)

\tcandidates := []Suggestion{}
\tseen := map[string]bool{}
\tadd := func(name string, values map[string]interface{}, reasoning string) {
\t\tvalues = sanitizeFlagValues(values, nil)
\t\tif len(values) == 0 {
\t\t\treturn
\t\t}
\t\tkey := suggestionKey(values)
\t\tif seen[key] {
\t\t\treturn
\t\t}
\t\tseen[key] = true
\t\tcandidates = append(candidates, Suggestion{
\t\t\tName:       name,
\t\t\tFlagValues: values,
\t\t\tFlags:      flagValuesToArgs(values),
\t\t\tReasoning:  reasoning,
\t\t})
\t}

\t// Batch
\tif isMoEOffload && batch > 1024 {
\t\tadd("ref-smaller-moe-batch", map[string]interface{}{"-b": fmt.Sprintf("%d", maxInt(batch/2, 1024))}, "refine: lower MoE batch to reduce expert handoff pressure")
\t\tif batch > 1536 {
\t\t\tadd("ref-moe-batch-1536", map[string]interface{}{"-b": "1536"}, "refine: middle MoE batch size")
\t\t}
\t}
\tif !isMoEOffload && batch > 0 && batch < 8192 {
\t\tadd("ref-larger-batch", map[string]interface{}{"-b": fmt.Sprintf("%d", minInt(batch*2, 8192))}, "refine: larger batch for prefill throughput")
\t}
\tif !isMoEOffload && batch > 0 && batch < 16384 {
\t\tadd("ref-max-prefill-batch", map[string]interface{}{"-b": "16384"}, "refine: aggressive prefill batch")
\t}

\t// Ubatch
\tif isMoEOffload && ubatch > 384 {
\t\tadd("ref-moe-ubatch-384", map[string]interface{}{"-ub": "384"}, "refine: moderate MoE microbatch")
\t}
\tif ubatch > 256 {
\t\tadd("ref-smaller-ubatch", map[string]interface{}{"-ub": fmt.Sprintf("%d", maxInt(ubatch/2, 256))}, "refine: smaller microbatch to reduce compute buffer pressure")
\t}
\tub, _ := strconv.Atoi(base["-ub"])
\tif ub > 0 {
\t\tadd("ref-larger-ubatch-2x", map[string]interface{}{"-ub": fmt.Sprintf("%d", minInt(ub*2, 4096))}, "refine: 2x ubatch for prefill speed")
\t\tadd("ref-larger-ubatch-4x", map[string]interface{}{"-ub": fmt.Sprintf("%d", minInt(ub*4, 4096))}, "refine: 4x ubatch for prefill speed")
\t}

\t// Threads
\tif caps != nil {
\t\tif caps.CPU.Cores > 0 && atoiDefault(base["--threads"], 0) != caps.CPU.Cores {
\t\t\tadd("ref-threads-physical", map[string]interface{}{"--threads": fmt.Sprintf("%d", caps.CPU.Cores)}, "refine: pin decode threads to physical cores")
\t\t}
\t\tif caps.CPU.Threads > caps.CPU.Cores && atoiDefault(base["--threads-batch"], 0) != caps.CPU.Threads {
\t\t\tadd("ref-threads-batch-logical", map[string]interface{}{"--threads-batch": fmt.Sprintf("%d", caps.CPU.Threads)}, "refine: logical threads for prefill")
\t\t}
\t}

\treturn candidates
}

// mergeOverrides returns the union of two override maps. Extra wins on conflict.
func mergeOverrides(base, extra map[string]interface{}) map[string]interface{} {
\tmerged := make(map[string]interface{}, len(base)+len(extra))
\tfor k, v := range base {
\t\tmerged[k] = v
\t}
\tfor k, v := range extra {
\t\tmerged[k] = v
\t}
\treturn merged
}

func guardRiskyMoEOverrides(overrides map[string]interface{}, baseFlags []string) map[string]interface{} {"""
eng, ok = replace_once(eng, old, new, "add refinementPlan and mergeOverrides functions", eng_path)
changes += ok

write(eng_path, eng)

# ============================================================================
# 2. main.go — set RefinementRounds in cmdTune
# ============================================================================
print("\n=== main.go ===")
main_path = os.path.join("cmd", "ggrun", "main.go")
main = read(main_path)

old = """\tengine := &tune.Engine{
\t\tBaseURL:          fmt.Sprintf("http://localhost:%d", req.Port),
\t\tModel:            filepath.Base(req.ModelPath),
\t\tRounds:           rounds,
\t\tCache:            cache,"""
new = """\tengine := &tune.Engine{
\t\tBaseURL:          fmt.Sprintf("http://localhost:%d", req.Port),
\t\tModel:            filepath.Base(req.ModelPath),
\t\tRounds:           rounds,
\t\tRefinementRounds: 4,
\t\tCache:            cache,"""
main, ok = replace_once(main, old, new, "set RefinementRounds=4 in cmdTune", main_path)
changes += ok

write(main_path, main)

# ============================================================================
# Summary
# ============================================================================
print(f"\n{'='*60}")
if changes > 0:
    print(f"Applied {changes} change(s). Rebuild with:")
    print(f'  $env:CGO_ENABLED="0"; go build ./...')
else:
    print("All changes already applied. Nothing to do.")
print(f"{'='*60}")