#!/usr/bin/env python3
"""Quality-first KV type downgrade with asymmetric support.
When model fits on GPU but model+KV doesn't, try progressively more
aggressive KV quantization (preserving K quality first, then V) before
falling back to CPU KV. Works for both dense and MoE models.

Quality ladder (gentlest → most aggressive):
  1. q8_0/q5_1   — keep K lossless, gentle V reduction  (~15% VRAM savings)
  2. q8_0/q4_1   — keep K lossless, moderate V reduction (~21%)
  3. q8_0/q4_0   — keep K lossless, aggressive V          (~24%)
  4. q5_1/q4_1   — gentle K reduction, moderate V         (~35%)
  5. q5_0/q4_0   — moderate both                          (~41%)
  6. q4_0/q4_0   — aggressive both                        (~47%)
  7. turbo4/turbo4 — turboquant symmetric (if supported)   (~53%)
  8. turbo4/turbo3 — turboquant asymmetric (if supported)  (~59%)

Run from go/ module root: python fix_kv_downgrade.py"""

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
# 1. placement.go
# ============================================================================
print("\n=== placement.go ===")
plc_path = os.path.join("pkg", "placement", "placement.go")
plc = read(plc_path)

# 1a. Add BackendHelp to Options struct
old = "\tBackendTag      string"
new = "\tBackendTag      string\n\tBackendHelp     string // backend --help output; gates turbo KV types"
plc, ok = replace_once(plc, old, new, "add BackendHelp to Options struct", plc_path)
changes += ok

# 1b. Add KVTypeV to Strategy struct
old = '\tKVType          string'
new = '\tKVType          string\n\tKVTypeV         string // V cache type for asymmetric KV (empty = same as KVType)'
plc, ok = replace_once(plc, old, new, "add KVTypeV to Strategy struct", plc_path)
changes += ok

# 1c. Add tryKVDowngradeForGPU before resolveAutoKVPlacement
old = "func resolveAutoKVPlacement("
new = """// tryKVDowngradeForGPU finds the highest-quality KV cache type combination
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
\tfreeVRAM := 0
\tfor _, g := range caps.GPUs {
\t\tfreeVRAM += g.VRAMFreeMB()
\t}
\tif freeVRAM <= 0 {
\t\treturn currentKVType, "", false
\t}
\tfitOnGPU := int(float64(freeVRAM-vramOverheadMB) / 0.90)

\t// VRAM the model occupies on GPU. For MoE, only non-expert weights
\t// (attention, norms, embeddings) are GPU-resident; expert FFN weights
\t// are on CPU via --n-cpu-moe.
\tmodelOnGPU := totalSizeMB
\tif model.IsMoE && model.NonExpertBytes > 0 {
\t\tmodelOnGPU = bytesToMiBCeil(model.NonExpertBytes)
\t}
\tif modelOnGPU > fitOnGPU {
\t\treturn currentKVType, "", false
\t}

\t// Quality-first ladder: gentlest reduction first. Keys are more
\t// quantization-sensitive than values, so we keep K at higher quality
\t// and compress V more aggressively before touching K.
\ttype kvCombo struct{ k, v string }
\tcombos := []kvCombo{
\t\t{"q8_0", "q5_1"},  // K lossless, gentle V reduction    (~15% savings)
\t\t{"q8_0", "q4_1"},  // K lossless, moderate V reduction  (~21%)
\t\t{"q8_0", "q4_0"},  // K lossless, aggressive V          (~24%)
\t\t{"q5_1", "q4_1"},  // gentle K, moderate V              (~35%)
\t\t{"q5_0", "q4_0"},  // moderate both                     (~41%)
\t\t{"q4_0", "q4_0"},  // aggressive both, smallest universal (~47%)
\t}
\t// Turbo types are gated on backend support (turboquant forks only).
\t// Placed last: they save the most VRAM but quality is backend-dependent.
\tif strings.Contains(backendHelp, "turbo") {
\t\tcombos = append(combos,
\t\t\tkvCombo{"turbo4", "turbo4"}, // symmetric turbo  (~53%)
\t\t\tkvCombo{"turbo4", "turbo3"}, // asymmetric turbo (~59%)
\t\t)
\t}
\tfor _, c := range combos {
\t\t// Skip if this combo isn't a downgrade from the current type
\t\tif c.k == currentKVType && c.v == currentKVType {
\t\t\tcontinue
\t\t}
\t\ttryKVMB := computeKVTotalMBAsymmetric(model, ctxSize, c.k, c.v)
\t\tif modelOnGPU+tryKVMB <= fitOnGPU {
\t\t\treturn c.k, c.v, true
\t\t}
\t}
\treturn currentKVType, "", false
}

""" + old
plc, ok = replace_once(plc, old, new, "add quality-first tryKVDowngradeForGPU", plc_path)
changes += ok

# 1d. Modify Compute() to call tryKVDowngradeForGPU when placement is "cpu"
old = """\tif opts.KVPlacement == "auto" {
\t\ts.KVPlacement = resolveAutoKVPlacement(caps, model, totalSizeMB, kvTotalMB, vramOverheadMB)
\t} else {
\t\ts.KVPlacement = opts.KVPlacement
\t}"""
new = """\tif opts.KVPlacement == "auto" {
\t\ts.KVPlacement = resolveAutoKVPlacement(caps, model, totalSizeMB, kvTotalMB, vramOverheadMB)
\t\t// Before accepting CPU KV, try progressively smaller KV types
\t\t// (quality-first: gentlest reduction that fits). KV on GPU is
\t\t// always faster than KV on CPU (~10x bandwidth advantage).
\t\tif s.KVPlacement == "cpu" {
\t\t\tif ktK, ktV, fits := tryKVDowngradeForGPU(caps, model, totalSizeMB, vramOverheadMB, s.ContextSize, s.KVType, opts.BackendHelp); fits {
\t\t\t\ts.KVType = ktK
\t\t\t\ts.KVTypeV = ktV
\t\t\t\ts.KVPlacement = "gpu"
\t\t\t\tkvTotalMB = computeKVTotalMBAsymmetric(model, s.ContextSize, ktK, ktV)
\t\t\t}
\t\t}
\t} else {
\t\ts.KVPlacement = opts.KVPlacement
\t}"""
plc, ok = replace_once(plc, old, new, "add KV downgrade fallback in Compute()", plc_path)
changes += ok

# 1e. Update Args() for asymmetric KV emission
old_args = '\tif s.KVType != "" {\n\t\targs = append(args, "--cache-type-k", s.KVType)\n\t\targs = append(args, "--cache-type-v", s.KVType)\n\t}'
new_args = '\tif s.KVType != "" {\n\t\tkvV := s.KVTypeV\n\t\tif kvV == "" {\n\t\t\tkvV = s.KVType\n\t\t}\n\t\targs = append(args, "--cache-type-k", s.KVType, "--cache-type-v", kvV)\n\t}'
plc, ok = replace_once(plc, old_args, new_args, "update Args() for asymmetric KV", plc_path)
if not ok:
    old_args2 = '\t\targs = append(args, "--cache-type-k", s.KVType, "--cache-type-v", s.KVType)'
    new_args2 = '\t\tkvV := s.KVTypeV\n\t\tif kvV == "" {\n\t\t\tkvV = s.KVType\n\t\t}\n\t\targs = append(args, "--cache-type-k", s.KVType, "--cache-type-v", kvV)'
    plc, ok = replace_once(plc, old_args2, new_args2, "update Args() for asymmetric KV (alt)", plc_path)
changes += ok

write(plc_path, plc)

# ============================================================================
# 2. main.go — pass BackendHelp to placement.Options
# ============================================================================
print("\n=== main.go ===")
main_path = os.path.join("cmd", "ggrun", "main.go")
main = read(main_path)

old = "\t\tBackendTag:      be.Tag,"
new = "\t\tBackendTag:      be.Tag,\n\t\tBackendHelp:     be.Help,"
main, ok = replace_once(main, old, new, "pass BackendHelp to placement.Options", main_path)
changes += ok

write(main_path, main)

# ============================================================================
print(f"\n{'='*60}")
if changes > 0:
    print(f"Applied {changes} change(s). Rebuild with:")
    print(f'  $env:CGO_ENABLED="0"; go build -o bin\\ggrun.exe ./cmd/ggrun/')
    print()
    print("Test dense model (should show turbo4/turbo3 KV on GPU):")
    print(f'  .\\bin\\ggrun.exe --model $dense --dry-run --ctx 65536 --server-bin $server')
    print()
    print("Test MoE model (should show best asymmetric KV that fits):")
    print(f'  .\\bin\\ggrun.exe --model $moe --dry-run --ctx 65536 --server-bin $server')
else:
    print("All changes already applied.")
print(f"{'='*60}")