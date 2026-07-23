#!/usr/bin/env python3
"""Fix PredictVRAMUsage compute buffer scaling + remove --fit from gpu-kv combos.
Run from go/ module root: python fix_compute_buf.py"""

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
# 1. placement.go — context-aware compute buffer in PredictVRAMUsage
# ============================================================================
print("\n=== placement.go ===")
plc_path = os.path.join("pkg", "placement", "placement.go")
plc = read(plc_path)

old_compute = """\t// 3. Compute buffer (scales with ubatch)
\tubatch := 512
\tif v, ok := flags["-ub"]; ok {
\t\tif n, err := strconv.Atoi(v); err == nil && n > 0 {
\t\t\tubatch = n
\t\t}
\t}
\tneededMB += firstLaunchComputeBufMB(model, ubatch)"""

new_compute = """\t// 3. Compute buffer (scales with ubatch AND context size)
\tubatch := 512
\tif v, ok := flags["-ub"]; ok {
\t\tif n, err := strconv.Atoi(v); err == nil && n > 0 {
\t\t\tubatch = n
\t\t}
\t}
\tcomputeMB := firstLaunchComputeBufMB(model, ubatch)
\t// Attention graph buffers scale linearly with context size.
\t// At 262144 ctx the compute buffer is ~4x the 65536 baseline
\t// (1320 MB vs ~330 MB on a 9B model). Scale accordingly.
\tif ctxSize > 65536 {
\t\tcomputeMB = computeMB * ctxSize / 65536
\t}
\tneededMB += computeMB"""

plc, ok = replace_once(plc, old_compute, new_compute, "context-aware compute buffer in PredictVRAMUsage", plc_path)
changes += ok

write(plc_path, plc)

# ============================================================================
# 2. engine.go — remove --fit from gpu-kv-turbo4 combos
#    (-ngl 999 is protected and overrides --fit, making it useless)
# ============================================================================
print("\n=== engine.go ===")
eng_path = os.path.join("pkg", "tune", "engine.go")
eng = read(eng_path)

# 2a. Main gpu-kv-turbo4 combo
old1 = '"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "--flash-attn": true, "--fit": "on"},\n\t\t\t"test GPU KV + turbo4 + flash-attn + fit to reclaim GPU speed on constrained VRAM")'
new1 = '"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "--flash-attn": true},\n\t\t\t"test GPU KV + turbo4 + flash-attn to reclaim GPU speed on constrained VRAM")'
eng, ok = replace_once(eng, old1, new1, "remove --fit from gpu-kv-turbo4", eng_path)
changes += ok

# 2b. gpu-kv-turbo4-small-batch
old2 = '"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "--flash-attn": true, "--fit": "on", "-b": fmt.Sprintf("%d", maxInt(batch/2, 1024))},\n\t\t\t\t"test GPU KV + turbo4 + FA + fit with halved batch to fit constrained VRAM")'
new2 = '"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "--flash-attn": true, "-b": fmt.Sprintf("%d", maxInt(batch/2, 1024))},\n\t\t\t\t"test GPU KV + turbo4 + FA with halved batch to fit constrained VRAM")'
eng, ok = replace_once(eng, old2, new2, "remove --fit from gpu-kv-turbo4-small-batch", eng_path)
changes += ok

# 2c. gpu-kv-turbo4-small-ubatch
old3 = '"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "--flash-attn": true, "--fit": "on", "-ub": fmt.Sprintf("%d", maxInt(ubatch/2, 256))},\n\t\t\t\t"test GPU KV + turbo4 + FA + fit with halved ubatch to fit constrained VRAM")'
new3 = '"--no-kv-offload": false, "--cache-type-k": "turbo4", "--cache-type-v": "turbo3", "--flash-attn": true, "-ub": fmt.Sprintf("%d", maxInt(ubatch/2, 256))},\n\t\t\t\t"test GPU KV + turbo4 + FA with halved ubatch to fit constrained VRAM")'
eng, ok = replace_once(eng, old3, new3, "remove --fit from gpu-kv-turbo4-small-ubatch", eng_path)
changes += ok

write(eng_path, eng)

# ============================================================================
print(f"\n{'='*60}")
if changes > 0:
    print(f"Applied {changes} fix(es). Rebuild and test:")
    print(f'  $env:CGO_ENABLED="0"; go build -o bin\\ggrun.exe ./cmd/ggrun/')
    print(f'  .\\bin\\ggrun.exe tune --model $dense --retune --ctx 262144 --rounds 2 --server-bin $server')
else:
    print("All changes already applied.")
print(f"{'='*60}")