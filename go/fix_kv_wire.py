#!/usr/bin/env python3
"""Wire up KV downgrade: Compute() call site + main.go BackendHelp.
Run from go/ module root: python fix_kv_wire.py"""

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
# 1. placement.go — add downgrade call after resolveAutoKVPlacement
# ============================================================================
print("\n=== placement.go ===")
plc_path = os.path.join("pkg", "placement", "placement.go")
plc = read(plc_path)

old = """\t\t\ts.KVPlacement = resolveAutoKVPlacement(caps, model, totalSizeMB, kvNeedMB, perGPUOH*len(caps.GPUs))
\t\t}
\t}"""

new = """\t\t\ts.KVPlacement = resolveAutoKVPlacement(caps, model, totalSizeMB, kvNeedMB, perGPUOH*len(caps.GPUs))
\t\t\t// Before accepting CPU KV, try progressively smaller KV types
\t\t\t// (quality-first: gentlest reduction that fits). KV on GPU is
\t\t\t// always faster than KV on CPU (~10x bandwidth advantage).
\t\t\tif s.KVPlacement == "cpu" {
\t\t\t\tif ktK, ktV, fits := tryKVDowngradeForGPU(caps, model, totalSizeMB, perGPUOH*len(caps.GPUs), s.ContextSize, s.KVType, opts.BackendHelp); fits {
\t\t\t\t\ts.KVType = ktK
\t\t\t\t\ts.KVTypeK = ktK
\t\t\t\t\ts.KVTypeV = ktV
\t\t\t\t\ts.KVPlacement = "gpu"
\t\t\t\t}
\t\t\t}
\t\t}
\t}"""

plc, ok = replace_once(plc, old, new, "add KV downgrade after resolveAutoKVPlacement", plc_path)
changes += ok

write(plc_path, plc)

# ============================================================================
# 2. main.go — pass BackendHelp to placement.Options
# ============================================================================
print("\n=== main.go ===")
main_path = os.path.join("cmd", "ggrun", "main.go")
main = read(main_path)

old = "\t\t\tBackendTag:             backendDialect(be),"
new = "\t\t\tBackendTag:             backendDialect(be),\n\t\t\tBackendHelp:            be.Help,"
main, ok = replace_once(main, old, new, "pass BackendHelp to placement.Options", main_path)
changes += ok

write(main_path, main)

# ============================================================================
print(f"\n{'='*60}")
if changes > 0:
    print(f"Applied {changes} change(s). Rebuild:")
    print(f'  $env:CGO_ENABLED="0"; go build -o bin\\ggrun.exe ./cmd/ggrun/')
else:
    print("All changes already applied.")
print(f"{'='*60}")