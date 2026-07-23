#!/usr/bin/env python3
"""Fix the missing PredictOOM pre-launch check in the main tune loop.
Uses line-based insertion instead of pattern matching.
Run from go/ module root: python fix_oom_check.py"""

import os

eng_path = os.path.join("pkg", "tune", "engine.go")
with open(eng_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the anchor line
anchor = None
for i, line in enumerate(lines):
    if "Candidate flags need their own backend process" in line:
        anchor = i
        break

if anchor is None:
    print("[FAIL] Could not find 'Candidate flags need their own backend process' in engine.go")
    exit(1)

# Check if PredictOOM check is already inserted
already = any("PredictOOM" in lines[j] and "predicted OOM" in lines[j] for j in range(max(0, anchor-20), anchor))
if already:
    print("[skip] PredictOOM pre-launch check already present")
    exit(0)

# Detect indentation from the anchor line
indent = ""
for ch in lines[anchor]:
    if ch in ("\t", " "):
        indent += ch
    else:
        break

print(f"  Found anchor at line {anchor+1}, indent = {repr(indent)}")

# Build the insertion block
insert = f"""{indent}// Pre-launch OOM prediction: skip candidates that are mathematically
{indent}// guaranteed to OOM, saving 30-60s per skipped candidate.
{indent}if e.PredictOOM != nil && e.PredictOOM(candidateFlags) {{
{indent}\tif e.OnProgress != nil {{
{indent}\t\te.OnProgress(fmt.Sprintf("AI-tune: skipping candidate %q (predicted OOM)", suggestion.Name))
{indent}\t}}
{indent}\tcrashed := Entry{{
{indent}\t\tTimestamp:     Now(),
{indent}\t\tModelPath:     modelPath,
{indent}\t\tModelName:     e.Model,
{indent}\t\tHardwareHash:  e.hardwareHash(),
{indent}\t\tBackend:       e.Backend,
{indent}\t\tVision:        e.Vision,
{indent}\t\tRound:         round,
{indent}\t\tName:          suggestion.Name,
{indent}\t\tFlags:         flagMap(candidateFlags),
{indent}\t\tOverrideFlags: overrides,
{indent}\t\tStatus:        "predicted-oom",
{indent}\t}}
{indent}\te.addCache(&crashed)
{indent}\tentries = append(entries, crashed)
{indent}\tcrashedFlagSets = append(crashedFlagSets, overrides)
{indent}\te.saveTuneProgress(modelPath, baseline, best, entries, minImprovementPct, false)
{indent}\tround++
{indent}\tcontinue
{indent}}}
{indent}
"""

# Insert before the anchor line
lines.insert(anchor, insert)

with open(eng_path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(lines)

print("[ok]   Inserted PredictOOM pre-launch check before main tune loop launch")
print(f"       {len(insert.splitlines())} lines inserted at line {anchor+1}")