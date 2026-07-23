#!/usr/bin/env python3
"""Fix PredictVRAMUsage: remove KV from prediction, tighten margin.
Run from go/ module root: python fix_predict.py"""

path = "pkg/placement/placement.go"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the line "// 2. KV cache on GPU" inside PredictVRAMUsage
start = None
end = None
for i, line in enumerate(lines):
    if "// 2. KV cache on GPU" in line and start is None:
        start = i
    if start is not None and line.strip() == "return neededMB, freeMB":
        end = i
        break

if start is None or end is None:
    print(f"FAIL: could not find markers (start={start}, end={end})")
    exit(1)

print(f"Replacing lines {start+1}-{end+1}")

new_block = """\t// 2. Context size (needed for compute buffer estimate)
\tctxSize := model.ContextSize
\tif v, ok := flags["--ctx-size"]; ok {
\t\tif n, err := strconv.Atoi(v); err == nil && n > 0 {
\t\t\tctxSize = n
\t\t}
\t}

\t// 3. Compute buffer: empirical estimate from llama.cpp measurements.
\t// ~330 MB at 65536 ctx on a 9B model, scales linearly with context.
\tcomputeMB := ctxSize / 200
\tif computeMB < 128 {
\t\tcomputeMB = 128
\t}

\t// 4. Overhead: CUDA context + internal buffers + fragmentation.
\t// Measured ~300-500 MB on NVIDIA GPUs; 400 MB is a safe middle ground.
\tconst overheadMB = 400

\t// Predict the actual allocation that fails: model weights + compute
\t// buffer + overhead. KV cache is allocated separately and the backend
\t// handles KV pressure via --fit or context reduction. Excluding KV
\t// avoids false positives on MoE models where model+KV exceeds free
\t// VRAM but the backend still fits both via lazy/mmap allocation.
\tneededMB = neededMB + computeMB + overheadMB
\tneededMB = neededMB * 105 / 100 // 5% margin for alignment

\treturn neededMB, freeMB
"""

lines[start:end+1] = [new_block]

with open(path, "w", encoding="utf-8", newline="\n") as f:
    f.writelines(lines)

print("OK — replaced KV+compute+CUDA+margin with compute+overhead+5% margin")