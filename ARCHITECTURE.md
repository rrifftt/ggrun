# Architecture

## Pipeline

```
flowchart LR
    A[1. Detect<br/>GPU/CPU/RAM/backends] --> B[2. Parse GGUF<br/>layers, experts, KV heads]
    B --> C[3. Place<br/>pick strategy]
    C --> D[4. Tune<br/>benchmark flag combos<br/>in impact order]
    D --> E[5. Refine<br/>2nd pass on compute knobs<br/>b/ub/t/tb]
    E --> F[6. Confirm<br/>re-measure vs baseline]
    F --> G[7. Cache<br/>reuse on next launch]
    C -.-> |SingleGPU / MoEOffload<br/>DenseCPUOffload / CPUOnly| D
```

## Packages

| Package | Description |
|---------|-------------|
| `cmd/ggrun` | CLI dispatch, launch orchestration, OOM recovery |
| `pkg/placement` | Strategy computation, KV sizing, MoE expert packing, VRAM predictor |
| `pkg/tune` | Auto-tune engine, tiered deterministic plan, refinement pass, cache format |
| `pkg/gguf` | Pure-Go GGUF metadata parser |
| `pkg/detect` | Hardware detection (GPU, CPU, RAM, backends) |
| `pkg/benchmark` | HTTP benchmark runner |
| `pkg/server` | Backend process management |
| `pkg/probe` | Live VRAM/RAM measurement |
| `pkg/recovery` | Crash detection and restart |
| `pkg/backends` | Registered fork backend routing |

## Invariants

**Quality vs Performance Flags:** The tune engine must never override flags that affect output quality or effective context (`--cache-type-k/-v`, `--parallel`, `--ctx-size`, placement, tensor-split). These are user-owned.

**Cache Scoping:** Tune caches are keyed by model path + hardware hash + backend + vision. Speculative profiles add a finer scope.

**VRAM Accounting:** `EstimateVRAMNeed` is the single source of truth for VRAM fitting. Both the placement engine and the tune predictor must use it.
