# ggrun

auto-tuned llama.cpp launcher with automatic GPU/CPU placement, MoE expert offload, and hardware-aware KV cache management. Point it at a GGUF model and it figures out the optimal `llama-server` flags for your exact hardware.

## Results

Discovered via `ggrun tune` on an RTX 3070 (8 GB) + Ryzen 7 5800X:

| Model | Baseline | Tuned | Speedup | Key flags discovered |
|-------|----------|-------|---------|---------------------|
| Dense 9B Q4_K_M | 32 tok/s | **70.9 tok/s** | 2.2× | turbo4 KV + GPU KV + flash attention |
| MoE 28B Q4_K_M | 17 tok/s | **49.3 tok/s** | 2.9× | `--n-cpu-moe 27` + `--no-mmap` + `-ub 384` |

## How It Works

1. **Detect** — Probes GPUs, CPU, RAM, and available backends
2. **Parse** — Reads GGUF metadata (layers, experts, KV heads, architecture)
3. **Place** — Picks a strategy: SingleGPU, MoEOffload (`--n-cpu-moe`), DenseCPUOffload (`--fit on`), or CPUOnly
4. **Tune** — Benchmarks flag combos in impact order: MoE expert placement → KV type/placement → `--no-mmap` → threads → batch
5. **Refine** — 2nd pass tests compute knobs (b/ub/t/tb) on top of the winning config
6. **Confirm** — Re-measures the winner against baseline to filter noise
7. **Cache** — Saves the result; future launches apply it automatically

## Features

- **Multi-backend**: llama.cpp, ik_llama, Vulkan, Metal, TurboQuant
- **MoE-aware**: strict expert offload — attention stays on GPU, only FFN experts on CPU
- **Turbo KV**: sizes and places turboquant cache types (turbo4/turbo3/turbo2/turbo1)
- **VRAM predictor**: pre-launch OOM prediction skips doomed candidates instantly
- **OOM recovery**: detects CUDA OOM, re-plans with fewer GPU layers, retries
- **Speculative decoding**: MTP, EAGLE3, n-gram, draft model profiles
- **Vision/multimodal**, **community tune sharing**
- **Cross-platform**: Windows, Linux, macOS

## Build

```bash
cd go
CGO_ENABLED=0 go build -o ggrun ./cmd/ggrun/
```

## Usage

```bash
./ggrun detect                          # detect hardware
./ggrun dry-run model.gguf --ctx 65536  # preview flags without launching
./ggrun tune model.gguf --rounds 8      # auto-tune (~5 min)
./ggrun model.gguf --ctx 65536          # launch with cached tune
./ggrun model.gguf --gpus 0,2           # multi-GPU
```

## Project Structure

```
go/
├── cmd/ggrun/       CLI, launch orchestration, OOM recovery
└── pkg/
    ├── placement/   Strategy computation, KV sizing, MoE expert packing, VRAM predictor
    ├── tune/        auto-tune engine, tiered deterministic plan, refinement pass, cache format
    ├── gguf/        Pure-Go GGUF parser
    ├── detect/      Hardware detection (GPU, CPU, RAM, backends)
    ├── benchmark/   HTTP benchmark runner
    ├── server/      Backend process management
    ├── probe/       VRAM/RAM measurement
    ├── recovery/    Crash detection and restart
    ├── config/      User configuration
    └── backends/    Registered fork backend routing
```

## Credits

Originally created by [raketenkater](https://github.com/raketenkater/ggrun). This fork adds the tiered deterministic tune plan, strict MoE expert offload, CPU-aware KV placement, TurboQuant cache type support, refinement pass, and VRAM predictor.

## License

See [LICENSE](LICENSE).
