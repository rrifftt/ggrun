# Contributing

## Build & Test

```bash
cd go
CGO_ENABLED=0 go build ./...
CGO_ENABLED=0 go vet ./...
go test ./... -count=1
```

CI runs on push; keep it green.

## Commits

One task per commit. Message prefix convention: `[phaseN-taskM] description`.

## Extending

- **New backends:** Add to `builtinRecipes` in `pkg/backends/backends.go`.
- **New tune candidates:** Add to `deterministicPlan` in `pkg/tune/engine.go`.
- **New hardware detection:** Add to `pkg/detect/detect.go`.
