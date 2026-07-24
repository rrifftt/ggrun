const fs = require('fs');

const detectPath = 'C:/Users/raile/ggrun/go/pkg/detect/detect.go';
const placementPath = 'C:/Users/raile/ggrun/go/pkg/placement/placement.go';

// 1. Update detect.go to warn on nvidia-smi failure
let detect = fs.readFileSync(detectPath, 'utf8');
const oldNvidia = `func detectNVIDIA() []GPU {
    out, err := exec.Command("nvidia-smi",
        "--query-gpu=index,pci.bus_id,name,memory.total,memory.used,driver_version,compute_cap",
        "--format=csv,noheader,nounits").Output()
    if err != nil {
        return nil
    }`;
const newNvidia = `func detectNVIDIA() []GPU {
    out, err := exec.Command("nvidia-smi",
        "--query-gpu=index,pci.bus_id,name,memory.total,memory.used,driver_version,compute_cap",
        "--format=csv,noheader,nounits").Output()
    if err != nil {
        if !errors.Is(err, exec.ErrNotFound) {
            fmt.Fprintf(os.Stderr, "ggrun: nvidia-smi ran but failed: %v\\n", err)
        }
        return nil
    }`;
detect = detect.replace(oldNvidia, newNvidia);
// Need to add "errors" to imports if not present
if (!detect.includes('"errors"')) {
    detect = detect.replace('"encoding/json"', '"encoding/json"\n\t"errors"');
}
fs.writeFileSync(detectPath, detect);

// 2. Remove dead constants from placement.go
let placement = fs.readFileSync(placementPath, 'utf8');
placement = placement.replace('\tcomputePerGPUMB     = 512  // legacy; non-MoE single-GPU sizing only\n', '');
placement = placement.replace('\tvramOverheadPercent = 130  // model size * this / 100 = estimated VRAM needed\n', '');
fs.writeFileSync(placementPath, placement);

console.log('Phase 7 code edits applied.');
