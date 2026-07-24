const fs = require('fs');
const path = require('path');

const dir = 'C:/Users/raile/ggrun/go/pkg/placement';
const placementPath = path.join(dir, 'placement.go');
const cachePath = path.join(dir, 'cache.go');

let placement = fs.readFileSync(placementPath, 'utf8');
let cache = fs.existsSync(cachePath) ? fs.readFileSync(cachePath, 'utf8') : '';

const importMatch = placement.match(/import \(([\s\S]*?)\)/);
const imports = importMatch ? `package placement\n\nimport (\n${importMatch[1]}\n)\n\n` : 'package placement\n\n';

function extractFunc(content, funcName) {
    const regex = new RegExp(`func (\\([^\\)]*\\) )?${funcName}\\(`, 'g');
    const match = regex.exec(content);
    if (!match) return null;

    let start = match.index;
    let braceIndex = content.indexOf('{', start);
    if (braceIndex === -1) return null;

    let depth = 1;
    let i = braceIndex + 1;
    while (depth > 0 && i < content.length) {
        if (content[i] === '{') depth++;
        else if (content[i] === '}') depth--;
        i++;
    }
    
    const block = content.substring(start, i);
    const newContent = content.replace(block, '');
    return { block, newContent };
}

const targets = {
    'gpu_order.go': ['orderGPUsByBandwidth', 'normalizeSplit', 'splitCompactKey', 'ceilDivInt', 'bytesToMiBCeil', 'maxInt', 'minInt'],
    'vram.go': ['PredictVRAMUsage', 'ParseFlagsToMap', 'checkMemoryOrDie', 'firstLaunchComputeBufMB', 'measuredCUDAOverheadMB'],
    'cram.go': ['computeCRAM'],
    'kv.go': ['computeKVTotalMB', 'computeKVTotalMBAsymmetric', 'parseKVType', 'resolveAutoKVPlacement', 'computeAutoContextSizeKVPlacement'],
    'moe.go': ['buildMoEOffload', 'maximizeMoEGPUFitByUBatch', 'buildOTStringFromAssignments'],
    'dense.go': ['buildCPUOnly', 'buildSingleGPU', 'buildMultiGPUDense', 'buildDenseCPUOffload'],
    'probe_cache.go': ['loadProbeCache', 'saveProbeCache', 'loadSystemProbe', 'PlacementCachePathFor', 'placementCachePathForSpecMode', 'LoadPlacementCache', 'SavePlacementCache', 'StrategyToCacheEntry', 'SystemCUDAOverheadByGPU', 'parseGPUAssignments', 'parseTensorSplit']
};

let probeCacheContent = '';

for (const funcName of targets['probe_cache.go']) {
    const res = extractFunc(cache, funcName);
    if (res) {
        probeCacheContent += res.block + '\n\n';
        cache = res.newContent;
    }
}

for (const [file, funcs] of Object.entries(targets)) {
    let content = '';
    for (const funcName of funcs) {
        const res = extractFunc(placement, funcName);
        if (res) {
            content += res.block + '\n\n';
            placement = res.newContent;
        }
    }
    if (file === 'probe_cache.go') {
        fs.writeFileSync(path.join(dir, file), imports + probeCacheContent + content);
    } else {
        fs.writeFileSync(path.join(dir, file), imports + content);
    }
}

fs.writeFileSync(placementPath, placement);
if (fs.existsSync(cachePath)) fs.unlinkSync(cachePath);

console.log('Phase 3 split complete.');
