const fs = require('fs');
const path = require('path');

const root = 'C:/Users/raile/ggrun/go';
const tuneDir = path.join(root, 'pkg', 'tune');
const placementDir = path.join(root, 'pkg', 'placement');

// 1. Create flock_unix.go
const flockUnix = `//go:build !windows

package tune

import (
    "os"
    "syscall"
)

func lockFile(f *os.File) error {
    return syscall.Flock(int(f.Fd()), syscall.LOCK_EX)
}

func unlockFile(f *os.File) error {
    return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
`;
fs.writeFileSync(path.join(tuneDir, 'flock_unix.go'), flockUnix);

// 2. Create flock_windows.go
const flockWindowsFixed = `//go:build windows

package tune

import (
    "os"
    "syscall"
    "unsafe"
)

var (
    kernel32       = syscall.NewLazyDLL("kernel32.dll")
    procLockFile   = kernel32.NewProc("LockFile")
    procUnlockFile = kernel32.NewProc("UnlockFile")
)

func lockFile(f *os.File) error {
    const LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    const LOCKFILE_FAIL_IMMEDIATELY = 0x00000001
    var ol syscall.Overlapped
    ret, _, err := procLockFile.Call(f.Fd(), 1, 0, 0, 1, uintptr(LOCKFILE_EXCLUSIVE_LOCK), 0, 0, uintptr(unsafe.Pointer(&ol)))
    if ret == 0 {
        return err
    }
    return nil
}

func unlockFile(f *os.File) error {
    var ol syscall.Overlapped
    ret, _, err := procUnlockFile.Call(f.Fd(), 0, 0, 1, 0, 0, 0, uintptr(unsafe.Pointer(&ol)))
    if ret == 0 {
        return err
    }
    return nil
}
`;
fs.writeFileSync(path.join(tuneDir, 'flock_windows.go'), flockWindowsFixed);

// 3. Create flock.go (wrapper)
const flockGo = `package tune

import (
    "os"
)

func AcquireLock(path string) (*os.File, func(), error) {
    f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE, 0644)
    if err != nil {
        return nil, nil, err
    }
    if err := lockFile(f); err != nil {
        f.Close()
        return nil, nil, err
    }
    return f, func() {
        unlockFile(f)
        f.Close()
    }, nil
}
`;
fs.writeFileSync(path.join(tuneDir, 'flock.go'), flockGo);

// 4. Update tune.go (Add schema version and locking)
const tunePath = path.join(tuneDir, 'tune.go');
let tune = fs.readFileSync(tunePath, 'utf8');

// Add schema version constant
tune = tune.replace('type Entry struct {', 'const TuneSchemaVersion = 2\n\ntype Entry struct {');
// Add SchemaVersion field to Entry
tune = tune.replace('Best          bool                   `json:"best"`', 'Best          bool                   `json:"best"`\n\tSchemaVersion int                    `json:"schema_version"`');

// Update Load to check version
tune = tune.replace(`func (c *Cache) Load() ([]Entry, error) {
    data, err := os.ReadFile(c.path)
    if err != nil {
        if os.IsNotExist(err) {
            return []Entry{}, nil
        }
        return nil, err
    }
    var entries []Entry
    if err := json.Unmarshal(data, &entries); err != nil {
        return nil, err
    }
    return entries, nil
}`, `func (c *Cache) Load() ([]Entry, error) {
    data, err := os.ReadFile(c.path)
    if err != nil {
        if os.IsNotExist(err) {
            return []Entry{}, nil
        }
        return nil, err
    }
    var entries []Entry
    if err := json.Unmarshal(data, &entries); err != nil {
        return nil, err
    }
    // Validate schema version
    for i := range entries {
        if entries[i].SchemaVersion > TuneSchemaVersion {
            return nil, fmt.Errorf("cache schema version %d is newer than supported %d", entries[i].SchemaVersion, TuneSchemaVersion)
        }
    }
    return entries, nil
}`);

// Update Add to use locking and set version
tune = tune.replace(`func (c *Cache) Add(entry Entry) error {
    entries, err := c.Load()
    if err != nil {
        return err
    }`, `func (c *Cache) Add(entry Entry) error {
    entry.SchemaVersion = TuneSchemaVersion
    
    // Acquire exclusive lock
    f, unlock, err := AcquireLock(c.path)
    if err != nil {
        return err
    }
    defer unlock()
    
    // Re-read under lock
    entries, err := c.Load()
    if err != nil {
        return err
    }`);

fs.writeFileSync(tunePath, tune);

// 5. Update placement/cache.go (Add schema version to CacheEntry)
const cachePath = path.join(placementDir, 'cache.go');
if (fs.existsSync(cachePath)) {
    let cache = fs.readFileSync(cachePath, 'utf8');
    cache = cache.replace('type CacheEntry struct {', 'const PlacementSchemaVersion = 2\n\ntype CacheEntry struct {');
    fs.writeFileSync(cachePath, cache);
}

console.log('Phase 5 edits applied.');
