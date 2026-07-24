//go:build windows

package tune

import (
    "os"
    "syscall"
    "unsafe"
)

var (
    kernel32         = syscall.NewLazyDLL("kernel32.dll")
    procLockFileEx   = kernel32.NewProc("LockFileEx")
    procUnlockFileEx = kernel32.NewProc("UnlockFileEx")
)

func lockFile(f *os.File) error {
    const LOCKFILE_EXCLUSIVE_LOCK = 0x00000002
    var ol syscall.Overlapped
    ret, _, err := procLockFileEx.Call(
        f.Fd(),
        uintptr(LOCKFILE_EXCLUSIVE_LOCK),
        0,
        1,
        0,
        uintptr(unsafe.Pointer(&ol)),
    )
    if ret == 0 {
        return err
    }
    return nil
}

func unlockFile(f *os.File) error {
    var ol syscall.Overlapped
    ret, _, err := procUnlockFileEx.Call(
        f.Fd(),
        0,
        1,
        0,
        uintptr(unsafe.Pointer(&ol)),
    )
    if ret == 0 {
        return err
    }
    return nil
}
