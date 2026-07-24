//go:build windows

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
