package tune

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
