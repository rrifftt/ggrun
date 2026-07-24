package probe

import (
	"testing"
)

func TestProbeReturnsMemoryOrError(t *testing.T) {
	m, err := Probe()
	if m == nil && err == nil {
		t.Fatalf("expected either memory or error, got both nil")
	}
	if m != nil {
		_ = m.TotalFree()
		_ = m.String()
	}
}
