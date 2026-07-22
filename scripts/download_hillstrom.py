"""Download and verify the official Hillstrom email experiment CSV."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen

from causal_benchmark.hillstrom import HILLSTROM_SHA256, HILLSTROM_SOURCE_URL

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "hillstrom_email_rct.csv"


def main() -> None:
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(HILLSTROM_SOURCE_URL, timeout=120) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest().upper()
    if digest != HILLSTROM_SHA256:
        raise ValueError(f"SHA-256 mismatch: expected {HILLSTROM_SHA256}, received {digest}")
    DESTINATION.write_bytes(payload)
    print(f"Wrote {len(payload):,} bytes to {DESTINATION}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
