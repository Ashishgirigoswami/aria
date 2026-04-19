"""Hold a PJRT TPU slice slot indefinitely so other workers can keep running.

On torch_xla 2.9 multi-host v4-32, the runtime initializes a full 4-host
ICI mesh at ``torch_xla.device()`` time. If any host exits its python
process, PJRT tears down the slice and kills the remaining workers.

When one eval job finishes early while others are still running (e.g.
winogrande done in 40 min, hellaswag needs 6 h), the early worker must
keep a python process alive that has claimed its TPU chip — this script
is that idle placeholder. It allocates a single XLA tensor to force PJRT
init, then sleeps until SIGTERM.

Usage:
    setsid nohup python3 scripts/slice_keeper.py &
"""
from __future__ import annotations

import signal
import sys
import time

import torch

import torch_xla

_RUNNING = True


def _handle_term(_signum: int, _frame) -> None:
    global _RUNNING
    _RUNNING = False


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_term)
    signal.signal(signal.SIGINT, _handle_term)

    device = torch_xla.device()
    # Tiny tensor forces the compiler to bind this process to the chip.
    _anchor = torch.ones(1, device=device)
    torch_xla.sync()
    print(f"slice_keeper: holding {device} with anchor on process PID={__import__('os').getpid()}", flush=True)

    # Sleep in small increments so SIGTERM is responsive.
    while _RUNNING:
        time.sleep(10)
    print("slice_keeper: exiting on signal", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
