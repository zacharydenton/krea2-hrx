"""The Krea 2 transformer blocks in Loom, behind the same shape of API as the sibling
repos: a resident session per sequence length, one ctypes call per forward."""
from __future__ import annotations

import ctypes
import json
import os
import time
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
from scripts.build_kernels import build as build_kernels

_ABI = 2
_ERR = 4096
_U16P, _F32P = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float)


class Krea2Error(RuntimeError):
    pass


class Krea2Blocks:
    def __init__(self, tokens: int, layers: int = 28, weights: str | Path | None = None, library: str | Path | None = None):
        self.tokens, self.layers = tokens, layers
        if not 1 <= layers <= 28:
            raise ValueError("layers must be 1..28")
        weights = Path(weights or ROOT / "build/weights")
        library = Path(library or ROOT / "build/libkrea2.so")
        native = ctypes.CDLL(str(library))
        native.krea2_abi_version.restype = ctypes.c_uint32
        if native.krea2_abi_version() != _ABI:
            raise Krea2Error("ABI mismatch; rebuild with scripts/build_host.sh")
        config = weights / "config.json"
        bits = json.loads(config.read_text()).get("bits", 4) if config.is_file() else 4
        kernels = build_kernels(tokens, bits)
        native.krea2_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        native.krea2_run.argtypes = [ctypes.c_void_p, _U16P, ctypes.c_size_t, _F32P, ctypes.c_size_t, _F32P, _F32P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
        native.krea2_run_range.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, *native.krea2_run.argtypes[1:]]
        native.krea2_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        native.krea2_destroy.argtypes = [ctypes.c_void_p]
        self._native = native
        handle = ctypes.c_void_p(); err = ctypes.create_string_buffer(_ERR)
        if native.krea2_create(os.fsencode(weights), os.fsencode(kernels), tokens, layers, ctypes.byref(handle), err, _ERR):
            raise Krea2Error(err.value.decode())
        self._handle = handle

    def close(self):
        if getattr(self, "_handle", None):
            self._native.krea2_destroy(self._handle); self._handle = None

    def __del__(self):
        try: self.close()
        except Exception: pass

    def forward(self, x: torch.Tensor, mods: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                *, first_block: int = 0, block_count: int | None = None) -> torch.Tensor:
        """x [tokens][6144] (any float dtype) -> f16 residual stream after `layers` blocks.
        mods [layers][6][6144] f32, cos/sin [tokens][128] f32.
        An optional contiguous block range uses the same full-session mods layout."""
        block_count = self.layers - first_block if block_count is None else block_count
        if first_block < 0 or block_count < 1 or first_block + block_count > self.layers:
            raise ValueError("block range must be within the loaded layers")
        timing = os.environ.get("KREA2_TIMING") == "1"
        t0 = time.time()
        xa = np.array(x.detach().to(torch.float16).cpu().numpy(), copy=True, order="C")
        ma = np.ascontiguousarray(mods.detach().float().cpu().numpy()[: self.layers])
        ca = np.ascontiguousarray(cos.detach().float().cpu().numpy()); sa = np.ascontiguousarray(sin.detach().float().cpu().numpy())
        t1 = time.time()
        if not (xa.shape == (self.tokens, 6144) and ma.shape == (self.layers, 6, 6144) and ca.shape == sa.shape == (self.tokens, 128)):
            raise ValueError("x, mods and cos/sin must match the session's token and layer counts")
        err = ctypes.create_string_buffer(_ERR)
        args = (xa.ctypes.data_as(_U16P), xa.size, ma.ctypes.data_as(_F32P), ma.size,
                ca.ctypes.data_as(_F32P), sa.ctypes.data_as(_F32P), ca.size, err, _ERR)
        if first_block == 0 and block_count == self.layers:
            rc = self._native.krea2_run(self._handle, *args)
        else:
            rc = self._native.krea2_run_range(self._handle, first_block, block_count, *args)
        if rc:
            raise Krea2Error(err.value.decode())
        t2 = time.time()
        out = torch.from_numpy(xa)
        if timing:
            print(f"  loom wrapper: to-host {t1 - t0:.3f} s, krea2_run {t2 - t1:.3f} s, from-host {time.time() - t2:.3f} s", file=sys.stderr)
        return out

    def profile(self, enable: bool = True):
        self._native.krea2_profile(self._handle, int(enable))
