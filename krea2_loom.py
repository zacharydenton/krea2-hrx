"""The Krea 2 transformer blocks in Loom, behind the same shape of API as the sibling
repos: a resident session per sequence length, one ctypes call per forward."""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "reference"))
import krea2_ref as R

_ABI = 1
_ERR = 4096
_U16P, _F32P = ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float)


class Krea2Error(RuntimeError):
    pass


class Krea2Blocks:
    def __init__(self, tokens: int, layers: int = 28, weights: str | Path | None = None, library: str | Path | None = None):
        self.tokens, self.layers = tokens, layers
        weights = Path(weights or ROOT / "build/weights")
        library = Path(library or ROOT / "build/libkrea2.so")
        kernels = ROOT / "build/kernels" / f"T{tokens}"
        if not (kernels / "attention.hsaco").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/build_kernels.py"), str(tokens)], check=True, capture_output=True)
        native = ctypes.CDLL(str(library))
        native.krea2_abi_version.restype = ctypes.c_uint32
        if native.krea2_abi_version() != _ABI:
            raise Krea2Error("ABI mismatch; rebuild with scripts/build_host.sh")
        native.krea2_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_size_t]
        native.krea2_run.argtypes = [ctypes.c_void_p, _U16P, ctypes.c_size_t, _F32P, ctypes.c_size_t, _F32P, _F32P, ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t]
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

    def forward(self, x: torch.Tensor, mods: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """x [tokens][6144] (any float dtype) -> f16 residual stream after `layers` blocks.
        mods [layers][6][6144] f32, cos/sin [tokens][128] f32."""
        xa = np.ascontiguousarray(x.detach().to(torch.float16).cpu().numpy())
        ma = np.ascontiguousarray(mods.detach().float().cpu().numpy()[: self.layers])
        ca = np.ascontiguousarray(cos.detach().float().cpu().numpy()); sa = np.ascontiguousarray(sin.detach().float().cpu().numpy())
        assert xa.shape == (self.tokens, 6144) and ma.shape == (self.layers, 6, 6144) and ca.shape == sa.shape == (self.tokens, 128)
        err = ctypes.create_string_buffer(_ERR)
        rc = self._native.krea2_run(self._handle, xa.ctypes.data_as(_U16P), xa.size, ma.ctypes.data_as(_F32P), ma.size,
                                    ca.ctypes.data_as(_F32P), sa.ctypes.data_as(_F32P), ca.size, err, _ERR)
        if rc:
            raise Krea2Error(err.value.decode())
        return torch.from_numpy(xa.copy())

    def profile(self, enable: bool = True):
        self._native.krea2_profile(self._handle, int(enable))
