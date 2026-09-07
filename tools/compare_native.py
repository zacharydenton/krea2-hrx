"""Render one comparison through the native C API using ComfyUI's saved noise."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parent.parent


def png_rgb(path, size, rgb):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    scanlines = b''.join(b'\0' + rgb[y * size * 3:(y + 1) * size * 3] for y in range(size))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' +
                     chunk(b'IHDR', struct.pack('>IIBBBBB', size, size, 8, 2, 0, 0, 0)) +
                     chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))


def main():
    directory = Path(sys.argv[1])
    job = json.loads((directory / 'job.json').read_text())
    n = job['size']
    raw = (directory / 'noise.bin').read_bytes()
    if len(raw) != (n // 16) ** 2 * 64 * 4:
        raise ValueError('Initial noise has the wrong size')
    noise = (C.c_float * (len(raw) // 4)).from_buffer_copy(raw)
    rgb = (C.c_uint8 * (n * n * 3))()
    lib = C.CDLL(str(ROOT / 'build/libkrea2_pipeline.so'))
    ptr, size, char = C.c_void_p, C.c_size_t, C.c_char_p
    lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, size]
    lib.krea2_pipeline_destroy.argtypes = [ptr]
    lib.krea2_pipeline_destroy.restype = None
    lib.krea2_generate.argtypes = [ptr, char, C.c_int, C.c_int, C.c_int, C.c_uint64,
                                  ptr, size, ptr, size, char, size]
    session, error = ptr(), C.create_string_buffer(4096)

    def call(fn, *params):
        if fn(*params, error, len(error)):
            raise RuntimeError(error.value.decode())

    start = time.perf_counter()
    bundle = Path(os.environ.get('KREA2_MODEL') or Path.home() / 'comfy-models/diffusion_models/krea2_turbo_int8_convrot.safetensors')
    call(lib.krea2_pipeline_create, str(bundle).encode(), None, C.byref(session))
    loaded = time.perf_counter()
    try:
        call(lib.krea2_generate, session, job['prompt'].encode(), n, n, job['steps'],
             job['seed'], noise, len(noise), rgb, len(rgb))
        finished = time.perf_counter()
        pixels = bytes(rgb)
        png_rgb(directory / 'loom.png', n, pixels)
        result = dict(runtime="HRX", load_seconds=loaded - start, seconds=finished - loaded,
                      noise_sha256=hashlib.sha256(raw).hexdigest(),
                      rgb_sha256=hashlib.sha256(pixels).hexdigest())
        (directory / 'loom.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    finally:
        lib.krea2_pipeline_destroy(session)


if __name__ == '__main__':
    main()
