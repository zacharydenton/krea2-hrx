"""Compare native pipeline components with installed model implementations."""
import ctypes as C
import gc
import os
from pathlib import Path
import sys
import subprocess
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(os.environ.get("KREA2_TEST_BIN", ROOT / "build"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reference"))


def main():
    lib = C.CDLL(str(BIN / "libkrea2.so"))
    ptr, size, char = C.c_void_p, C.c_size_t, C.c_char_p
    lib.krea2_pipeline_create.argtypes = [char, char, C.POINTER(ptr), char, size]
    lib.krea2_pipeline_destroy.argtypes = [ptr]
    lib.krea2_tokenize.argtypes = [ptr, char, ptr, size, C.POINTER(size), char, size]
    lib.krea2_encode.argtypes = [ptr, char, ptr, size, C.POINTER(size), char, size]
    lib.krea2_transformer.argtypes = [ptr, ptr, size, C.c_int, ptr, size, C.c_int, C.c_int, C.c_float, ptr, size, char, size]
    lib.krea2_generate.argtypes = [ptr, char, C.c_int, C.c_int, C.c_int, C.c_uint64, ptr, size, ptr, size, char, size]
    lib.krea2_decode.argtypes = [ptr, ptr, size, C.c_int, C.c_int, ptr, size, char, size]
    err, session = C.create_string_buffer(4096), ptr()
    compiler = os.environ.get("LOOM_COMPILE", str(Path.home() / "code/hrx-system/build-cuda/loom/src/loom/tools/loom-compile/loom-compile"))
    def call(fn, *args):
        rc = fn(*args, err, len(err))
        if rc:
            raise RuntimeError(err.value.decode())
    from krea2_loom import DEFAULT_MODEL
    call(lib.krea2_pipeline_create, os.fsencode(DEFAULT_MODEL), compiler.encode(), C.byref(session))
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(Path.home() / "krea2-models/qwen3-vl-4b"))
        for text in ("a red fox", "Hello, world!\n12345", "räksmörgås 日本語 🦊", "<|im_start|>user\nhello<|im_end|>", "", "  white\tspace\n\n", "cafe\u0301 A\u030a", "a " * 600):
            ids = np.zeros(10000, np.int32); written = size()
            call(lib.krea2_tokenize, session, text.encode(), ids.ctypes.data, ids.size, C.byref(written))
            want = tokenizer.encode(text, add_special_tokens=False)
            assert ids[:written.value].tolist() == want, (text, ids[:written.value].tolist(), want)
        print("PASS native tokenizer", flush=True)
        from tools.pipeline import ReferenceForward
        # Only text encoding and VAE methods are needed from the pipeline.
        # Keep a meta transformer for its configuration, avoiding a second
        # resident 13B transformer alongside the independent reference.
        from diffusers import Krea2Pipeline, FlowMatchEulerDiscreteScheduler, AutoencoderKLQwenImage
        from diffusers.models.transformers.transformer_krea2 import Krea2Transformer2DModel
        from transformers import Qwen3VLModel
        from tools.pipeline import QWEN, VAE, SCHEDULER
        with torch.device("meta"):
            config_transformer = Krea2Transformer2DModel()
        text_encoder = Qwen3VLModel.from_pretrained(str(QWEN), torch_dtype=torch.bfloat16).cuda()
        pipe = Krea2Pipeline(scheduler=FlowMatchEulerDiscreteScheduler(**SCHEDULER),
                            vae=None, text_encoder=text_encoder, tokenizer=tokenizer,
                            transformer=config_transformer, is_distilled=True)
        del text_encoder, config_transformer
        prompt = "a red fox"
        with torch.no_grad():
            states, mask = pipe.get_text_hidden_states(prompt)
        states = states[:, mask[0]].contiguous()
        count = size()
        call(lib.krea2_encode, session, prompt.encode(), None, 0, C.byref(count))
        assert count.value == states.shape[1]
        taps = np.empty((count.value, 12, 2560), np.float32)
        call(lib.krea2_encode, session, prompt.encode(), taps.ctypes.data, taps.size, C.byref(count))
        want = states[0].float().cpu().numpy()
        cosine = float(np.dot(taps.ravel(), want.ravel()) / (np.linalg.norm(taps) * np.linalg.norm(want)))
        print(f"text encoder cosine {cosine:.8f}, max_abs {np.max(np.abs(taps-want)):.5g}", flush=True)
        assert cosine > 0.995
        pipe.text_encoder = None
        gc.collect()
        torch.cuda.empty_cache()
        # Feed identical text states and latents to isolate the full transformer.
        from safetensors.torch import load_file
        import krea2_ref as R
        weights = load_file(str(Path.home() / "krea2-models/krea2_turbo_bf16.safetensors"), device="cpu")
        reference = R.Krea2Ref(weights, quant="none")
        adapter = ReferenceForward(reference, None, "loom")
        with tempfile.TemporaryDirectory() as td, torch.no_grad():
            td = Path(td)
            want.tofile(td / "text.bin")
            hidden = torch.linspace(-2, 2, 6144, device="cuda").bfloat16()[None, None]
            hidden.float().cpu().numpy().tofile(td / "hidden.bin")
            subprocess.run([str(BIN / "krea2-native-components"), str(DEFAULT_MODEL), str(td)], check=True)
            e, m = reference.time_embed(torch.tensor([0.75], device="cuda", dtype=torch.bfloat16))
            for name, expected_component in (("condition", reference.text_in(states)), ("temb", e), ("mod", m), ("final", reference.final(hidden, e))):
                a = np.fromfile(td / f"{name}.bin", np.float32)
                b = expected_component.float().cpu().numpy().ravel()
                cs = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
                print(f"{name} cosine {cs:.8f}, max_abs {np.max(np.abs(a-b)):.5g}", flush=True)
                assert cs > 0.999
        torch.manual_seed(7)
        latents = torch.randn(1, 16, 64, device="cuda", dtype=torch.bfloat16)
        positions = R.position_ids(states.shape[1], 4, 4, "cuda")
        expected = adapter(latents, states, torch.tensor([0.75], device="cuda", dtype=torch.bfloat16), positions).sample
        host_latents = latents[0].float().cpu().numpy()
        output = np.empty_like(host_latents)
        call(lib.krea2_transformer, session, want.ctypes.data, want.size, count.value,
             host_latents.ctypes.data, host_latents.size, 64, 64, 0.75, output.ctypes.data, output.size)
        wanted = expected[0].float().cpu().numpy()
        cosine = float(np.dot(output.ravel(), wanted.ravel()) / (np.linalg.norm(output) * np.linalg.norm(wanted)))
        print(f"complete transformer cosine {cosine:.8f}, max_abs {np.max(np.abs(output-wanted)):.5g}", flush=True)
        # Full independent W4A4 trajectories amplify activation-code rounding.
        # Components above have stricter, identical-input accuracy checks.
        assert cosine > 0.95
        # Equal-area rectangles must invalidate the rotary cache despite having
        # the same total token count. Returning to a shape must remain exact.
        rectangle_latents = torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16)
        rectangle_host = rectangle_latents[0].float().cpu().numpy()
        rectangle_output = np.empty_like(rectangle_host)
        first_rectangle = None
        for width, height in ((64, 128), (128, 64), (64, 128)):
            call(lib.krea2_transformer, session, want.ctypes.data, want.size, count.value,
                 rectangle_host.ctypes.data, rectangle_host.size, width, height, 0.75,
                 rectangle_output.ctypes.data, rectangle_output.size)
            if first_rectangle is None:
                first_rectangle = rectangle_output.copy()
            elif width == 64:
                np.testing.assert_array_equal(rectangle_output, first_rectangle)
            else:
                assert not np.array_equal(rectangle_output, first_rectangle)
            with torch.no_grad():
                position = R.position_ids(count.value, height // 16, width // 16, "cuda")
                rectangle_expected = adapter(rectangle_latents, states,
                    torch.tensor([0.75], device="cuda", dtype=torch.bfloat16), position).sample
            a = rectangle_output.ravel()
            b = rectangle_expected[0].float().cpu().numpy().ravel()
            assert float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))) > 0.95
        print("PASS native rotary cache across equal-area rectangles", flush=True)
        if adapter.loom is not None:
            adapter.loom.close()
        del adapter, reference, weights
        gc.collect()
        torch.cuda.empty_cache()
        # Load the VAE only for its phase, after releasing transformer weights.
        pipe.vae = AutoencoderKLQwenImage.from_pretrained(str(VAE), torch_dtype=torch.bfloat16).cuda()
        pipe.vae.enable_tiling()
        # Check the untiled boundary, then overlapping tiles and short edges.
        for height, width in ((64, 64), (256, 256), (272, 320)):
            decode_latents = torch.randn(1, height // 16 * (width // 16), 64, device="cuda", dtype=torch.bfloat16)
            host_decode = decode_latents[0].float().cpu().numpy()
            rgb = np.empty((height, width, 3), np.uint8)
            call(lib.krea2_decode, session, host_decode.ctypes.data, host_decode.size, width, height, rgb.ctypes.data, rgb.size)
            z = pipe._unpack_latents(decode_latents, height, width).to(pipe.vae.dtype)
            mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda").view(1, 16, 1, 1, 1).to(z.dtype)
            invstd = 1 / torch.tensor(pipe.vae.config.latents_std, device="cuda").view(1, 16, 1, 1, 1).to(z.dtype)
            with torch.no_grad():
                image = pipe.vae.decode(z / invstd + mean, return_dict=False)[0][:, :, 0]
            wanted = np.array(pipe.image_processor.postprocess(image, output_type="pil")[0])
            mae = np.abs(rgb.astype(float) - wanted.astype(float)).mean()
            print(f"VAE {width}x{height} RGB mean absolute error {mae:.4f}/255", flush=True)
            # Preserve identical-input evidence when an oracle comparison fails.
            if mae >= 3:
                failure = ROOT / "build/native-validation"
                failure.mkdir(exist_ok=True)
                host_decode.tofile(failure / f"input-{width}x{height}.bin")
                rgb.tofile(failure / f"native-{width}x{height}.bin")
                wanted.tofile(failure / f"reference-{width}x{height}.bin")
            assert mae < 3
        # Compare generation with an independently scheduled sequence of component
        # calls, using identical initial noise and native text states.
        from diffusers import FlowMatchEulerDiscreteScheduler
        from tools.pipeline import SCHEDULER
        scheduler = FlowMatchEulerDiscreteScheduler(**SCHEDULER)
        scheduler.set_timesteps(sigmas=np.linspace(1, 0.5, 2), mu=1.15, device="cuda")
        # Match the real pipeline: CUDA sigmas round dt to the bf16 velocity
        # dtype before multiplication; a CPU scalar keeps its fp32 value.
        current = torch.from_numpy(host_latents).to("cuda", torch.bfloat16)
        for timestep in scheduler.timesteps:
            current_host = current.float().cpu().numpy()
            call(lib.krea2_transformer, session, taps.ctypes.data, taps.size, count.value,
                 current_host.ctypes.data, current_host.size, 64, 64, float((timestep / 1000).bfloat16()), output.ctypes.data, output.size)
            current = scheduler.step(torch.from_numpy(output).to("cuda", torch.bfloat16), timestep, current, return_dict=False)[0]
        current_host = current.float().cpu().numpy()
        expected_rgb = np.empty((64, 64, 3), np.uint8)
        call(lib.krea2_decode, session, current_host.ctypes.data, current_host.size, 64, 64, expected_rgb.ctypes.data, expected_rgb.size)
        generated = np.empty_like(expected_rgb)
        call(lib.krea2_generate, session, prompt.encode(), 64, 64, 2, 0,
             host_latents.ctypes.data, host_latents.size, generated.ctypes.data, generated.size)
        np.testing.assert_array_equal(generated, expected_rgb)
        print("PASS native generation vs independent scheduler", flush=True)
        # Buffer errors and malformed input must return through the C ABI.
        assert lib.krea2_tokenize(session, b"\xff", None, 0, C.byref(count), err, len(err)) != 0
        assert lib.krea2_decode(session, host_latents.ctypes.data, host_latents.size, 64, 64, generated.ctypes.data, 1, err, len(err)) != 0
        assert lib.krea2_generate(session, prompt.encode(), 65, 64, 2, 0, None, 0, generated.ctypes.data, generated.size, err, len(err)) != 0
        count = size()
        call(lib.krea2_encode, session, b"a " * 1000, None, 0, C.byref(count))
        assert count.value == 512
        print("PASS native input validation and prompt truncation", flush=True)
        # Sessions may be called from different host threads. Their temporary
        # buffers must remain ordered and isolated behind the session lock.
        from concurrent.futures import ThreadPoolExecutor
        def threaded_decode():
            local_error = C.create_string_buffer(4096)
            result = np.empty((64, 64, 3), np.uint8)
            rc = lib.krea2_decode(session, host_latents.ctypes.data, host_latents.size,
                                 64, 64, result.ctypes.data, result.size,
                                 local_error, len(local_error))
            assert rc == 0, local_error.value.decode()
            return result
        expected_threaded = threaded_decode()
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = [executor.submit(threaded_decode) for _ in range(2)]
            for result in results:
                np.testing.assert_array_equal(result.result(), expected_threaded)
        print("PASS native temporary buffers across host threads", flush=True)
        print("PASS native pipeline components", flush=True)
    finally:
        lib.krea2_pipeline_destroy(session)


if __name__ == "__main__":
    main()
