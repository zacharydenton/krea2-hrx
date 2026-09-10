#!/usr/bin/env python3
"""Qualify the real fusion projection against immutable model reference fixtures."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def executable(command, name):
    result = subprocess.run(command + ["--message-format=json"], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, check=True)
    paths = [entry["executable"] for line in result.stdout.splitlines()
             if (entry := json.loads(line)).get("reason") == "compiler-artifact"
             and entry.get("executable") and entry["target"]["name"] == name]
    if len(paths) != 1:
        raise RuntimeError(f"expected one executable, got {paths}")
    return paths[0]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.flush()
        os.fsync(file.fileno())
        temporary = Path(file.name)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolchain", required=True, type=Path)
    parser.add_argument("--fixture", type=Path, default=ROOT / "build/quality")
    parser.add_argument("--manifest", type=Path, default=ROOT / "tests/fixtures/unquantized.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--profiles", type=Path)
    args = parser.parse_args()
    if os.environ.get("KREA2_QUALITY_MINT"):
        parser.error("qualification cannot mint or replace a reference baseline")
    work_root = ROOT / "build/fusion-qualification"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="run-", dir=work_root))
    profiles = args.profiles or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hrx/krea2-fusion"
    worker = executable(["cargo", "build", "--locked", "--release", "--example", "fusion_worker"], "fusion_worker")
    quality = executable(["cargo", "test", "--locked", "--release", "--test", "unquantized_parity", "--no-run"], "unquantized_parity")
    env = dict(os.environ, KREA2_QUALITY_FIXTURE=str(args.fixture.resolve()),
               KREA2_QUALITY_MANIFEST=str(args.manifest.resolve()),
               KREA2_FUSION_PROFILES=str(work / "candidates"))
    if args.checkpoint:
        env["KREA2_CHECKPOINT"] = str(args.checkpoint.resolve())

    print(f"Evidence: {work}", flush=True)

    def quality_run(backend, label, capture=False):
        print(f"Reference quality and production generation: {label}", flush=True)
        output = work / f"{label}.json"
        child = dict(env, KREA2_QUALITY_FUSION_BACKEND=backend,
                     KREA2_QUALITY_RESULT=str(output))
        child.pop("KREA2_FUSION_CAPTURE", None)
        if capture:
            child["KREA2_FUSION_CAPTURE"] = str(work / "cases")
        with (work / f"{label}.log").open("w") as log:
            subprocess.run([quality, "--ignored", "--nocapture", "--test-threads=1"],
                           cwd=ROOT, env=child, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = json.loads(output.read_text())
        if backend == "npu" and not result["selection"].startswith("NPU:"):
            raise RuntimeError("quality run did not execute the NPU projection")
        return result

    quality_run("gpu", "capture", capture=True)
    cases = sorted((work / "cases").glob("*/case.json"))
    if not cases:
        raise RuntimeError("reference trajectory did not capture the fusion operation")
    records = []
    for case in cases:
        result = subprocess.run([worker, "compile", str(case.parent), str(args.toolchain.resolve())],
                                cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, check=True)
        record_path = Path(result.stdout.strip())
        record = json.loads(record_path.read_text())
        for process in range(5):
            output = work / f"{case.parent.name}-stage-{process}.json"
            print(f"Stage {case.parent.name}, process {process + 1}/5", flush=True)
            with output.with_suffix(".log").open("w") as log:
                subprocess.run([worker, "measure", str(case.parent), str(record_path), str(output)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            print(output.with_suffix(".log").read_text().strip(), flush=True)
            record["processes"].append(json.loads(output.read_text()))
        records.append((record_path, record))

    # Each process checks reference quality, then warms and measures production generation.
    gpu, npu = [], []
    for process in range(5):
        for backend in (["gpu", "npu"] if process % 2 == 0 else ["npu", "gpu"]):
            result = quality_run(backend, f"{backend}-{process}")
            (gpu if backend == "gpu" else npu).append(result)
    reference = gpu[0]["reference_files"]
    if any(result["reference_files"] != reference for result in gpu + npu):
        raise RuntimeError("reference identity changed during qualification")
    evidence = {"gpu_seconds": [r["seconds"] for r in gpu],
                "npu_seconds": [r["seconds"] for r in npu],
                "reference_files": reference,
                "relative_rms_loss_db": max(r["relative_rms_loss_db"] for r in npu),
                "image_psnr_loss_db": max(r["image_psnr_loss_db"] for r in npu)}
    for candidate, record in records:
        record["generation"] = evidence
        write(candidate, record)
        # Rust owns the selection rule used by both qualification and inference.
        result = subprocess.run([worker, "status", str(candidate)], env=env,
                                text=True, stdout=subprocess.PIPE, check=True)
        print(f"{candidate.stem}: {result.stdout.strip()}")
        write(profiles / candidate.name, record)
    print(f"Evidence: {work}")
    print(f"Profiles: {profiles}")


if __name__ == "__main__":
    main()
