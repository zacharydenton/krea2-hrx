# Release preparation

The repository and Cargo package are `krea2-hrx`. The CLI and Rust library remain
`krea2`. Cargo publishing is disabled; installation is from a source checkout.

## Discovery metadata

Use this description for the GitHub About field and Cargo package:

> Krea 2 image generation on AMD Strix Halo, powered by Loom kernels and HRX

GitHub topics:

```text
loom hrx krea krea2 text-to-image image-generation generative-ai diffusion-models
gpu-computing inference amd radeon strix-halo gfx1151 comfyui quantization
```

Cargo keywords are the five focused terms in `Cargo.toml`. Keep claims scoped to
the supported GPU and link performance claims to the recorded methodology.

## Before making a release public

- Run `scripts/test.sh --cpu` and check the GitHub Actions result.
- Run `scripts/test.sh --gpu` on gfx1151. Run `scripts/parity.sh` with the frozen
  reference fixture and local weights; see [testing](testing.md).
- Follow the README quick start from a clean checkout and generate an image.
  Check default NPU-enabled and `--no-default-features` builds.
- Review the staged files and repository history for credentials, private data,
  model weights, and generated artifacts. Preserve third-party license notices.
- Confirm the repository name, description, topics, links, and version. Record
  the tested Rust/runtime versions and any known limitations in release notes.
- Make the repository public when publication is intended, then tag the tested
  commit and create a GitHub release. Preparing docs does not publish a release.

Historical benchmark logs retain the original checkout paths and project name
as provenance; they are not current installation instructions.
