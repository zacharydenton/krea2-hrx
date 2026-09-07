#include "native_ops.h"
namespace krea_native {
namespace {
void matmul(const std::string &name, const void *a, const void *b, void *out,
            int m, int n, int k, int batches, float alpha = 1,
            const void *bias = nullptr) {
  size_t as = size_t(m) * k, bs = size_t(n) * k, cs = size_t(m) * n;
  gpu::Args args;
  args.i32(m).f32(alpha).ptr(a).ptr(b).ptr(out);
  if (bias)
    args.ptr(bias);
  // Double the row tile when it adds no extra padded rows. Tiny matrices
  // retain enough independent workgroups to occupy the GPU.
  const bool wide =
      m >= 128 && n >= 64 && ((m + 63) / 64) % 2 == 0 &&
      (name == "gemm_bf16_bf16_nt" || name == "gemm_bf16_bf16_nt_bias");
  const int tile_m = wide ? 128 : 64;
  // The long convolution reductions stream the input once per column tile.
  // A 128-column tile halves those reads; short reductions favor more tiles.
  const bool square = wide && n >= 128 && ((n + 63) / 64) % 2 == 0 && k >= 128;
  const int tile_n = square ? 128 : 64;
  native_launch(square ? name + "_tiled"
                : wide ? name + "_wide"
                       : name,
                {{"m", size_t(m)},
                 {"n", size_t(n)},
                 {"k", size_t(k)},
                 {"asize", as * batches},
                 {"bsize", bs * batches},
                 {"csize", cs * batches},
                 {"astride", as},
                 {"bstride", bs}},
                args, (n + tile_n - 1) / tile_n,
                batches * ((m + tile_m - 1) / tile_m));
}
} // namespace
Tensor Ops::linear(const Tensor &x, const Weight &w, const B *bias) {
  if (w.shape.empty() || w.shape[0] < 1 || x.rows < 1 || x.cols < 1)
    throw std::invalid_argument("linear dimensions");
  int n = w.shape[0], k = x.cols;
  if (w.t.size() != size_t(n) * k)
    throw std::invalid_argument("linear dimensions");
  Tensor y(x.rows, n);
  matmul(bias ? "gemm_bf16_bf16_nt_bias" : "gemm_bf16_bf16_nt", x.ptr, w.t.ptr,
         y.ptr, x.rows, n, k, 1, 1, bias);
  return y;
}
Tensor Ops::norm(const Tensor &x, const Weight &w, int mode, float eps) {
  if (w.count() != size_t(x.cols) || mode < 0 || mode > 2)
    throw std::invalid_argument("norm dimensions");
  Tensor y(x.rows, x.cols);
  gpu::Args args;
  args.i32(x.rows).f32(eps).ptr(x.ptr).ptr(w.as_f32()).ptr(y.ptr);
  native_launch("norm_" + std::to_string(mode),
                {{"xsize", x.size()}, {"cols", size_t(x.cols)}}, args, x.rows);
  return y;
}
Tensor Ops::unary(const Tensor &x, int op) {
  if (op < 0 || op > 2)
    throw std::invalid_argument("unary operation");
  Tensor y(x.rows, x.cols);
  gpu::Args args;
  args.i32(x.size()).ptr(x.ptr).ptr(y.ptr);
  native_launch(op == 0   ? "unary_silu"
                : op == 1 ? "unary_gelu"
                          : "unary_sigmoid",
                {}, args, (x.size() + 255) / 256);
  return y;
}
Tensor Ops::binary(const Tensor &x, const Tensor &y, int op) {
  if (!y.size() || x.size() % y.size() || op < 0 || op > 1)
    throw std::invalid_argument("binary broadcast");
  Tensor z(x.rows, x.cols);
  gpu::Args args;
  args.i32(x.size()).ptr(x.ptr).ptr(y.ptr).ptr(z.ptr);
  native_launch(op == 0 ? "binary_add" : "binary_mul", {{"yn", y.size()}}, args,
                (x.size() + 255) / 256);
  return z;
}
Tensor Ops::rope(const Tensor &x, int tokens, int heads, float theta,
                 bool interleaved) {
  if (interleaved || tokens != x.rows || heads < 1 || x.cols % heads ||
      (x.cols / heads) % 2 || !std::isfinite(theta) || theta <= 0)
    throw std::invalid_argument("split-half rotary dimensions");
  Tensor y(x.rows, x.cols);
  gpu::Args args;
  args.i32(x.size()).f32(theta).ptr(x.ptr).ptr(y.ptr);
  native_launch("rope",
                {{"dim", size_t(x.cols / heads)}, {"heads", size_t(heads)}},
                args, (x.size() + 255) / 256);
  return y;
}
Tensor Ops::attention(const Tensor &q, const Tensor &k, const Tensor &v,
                      int batch, int s, int heads, int kv, int d, bool causal) {
  if (batch < 1 || s < 1 || heads < 1 || kv < 1 || d < 1 || heads % kv ||
      q.size() != size_t(batch) * s * heads * d ||
      k.size() != size_t(batch) * s * kv * d || v.size() != k.size())
    throw std::invalid_argument("attention dimensions");
  Tensor qp(batch * heads * s, d), kp(batch * heads * s, d),
      vp(batch * heads * s, d);
  for (auto pair :
       {std::pair{&q, &qp}, std::pair{&k, &kp}, std::pair{&v, &vp}}) {
    gpu::Args a;
    a.i32(qp.size()).ptr(pair.first->ptr).ptr(pair.second->ptr);
    native_launch("head_pack",
                  {{"dim", size_t(d)},
                   {"tokens", size_t(s)},
                   {"heads", size_t(heads)},
                   {"kv", size_t(pair.first == &q ? heads : kv)},
                   {"xsize", pair.first->size()},
                   {"ysize", qp.size()}},
                  a, (qp.size() + 255) / 256);
  }
  size_t count = size_t(batch) * heads * s * s;
  auto scores = device_storage(count * 4);
  matmul("gemm_bf16_f32_nt", qp.ptr, kp.ptr, scores.get(), s, s, d,
         batch * heads, 1 / std::sqrt(float(d)));
  Tensor probs(batch * heads * s, s), op(batch * heads * s, d),
      out(batch * s, heads * d);
  gpu::Args a;
  a.i32(batch * heads * s).ptr(scores.get()).ptr(probs.ptr);
  native_launch(causal ? "softmax_causal" : "softmax",
                {{"xsize", count}, {"tokens", size_t(s)}}, a,
                batch * heads * s);
  matmul("gemm_bf16_bf16_nn", probs.ptr, vp.ptr, op.ptr, s, d, s,
         batch * heads);
  gpu::Args b;
  b.i32(op.size()).ptr(op.ptr).ptr(out.ptr);
  native_launch("head_unpack",
                {{"dim", size_t(d)},
                 {"tokens", size_t(s)},
                 {"heads", size_t(heads)},
                 {"kv", size_t(heads)},
                 {"xsize", op.size()},
                 {"ysize", out.size()}},
                b, (op.size() + 255) / 256);
  return out;
}
Tensor Ops::conv(const Tensor &x, int h, int w, const Weight &weight,
                 const B *bias) {
  if (weight.shape.size() != 4 || weight.shape[1] != x.cols ||
      weight.shape[2] != weight.shape[3] || weight.shape[2] % 2 != 1 || h < 1 ||
      w < 1 || x.rows != h * w)
    throw std::invalid_argument("convolution dimensions");
  int kernel = weight.shape[2];
  if (kernel == 1)
    return linear(x, weight, bias);
  Tensor patches(h * w, x.cols * kernel * kernel);
  gpu::Args args;
  args.i32(patches.size()).ptr(x.ptr).ptr(patches.ptr);
  native_launch("im2col",
                {{"xsize", x.size()},
                 {"channels", size_t(x.cols)},
                 {"width", size_t(w)},
                 {"height", size_t(h)},
                 {"kernel", size_t(kernel)}},
                args, (patches.size() + 255) / 256);
  return linear(patches, weight, bias);
}
Tensor Ops::upsample(const Tensor &x, int h, int w) {
  if (h < 1 || w < 1 || size_t(h) * w != size_t(x.rows))
    throw std::invalid_argument("upsampling dimensions");
  Tensor y(h * w * 4, x.cols);
  gpu::Args args;
  args.i32(y.size()).ptr(x.ptr).ptr(y.ptr);
  native_launch(
      "upsample",
      {{"xsize", x.size()}, {"channels", size_t(x.cols)}, {"width", size_t(w)}},
      args, (y.size() + 255) / 256);
  return y;
}
void Ops::euler_step(Tensor &sample, const Tensor &velocity, float delta) {
  if (sample.rows != velocity.rows || sample.cols != velocity.cols)
    throw std::invalid_argument("scheduler tensor dimensions");
  gpu::Args args;
  args.i32(sample.size()).f32(delta).ptr(sample.ptr).ptr(velocity.ptr);
  native_launch("euler", {}, args, (sample.size() + 255) / 256);
}
void Ops::guidance(Tensor &cond, const Tensor &uncond, float scale) {
  if (cond.rows != uncond.rows || cond.cols != uncond.cols)
    throw std::invalid_argument("guidance tensor dimensions");
  gpu::Args args;
  args.i32(cond.size()).f32(scale).ptr(cond.ptr).ptr(uncond.ptr);
  native_launch("guidance", {}, args, (cond.size() + 255) / 256);
}
} // namespace krea_native
