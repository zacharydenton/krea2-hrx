#include "krea2_pipeline.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <vector>
int main(int argc, char **argv) {
  try {
    std::map<std::string, std::string> args;
    const std::set<std::string> options = {
        "--model",    "--text-encoder", "--vae",    "--prompt",   "--out",
        "--compiler", "--width",        "--height", "--steps",    "--seed",
        "--negative", "--guidance",     "--checkpoint"};
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc)
        throw std::runtime_error("every option needs a value");
      if (!options.count(argv[i]) || args.count(argv[i]))
        throw std::runtime_error("unknown or repeated option: " +
                                 std::string(argv[i]));
      args[argv[i]] = argv[i + 1];
    }
    if (!args.count("--model") || !args.count("--prompt") || !args.count("--out"))
      throw std::runtime_error(
          "usage: krea2 --model diffusion_models/krea2_turbo_int8_convrot.safetensors "
          "--prompt TEXT --out IMAGE.ppm [--text-encoder FILE] [--vae FILE] "
          "[--checkpoint turbo|raw] [--compiler PATH] [--width 1024] [--height 1024] "
          "[--steps N] [--seed 0] [--negative TEXT] [--guidance G]\n"
          "--model is ComfyUI's int8 ConvRot checkpoint; the text encoder and VAE are "
          "found beside it in ComfyUI's models layout.\n"
          "steps and guidance default to the checkpoint: Turbo 8 and 0 (no guidance), "
          "Raw 52 and 3.5");
    auto integer = [&](const char *key, int fallback) {
      if (!args.count(key))
        return fallback;
      size_t consumed = 0;
      int value = std::stoi(args[key], &consumed);
      if (consumed != args[key].size())
        throw std::runtime_error("invalid number: " + args[key]);
      return value;
    };
    int w = integer("--width", 1024), h = integer("--height", 1024),
        steps = integer("--steps", -1);
    if (w < 64 || h < 64 || w > 2048 || h > 2048 || w % 16 || h % 16 ||
        steps == 0 || steps > 100)
      throw std::runtime_error("invalid image dimensions");
    float guidance = -1;
    if (args.count("--guidance")) {
      size_t consumed = 0;
      guidance = std::stof(args["--guidance"], &consumed);
      if (consumed != args["--guidance"].size() || guidance < 0 ||
          guidance > 100)
        throw std::runtime_error("guidance must be in 0..100");
    }
    uint64_t seed = 0;
    if (args.count("--seed")) {
      const auto &s = args["--seed"];
      if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("seed must be an unsigned integer");
      seed = std::stoull(s);
    }
    krea2_pipeline *p = nullptr;
    char error[4096];
    auto start = std::chrono::steady_clock::now();
    const char *compiler =
        args.count("--compiler") ? args["--compiler"].c_str() : nullptr;
    int distilled = -1;
    if (args.count("--checkpoint")) {
      if (args["--checkpoint"] != "turbo" && args["--checkpoint"] != "raw")
        throw std::runtime_error("--checkpoint must be turbo or raw");
      distilled = args["--checkpoint"] == "turbo";
    }
    if (krea2_pipeline_create_files(
            args["--model"].c_str(),
            args.count("--text-encoder") ? args["--text-encoder"].c_str() : nullptr,
            args.count("--vae") ? args["--vae"].c_str() : nullptr, distilled,
            compiler, &p, error, sizeof(error)))
      throw std::runtime_error(error);
    struct Cleanup {
      krea2_pipeline *p;
      ~Cleanup() { krea2_pipeline_destroy(p); }
    } cleanup{p};
    std::cerr << "native model loaded\n";
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    if (krea2_generate_guided(
            p, args["--prompt"].c_str(),
            args.count("--negative") ? args["--negative"].c_str() : nullptr,
            guidance, w, h, steps, seed, nullptr, 0, rgb.data(), rgb.size(),
            error, sizeof(error)))
      throw std::runtime_error(error);
    std::ofstream out(args["--out"], std::ios::binary);
    out << "P6\n" << w << " " << h << "\n255\n";
    out.write((char *)rgb.data(), rgb.size());
    if (!out)
      throw std::runtime_error("cannot write output image");
    std::cout << "saved " << args["--out"] << " in "
              << std::chrono::duration<double>(
                     std::chrono::steady_clock::now() - start)
                     .count()
              << " s\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
