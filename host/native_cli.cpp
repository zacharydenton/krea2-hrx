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
    const std::set<std::string> options = {"--bundle",   "--prompt", "--out",
                                           "--compiler", "--width",  "--height",
                                           "--steps",    "--seed"};
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc)
        throw std::runtime_error("every option needs a value");
      if (!options.count(argv[i]) || args.count(argv[i]))
        throw std::runtime_error("unknown or repeated option: " +
                                 std::string(argv[i]));
      args[argv[i]] = argv[i + 1];
    }
    if (!args.count("--bundle") || !args.count("--prompt") ||
        !args.count("--out"))
      throw std::runtime_error(
          "usage: krea2-generate --bundle DIR --prompt TEXT --out IMAGE.ppm "
          "[--compiler PATH] [--width 1024] [--height 1024] [--steps 8] "
          "[--seed 0]");
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
        steps = integer("--steps", 8);
    if (w < 64 || h < 64 || w > 2048 || h > 2048 || w % 16 || h % 16 ||
        steps < 1 || steps > 100)
      throw std::runtime_error("invalid image dimensions");
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
    if (krea2_pipeline_create(
            args["--bundle"].c_str(),
            args.count("--compiler") ? args["--compiler"].c_str() : nullptr, &p,
            error, sizeof(error)))
      throw std::runtime_error(error);
    struct Cleanup {
      krea2_pipeline *p;
      ~Cleanup() { krea2_pipeline_destroy(p); }
    } cleanup{p};
    std::cerr << "native model loaded\n";
    std::vector<uint8_t> rgb(size_t(w) * h * 3);
    if (krea2_generate(p, args["--prompt"].c_str(), w, h, steps, seed, nullptr,
                       0, rgb.data(), rgb.size(), error, sizeof(error)))
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
