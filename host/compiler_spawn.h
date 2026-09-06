#pragma once
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>
extern char **environ;
namespace krea_native {
// Run the Loom compiler with an argument vector, sending its stderr to `log`.
// A failure throws with the log's tail, so a bad source or configuration is
// diagnosable from the C ABI error string.
inline void run_compiler(const std::vector<std::string> &command,
                         const std::filesystem::path &log,
                         const std::string &what) {
  std::vector<std::string> args = command;
  std::vector<char *> argv;
  for (auto &a : args)
    argv.push_back(a.data());
  argv.push_back(nullptr);
  posix_spawn_file_actions_t actions;
  posix_spawn_file_actions_init(&actions);
  struct Actions {
    posix_spawn_file_actions_t *p;
    ~Actions() { posix_spawn_file_actions_destroy(p); }
  } owner{&actions};
  posix_spawn_file_actions_addopen(&actions, 2, log.c_str(),
                                   O_WRONLY | O_CREAT | O_TRUNC,
                                   0600);
  pid_t pid;
  int rc = posix_spawnp(&pid, args[0].c_str(), &actions, nullptr, argv.data(),
                        environ);
  if (rc)
    throw std::runtime_error("cannot start Loom compiler '" + args[0] +
                             "': " + std::strerror(rc));
  int status;
  while (waitpid(pid, &status, 0) < 0)
    if (errno != EINTR)
      throw std::runtime_error("cannot wait for the Loom compiler");
  if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
    return;
  std::string tail;
  if (std::ifstream input(log, std::ios::binary); input) {
    std::string text((std::istreambuf_iterator<char>(input)), {});
    tail = text.size() > 600 ? text.substr(text.size() - 600) : text;
  }
  throw std::runtime_error("Loom compilation failed: " + what +
                           (tail.empty() ? "" : "\n" + tail));
}
} // namespace krea_native
