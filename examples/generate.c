// Compile as C, link against libkrea2_pipeline. The API returns RGB bytes;
// image encoding and language bindings do not require any model framework.
#include "../host/krea2_pipeline.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s BUNDLE PROMPT OUTPUT.ppm\n", argv[0]);
    return 2;
  }
  if (krea2_pipeline_abi_version() != KREA2_PIPELINE_ABI_VERSION)
    return 2;
  char error[4096];
  krea2_pipeline *p = NULL;
  if (krea2_pipeline_create(argv[1], NULL, &p, error, sizeof(error))) {
    fprintf(stderr, "%s\n", error);
    return 1;
  }
  const int width = 256, height = 256;
  size_t bytes = (size_t)width * height * 3;
  uint8_t *rgb = malloc(bytes);
  if (!rgb) {
    krea2_pipeline_destroy(p);
    return 1;
  }
  int rc = krea2_generate(p, argv[2], width, height, 8, 0, NULL, 0, rgb, bytes,
                          error, sizeof(error));
  if (rc)
    fprintf(stderr, "%s\n", error);
  else {
    FILE *f = fopen(argv[3], "wb");
    if (!f)
      rc = 1;
    else {
      fprintf(f, "P6\n%d %d\n255\n", width, height);
      if (fwrite(rgb, 1, bytes, f) != bytes)
        rc = 1;
      if (fclose(f))
        rc = 1;
    }
  }
  free(rgb);
  krea2_pipeline_destroy(p);
  return rc;
}
