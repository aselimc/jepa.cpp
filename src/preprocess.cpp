// Image / video preprocessing (scaffold). stb_image + stb_image_resize2 are vendored in third_party/.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_resize2.h"
#include "jepa-internal.h"
void jepa_free(void * p) { free(p); }
