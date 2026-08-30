// Internal shared declarations. Public API lives in include/jepa.h.
#pragma once
#include "jepa.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <string>
#include <vector>
#include <map>

#define JEPA_VERSION "0.1.0-dev"

// Log helper
void jepa_log(const char * fmt, ...);
