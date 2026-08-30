// Minimal NumPy .npy reader/writer (v1.0 / v2.0 headers, C-order only).
// Supports float32, float16 (raw), int32, int64, uint8. Header-only; no dependencies.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace npy {

struct Array {
    std::string dtype;              // "<f4", "<f2", "<i4", "<i8", "|u1"
    std::vector<int64_t> shape;
    std::vector<uint8_t> bytes;     // raw C-order data

    int64_t numel() const { int64_t n = 1; for (auto s : shape) n *= s; return n; }
    size_t item_size() const {
        if (dtype == "<f4" || dtype == "<i4") return 4;
        if (dtype == "<f2") return 2;
        if (dtype == "<i8" || dtype == "<f8") return 8;
        if (dtype == "|u1" || dtype == "|i1" || dtype == "|b1") return 1;
        throw std::runtime_error("npy: unsupported dtype " + dtype);
    }
    const float   * f32() const { if (dtype != "<f4") throw std::runtime_error("npy: not float32: " + dtype); return (const float *) bytes.data(); }
    const int32_t * i32() const { if (dtype != "<i4") throw std::runtime_error("npy: not int32: " + dtype);   return (const int32_t *) bytes.data(); }
    const int64_t * i64() const { if (dtype != "<i8") throw std::runtime_error("npy: not int64: " + dtype);   return (const int64_t *) bytes.data(); }
    const uint8_t * u8()  const { if (dtype != "|u1") throw std::runtime_error("npy: not uint8: " + dtype);   return bytes.data(); }

    // Convert any supported numeric dtype to a float vector (for tolerant comparisons).
    std::vector<float> to_f32() const {
        std::vector<float> out(numel());
        if (dtype == "<f4") { memcpy(out.data(), bytes.data(), out.size() * 4); }
        else if (dtype == "<f8") { const double * p = (const double *) bytes.data(); for (size_t i = 0; i < out.size(); i++) out[i] = (float) p[i]; }
        else if (dtype == "<i4") { const int32_t * p = i32(); for (size_t i = 0; i < out.size(); i++) out[i] = (float) p[i]; }
        else if (dtype == "<i8") { const int64_t * p = i64(); for (size_t i = 0; i < out.size(); i++) out[i] = (float) p[i]; }
        else if (dtype == "|u1") { for (size_t i = 0; i < out.size(); i++) out[i] = (float) bytes[i]; }
        else throw std::runtime_error("npy: to_f32 unsupported dtype " + dtype);
        return out;
    }
};

inline Array load(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot open " + path);
    char magic[6]; f.read(magic, 6);
    if (memcmp(magic, "\x93NUMPY", 6) != 0) throw std::runtime_error("npy: bad magic in " + path);
    uint8_t ver[2]; f.read((char *) ver, 2);
    uint32_t hlen = 0;
    if (ver[0] == 1) { uint16_t h; f.read((char *) &h, 2); hlen = h; }
    else             { f.read((char *) &hlen, 4); }
    std::string hdr(hlen, '\0'); f.read(&hdr[0], hlen);

    Array a;
    auto find_val = [&](const char * key) -> std::string {
        size_t p = hdr.find(key); if (p == std::string::npos) throw std::runtime_error(std::string("npy: header missing ") + key);
        p = hdr.find(':', p) + 1;
        while (p < hdr.size() && hdr[p] == ' ') p++;
        return hdr.substr(p);
    };
    { std::string v = find_val("'descr'"); size_t q1 = v.find('\''), q2 = v.find('\'', q1 + 1); a.dtype = v.substr(q1 + 1, q2 - q1 - 1); }
    { std::string v = find_val("'fortran_order'"); if (v.rfind("True", 0) == 0) throw std::runtime_error("npy: fortran order unsupported: " + path); }
    { std::string v = find_val("'shape'"); size_t p1 = v.find('('), p2 = v.find(')');
      std::string s = v.substr(p1 + 1, p2 - p1 - 1);
      size_t i = 0;
      while (i < s.size()) {
          while (i < s.size() && (s[i] == ' ' || s[i] == ',')) i++;
          if (i >= s.size()) break;
          size_t j = i; while (j < s.size() && isdigit((unsigned char) s[j])) j++;
          if (j > i) a.shape.push_back(std::stoll(s.substr(i, j - i)));
          i = j;
      } }
    a.bytes.resize((size_t) a.numel() * a.item_size());
    f.read((char *) a.bytes.data(), a.bytes.size());
    if (!f) throw std::runtime_error("npy: truncated data in " + path);
    return a;
}

inline void save(const std::string & path, const std::string & dtype, const std::vector<int64_t> & shape, const void * data) {
    std::string shp = "(";
    for (size_t i = 0; i < shape.size(); i++) { shp += std::to_string(shape[i]); shp += (shape.size() == 1 ? "," : (i + 1 < shape.size() ? ", " : "")); }
    shp += ")";
    std::string hdr = "{'descr': '" + dtype + "', 'fortran_order': False, 'shape': " + shp + ", }";
    size_t total = 10 + hdr.size() + 1; size_t pad = (16 - total % 16) % 16;
    hdr += std::string(pad, ' '); hdr += '\n';
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("npy: cannot write " + path);
    f.write("\x93NUMPY\x01\x00", 8);
    uint16_t hlen = (uint16_t) hdr.size(); f.write((const char *) &hlen, 2);
    f.write(hdr.data(), hdr.size());
    size_t isz = dtype == "<f4" || dtype == "<i4" ? 4 : dtype == "<f2" ? 2 : dtype == "<i8" || dtype == "<f8" ? 8 : 1;
    int64_t n = 1; for (auto s : shape) n *= s;
    f.write((const char *) data, (size_t) n * isz);
}

inline void save_f32(const std::string & path, const std::vector<int64_t> & shape, const float * data) { save(path, "<f4", shape, data); }

} // namespace npy
