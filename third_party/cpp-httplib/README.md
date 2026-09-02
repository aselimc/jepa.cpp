# cpp-httplib

Vendored single header: [`yhirose/cpp-httplib`](https://github.com/yhirose/cpp-httplib) **v0.54.1**
(`httplib.h`, MIT, `LICENSE` alongside it).

`tools/jepa-server.cpp` is the only consumer. The header is used in its plain form — no OpenSSL, no
zlib, no Brotli — so the server speaks HTTP/1.1 on a loopback socket and nothing else links in.

Refresh with:

```
curl -L -o third_party/cpp-httplib/httplib.h \
  https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.54.1/httplib.h
curl -L -o third_party/cpp-httplib/LICENSE \
  https://raw.githubusercontent.com/yhirose/cpp-httplib/v0.54.1/LICENSE
```
