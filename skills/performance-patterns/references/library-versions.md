<!-- (C) 2026 Intel Corporation, MIT license -->
# Library version reference

This file is loaded by `patterns/library-version-upgrade.md` when a hot symbol
from a system library has been identified in profiling output.

Each row maps an internal symbol (as it appears in `perf report` / `perf
annotate`) to the library that contains it, a suggested minimum version, and
a brief reason.  When a symbol matches, detect the installed version using the
primitives at the bottom of this file and report the gap.

---

## Version table

| Symbol (as seen in perf) | `.so` file | Library | Suggested minimum version | Reason |
|--------------------------|-----------|---------|--------------------------|--------|
| `gcm_init_avx` | `libcrypto.so.3` | OpenSSL | 3.3 | Significant AVX-512 optimizations for AES-GCM in 3.3+ (Sapphire Rapids and later) |
| `ossl_aes_gcm_encrypt_avx512` | `libcrypto.so.3` | OpenSSL | 3.3 | AVX-512 AES-GCM encrypt path present in earlier versions but significantly further optimized in 3.3+; if hot on a pre-3.3 system, upgrade for a material speedup |
| `ossl_aes_gcm_init_avx512` | `libcrypto.so.3` | OpenSSL | 3.3 | AVX-512 AES-GCM init path present in earlier versions but significantly further optimized in 3.3+; if hot on a pre-3.3 system, upgrade for a material speedup |
| `aesni_xts_256_encrypt_avx512` | `libcrypto.so.3` | OpenSSL | 3.3 | AVX-512 AES-XTS-256 path present in earlier versions but significantly further optimized in 3.3+; if hot on a pre-3.3 system, upgrade for a material speedup |
| `aesni_xts_encrypt` | `libcrypto.so.3` | OpenSSL | 3.3 | Non-AVX-512 AES-XTS path; seeing this hot in profiles on a modern CPU means the AVX-512 variant (`aesni_xts_256_encrypt_avx512`) is absent — upgrade to OpenSSL 3.3 to get the faster implementation |

---

## Version detection primitives

When a symbol matches a row above, first try the **generic** method below.
If it does not produce a useful version string, use the library-specific
primitive that follows.

### Generic: resolve the `.so` symlink

Most system libraries encode their full version in the real filename.
`readlink` on the `.so` name reveals it without any library-specific tooling:

```bash
readlink -f /path/to/libfoo.so.N
```

Example (zlib):
```
$ readlink -f /usr/lib/x86_64-linux-gnu/libz.so.1
/usr/lib/x86_64-linux-gnu/libz.so.1.2.11
```
The real filename `libz.so.1.2.11` gives version `1.2.11` directly.

To find the `.so` path first:
```bash
ldconfig -p | grep libfoo
# or, given the binary that links it:
ldd <binary> | grep libfoo
```

**OpenSSL is a notable exception:** `libcrypto.so.3` resolves to a filename
that still only shows the major version, so `readlink` is not useful there —
use the OpenSSL-specific primitive below instead.

---

### OpenSSL

**Preferred:**
```bash
openssl version
```
Example output:
```
OpenSSL 3.6.2 7 Apr 2026 (Library: OpenSSL 3.6.2 7 Apr 2026)
```

**Fallback** (replace path with the actual `.so` location from `ldd`):
```bash
strings /usr/lib/x86_64-linux-gnu/libcrypto.so.3 | grep "^OpenSSL 3"
```
Example output:
```
OpenSSL 3.6.2 7 Apr 2026
```

To find the actual `.so` path on the current system:
```bash
ldconfig -p | grep libcrypto
# or, if you have the binary that links it:
ldd <binary> | grep libcrypto
```
