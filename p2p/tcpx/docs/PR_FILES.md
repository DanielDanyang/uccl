# Files for PR

## Core Implementation (5 files)
```
p2p/tcpx/Makefile
p2p/tcpx/run_tcpx_test.sh
p2p/tcpx/tcpx_interface.h
p2p/tcpx/tcpx_impl.cc
p2p/tcpx/include/tcpx_structs.h
```

## RX Metadata Parsing (4 files)
```
p2p/tcpx/rx/rx_cmsg_parser.h
p2p/tcpx/rx/rx_cmsg_parser.cc
p2p/tcpx/rx/rx_descriptor.h
p2p/tcpx/rx/rx_descriptor.cc
```

## Tests (4 files)
```
p2p/tcpx/tests/test_tcpx_transfer.cc
p2p/tcpx/tests/test_connection.cc
p2p/tcpx/tests/test_rx_cmsg_parser.cc
p2p/tcpx/tests/test_rx_descriptor.cc
```

## Documentation (3 files)
```
p2p/tcpx/README.md
p2p/tcpx/docs/TCPX_LOGIC_MAPPING.md
p2p/tcpx/docs/tcpx_transfer.md
```

---

## Files to EXCLUDE (kernel implementation - WIP)
```
p2p/tcpx/device/unpack_kernels.cu
p2p/tcpx/device/unpack_launch.cu
p2p/tcpx/device/unpack_launch.h
```

## Files to EXCLUDE (reference code)
```
p2p/tcpx/reference/
```

---

## Git Commands

```bash
cd p2p/tcpx

# Add files
git add \
  README.md \
  Makefile \
  run_tcpx_test.sh \
  tcpx_interface.h \
  tcpx_impl.cc \
  include/tcpx_structs.h \
  rx/rx_cmsg_parser.h \
  rx/rx_cmsg_parser.cc \
  rx/rx_descriptor.h \
  rx/rx_descriptor.cc \
  tests/test_tcpx_transfer.cc \
  tests/test_connection.cc \
  tests/test_rx_cmsg_parser.cc \
  tests/test_rx_descriptor.cc \
  docs/TCPX_LOGIC_MAPPING.md \
  docs/tcpx_transfer.md

# Verify
git status

# Commit
git commit -m "feat(tcpx): Add GPU-to-GPU transfer with D2D and host unpack

- TCPX connection management (listen/accept/connect)
- CUDA device buffer registration
- Async send/receive operations
- RX metadata parsing (CMSG scatter-gather lists)
- Descriptor block construction
- D2D unpack implementation (default)
- Host-mediated unpack (fallback)
- End-to-end transfer test with validation

Tested on H100 GPUs with TCPX plugin v3.1.6.
"
```

---

Total: 16 files

