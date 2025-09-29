# TCPX Logic Mapping (NCCL Plugin -> UCCL Integration)

## 1. Overview
This note cross-references the upstream NCCL TCPX plugin with the TCPX helpers that live under `p2p/tcpx/`. It lists the major call chains, the matching components in our tree, and places where we can lean on the plugin API instead of maintaining parallel code.

## 2. Upstream NCCL TCPX Call Flow
The exported table in `nccl-plugin-gpudirecttcpx/src/net_tcpx.h` is thin; most behaviour is implemented in the companion `.cc` and socket helpers.

| Stage | NCCL entry point(s) | Supporting implementation |
| --- | --- | --- |
| Device discovery & metadata | `tcpxDevices`, `tcpxGetProperties` (`nccl-plugin-gpudirecttcpx/src/net_tcpx.cc:742`) | enumerates NICs and reports `DEV_UNPACK` capabilities |
| Listen / connect | `tcpxListen`, `tcpxConnect_v5`, `tcpxAccept_v5` (`.../net_tcpx.cc:755-820`) | uses `tcpxConnectionSetup` to exchange NCCL handles and allocate comm state |
| Memory registration | `tcpxRegMr`, `tcpxDeregMr` (`.../net_tcpx.cc:821-889`) | CUDA path registers dmabuf pages for devmem-tcp |
| Request posting | `tcpxIsend_v5` (`.../net_tcpx.cc:898`), `tcpxIrecv_v5` (`.../net_tcpx.cc:921`) | pull a free `tcpxRequest` from `tcpxRequestQueue` (`.../work_queue.h:114`) |
| RX scatter gather | `gpudirectTCPXRecv` -> `process_recv_cmsg` (`.../sock/tcpx.h:136`) | decodes SCM_DEVMEM control data into `tcpxDataPipe::scatter_list` |
| Unpack queue population | `tcpxRecvEnqueue` (`.../net_tcpx.cc:1128`) | copies scatter entries into `tcpxRequest::unpack_slot.mem` and bumps `cnt_cache` |
| Completion polling | `tcpxTest` (`.../net_tcpx.cc:1311`) | advances the comm, checks `REQUEST_DONE`, writes `*(r->unpack_slot.cnt)` for GPU receives |
| Device queue export | `tcpxGetDeviceHandle` (`.../net_tcpx.cc:1452`) | allocates a `tcpxNetDeviceQueue`, maps it to CUDA, returns `unpackNetDeviceHandle` |
| Device unpack | NCCL device kernels (`p2p/tcpx/reference/unpack/unpack.h`) | read `loadMeta` descriptors straight on the GPU |

Important structs:
- `struct tcpxRequest` (`nccl-plugin-gpudirecttcpx/src/work_queue.h:63`) carries buffer pointers and the `unpackSlot`.
- `struct unpackSlot` (`.../devcomm/unpack_defs1.h:61`) holds the per-request queue state: `mem`, `cnt`, `cnt_cache`, fd/token arrays.
- `struct tcpxNetDeviceQueue` (`.../devcomm/unpack_defs1.h:38`) is the host-visible queue that mirrors what the device kernel consumes.

## 3. Current UCCL TCPX Code Path
The transfer test (`p2p/tcpx/tests/test_tcpx_transfer.cc`) exercises the exported interface in `p2p/tcpx/tcpx_interface.h` plus several local helper layers.

| Stage | UCCL component | Notes |
| --- | --- | --- |
| Plugin bootstrap | `tcpx_load_plugin`, `tcpx_get_device_count` (`p2p/tcpx/tcpx_impl.cc:56`) | loads the v7 table, installs a debug logger |
| Listen / connect | `tcpx_listen`, `tcpx_connect_v5`, `tcpx_accept_v5` (`p2p/tcpx/tcpx_impl.cc:141-172`) | thin wrappers around the plugin function table |
| Memory registration | `tcpx_reg_mr`, `tcpx_dereg_mr` (`p2p/tcpx/tcpx_impl.cc:179-210`) | forwarders with extra logging |
| Request post / poll | `tcpx_isend`, `tcpx_irecv`, `tcpx_test`, `tcpx_irecv_consumed` (`p2p/tcpx/tcpx_impl.cc:212-273`) | no extra logic beyond null-checks |
| Control-message parsing | `tcpx::rx::CmsgParser` (`p2p/tcpx/rx/rx_cmsg_parser.cc`) | user-space reimplementation of `process_recv_cmsg` |
| Descriptor construction | `tcpx::rx::DescriptorBuilder` (`p2p/tcpx/rx/rx_descriptor.cc`) | converts a `ScatterList` into an `UnpackDescriptorBlock` |
| CUDA unpack launch | `tcpx::device::UnpackLauncher` + `tcpxUnpackKernel` (`p2p/tcpx/device/unpack_launch.cu`, `unpack_kernels.cu`) | adapted copy of the NCCL kernel with simplified scheduling |
| Completion handling | polling loops in the test (`p2p/tcpx/tests/test_tcpx_transfer.cc:382, 636`) | now stop after the first `done == 1` event and accept `size == 0` for GPU buffers |

The full NCCL reference kernel remains in `p2p/tcpx/reference/unpack/` for comparison; our runtime copies a reduced version in `device/unpack_kernels.cu`.

## 4. Mapping Notes
### 4.1 Connection and requests
- `tcpx_irecv` and `tcpx_isend` hand back opaque request handles. The test recasts them to a local `struct TcpxRequest` (`p2p/tcpx/tests/test_tcpx_transfer.cc:414`) that mirrors `nccl-plugin-gpudirecttcpx/src/work_queue.h:63`. This works today but ties us to the plugin's internal layout.
- `tcpxTest` fills `*(r->unpack_slot.cnt)` when a GPU receive completes. With the recent poll-loop fix we copy that value once and launch the device unpack without invoking `tcpx_test` again.

### 4.2 Scatter metadata
- `process_recv_cmsg` in the plugin (`nccl-plugin-gpudirecttcpx/src/sock/tcpx.h:136`) already produces the scatter list used by `tcpxRecvEnqueue`. Our `CmsgParser` duplicates this logic, including token coalescing and host bounce buffer handling.
- The test currently reads `rx_req->unpack_slot.mem` and casts it to an array of `LoadMetaEntry` that matches NCCL's `loadMeta` (`p2p/tcpx/reference/unpack/unpack_defs.h:33`). The duplication could be removed by sharing the upstream headers.

### 4.3 Device unpack
- NCCL's `ncclNetDeviceUnpack` kernels (see `reference/unpack/unpack.h`) schedule work per warp and reuse shared memory buffers. Our custom kernel (`p2p/tcpx/device/unpack_kernels.cu`) performs a simpler block-per-descriptor copy with similar alignment heuristics but without the shared-memory helpers.

## 5. Places to Reuse Existing TCPX Code
1. **Reuse struct definitions instead of copying them.** Import `work_queue.h` and `devcomm/unpack_defs1.h`, or create a small shim header, so we can remove the local copies of `TcpxRequest`, `TcpxUnpackSlot`, and `UnpackNetDeviceHandle` in `p2p/tcpx/tests/test_tcpx_transfer.cc:98`.
2. **Call the upstream control-message helper.** Expose a wrapper around `process_recv_cmsg` (from `nccl-plugin-gpudirecttcpx/src/sock/tcpx.h:136`) and retire `tcpx::rx::CmsgParser` for the hot path. This keeps token coalescing and host-bounce warnings in sync with the plugin.
3. **Consume the plugin's `loadMeta` queue directly.** Since `tcpxGetDeviceHandle` already maps the queue (`nccl-plugin-gpudirecttcpx/src/net_tcpx.cc:1452`), we can launch the CUDA kernel directly on those descriptors and drop most of `DescriptorBuilder`.
4. **Link against the original CUDA kernel.** Rather than maintaining `tcpxUnpackKernel`, build the reference NCCL sources from `p2p/tcpx/reference/unpack/` and invoke `ncclNetDeviceUnpack`. This avoids drift when NVIDIA updates the kernel.
5. **Encapsulate poll semantics.** Provide a helper (e.g. `tcpx_wait_request`) that loops on `tcpx_test`, stops on the first `done == 1`, and returns metadata (size, descriptor pointer). Future callers then inherit the correct behaviour automatically.

## 6. Behaviour Differences to Align
- Earlier we re-ran `tcpx_test` after `done` flipped, which cleared the queue and produced the `received_size == 0` timeout seen in `server_log.txt`. The new polling logic keeps the plugin contract.
- Host bounce-buffer tracking (`user_buffer_count` in `tcpx.h`) is currently lost. Using the upstream parser preserves that diagnostic.
- Descriptor coalescing rules in `DescriptorBuilder::mergeDescriptors` should match the upstream copy. Using the plugin data directly avoids divergence.
- Our CUDA launch heuristics (`device/unpack_launch.cu`) differ from NCCL's warp scheduling. Reusing the original kernel maintains performance characteristics.

## 7. Recommended Next Steps
1. Add a shim header that exposes the plugin's internal structs (or include the originals) and update the test to stop guessing layouts.
2. Wrap `process_recv_cmsg` inside `tcpx_impl.cc` and delete the bespoke parser from the hot path.
3. Evaluate linking the reference unpack kernel and switching the test to call it.
4. Introduce a small polling helper so every caller follows the same completion pattern.
5. After the refactor, clean up redundant code paths (descriptor builder, parser) that are only needed for debug tooling.
