#!/usr/bin/env python3
"""TCPX local smoke test using the uccl_engine_tcpx C API via ctypes."""

from __future__ import annotations

import ctypes
import multiprocessing
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import torch

LIB_NAME = "libuccl_tcpx_engine.so"
LIB_PATH = Path(__file__).resolve().parent / LIB_NAME
if not LIB_PATH.exists():
    raise FileNotFoundError(
        f"{LIB_NAME} not found at {LIB_PATH}. Run 'make' in p2p/tcpx first."
    )

_lib = ctypes.CDLL(str(LIB_PATH))

# ctypes aliases
c_void_p = ctypes.c_void_p
c_char_p = ctypes.c_char_p
c_size_t = ctypes.c_size_t
c_uint64 = ctypes.c_uint64
c_int = ctypes.c_int
c_bool = ctypes.c_bool
c_uint8_p = ctypes.POINTER(ctypes.c_uint8)

# Function prototypes
_lib.uccl_engine_create.argtypes = [c_int, c_int]
_lib.uccl_engine_create.restype = c_void_p

_lib.uccl_engine_destroy.argtypes = [c_void_p]
_lib.uccl_engine_destroy.restype = None

_lib.uccl_engine_get_metadata.argtypes = [c_void_p, ctypes.POINTER(c_char_p)]
_lib.uccl_engine_get_metadata.restype = c_int

_lib.uccl_engine_free_endpoint_metadata.argtypes = [c_uint8_p]
_lib.uccl_engine_free_endpoint_metadata.restype = None

_lib.uccl_engine_accept.argtypes = [c_void_p, ctypes.c_void_p, c_size_t, ctypes.POINTER(c_int)]
_lib.uccl_engine_accept.restype = c_void_p

_lib.uccl_engine_connect.argtypes = [c_void_p, c_char_p, c_int, c_int]
_lib.uccl_engine_connect.restype = c_void_p

_lib.uccl_engine_conn_destroy.argtypes = [c_void_p]
_lib.uccl_engine_conn_destroy.restype = None

_lib.uccl_engine_write.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t, ctypes.POINTER(c_uint64)]
_lib.uccl_engine_write.restype = c_int

_lib.uccl_engine_recv.argtypes = [c_void_p, c_void_p, c_void_p, c_size_t]
_lib.uccl_engine_recv.restype = c_int

_lib.uccl_engine_xfer_status.argtypes = [c_void_p, c_uint64]
_lib.uccl_engine_xfer_status.restype = c_bool

# Environment defaults
os.environ.setdefault("UCCL_RCMODE", "1")
PLUGIN_PATH = os.environ.get("UCCL_TCPX_PLUGIN_PATH", "libnccl-net.so")
PLUGIN_DEV = os.environ.get("UCCL_TCPX_DEV", "0")
print(f"Using TCPX plugin: {PLUGIN_PATH}")
print(f"Using TCPX device: {PLUGIN_DEV}")


class TcpxEndpoint:
    """Minimal TCPX engine wrapper mirroring the RDMA Endpoint surface."""

    def __init__(self, local_gpu_idx: int = 0, num_cpus: int = 4) -> None:
        self._engine = _lib.uccl_engine_create(local_gpu_idx, num_cpus)
        if not self._engine:
            raise RuntimeError("uccl_engine_create failed")
        self._next_conn_id = 1
        self._conns: dict[int, c_void_p] = {}

    def close(self) -> None:
        for handle in list(self._conns.values()):
            _lib.uccl_engine_conn_destroy(handle)
        self._conns.clear()
        if self._engine:
            _lib.uccl_engine_destroy(self._engine)
            self._engine = None

    def __del__(self) -> None:  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def get_metadata(self) -> bytes:
        if not self._engine:
            raise RuntimeError("engine destroyed")
        buf = c_char_p()
        rc = _lib.uccl_engine_get_metadata(self._engine, ctypes.byref(buf))
        if rc != 0:
            raise RuntimeError(f"uccl_engine_get_metadata failed (rc={rc})")
        try:
            return ctypes.string_at(buf)
        finally:
            _lib.uccl_engine_free_endpoint_metadata(
                ctypes.cast(buf, c_uint8_p)
            )

    def _register_conn(self, conn_handle: c_void_p) -> int:
        conn_id = self._next_conn_id
        self._next_conn_id += 1
        self._conns[conn_id] = conn_handle
        return conn_id

    def accept(self) -> tuple[int, str, int]:
        if not self._engine:
            raise RuntimeError("engine destroyed")
        ip_buf = ctypes.create_string_buffer(64)
        remote_gpu = c_int()
        conn = _lib.uccl_engine_accept(
            self._engine, ip_buf, c_size_t(len(ip_buf)), ctypes.byref(remote_gpu)
        )
        if not conn:
            raise RuntimeError("uccl_engine_accept failed")
        conn_id = self._register_conn(conn)
        return conn_id, ip_buf.value.decode("ascii"), remote_gpu.value

    def connect(self, ip: str, remote_gpu_idx: int, remote_port: int) -> int:
        if not self._engine:
            raise RuntimeError("engine destroyed")
        conn = _lib.uccl_engine_connect(
            self._engine, ip.encode("ascii"), c_int(remote_gpu_idx), c_int(remote_port)
        )
        if not conn:
            raise RuntimeError("uccl_engine_connect failed")
        return self._register_conn(conn)

    def write_async(self, conn_id: int, ptr: int, size: int) -> int:
        handle = self._conns.get(conn_id)
        if not handle:
            raise KeyError(f"unknown conn_id {conn_id}")
        transfer_id = c_uint64()
        rc = _lib.uccl_engine_write(
            handle,
            c_void_p(),
            c_void_p(ptr),
            c_size_t(size),
            ctypes.byref(transfer_id),
        )
        if rc != 0:
            raise RuntimeError(f"uccl_engine_write failed (rc={rc})")
        return transfer_id.value

    def wait(self, conn_id: int, transfer_id: int, poll_interval: float = 0.001) -> None:
        handle = self._conns.get(conn_id)
        if not handle:
            raise KeyError(f"unknown conn_id {conn_id}")
        while True:
            done = _lib.uccl_engine_xfer_status(handle, c_uint64(transfer_id))
            if done:
                return
            time.sleep(poll_interval)

    def recv(self, conn_id: int, ptr: int, size: int) -> None:
        handle = self._conns.get(conn_id)
        if not handle:
            raise KeyError(f"unknown conn_id {conn_id}")
        rc = _lib.uccl_engine_recv(handle, c_void_p(), c_void_p(ptr), c_size_t(size))
        if rc != 0:
            raise RuntimeError(f"uccl_engine_recv failed (rc={rc})")

    def close_conn(self, conn_id: int) -> None:
        handle = self._conns.pop(conn_id, None)
        if handle:
            _lib.uccl_engine_conn_destroy(handle)


def parse_endpoint_meta(meta: bytes) -> Tuple[str, int, int]:
    """Return (ip, port, remote_gpu_idx)."""
    text = meta.decode("ascii")
    try:
        addr, gpu_str = text.split("?")
        ip, port_str = addr.split(":")
    except ValueError as exc:  # pragma: no cover
        raise ValueError(f"Unexpected metadata format: {text}") from exc
    return ip, int(port_str), int(gpu_str)


def test_local() -> None:
    print("Running TCPX local write test")
    meta_q: multiprocessing.Queue[bytes] = multiprocessing.Queue()
    finished_q: multiprocessing.Queue[None] = multiprocessing.Queue()

    def server_proc(ep_meta_q: multiprocessing.Queue[bytes],
                    done_q: multiprocessing.Queue[None]) -> None:
        ep_meta = ep_meta_q.get(timeout=10)
        ip, port, remote_gpu = parse_endpoint_meta(ep_meta)
        endpoint = TcpxEndpoint(local_gpu_idx=0, num_cpus=4)
        try:
            conn_id = endpoint.connect(ip, remote_gpu, port)
            tensor = torch.full((1024,), 42.0, dtype=torch.float32, device="cuda:0")
            transfer_id = endpoint.write_async(
                conn_id, tensor.data_ptr(), tensor.numel() * tensor.element_size()
            )
            endpoint.wait(conn_id, transfer_id)
            torch.cuda.synchronize()
            print(f"[Server] write done, sample={tensor[:4].tolist()}")
            endpoint.close_conn(conn_id)
        finally:
            endpoint.close()
        done_q.put(None)

    def client_proc(ep_meta_q: multiprocessing.Queue[bytes],
                    done_q: multiprocessing.Queue[None]) -> None:
        endpoint = TcpxEndpoint(local_gpu_idx=0, num_cpus=4)
        try:
            metadata = endpoint.get_metadata()
            ep_meta_q.put(metadata)
            conn_id, _, _ = endpoint.accept()
            tensor = torch.zeros(1024, dtype=torch.float32, device="cuda:0")
            endpoint.recv(conn_id, tensor.data_ptr(), tensor.numel() * tensor.element_size())
            torch.cuda.synchronize()
            print(f"[Client] received sample={tensor[:4].tolist()}")
            expected = torch.full_like(tensor, 42.0)
            if not torch.allclose(tensor, expected):
                raise RuntimeError("Data mismatch in TCPX write test")
            print("[Client] data verified")
            endpoint.close_conn(conn_id)
        finally:
            endpoint.close()
        done_q.put(None)

    server = multiprocessing.Process(target=server_proc, args=(meta_q, finished_q))
    client = multiprocessing.Process(target=client_proc, args=(meta_q, finished_q))

    server.start()
    time.sleep(1)
    client.start()

    finished_q.get(timeout=30)
    finished_q.get(timeout=30)

    server.join(timeout=5)
    client.join(timeout=5)

    if server.exitcode != 0:
        raise SystemExit(f"server failed with code {server.exitcode}")
    if client.exitcode != 0:
        raise SystemExit(f"client failed with code {client.exitcode}")

    print("Local TCPX write test passed\n")


if __name__ == "__main__":
    try:
        test_local()
    except KeyboardInterrupt:
        print("\nInterrupted, terminating")
        sys.exit(1)
